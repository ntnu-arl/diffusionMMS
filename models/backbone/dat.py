import math
import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple
from utils.logger import get_root_logger
from models.dat_utils.dat_blocks import *
from models.dat_utils.nat import NeighborhoodAttention2D
from models.dat_utils.slide import SlideAttention

import torch.utils.checkpoint as cp
logger = get_root_logger()

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.fp16_enabled = False

    def forward(self, x):
        B, C, H, W = x.size()
        gamma = self.gamma[None, :, None, None].expand(B, C, H, W)
        return x * gamma


class TransformerStage(nn.Module):
    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 local_orf, local_kv_size,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate,
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation, deform_groups,
                 layer_scale_value,
                 use_lpu, use_cmt_mlp, log_cpb,
                 stage_i,
                 use_checkpoint, prompt_tuning_config=None):

        super().__init__()
        self.fp16_enabled = False
        self.use_checkpoint = use_checkpoint
        fmap_size = to_2tuple(fmap_size)
        local_kv_size = to_2tuple(local_kv_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1,
                              0) if dim_in != dim_embed else nn.Identity()
        self.use_lpu = use_lpu
        self.stage_spec = stage_spec

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed)
             for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] !=
             'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLP
        if use_dwc_mlp:
            if use_cmt_mlp:
                mlp_fn = TransformerMLPWithConv_CMT
            else:
                mlp_fn = TransformerMLPWithConv

        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(
                    dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1,
                          padding=1, groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size,
                                   attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(
                        fmap_size,
                        fmap_size,
                        heads,
                        hc,
                        n_groups,
                        attn_drop,
                        proj_drop,
                        stride,
                        offset_range_factor,
                        use_pe,
                        dwc_pe,
                        no_off,
                        fixed_pe,
                        ksize,
                        log_cpb,
                        stage_i
                    )
                )

            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(
                        dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(
                        dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'A':
                self.attns.append(
                    LDABaseline(fmap_size, local_kv_size, heads, hc,
                                n_groups, use_pe, no_off, local_orf)
                )
            elif stage_spec[i] == 'P':
                self.attns.append(
                    PyramidAttention(dim_embed, heads,
                                     attn_drop, proj_drop, sr_ratio)
                )
            elif self.stage_spec[i] == 'X':
                self.attns.append(
                    nn.Conv2d(dim_embed, dim_embed, kernel_size=window_size,
                              padding=window_size // 2, groups=dim_embed)
                )
            elif self.stage_spec[i] == 'E':
                self.attns.append(
                    SlideAttention(dim_embed, heads, 3)
                )
            else:
                raise NotImplementedError(
                    f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(
                DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def _inner_forward(self, x):
        x = self.proj(x)

        for d in range(self.depths):
            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x.contiguous())
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x

    def forward(self, x):
        if self.training and x.requires_grad and self.use_checkpoint:
            return cp.checkpoint(self._inner_forward, x)
        else:
            return self._inner_forward(x)

