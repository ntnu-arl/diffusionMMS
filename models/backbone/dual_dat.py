import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn

from ..dat_utils.dat_blocks import *
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
from ..backbone.dat import LayerNormProxy, TransformerStage
from utils.logger import get_root_logger
from utils.misc import load_dual_dat_pretrained_model
logger = get_root_logger()


class Dual_DAT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 4, 18, 2],
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.5,
                 strides=[8, 4, 2, 1],
                 offset_range_factor=[-1, -1, -1, -1],
                 local_orf=[-1, -1, -1, -1],
                 local_kv_sizes=[-1, -1, -1, -1],
                 offset_pes=[False, False, False, False],
                 stage_spec=[
                     ["N", "D"],
                     ["N", "D", "N", "D"],
                     ["N", "D", "N", "D", "N", "D", "N", "D", "N",
                      "D", "N", "D", "N", "D", "N", "D", "N", "D"],
                     ["D", "D"]],
                 groups=[1, 2, 3, 6],
                 use_pes=[True, True, True, True],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[True, True, True, True],
                 use_conv_patches=True,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 deform_groups=[0, 0, 0, 0],
                 nat_ksizes=[7, 7, 7, 7],
                 layer_scale_values=[-1, -1, -1, -1],
                 use_lpus=[True, True, True, True],
                 use_cmt_mlps=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=True,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.dims = dims
        self.num_heads = heads
        self.out_indices = out_indices

        self.log_cpb = log_cpb[0]
        self.dwc_pe = dwc_pes[0]
        self.slide = stage_spec[0][0] == "E"
        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(nn.Conv2d(3, dim_stem, patch_size, patch_size, 0), LayerNormProxy(dim_stem))
        self.patch_proj_d = nn.Sequential(
            nn.Conv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(nn.Conv2d(3, dim_stem, patch_size, patch_size, 0), LayerNormProxy(dim_stem))

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.stages_d = nn.ModuleList()
        self.norms_d = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    local_orf[i], local_kv_sizes[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i],
                    ksize_qnas[i],
                    nqs[i],
                    qna_activation,
                    deform_groups[i],
                    layer_scale_values[i],
                    use_lpus[i],
                    use_cmt_mlps[i],
                    log_cpb[i],
                    i, use_checkpoint
                )
            )
            if i in self.out_indices:
                self.norms.append(
                    LayerNormProxy(dim2)
                )
            else:
                self.norms.append(nn.Identity())
            self.stages_d.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    local_orf[i], local_kv_sizes[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i],
                    ksize_qnas[i],
                    nqs[i],
                    qna_activation,
                    deform_groups[i],
                    layer_scale_values[i],
                    use_lpus[i],
                    use_cmt_mlps[i],
                    log_cpb[i],
                    i, use_checkpoint
                )
            )
            if i in self.out_indices:
                self.norms_d.append(
                    LayerNormProxy(dim2)
                )
            else:
                self.norms_d.append(nn.Identity())

            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        self.down_projs_d = nn.ModuleList()
        self.FRMs = self.build_FRM()
        self.FFMs = self.build_FFM()

        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
            self.down_projs_d.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.lower_lr_kvs = lower_lr_kvs
        self.pretrained = pretrained
        self.reset_parameters()

    def build_FRM(self):
        layers = nn.ModuleList()
        for i in range(4):
            layer = FRM(self.dims[i], reduction=1)
            layers.append(layer)

        return layers

    def build_FFM(self):
        layers = nn.ModuleList()
        for i in range(4):
            layer = FFM(
                dim=int(self.dims[i]),
                reduction=1,
                num_heads=self.num_heads[i]
            )
            layers.append(layer)

        return layers

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def init_weights(self, pretrained):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

        if isinstance(pretrained, str):
            load_dual_dat_pretrained_model(self, pretrained)
            logger.info("DAT backbone has been loaded successfully!")

    def forward(self, x, x_d):
        x = self.patch_proj(x)
        x_d = self.patch_proj_d(x_d)

        outs = []
        for i in range(4):
            x = self.stages[i](x)
            x_d = self.stages_d[i](x_d)
            x, x_d = self.FRMs[i](x.contiguous(), x_d.contiguous())
            y = self.norms[i](x)
            y_d = self.norms_d[i](x_d)
            out = self.FFMs[i](y.contiguous(), y_d.contiguous())
            outs.append(out.contiguous())
            if i < 3:
                x = self.down_projs[i](x)
                x_d = self.down_projs_d[i](x_d)

        return outs


class dual_dat_s(Dual_DAT):
    def __init__(self, img_size=224, patch_size=4, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 4, 18, 2],
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.5,
                 strides=[8, 4, 2, 1],
                 offset_range_factor=[-1, -1, -1, -1],
                 local_orf=[-1, -1, -1, -1],
                 local_kv_sizes=[-1, -1, -1, -1],
                 offset_pes=[False, False, False, False],
                 stage_spec=[
                     ["N", "D"],
                     ["N", "D", "N", "D"],
                     ["N", "D", "N", "D", "N", "D", "N", "D", "N",
                      "D", "N", "D", "N", "D", "N", "D", "N", "D"],
                     ["D", "D"]],
                 groups=[1, 2, 3, 6],
                 use_pes=[True, True, True, True],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[True, True, True, True],
                 use_conv_patches=True,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 deform_groups=[0, 0, 0, 0],
                 nat_ksizes=[7, 7, 7, 7],
                 layer_scale_values=[-1, -1, -1, -1],
                 use_lpus=[True, True, True, True],
                 use_cmt_mlps=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=True,
                 pretrained=None,
                 **kwargs):
        super().__init__(img_size, patch_size,  expansion, dim_stem, dims, depths, heads, heads_q, window_sizes, drop_rate, attn_drop_rate, drop_path_rate, strides, offset_range_factor, local_orf, local_kv_sizes, offset_pes, stage_spec, groups, use_pes, dwc_pes,
                         sr_ratios, lower_lr_kvs, fixed_pes, no_offs, ns_per_pts, use_dwc_mlps, use_conv_patches, ksizes, ksize_qnas, nqs, qna_activation, deform_groups, nat_ksizes, layer_scale_values, use_lpus, use_cmt_mlps, log_cpb, out_indices, use_checkpoint, pretrained, **kwargs)


class dual_dat_b(Dual_DAT):
    def __init__(self, img_size=224, patch_size=4, expansion=4,
                 dim_stem=128, dims=[128, 256, 512, 1024], depths=[2, 4, 18, 2],
                 heads=[4, 8, 16, 32], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.7,
                 strides=[8, 4, 2, 1],
                 offset_range_factor=[-1, -1, -1, -1],
                 local_orf=[-1, -1, -1, -1],
                 local_kv_sizes=[-1, -1, -1, -1],
                 offset_pes=[False, False, False, False],
                 stage_spec=[["N", "D"],
                             ["N", "D", "N", "D"],
                             ["N", "D", "N", "D", "N", "D", "N", "D", "N",
                              "D", "N", "D", "N", "D", "N", "D", "N", "D"],
                             ["D", "D"]],
                 groups=[2, 4, 8, 16],
                 use_pes=[True, True, True, True],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[True, True, True, True],
                 use_conv_patches=True,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 deform_groups=[0, 0, 0, 0],
                 nat_ksizes=[7, 7, 7, 7],
                 layer_scale_values=[-1, -1, -1, -1],
                 use_lpus=[True, True, True, True],
                 use_cmt_mlps=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=True,
                 pretrained=None,
                 **kwargs):
        super().__init__(img_size, patch_size, expansion, dim_stem, dims, depths, heads, heads_q, window_sizes, drop_rate, attn_drop_rate, drop_path_rate, strides, offset_range_factor, local_orf, local_kv_sizes, offset_pes, stage_spec, groups, use_pes, dwc_pes,
                         sr_ratios, lower_lr_kvs, fixed_pes, no_offs, ns_per_pts, use_dwc_mlps, use_conv_patches, ksizes, ksize_qnas, nqs, qna_activation, deform_groups, nat_ksizes, layer_scale_values, use_lpus, use_cmt_mlps, log_cpb, out_indices, use_checkpoint, pretrained, **kwargs)


class dual_dat_t(Dual_DAT):
    def __init__(self, img_size=224, patch_size=4, expansion=4,
                 dim_stem=64, dims=[64, 128, 256, 512], depths=[2, 4, 18, 2],
                 heads=[2, 4, 8, 16], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.3,
                 strides=[8, 4, 2, 1],
                 offset_range_factor=[-1, -1, -1, -1],
                 local_orf=[-1, -1, -1, -1],
                 local_kv_sizes=[-1, -1, -1, -1],
                 offset_pes=[False, False, False, False],
                 stage_spec=[
                     ["N", "D"],
                     ["N", "D", "N", "D"],
                     ["N", "D", "N", "D", "N", "D", "N", "D", "N",
                      "D", "N", "D", "N", "D", "N", "D", "N", "D"],
                     ["D", "D"]],
                 groups=[1, 2, 4, 8],
                 use_pes=[True, True, True, True],
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1],
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[True, True, True, True],
                 use_conv_patches=True,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 deform_groups=[0, 0, 0, 0],
                 nat_ksizes=[7, 7, 7, 7],
                 layer_scale_values=[-1, -1, -1, -1],
                 use_lpus=[True, True, True, True],
                 use_cmt_mlps=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=True,
                 pretrained=None,
                 **kwargs):
        super().__init__(img_size, patch_size, expansion, dim_stem, dims, depths, heads, heads_q, window_sizes, drop_rate, attn_drop_rate, drop_path_rate, strides, offset_range_factor, local_orf, local_kv_sizes, offset_pes, stage_spec, groups, use_pes, dwc_pes,
                         sr_ratios, lower_lr_kvs, fixed_pes, no_offs, ns_per_pts, use_dwc_mlps, use_conv_patches, ksizes, ksize_qnas, nqs, qna_activation, deform_groups, nat_ksizes, layer_scale_values, use_lpus, use_cmt_mlps, log_cpb, out_indices, use_checkpoint, pretrained, **kwargs)
