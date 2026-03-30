"""
Pure PyTorch replacements for mmcv/mmseg/mmengine modules.

All classes produce state_dict keys identical to the original mmcv/mmseg
implementations, ensuring checkpoint compatibility.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm / Activation builders
# ---------------------------------------------------------------------------

def build_norm_layer(cfg, num_features):
    """Build a normalization layer from a config dict.

    Returns:
        tuple: (name, layer) where name is the abbreviated type string
        used as the module attribute name (matching mmcv convention).
    """
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    requires_grad = cfg.pop("requires_grad", True)

    name_map = {"LN": "ln", "BN": "bn", "GN": "gn", "SyncBN": "bn"}
    if layer_type == "LN":
        layer = nn.LayerNorm(num_features, **cfg)
    elif layer_type == "BN":
        layer = nn.BatchNorm2d(num_features, **cfg)
    elif layer_type == "GN":
        layer = nn.GroupNorm(num_channels=num_features, **cfg)
    elif layer_type == "SyncBN":
        layer = nn.SyncBatchNorm(num_features, **cfg)
    else:
        raise ValueError(f"Unsupported norm type: {layer_type}")

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return (name_map[layer_type], layer)


def build_activation(cfg):
    """Build an activation layer from a config dict."""
    if cfg is None:
        return None
    cfg = cfg.copy()
    act_type = cfg.pop("type")
    if act_type == "ReLU":
        return nn.ReLU(**cfg)
    elif act_type == "GELU":
        return nn.GELU()
    elif act_type == "SiLU":
        return nn.SiLU(**cfg)
    raise ValueError(f"Unsupported activation type: {act_type}")


# ---------------------------------------------------------------------------
# ConvModule  (replaces mmcv.cnn.ConvModule)
# ---------------------------------------------------------------------------

class ConvModule(nn.Module):
    """Conv + optional Norm + optional Activation.

    State-dict compatible with mmcv.cnn.ConvModule:
    - conv stored as ``self.conv``
    - norm stored under abbreviated name (``self.gn``, ``self.bn``, …)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        order=("conv", "norm", "act"),
        **kwargs,
    ):
        super().__init__()
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        if bias == "auto":
            bias = not self.with_norm

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if self.with_norm:
            norm_channels = (
                out_channels if order.index("norm") > order.index("conv") else in_channels
            )
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        if self.with_activation:
            self.activate = build_activation(act_cfg)

    def forward(self, x):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and self.with_norm:
                x = getattr(self, self.norm_name)(x)
            elif layer == "act" and self.with_activation:
                x = self.activate(x)
        return x


# ---------------------------------------------------------------------------
# FFN  (replaces mmcv.cnn.bricks.transformer.FFN)
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Feed-forward network used in transformer layers.

    State-dict keys: ``layers.0.0.{weight,bias}``, ``layers.1.{weight,bias}``
    (for default ``num_fcs=2``).
    """

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.add_identity = add_identity

        activation = build_activation(act_cfg)

        layers = []
        in_ch = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(nn.Linear(in_ch, feedforward_channels), activation, nn.Dropout(ffn_drop))
            )
            in_ch = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (pure PyTorch, replaces mmcv.ops)
# ---------------------------------------------------------------------------

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure PyTorch implementation of multi-scale deformable attention core."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    # Move spatial shapes to CPU once to avoid per-level GPU sync
    spatial_shapes_cpu = value_spatial_shapes.cpu().tolist()

    value_list = value.split(
        [H * W for H, W in spatial_shapes_cpu], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(spatial_shapes_cpu):
        # (bs, H_*W_, num_heads, embed_dims) -> (bs*num_heads, embed_dims, H_, W_)
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # (bs, num_queries, num_heads, num_points, 2) -> (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # (bs, num_queries, num_heads, num_levels, num_points)
    # -> (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention (pure PyTorch).

    State-dict keys match ``mmcv.ops.MultiScaleDeformableAttention``:
    ``sampling_offsets``, ``attention_weights``, ``value_proj``, ``output_proj``.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        spatial_shapes = kwargs["spatial_shapes"]
        reference_points = kwargs["reference_points"]

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


# ---------------------------------------------------------------------------
# BaseDecodeHead  (replaces mmseg.models.decode_heads.decode_head.BaseDecodeHead)
# ---------------------------------------------------------------------------

class BaseDecodeHead(nn.Module):
    """Minimal base class for decode heads.

    Provides ``conv_seg``, ``in_channels``, ``channels``, ``num_classes``
    with state-dict keys matching mmseg's ``BaseDecodeHead``.
    """

    def __init__(
        self,
        in_channels,
        channels,
        num_classes=None,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        in_index=-1,
        input_transform=None,
        loss_decode=None,
        ignore_index=255,
        align_corners=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels if isinstance(in_channels, (list, tuple)) else [in_channels]
        self.channels = channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners

        if num_classes is not None:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None


# ---------------------------------------------------------------------------
# FPN  (replaces mmseg.models.necks.FPN)
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    """Feature Pyramid Network.

    State-dict keys match mmseg's FPN:
    ``lateral_convs.{i}.conv.weight``, ``lateral_convs.{i}.gn.weight``, …
    ``fpn_convs.{i}.conv.weight``, ``fpn_convs.{i}.gn.weight``, …
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level + 1

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                act_cfg=act_cfg,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_ch = in_channels[self.backbone_end_level - 1] if i == 0 else out_channels
                self.fpn_convs.append(
                    ConvModule(in_ch, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg
            )

        # Build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(self.lateral_convs))]

        # Extra levels (max-pool when no extra convs)
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)
