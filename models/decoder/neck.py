import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_layers import ConvModule, FPN


class MultiStageMerging(nn.Module):
    def __init__(
        self,
        in_channels=[256, 256, 256, 256],
        out_channels=256,
        kernel_size=1,
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=None,
        align_corners=False,
        init_cfg=None,
        **kwargs
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.down = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = list()
        size = inputs[0].shape[2:]
        for index, input in enumerate(inputs):
            input = F.interpolate(
                input, size=size, mode="bilinear", align_corners=self.align_corners
            )
            outs.append(input)
        out = torch.cat(outs, dim=1)
        out = self.down(out)
        return [out]


class FeaturePyramidNetwork(FPN):
    def __init__(self, in_channels=[96, 192, 384, 768], **kwargs):
        super(FeaturePyramidNetwork, self).__init__(
            in_channels=list(in_channels),
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type="GN", num_groups=32),
            num_outs=4,
        )
