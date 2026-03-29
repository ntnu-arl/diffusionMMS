import torch.nn as nn
import torch

from models.common_layers import BaseDecodeHead, MultiScaleDeformableAttention
from .transformer import DetrTransformerEncoder, SinePositionalEncoding


class DeformableHeadWithTime(BaseDecodeHead):
    """Implements the DeformableEncoder.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self, num_feature_levels, **kwargs):

        super().__init__(input_transform="multiple_select", **kwargs)

        self.num_feature_levels = num_feature_levels
        self.encoder = DetrTransformerEncoder(
            num_layers=6,
            transformerlayers=dict(
                type="BaseTransformerLayerWithTime",
                use_time_mlp=True,
                attn_cfgs=dict(
                    type="MultiScaleDeformableAttention",
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.0,
                ),
                ffn_cfgs=dict(
                    type="FFN",
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.0,
                    act_cfg=dict(type="GELU"),
                ),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
        )
        self.positional_encoding = SinePositionalEncoding(
            num_feats=128, normalize=True, offset=-0.5
        )
        self.embed_dims = self.encoder.embed_dims

        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, inputs, times):

        mlvl_feats = inputs[-self.num_feature_levels :]

        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            feat = feat.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        reference_points = self.get_reference_points(spatial_shapes, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2
        )  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            time=times,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
        )
        memory = memory.permute(1, 2, 0)
        memory = memory.reshape(bs, c, h, w).contiguous()
        out = self.conv_seg(memory)
        return out


class MaskDecoder(DeformableHeadWithTime):
    def __init__(self, **kwargs):
        super(MaskDecoder, self).__init__(
            in_channels=[256],
            channels=256,
            in_index=[0],
            dropout_ratio=0.0,
            norm_cfg=dict(type="BN", requires_grad=True),
            align_corners=False,
            num_feature_levels=1,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
            **kwargs
        )


if __name__ == "__main__":
    net = DeformableHeadWithTime(
        in_channels=[256],
        channels=256,
        in_index=[0],
        dropout_ratio=0.0,
        num_classes=150,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        num_feature_levels=1,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    )
