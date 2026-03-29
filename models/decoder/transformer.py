import copy
import math
import warnings
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_layers import (
    build_norm_layer,
    build_activation,
    FFN,
    MultiScaleDeformableAttention,
)


def _build_attention(cfg):
    """Build an attention module from a config dict."""
    cfg = cfg.copy()
    attn_type = cfg.pop("type")
    if attn_type == "MultiScaleDeformableAttention":
        return MultiScaleDeformableAttention(**cfg)
    raise ValueError(f"Unknown attention type: {attn_type}")


def _build_ffn(cfg):
    """Build a feed-forward network from a config dict."""
    cfg = cfg.copy()
    ffn_type = cfg.pop("type")
    if ffn_type == "FFN":
        return FFN(**cfg)
    raise ValueError(f"Unknown FFN type: {ffn_type}")


class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
    """

    def __init__(
        self,
        num_feats,
        temperature=10000,
        normalize=False,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
        init_cfg=None,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (
                (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            )
            x_embed = (
                (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
            )
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats={self.num_feats}, "
        repr_str += f"temperature={self.temperature}, "
        repr_str += f"normalize={self.normalize}, "
        repr_str += f"scale={self.scale}, "
        repr_str += f"eps={self.eps})"
        return repr_str


class BaseTransformerLayerWithTime(nn.Module):
    """Base `TransformerLayer` for vision transformer.

    Supports configurable self/cross attention, FFN, and norm layers
    via operation_order, with optional time-conditioned modulation.

    Args:
        attn_cfgs: Config dicts for attention modules.
        ffn_cfgs: Config dict(s) for FFN modules.
        use_time_mlp (bool): Whether to apply time-conditioned modulation.
        operation_order (tuple[str]): Execution order, e.g.
            ('self_attn', 'norm', 'ffn', 'norm').
        norm_cfg (dict): Config for normalization layers.
        batch_first (bool): If True, tensors are (B, N, C); else (N, B, C).
    """

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        use_time_mlp=False,
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):

        deprecated_args = dict(
            feedforward_channels="feedforward_channels",
            ffn_dropout="ffn_drop",
            ffn_num_fcs="num_fcs",
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. ",
                    DeprecationWarning,
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__()

        self.batch_first = batch_first

        assert set(operation_order) & {"self_attn", "norm", "ffn", "cross_attn"} == set(
            operation_order
        ), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        num_attn = operation_order.count("self_attn") + operation_order.count(
            "cross_attn"
        )
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        self.use_time_mlp = use_time_mlp

        self.time_mlp = (
            nn.Sequential(  # [2, 1024]
                nn.SiLU(),
                nn.Linear(
                    ffn_cfgs["embed_dims"] * 4, ffn_cfgs["embed_dims"] * 2
                ),  # [2, 512]
            )
            if use_time_mlp
            else None
        )

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                attention = _build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims
            self.ffns.append(_build_ffn(ffn_cfgs[ffn_index]))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(
        self,
        query,
        key=None,
        value=None,
        time=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        if self.time_mlp is not None and time is not None:
            time = self.time_mlp(time)  # [2, 1024] -> [2, 512]
            time = rearrange(time, "b c -> 1 b c")  # [2, 512] -> [1, 2, 512]
            scale, shift = time.chunk(2, dim=2)  # [1, 2, 256] * 2
            query = query * (scale + 1) + shift

        return query


class TransformerLayerSequence(nn.Module):
    """A sequence of transformer layers (replaces mmcv version).

    Builds ``num_layers`` copies of the layer described by
    ``transformerlayers`` config dict.  State-dict key prefix: ``layers.{i}.…``
    """

    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            cfg = copy.deepcopy(transformerlayers)
            layer_type = cfg.pop("type")
            if layer_type == "BaseTransformerLayerWithTime":
                self.layers.append(BaseTransformerLayerWithTime(**cfg))
            else:
                raise ValueError(f"Unknown transformer layer type: {layer_type}")
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        return query


class DetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type="LN"), **kwargs):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = (
                build_norm_layer(post_norm_cfg, self.embed_dims)[1]
                if self.pre_norm
                else None
            )
        else:
            assert not self.pre_norm, (
                f"Use prenorm in "
                f"{self.__class__.__name__},"
                f"Please specify post_norm_cfg"
            )
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x
