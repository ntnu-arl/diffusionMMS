"""EUPE backbone wrapper (ViT and ConvNeXt variants).

Requires the EUPE repo to be cloned locally. Set the path via the
``eupe_repo_path`` parameter or the ``EUPE_REPO_PATH`` environment variable.
Default: ``/tmp/EUPE``.

Clone:  git clone https://github.com/facebookresearch/EUPE /tmp/EUPE

Supported variants:
    ViT:
        eupe_vits16  — ViT-S/16,  embed_dim=384,  12 blocks, 21M params
        eupe_vitb16  — ViT-B/16,  embed_dim=768,  12 blocks, 86M params
    ConvNeXt (natively multi-scale — no adapter layers needed):
        eupe_convnext_tiny  — dims=[96,192,384,768],   29M params
        eupe_convnext_small — dims=[96,192,384,768],   50M params
        eupe_convnext_base  — dims=[128,256,512,1024], 89M params
"""

import os
import sys

import torch
import torch.nn as nn
from utils.logger import get_root_logger

logger = get_root_logger()

_DEFAULT_EUPE_PATH = "/tmp/EUPE"


def _ensure_eupe_importable(repo_path: str | None = None):
    """Add EUPE repo to sys.path if not already importable."""
    try:
        import eupe  # noqa: F401
        return
    except ImportError:
        pass
    repo_path = repo_path or os.environ.get("EUPE_REPO_PATH", _DEFAULT_EUPE_PATH)
    if not os.path.isdir(repo_path):
        raise RuntimeError(
            f"EUPE repo not found at {repo_path}. "
            "Clone it: git clone https://github.com/facebookresearch/EUPE /tmp/EUPE"
        )
    sys.path.insert(0, repo_path)
    logger.info(f"Added EUPE repo to sys.path: {repo_path}")


# ---------- ViT variants (need spatial adapters, like DINOv2) ----------

_VIT_OUT_INDICES = {
    "eupe_vits16": [2, 5, 8, 11],
    "eupe_vitb16": [2, 5, 8, 11],
}

_VIT_OUT_CHANNELS = {
    "eupe_vits16": [96, 192, 384, 384],
    "eupe_vitb16": [128, 256, 512, 1024],
}


class EUPEViTBackbone(nn.Module):
    """EUPE ViT backbone with spatial adapters for FPN compatibility.

    Produces a 4-level feature pyramid from intermediate ViT blocks:
        Level 0: 2x upsample  → effective stride ~8
        Level 1: keep          → effective stride 16
        Level 2: 2x downsample → effective stride 32
        Level 3: 2x downsample → effective stride 32
    """

    def __init__(
        self,
        model_name="eupe_vitb16",
        out_indices=None,
        out_channels=None,
        freeze_backbone=False,
        weights=None,
        eupe_repo_path=None,
    ):
        super().__init__()
        _ensure_eupe_importable(eupe_repo_path)
        from eupe.hub.backbones import (
            eupe_vits16 as _eupe_vits16,
            eupe_vitb16 as _eupe_vitb16,
        )

        factory = {"eupe_vits16": _eupe_vits16, "eupe_vitb16": _eupe_vitb16}
        kwargs = {"pretrained": True}
        if weights is not None:
            kwargs["weights"] = weights
        self.vit = factory[model_name](**kwargs)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_size
        self.out_indices = out_indices or _VIT_OUT_INDICES[model_name]
        out_channels = out_channels or _VIT_OUT_CHANNELS[model_name]

        logger.info(
            f"EUPE ViT backbone: {model_name}, embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}, out_indices={self.out_indices}, "
            f"out_channels={out_channels}"
        )

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            logger.info("EUPE ViT backbone frozen")

        # Spatial adapters (same approach as DINOv2 backbone)
        self.adapt0 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, out_channels[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels[0]),
            nn.GELU(),
        )
        self.adapt1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels[1], kernel_size=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.GELU(),
        )
        self.adapt2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.GELU(),
        )
        self.adapt3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.GELU(),
        )
        self._adapters = nn.ModuleList([self.adapt0, self.adapt1, self.adapt2, self.adapt3])

    def init_weights(self, pretrained=None):
        logger.info("EUPE ViT pretrained weights already loaded")

    def forward(self, x):
        features = self.vit.get_intermediate_layers(
            x, n=self.out_indices, reshape=True
        )
        return [self._adapters[i](feat) for i, feat in enumerate(features)]


# ---------- ConvNeXt variants (natively multi-scale) ----------

class EUPEConvNeXtBackbone(nn.Module):
    """EUPE ConvNeXt backbone — natively produces multi-scale features.

    Output strides: [4, 8, 16, 32] with per-stage channel dims.
    ConvNeXt-Base dims [128, 256, 512, 1024] match the DAT-B FPN config
    directly, requiring no adapter layers.
    """

    def __init__(
        self,
        model_name="eupe_convnext_base",
        freeze_backbone=False,
        weights=None,
        eupe_repo_path=None,
    ):
        super().__init__()
        _ensure_eupe_importable(eupe_repo_path)
        from eupe.hub.backbones import (
            eupe_convnext_tiny as _cnx_t,
            eupe_convnext_small as _cnx_s,
            eupe_convnext_base as _cnx_b,
        )

        factory = {
            "eupe_convnext_tiny": _cnx_t,
            "eupe_convnext_small": _cnx_s,
            "eupe_convnext_base": _cnx_b,
        }
        kwargs = {"pretrained": True}
        if weights is not None:
            kwargs["weights"] = weights
        self.convnext = factory[model_name](**kwargs)
        self.embed_dims = self.convnext.embed_dims
        self.out_indices = [0, 1, 2, 3]

        logger.info(
            f"EUPE ConvNeXt backbone: {model_name}, "
            f"dims={self.embed_dims}, strides=[4, 8, 16, 32]"
        )

        if freeze_backbone:
            for param in self.convnext.parameters():
                param.requires_grad = False
            logger.info("EUPE ConvNeXt backbone frozen")

    def init_weights(self, pretrained=None):
        logger.info("EUPE ConvNeXt pretrained weights already loaded")

    def forward(self, x):
        features = self.convnext.get_intermediate_layers(
            x, n=self.out_indices, reshape=True
        )
        # features: tuple of 4 tensors at strides [4, 8, 16, 32]
        return list(features)


# ---------- Factory functions ----------

def eupe_vits16(**kwargs):
    return EUPEViTBackbone(model_name="eupe_vits16", **kwargs)

def eupe_vitb16(**kwargs):
    return EUPEViTBackbone(model_name="eupe_vitb16", **kwargs)

def eupe_convnext_tiny(**kwargs):
    return EUPEConvNeXtBackbone(model_name="eupe_convnext_tiny", **kwargs)

def eupe_convnext_small(**kwargs):
    return EUPEConvNeXtBackbone(model_name="eupe_convnext_small", **kwargs)

def eupe_convnext_base(**kwargs):
    return EUPEConvNeXtBackbone(model_name="eupe_convnext_base", **kwargs)
