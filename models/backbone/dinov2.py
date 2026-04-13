"""DINOv2 backbone wrapper for multi-scale feature extraction.

Loads a pretrained DINOv2 ViT via torch.hub and extracts intermediate
block features, reshaping and projecting them to produce a 4-level
feature pyramid compatible with the existing FPN neck.

Supported variants:
    dinov2_vits14  — ViT-S/14,  embed_dim=384,  12 blocks
    dinov2_vitb14  — ViT-B/14,  embed_dim=768,  12 blocks
    dinov2_vitl14  — ViT-L/14,  embed_dim=1024, 24 blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_root_logger

logger = get_root_logger()

# Default intermediate block indices for each variant (4 equally-spaced taps)
_DEFAULT_INDICES = {
    "dinov2_vits14": [2, 5, 8, 11],
    "dinov2_vitb14": [2, 5, 8, 11],
    "dinov2_vitl14": [5, 11, 17, 23],
}

# Output channel dims that match FPN in_channels expectations
_DEFAULT_OUT_CHANNELS = {
    "dinov2_vits14": [96, 192, 384, 384],
    "dinov2_vitb14": [128, 256, 512, 1024],
    "dinov2_vitl14": [256, 512, 1024, 1024],
}


class DINOv2Backbone(nn.Module):
    """DINOv2 backbone producing 4-level multi-scale features.

    Features are extracted from 4 intermediate transformer blocks and
    projected + rescaled to create a spatial pyramid:
        Level 0: 2x upsample  → effective stride ~7  (≈ stride 8)
        Level 1: keep          → effective stride 14  (≈ stride 16)
        Level 2: 2x downsample → effective stride 28  (≈ stride 32)
        Level 3: 2x downsample → effective stride 28  (≈ stride 32)
    """

    def __init__(
        self,
        model_name="dinov2_vitl14",
        out_indices=None,
        out_channels=None,
        frozen_stages=-1,
        freeze_backbone=False,
    ):
        super().__init__()
        self.model_name = model_name

        # Load pretrained DINOv2
        self.dinov2 = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )
        self.embed_dim = self.dinov2.embed_dim
        self.patch_size = self.dinov2.patch_size
        self.out_indices = out_indices or _DEFAULT_INDICES[model_name]
        out_channels = out_channels or _DEFAULT_OUT_CHANNELS[model_name]

        logger.info(
            f"DINOv2 backbone: {model_name}, embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}, out_indices={self.out_indices}, "
            f"out_channels={out_channels}"
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            logger.info("DINOv2 backbone frozen")

        # Scale adapters: create spatial pyramid from uniform-resolution features
        # Level 0: upsample 2x (stride 14 → ~7, approximates stride 8)
        self.adapt0 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, out_channels[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels[0]),
            nn.GELU(),
        )
        # Level 1: keep spatial resolution (stride 14, approximates stride 16)
        self.adapt1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels[1], kernel_size=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.GELU(),
        )
        # Level 2: downsample 2x (stride 14 → 28, approximates stride 32)
        self.adapt2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.GELU(),
        )
        # Level 3: downsample 2x (stride 14 → 28, approximates stride 32)
        self.adapt3 = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_channels[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.GELU(),
        )
        self._adapters = nn.ModuleList([self.adapt0, self.adapt1, self.adapt2, self.adapt3])

    def init_weights(self, pretrained=None):
        """DINOv2 weights are loaded in __init__ via torch.hub. This is a
        no-op to satisfy the backbone interface (called by DiffusionMMS)."""
        logger.info("DINOv2 pretrained weights already loaded via torch.hub")

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: RGB tensor [B, 3, H, W]

        Returns:
            List of 4 feature maps at approximate strides [8, 16, 32, 32]
        """
        # get_intermediate_layers with reshape=True returns spatial feature maps
        features = self.dinov2.get_intermediate_layers(
            x, n=self.out_indices, reshape=True
        )
        # features: tuple of 4 tensors, each [B, embed_dim, H/14, W/14]

        outs = []
        for i, feat in enumerate(features):
            outs.append(self._adapters[i](feat))

        return outs


def dinov2_vits14(**kwargs):
    return DINOv2Backbone(model_name="dinov2_vits14", **kwargs)


def dinov2_vitb14(**kwargs):
    return DINOv2Backbone(model_name="dinov2_vitb14", **kwargs)


def dinov2_vitl14(**kwargs):
    return DINOv2Backbone(model_name="dinov2_vitl14", **kwargs)
