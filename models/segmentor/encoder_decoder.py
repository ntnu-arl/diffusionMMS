import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_root_logger
from utils.init_func import init_weight
from models import get_backbone, get_decoder

logger = get_root_logger()


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        backbone,
        decoder,
        aux_head=None,
        criterion=nn.CrossEntropyLoss(reduction="mean", ignore_index=255),
        norm_layer="BatchNorm2d",
        pretrained=None,
        train_cfg=None,
        eval=False,
    ):
        super(EncoderDecoder, self).__init__()

        self.backbone = get_backbone(backbone.name, **backbone.params)
        self.decoder = get_decoder(decoder.name, **decoder.params)
        if aux_head is not None:
            self.aux_head = get_decoder(aux_head.name, **aux_head.params)
        else:
            self.aux_head = None
        self.train_cfg = train_cfg
        if norm_layer == "BatchNorm2d":
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == "SyncBN":
            self.norm_layer = nn.SyncBatchNorm
        else:
            logger.error("unsupported batchnorm layer")

        self.criterion = criterion
        if not eval:
            self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        logger.info("Initing weights ...")
        init_weight(
            self.decoder,
            nn.init.kaiming_normal_,
            self.norm_layer,
            self.train_cfg.bn_eps,
            self.train_cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.aux_head:
            init_weight(
                self.aux_head,
                nn.init.kaiming_normal_,
                self.norm_layer,
                self.train_cfg.bn_eps,
                self.train_cfg.bn_momentum,
                mode="fan_in",
                nonlinearity="relu",
            )

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decoder.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.train_cfg.aux_index])
            aux_fm = F.interpolate(
                aux_fm, size=orisize[2:], mode="bilinear", align_corners=False
            )
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.train_cfg.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out
