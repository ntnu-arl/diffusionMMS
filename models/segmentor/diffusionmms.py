import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import expm1
import math
from models import get_backbone, get_decoder, get_neck
from einops import rearrange, repeat
from utils.logger import get_root_logger
from models.common_layers import ConvModule
from models.decoder.neck import MultiStageMerging

logger = get_root_logger()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class DiffusionMMS(nn.Module):
    def __init__(
        self,
        backbone,
        decoder,
        neck=None,
        aux_head=None,
        bit_scale=0.1,
        timesteps=1,
        randsteps=1,
        time_difference=1,
        learned_sinusoidal_dim=16,
        sample_range=(0, 0.999),
        noise_schedule="cosine",
        diffusion="ddim",
        accumulation=False,
        norm_layer="BatchNorm2d",
        criterion=nn.CrossEntropyLoss(reduction="mean", ignore_index=255),
        pretrained=None,
        train_cfg=None,
        eval=False,
        **kwargs,
    ):
        super(DiffusionMMS, self).__init__()
        self.backbone = get_backbone(backbone.name, **backbone.params)
        self.neck = get_neck(neck.name, **neck.params)
        self.merging = MultiStageMerging()
        self.decode_head = get_decoder(decoder.name, **decoder.params)
        self.num_classes = decoder.params.num_classes

        if aux_head is not None:
            self.aux_head = get_decoder(aux_head.name, **aux_head.params)
        else:
            self.aux_head = None
        if norm_layer == "BatchNorm2d":
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == "SyncBN":
            self.norm_layer = nn.SyncBatchNorm
        else:
            logger.error("unsupported batchnorm layer")

        self.criterion = criterion
        self.train_cfg = train_cfg

        # Diffusion parameters
        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.accumulation = accumulation

        self.embedding_table = nn.Embedding(
            self.num_classes + 1, self.decode_head.in_channels[0]
        )

        logger.info(
            f" timesteps: {timesteps},"
            f" randsteps: {randsteps},"
            f" sample_range: {sample_range},"
            f" diffusion: {diffusion}"
        )

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

        self.transform = ConvModule(
            self.decode_head.in_channels[0] * 2,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
        )

        # time embeddings
        time_dim = self.decode_head.in_channels[0] * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim),  # [2, 1024]
        )

        if not eval:
            self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        logger.info("Initing weights ...")

    def extract_feat(self, rgb, depth):
        x = self.backbone(rgb, depth)
        x_outs = self.neck(x)
        x_out = self.merging(x_outs)

        return x_out[0]

    def sampling(self, rgb, depth):
        orisize = rgb.shape
        x = self.extract_feat(rgb, depth)

        if self.diffusion == "ddim":
            out = self.ddim_sample(x)
        elif self.diffusion == "ddpm":
            out = self.ddpm_sample(x)
        out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)
        return out

    def forward_train(self, rgb, depth, label):
        x = self.extract_feat(rgb, depth)

        b, c, h, w, device = *x.shape, x.device
        label_down = F.interpolate(
            label.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        label_down = label_down.long()
        label_down[label_down == 255] = self.num_classes

        label_down = self.embedding_table(label_down).squeeze(1).permute(0, 3, 1, 2)

        label_down = (torch.sigmoid(label_down) * 2 - 1) * self.bit_scale

        # Sample time
        times = (
            torch.zeros((b,), device=device)
            .float()
            .uniform_(self.sample_range[0], self.sample_range[1])
        )

        # random noise
        noise = torch.randn_like(label_down)
        noise_level = self.log_snr(times)

        padded_noise_level = self.right_pad_dims_to(rgb, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * label_down + sigma * noise

        # conditional input
        feat = torch.cat([x, noised_gt], dim=1)
        feat = self.transform(feat)

        input_times = self.time_mlp(noise_level)

        out = self.decode_head([feat], input_times)
        out = F.interpolate(
            out, size=rgb.shape[2:], mode="bilinear", align_corners=False
        )

        if self.aux_head:
            aux_fm = self.aux_head(x)
            aux_fm = F.interpolate(
                aux_fm, size=rgb.shape[2:], mode="bilinear", align_corners=False
            )
            return out, aux_fm
        return out

    def forward(self, rgb, depth, label):
        if self.aux_head:
            out, aux_fm = self.forward_train(rgb, depth, label)
        else:
            out = self.forward_train(rgb, depth, label)
        losses = dict()
        if label is not None:
            losses["loss_decode"] = self.criterion(out, label.long())
            if self.aux_head:
                losses["loss_aux"] = self.train_cfg.aux_rate * self.criterion(
                    aux_fm, label.long()
                )
            else:
                losses["loss_aux"] = 0
            losses["total_loss"] = losses["loss_aux"] + losses["loss_decode"]
            return losses
        return out

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(
                1
                - (step + 1 + self.time_difference)
                / self.timesteps
                * (1 - self.sample_range[0]),
                self.sample_range[0],
            )
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, "t -> t b", b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_sample(self, x):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = repeat(x, "b c h w -> (r b) c h w", r=self.randsteps)
        mask_t = torch.randn(
            (self.randsteps, self.decode_head.in_channels[0], h, w), device=device
        )
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = self.decode_head([feat], input_times)  # [bs, 150, ]

            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next

            if self.accumulation:
                outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit

    @torch.no_grad()
    def ddpm_sample(self, x):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        #
        x = repeat(x, "b c h w -> (r b) c h w", r=self.randsteps)
        mask_t = torch.randn(
            (self.randsteps, self.decode_head.in_channels[0], h, w), device=device
        )
        outs = list()
        for times_now, times_next in time_pairs:
            feat = torch.cat([x, mask_t], dim=1)

            feat = self.transform(feat)

            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = self.decode_head([feat], input_times)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (mask_t * (1 - c) / alpha + c * mask_pred)
            variance = (sigma_next**2) * c
            log_variance = log(variance)

            noise = torch.where(
                rearrange(times_next > 0, "b -> b 1 1 1"),
                torch.randn_like(mask_t),
                torch.zeros_like(mask_t),
            )
            mask_t = mean + (0.5 * log_variance).exp() * noise

            if self.accumulation:
                outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from engine import get_model

    cfg = OmegaConf.load("config/nyuv2/ddp_baseline.yaml")
    model = get_model(cfg.model.name, **cfg.model.params).cuda()
    h = 480
    w = 640
    bs = 1
    out = model.forward_train(
        torch.ones(bs, 3, h, w).cuda(), torch.ones(bs, h, w).long().cuda()
    )
