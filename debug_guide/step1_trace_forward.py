"""
STEP 1: Trace NaN through the forward pass
============================================

GOAL: Find which layer first produces NaN/Inf in a single forward pass.

STRATEGY:
  Run one forward pass under the SAME conditions as training
  (torch.amp.autocast("cuda") with float16), and check every
  intermediate tensor for NaN/Inf.

WHAT WE LEARNED:
  The forward pass is CLEAN -- no NaN in any stage with fresh weights.
  This rules out static issues (bad init, bad pretrained weights,
  numerically unstable functions like log/exp/softmax) and points
  toward a TRAINING DYNAMICS issue (something that appears after
  gradient updates, or is intermittent based on random inputs).

RUN:
  CUDA_VISIBLE_DEVICES=3 python debug_guide/step1_trace_forward.py
"""
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from engine import get_model


def check(name, tensor):
    """Print NaN/Inf stats for a tensor."""
    if tensor is None:
        print(f"  {name}: None")
        return False
    if not isinstance(tensor, torch.Tensor):
        print(f"  {name}: {tensor} (scalar)")
        return False
    has_nan = tensor.isnan().any().item()
    has_inf = tensor.isinf().any().item()
    flag = ""
    if has_nan:
        flag += " *** NaN ***"
    if has_inf:
        flag += " *** Inf ***"
    print(f"  {name}: shape={list(tensor.shape)} dtype={tensor.dtype} "
          f"min={tensor.min().item():.6g} max={tensor.max().item():.6g} "
          f"mean={tensor.float().mean().item():.6g}{flag}")
    return has_nan or has_inf


def main():
    cfg = OmegaConf.load("config/goose/ablation_best_combo.yaml")
    train_cfg = cfg.train

    # Load one batch of real data (small batch to avoid OOM)
    dataset = get_dataset(cfg.experiment_dataset, train_cfg.dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                        drop_last=True)
    samples = next(iter(loader))
    rgb = samples["rgb"].cuda()
    label = samples["label"].cuda()
    depth = samples["depth"].cuda() if "depth" in samples else None

    print("=== Input ===")
    check("rgb", rgb)
    check("label", label)

    model = get_model(model_name=cfg.model.name, **cfg.model.params).cuda()
    model.train()

    # ---- Step-by-step forward under autocast (same as training) ----
    with torch.amp.autocast("cuda"):

        # ---- Stage 1: Backbone (DAT-B) ----
        print("\n=== Stage 1: Backbone ===")
        if model.rgb_only:
            backbone_out = model.backbone(rgb)
        else:
            backbone_out = model.backbone(rgb, depth)
        for i, feat in enumerate(backbone_out):
            check(f"backbone[{i}]", feat)

        # ---- Stage 2: Neck (FPN) ----
        print("\n=== Stage 2: Neck (FPN) ===")
        neck_out = model.neck(backbone_out)
        for i, feat in enumerate(neck_out):
            check(f"neck[{i}]", feat)

        # ---- Stage 3: Multi-scale merging ----
        print("\n=== Stage 3: Merging ===")
        merged = model.merging(neck_out)
        x = merged[0]
        check("merged (x)", x)

        # ---- Stage 4: Diffusion embedding + noise ----
        print("\n=== Stage 4: Embedding table + diffusion noise ===")
        b, c, h, w = x.shape
        label_down = F.interpolate(
            label.float().unsqueeze(1), size=(h, w), mode="nearest"
        ).long()
        label_down[label_down == 255] = model.num_classes
        check("label_down", label_down)

        label_embed = model.embedding_table(label_down).squeeze(1).permute(0, 3, 1, 2)
        check("label_embed", label_embed)

        label_scaled = (torch.sigmoid(label_embed) * 2 - 1) * model.bit_scale
        check("label_scaled", label_scaled)

        times = torch.zeros((b,), device=x.device).float().uniform_(
            model.sample_range[0], model.sample_range[1])
        noise = torch.randn_like(label_scaled)
        noise_level = model.log_snr(times)
        check("noise_level (log_snr)", noise_level)

        from models.segmentor.diffusionmms import log_snr_to_alpha_sigma
        padded = model.right_pad_dims_to(rgb, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded)
        check("alpha", alpha)
        check("sigma", sigma)

        noised_gt = alpha * label_scaled + sigma * noise
        check("noised_gt", noised_gt)

        # ---- Stage 5: Transform conv (fuses features + noised label) ----
        print("\n=== Stage 5: Transform conv ===")
        feat = torch.cat([x, noised_gt], dim=1)
        feat = model.transform(feat)
        check("transform output", feat)

        # ---- Stage 6: Time MLP ----
        print("\n=== Stage 6: Time MLP ===")
        input_times = model.time_mlp(noise_level)
        check("time_mlp output", input_times)

        # ---- Stage 7: Decode head (6 transformer encoder layers) ----
        print("\n=== Stage 7: Decode head (layer-by-layer) ===")
        dh = model.decode_head
        # Prepare inputs (mirrors DeformableHeadWithTime.forward)
        bs_, c_, h_, w_ = feat.shape
        mask = torch.zeros((bs_, h_, w_), device=feat.device)
        pos_embed = dh.positional_encoding(mask)
        check("pos_embed", pos_embed)

        pos_flat = pos_embed.flatten(2).transpose(1, 2)
        feat_flat = feat.flatten(2).transpose(1, 2)
        spatial_shapes = torch.tensor([(h_, w_)], dtype=torch.long, device=feat.device)
        level_start = torch.zeros(1, dtype=torch.long, device=feat.device)
        ref_pts = dh.get_reference_points(spatial_shapes, device=feat.device)

        query = feat_flat.permute(1, 0, 2)
        key_pos = pos_flat.permute(1, 0, 2)

        for idx, layer in enumerate(dh.encoder.layers):
            query = layer(
                query, key=None, value=None, time=input_times,
                query_pos=key_pos,
                spatial_shapes=spatial_shapes,
                reference_points=ref_pts,
                level_start_index=level_start,
            )
            bad = check(f"encoder_layer[{idx}]", query)
            if bad:
                print(f"  >>> NaN first appears at encoder layer {idx}!")
                break

        # ---- Stage 8: Loss computation ----
        print("\n=== Stage 8: Full forward + loss ===")
        out = model.decode_head([feat], input_times)
        check("decode_head output", out)
        out_up = F.interpolate(out, size=rgb.shape[2:], mode="bilinear",
                               align_corners=False)
        check("interpolated output", out_up)

        loss = model.criterion(out_up.float(), label.long())
        check("loss_decode", loss)

    print("\n=== CONCLUSION ===")
    print("If no NaN/Inf above: forward pass is clean with fresh weights.")
    print("The NaN must come from training dynamics (gradient updates).")
    print("Proceed to step2_mini_training.py")


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
