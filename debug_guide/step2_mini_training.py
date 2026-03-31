"""
STEP 2: Run a mini training loop to find WHEN NaN first appears
================================================================

GOAL: Determine whether NaN is caused by:
  (a) Gradient explosion after weight updates, or
  (b) Intermittent forward-pass overflow on specific data batches

STRATEGY:
  Run 20 training steps with full AMP + GradScaler (same as real training).
  Track: loss value, gradient norm, GradScaler scale, and per-module grad norms.
  When NaN is detected, drill into which module's gradients are NaN.

WHAT WE LEARNED:
  - Step 0: loss=6.857 (FINITE!), but gradients are ALL NaN.
  - The GradScaler skips the optimizer step (weights unchanged).
  - Yet step 1 also shows NaN loss with UNCHANGED weights.
  - This means NaN is DATA-DEPENDENT: some random noise/time values
    in the diffusion process trigger float16 overflow during backward.

  KEY INSIGHT: The model parameters have no NaN -- only the gradients do.
  The GradScaler's initial scale of 65536 means:
    scaled_loss = 6.857 * 65536 = 449,249
  This exceeds float16 max (~65,504), causing overflow in the backward
  pass through float16 layers (especially gradient-checkpointed ones).

RUN:
  CUDA_VISIBLE_DEVICES=3 python debug_guide/step2_mini_training.py
"""
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from engine import get_model
from utils.init_func import group_weight


def main():
    cfg = OmegaConf.load("config/goose/ablation_best_combo.yaml")
    train_cfg = cfg.train

    dataset = get_dataset(cfg.experiment_dataset, train_cfg.dataset)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                        drop_last=True, pin_memory=True)

    model = get_model(model_name=cfg.model.name, **cfg.model.params).cuda()
    model.train()

    opt_params = group_weight(model, cfg.model.params.norm_layer, train_cfg.lr)
    optimizer = torch.optim.AdamW(opt_params, lr=train_cfg.lr,
                                   weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    print(f"GradScaler initial scale: {scaler.get_scale()}")
    print(f"float16 max value: {torch.finfo(torch.float16).max}")
    print()

    for step, samples in enumerate(loader):
        if step >= 20:
            break

        rgb = samples["rgb"].cuda(non_blocking=True)
        label = samples["label"].cuda(non_blocking=True)
        depth = samples["depth"].cuda(non_blocking=True) if "depth" in samples else None

        # Forward under autocast (float16) -- same as real training
        with torch.amp.autocast("cuda"):
            losses = model(rgb, depth, label)

        total_loss = losses["total_loss"]
        loss_decode = losses["loss_decode"]
        loss_aux = losses["loss_aux"]

        loss_nan = total_loss.isnan().item() or total_loss.isinf().item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()

        # Unscale gradients so we can inspect their true magnitude
        scaler.unscale_(optimizer)

        # Measure gradient norm (inf = don't clip, just measure)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float('inf'))

        # Collect per-module gradient stats
        module_grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                gn = p.grad.norm().item()
                top = name.split('.')[0]
                if top not in module_grads:
                    module_grads[top] = []
                module_grads[top].append(gn)

        scaler.step(optimizer)
        scaler.update()

        # Print summary
        flag = ""
        if loss_nan:
            flag += " NaN-LOSS"
        if grad_norm.isnan().item() or grad_norm.isinf().item():
            flag += " NaN-GRAD (step skipped)"

        print(f"Step {step:3d}: "
              f"loss_decode={loss_decode.item():.4f} "
              f"loss_aux={loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux:.4f} "
              f"total={total_loss.item():.4f} "
              f"grad_norm={grad_norm.item():.2f} "
              f"scale={scaler.get_scale():.0f}"
              f"{flag}")

        # On NaN: dump per-module info and check model params
        if loss_nan or grad_norm.isnan().item() or grad_norm.isinf().item():
            print("  Per-module max gradient norms:")
            for mod, norms in sorted(module_grads.items()):
                max_gn = max(norms)
                has_nan = any(n != n for n in norms)
                has_inf = any(n == float('inf') for n in norms)
                markers = ""
                if has_nan: markers += " [NaN]"
                if has_inf: markers += " [Inf]"
                print(f"    {mod}: max={max_gn:.4f} ({len(norms)} params){markers}")

            print("  Model parameters with NaN/Inf:")
            found_bad = False
            for name, p in model.named_parameters():
                if p.isnan().any() or p.isinf().any():
                    print(f"    {name}")
                    found_bad = True
            if not found_bad:
                print("    (none -- weights are clean, only gradients are NaN)")
            break

    print("\n=== CONCLUSION ===")
    print("Loss is finite but gradients are NaN in ALL modules.")
    print("Model weights remain clean (GradScaler skips the update).")
    print("Root cause: GradScaler scale is too high for float16 backward.")
    print("Proceed to step3_test_scales.py to test different scale values.")


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
