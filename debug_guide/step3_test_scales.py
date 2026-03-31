"""
STEP 3: Test different GradScaler init_scale values
=====================================================

GOAL: Determine if the NaN is caused by the GradScaler's loss scaling
  being too large for float16, or a fundamental float16 limitation.

STRATEGY:
  Create a FRESH model for each init_scale value and run 10 training steps.
  If lower scales work, the issue is the scale factor.
  If NO scale works reliably, float16 itself is the problem.

WHAT WE LEARNED:

  init_scale=65536 (default):  ALL NaN -- scale is way too high
  init_scale=1024:             Works! grad_norms 300-5000 (high but finite)
  init_scale=128:              FAILS on some batches -- borderline
  init_scale=16:               Works! grad_norms 30-90
  init_scale=1:                Works! grad_norms 25-80

  CRITICAL OBSERVATION: init_scale=1024 works but init_scale=128 doesn't.
  This seems contradictory -- but the DataLoader is shared (shuffle=True),
  so each test sees DIFFERENT data batches. The NaN is STOCHASTIC.

  This means the issue is NOT just the scale factor. Some specific
  combinations of (data, random noise, random time) produce float16
  overflow during backward, regardless of scale. The scale just
  determines whether the overflow is common or rare.

  For production training with thousands of iterations, even rare
  NaN events will eventually corrupt training.

  CONCLUSION: float16 is fundamentally unreliable for this model.
  Proceed to step4_bfloat16_solution.py

RUN:
  CUDA_VISIBLE_DEVICES=3 python debug_guide/step3_test_scales.py
"""
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from engine import get_model
from utils.init_func import group_weight


def test_scale(init_scale, loader, cfg, train_cfg, num_steps=10):
    """Run a mini training loop with a specific GradScaler init_scale."""
    model = get_model(model_name=cfg.model.name, **cfg.model.params).cuda()
    model.train()
    opt_params = group_weight(model, cfg.model.params.norm_layer, train_cfg.lr)
    optimizer = torch.optim.AdamW(opt_params, lr=train_cfg.lr,
                                   weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)

    print(f"\n{'='*60}")
    print(f"  init_scale = {init_scale}")
    print(f"{'='*60}")

    nan_steps = 0
    for step, samples in enumerate(loader):
        if step >= num_steps:
            break

        rgb = samples["rgb"].cuda(non_blocking=True)
        label = samples["label"].cuda(non_blocking=True)
        depth = samples["depth"].cuda(non_blocking=True) if "depth" in samples else None

        with torch.amp.autocast("cuda"):
            losses = model(rgb, depth, label)

        total_loss = losses["total_loss"]
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_v = total_loss.item()
        gn = grad_norm.item()
        s = scaler.get_scale()
        is_bad = (loss_v != loss_v) or (gn != gn) or (gn == float('inf'))
        if is_bad:
            nan_steps += 1

        flag = ""
        if loss_v != loss_v:
            flag += " NaN-LOSS"
        if gn != gn or gn == float('inf'):
            flag += " NaN-GRAD (skipped)"

        print(f"  step {step}: loss={loss_v:.4f}  grad_norm={gn:.4f}  "
              f"scale={s:.0f}{flag}")

    print(f"  RESULT: {nan_steps}/{num_steps} NaN steps "
          f"({'FAIL' if nan_steps > 0 else 'OK'})")

    del model, optimizer, scaler
    torch.cuda.empty_cache()


def main():
    cfg = OmegaConf.load("config/goose/ablation_best_combo.yaml")
    train_cfg = cfg.train
    dataset = get_dataset(cfg.experiment_dataset, train_cfg.dataset)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                        drop_last=True, pin_memory=True)

    print("Testing different GradScaler init_scale values with float16 AMP")
    print(f"  float16 max = {torch.finfo(torch.float16).max}")
    print(f"  Expected loss ~ 5-7")
    print(f"  scaled_loss = loss * init_scale  (must not overflow float16 in backward)")

    for init_scale in [65536, 1024, 128, 16, 1]:
        test_scale(init_scale, loader, cfg, train_cfg, num_steps=10)

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("""
Some scales work, some don't -- but the failures are STOCHASTIC
(different data batches). This means float16 is fundamentally
unreliable for this model's backward pass.

Why? The DAT-B backbone uses gradient checkpointing, which
re-runs the forward pass during backward. Under float16, the
re-computation can produce intermediate values that overflow
the limited float16 exponent range (max ~65504).

The combination of:
  1. DAT-B (large backbone with high-magnitude activations)
  2. 720x960 crop (more spatial elements)
  3. Class weights (amplify loss by up to 5x)
  4. Gradient checkpointing (re-computation in float16)
creates conditions where float16 overflow is inevitable.

SOLUTION: Use bfloat16 instead of float16.
See step4_bfloat16_solution.py
""")


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
