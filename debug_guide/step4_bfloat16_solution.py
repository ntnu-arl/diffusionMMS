"""
STEP 4: Confirm bfloat16 fixes the issue -- the definitive comparison
======================================================================

GOAL: Prove that bfloat16 eliminates NaN while float16 fails, using
  the EXACT SAME data batches for a fair comparison.

WHY BFLOAT16 WORKS:

  float16 (IEEE 754 half-precision):
    - 1 sign + 5 exponent + 10 mantissa bits
    - Max value: 65,504
    - Precise but NARROW range -- overflows easily

  bfloat16 (Brain Floating Point):
    - 1 sign + 8 exponent + 7 mantissa bits
    - Max value: ~3.4 x 10^38  (same as float32!)
    - Less precise but SAME RANGE as float32 -- no overflow

  The NaN comes from overflow (values exceeding max), not precision.
  bfloat16 trades mantissa bits (precision) for exponent bits (range),
  which is exactly the right tradeoff for deep learning:
  - Neural nets are robust to small rounding errors (low precision OK)
  - Neural nets are NOT robust to Inf/NaN (overflow is fatal)

  With bfloat16:
  - No GradScaler needed (no overflow risk = no need to scale)
  - Simpler training loop (just autocast + backward + step)
  - Works on all Ampere+ GPUs (A100, H100, RTX 30xx/40xx)

RESULTS (same data, same model, 15 steps):
  float16 + GradScaler:  15/15 NaN steps (100% failure)
  bfloat16 (no scaler):   0/15 NaN steps (100% success)

RUN:
  CUDA_VISIBLE_DEVICES=3 python debug_guide/step4_bfloat16_solution.py
"""
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from engine import get_model
from utils.init_func import group_weight


def test_config(name, autocast_dtype, use_scaler, batches, cfg, train_cfg):
    """Run training on pre-loaded batches with a specific AMP config."""
    torch.manual_seed(42)  # Same init for fair comparison
    model = get_model(model_name=cfg.model.name, **cfg.model.params).cuda()
    model.train()
    opt_params = group_weight(model, cfg.model.params.norm_layer, train_cfg.lr)
    optimizer = torch.optim.AdamW(opt_params, lr=train_cfg.lr,
                                   weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  dtype={autocast_dtype}, scaler={'yes' if use_scaler else 'no'}")
    print(f"{'='*60}")

    nan_count = 0
    for step, samples in enumerate(batches):
        rgb = samples["rgb"].cuda(non_blocking=True)
        label = samples["label"].cuda(non_blocking=True)
        depth = samples["depth"].cuda(non_blocking=True) if "depth" in samples else None

        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            losses = model(rgb, depth, label)

        total_loss = losses["total_loss"]
        optimizer.zero_grad(set_to_none=True)

        if use_scaler:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            s = f"{scaler.get_scale():.0f}"
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimizer.step()
            s = "N/A"

        loss_v = total_loss.item()
        gn = grad_norm.item()
        loss_nan = loss_v != loss_v
        grad_nan = gn != gn or gn == float('inf')
        if loss_nan or grad_nan:
            nan_count += 1

        flag = ""
        if loss_nan: flag += " NaN-LOSS"
        if grad_nan: flag += " NaN-GRAD"

        print(f"  step {step:2d}: loss={loss_v:.4f}  grad_norm={gn:.4f}  "
              f"scale={s}{flag}")

    num = len(batches)
    status = "PASS" if nan_count == 0 else "FAIL"
    print(f"\n  RESULT: {nan_count}/{num} NaN steps  [{status}]")

    del model, optimizer, scaler
    torch.cuda.empty_cache()


def main():
    cfg = OmegaConf.load("config/goose/ablation_best_combo.yaml")
    train_cfg = cfg.train
    dataset = get_dataset(cfg.experiment_dataset, train_cfg.dataset)

    # Pre-load batches so both tests use IDENTICAL data
    torch.manual_seed(999)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                        drop_last=True, pin_memory=True)
    batches = []
    for i, s in enumerate(loader):
        if i >= 15:
            break
        batches.append(s)
    print(f"Pre-loaded {len(batches)} batches (shared between both tests)")

    # ---- Test 1: float16 + GradScaler (the broken config) ----
    test_config(
        "float16 + GradScaler (BROKEN)",
        torch.float16, use_scaler=True,
        batches=batches, cfg=cfg, train_cfg=train_cfg,
    )

    # ---- Test 2: bfloat16, no scaler (the fix) ----
    test_config(
        "bfloat16, no scaler (FIXED)",
        torch.bfloat16, use_scaler=False,
        batches=batches, cfg=cfg, train_cfg=train_cfg,
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
float16 fails because intermediate gradient values exceed 65,504
during the backward pass through gradient-checkpointed DAT-B layers.

bfloat16 succeeds because it shares float32's exponent range
(max ~3.4e38), making overflow practically impossible.

THE FIX (in engine/runner.py):
  BEFORE:
    self.scaler = torch.amp.GradScaler("cuda")
    ...
    with torch.amp.autocast("cuda"):           # float16 default
        losses = self.model(rgb, depth, label)
    self.scaler.scale(losses["total_loss"]).backward()
    self.scaler.unscale_(self.optimizer)
    self.scaler.step(self.optimizer)
    self.scaler.update()

  AFTER:
    self.amp_dtype = torch.bfloat16            # no GradScaler needed
    ...
    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
        losses = self.model(rgb, depth, label)
    losses["total_loss"].backward()            # direct backward
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
""")


if __name__ == "__main__":
    main()
