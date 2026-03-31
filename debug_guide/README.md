# Debugging NaN Loss in Mixed-Precision Training

A step-by-step case study of diagnosing and fixing NaN loss in a diffusion-based
semantic segmentation model trained with PyTorch AMP.

## The Problem

```
CUDA_VISIBLE_DEVICES=3 python train.py --config config/goose/ablation_best_combo.yaml
```

Training immediately produces `total_loss = NaN`. This config combines three
changes from the baseline that individually work fine:

| Ablation Config       | Backbone     | Crop     | Class Weights | Works? |
|-----------------------|-------------|----------|---------------|--------|
| `ablation_baseline_v2`| single_dat_s | 480x640  | No            | Yes    |
| `ablation_backbone_b` | single_dat_b | 480x640  | No            | Yes    |
| `ablation_crop720`    | single_dat_s | 720x960  | No            | Yes    |
| `ablation_classweight`| single_dat_s | 480x640  | Yes           | Yes    |
| **ablation_best_combo** | **single_dat_b** | **720x960** | **Yes** | **NaN** |

## Debugging Methodology

### Principle: Isolate, don't guess

NaN can come from many sources. Rather than guessing and re-running the full
training each time (slow!), we write small targeted scripts that test one
hypothesis at a time.

### Step 1: Is the forward pass itself broken?

**Script:** [`step1_trace_forward.py`](step1_trace_forward.py)

**Hypothesis:** Some layer produces NaN/Inf on the very first forward pass
(e.g., bad initialization, numerically unstable function, pretrained weight
mismatch).

**Method:** Run a single forward pass under `torch.amp.autocast("cuda")`
(same as training), check every intermediate tensor.

**Result:** All clean. Every stage from backbone through loss outputs
finite values. **Forward pass is not the problem.**

### Step 2: Does NaN appear after gradient updates?

**Script:** [`step2_mini_training.py`](step2_mini_training.py)

**Hypothesis:** Gradients explode after a few weight updates, corrupting
the model and producing NaN on subsequent forward passes.

**Method:** Run 20 training steps with the full AMP pipeline. Track loss,
gradient norms, and GradScaler state.

**Result:**
```
Step 0: loss=6.857 grad_norm=inf scale=32768  NaN-GRAD (step skipped)
```

**Surprise!** Loss is finite (6.857) but ALL gradients are NaN on the
very first step. The model weights are clean -- only gradients are broken.

**Key insight:** The GradScaler multiplies the loss by 65,536 before backward:
```
scaled_loss = 6.857 * 65536 = 449,249
```
This exceeds float16's maximum value of 65,504, causing overflow during
the backward pass through float16 layers.

### Step 3: Is it just the scale factor?

**Script:** [`step3_test_scales.py`](step3_test_scales.py)

**Hypothesis:** A lower GradScaler `init_scale` will avoid the overflow.

**Method:** Test init_scale values of 65536, 1024, 128, 16, and 1.

**Result:**
```
init_scale=65536:  ALL NaN
init_scale=1024:   OK (but grad_norms 300-5000)
init_scale=128:    FAILS on some batches
init_scale=16:     OK (grad_norms 30-90)
init_scale=1:      OK (grad_norms 25-80)
```

**Surprise again!** `init_scale=1024` works but `init_scale=128` doesn't.
The failures are **stochastic** -- they depend on which data batches are
drawn. Some specific (data, random noise, random time) combinations
trigger float16 overflow during backward regardless of scale.

**Conclusion:** No `init_scale` value is reliably safe. Float16 itself is
the problem for this model configuration.

### Step 4: The fix -- bfloat16

**Script:** [`step4_bfloat16_solution.py`](step4_bfloat16_solution.py)

**Hypothesis:** bfloat16 has the same exponent range as float32 and
cannot overflow in the same way.

**Method:** Run the exact same 15 data batches with both float16 and
bfloat16. Fixed random seed for fair comparison.

**Result:**
```
float16 + GradScaler:  15/15 NaN steps (100% failure)
bfloat16 (no scaler):   0/15 NaN steps (100% success)
```

## Root Cause

```
float16 (IEEE 754 half):     1 sign + 5 exponent + 10 mantissa  →  max 65,504
bfloat16 (Brain Float):      1 sign + 8 exponent + 7 mantissa   →  max ~3.4e38
float32 (IEEE 754 single):   1 sign + 8 exponent + 23 mantissa  →  max ~3.4e38
```

The NaN is caused by **float16 overflow during the backward pass**:

1. The DAT-B backbone uses **gradient checkpointing** (`torch.utils.checkpoint`),
   which discards activations during forward and recomputes them during backward.

2. Under `torch.amp.autocast("cuda")` (default = float16), the recomputation
   produces intermediate values in float16.

3. The combination of:
   - **DAT-B** (larger backbone → higher-magnitude activations)
   - **720x960 crop** (more spatial elements → larger tensors)
   - **Class weights up to 5x** (amplified loss → larger gradients)

   creates conditions where backward-pass intermediates exceed float16's
   maximum of 65,504, producing Inf → NaN → cascading failure.

4. bfloat16 shares float32's exponent range (8 bits → max ~3.4e38),
   making overflow practically impossible. It trades mantissa precision
   (7 vs 10 bits) but neural networks are robust to small rounding errors.

## The Fix

```python
# engine/runner.py

# BEFORE (broken):
self.scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):                  # default = float16
    losses = self.model(rgb, depth, label)
self.scaler.scale(losses["total_loss"]).backward()
self.scaler.step(self.optimizer)
self.scaler.update()

# AFTER (fixed):
self.amp_dtype = torch.bfloat16                    # no GradScaler needed
with torch.amp.autocast("cuda", dtype=self.amp_dtype):
    losses = self.model(rgb, depth, label)
losses["total_loss"].backward()                    # direct backward
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
self.optimizer.step()
```

## Lessons Learned

1. **NaN does not always mean the loss function is wrong.** Here the loss was
   finite but the backward pass overflowed.

2. **GradScaler cannot fix fundamental float16 limitations.** It adjusts the
   scale to avoid underflow, but if the model's activations inherently exceed
   float16 range during backward, no scale will help.

3. **Individual ablations passing does not guarantee the combination passes.**
   DAT-B alone was fine. 720x960 alone was fine. Class weights alone were fine.
   The NaN only appeared when all three were combined.

4. **bfloat16 > float16 for training stability.** If your GPU supports it
   (Ampere+: A100, H100, RTX 30xx/40xx), prefer bfloat16. You get the speed
   benefits of mixed precision without the overflow risks, and you can drop
   GradScaler entirely.

5. **Debug systematically, not by re-running.** Each script above tests exactly
   one hypothesis and takes < 2 minutes to run, vs. the full training which
   takes hours. The total debug time was ~10 minutes.

