# DiffusionMMS - GOOSE Dataset Training Instructions

## Training

### Full training run

```bash
# DAT-Small
python train.py --config config/goose/goose_dat_s_epoch_100.yaml

# DAT-Tiny
python train.py --config config/goose/goose_dat_t_epoch_100.yaml

# DAT-Base
python train.py --config config/goose/goose_dat_b_epoch_100.yaml
```

### Quick dry run

Copy a config and override these values for a fast test:

| Setting | Default | Quick test |
|---|---|---|
| `train.num_epochs` | 100 | 3 |
| `train.eval_last_n_epochs` | 10 | 2 |
| `train.batch_size` | 4 | 2 |
| `train.num_workers` | 16 | 4 |
| `train.warmup_iter` | 1500 | 10 |

```bash
cp config/goose/goose_dat_s_epoch_100.yaml config/goose/goose_dat_s_dryrun.yaml
# Edit the copy with the values above, then:
python train.py --config config/goose/goose_dat_s_dryrun.yaml
```

## Evaluation

### Standalone evaluation (range of epochs)

```bash
python eval.py --config config/goose/goose_dat_s_epoch_100.yaml --fr 91 --to 100
```

### Single epoch evaluation with visualization

```bash
python test.py --config config/goose/goose_dat_s_epoch_100.yaml --epoch 100 --show
```

## Output structure

After training, outputs are saved to:

```
output_dir/goose/<experiment_name>/
  checkpoint-1.pth
  checkpoint-2.pth
  ...
  checkpoint-100.pth
  best.pt              # symlink to the best checkpoint (by mIoU)
  eval_results.txt     # per-epoch mIoU + per-class IoU for evaluated epochs
```

## Monitoring

```bash
tensorboard --logdir log/
```

Logged metrics:
- `train_loss` (per epoch)
- `lr` (per epoch)
- `val_mIoU` (during evaluated epochs)

## Configuration reference

Key config parameters:

- `model.params.backbone.name`: `single_dat_t`, `single_dat_s`, or `single_dat_b`
- `model.params.pretrained`: path to pretrained DAT weights
- `model.params.decoder.params.num_classes`: 64 (GOOSE classes)
- `train.eval_last_n_epochs`: number of final epochs to evaluate during training (default: 10)
- `train.dataset.common_transforms.semseg_transform.params.crop_size`: training crop size `[H, W]`

## Dataset

GOOSE dataset expected at `data/goose_dataset/` with structure:

```
data/goose_dataset/
  images/
    train/   (sequence dirs with *_vis.png, *_realsense.png, *_front.png, *_camera_left.png)
    val/
  labels/
    train/   (sequence dirs with *_labelids.png)
    val/
```

- 64 semantic classes (class 0 = "undefined", mapped to ignore)
- RGB camera types used: windshield_vis, realsense, front, camera_left
- Train: 11,234 image-label pairs | Val: 1,369 pairs

## Training optimizations

The following optimizations are applied to reduce training time. No config changes are needed; they are all enabled by default.

### Automatic Mixed Precision (AMP)

**Files:** `engine/runner.py`, `engine/evaluator.py`

The forward pass and loss computation run under `torch.amp.autocast("cuda")`, which automatically uses float16 for eligible operations (convolutions, linear layers, attention) while keeping float32 for numerically sensitive operations (loss, normalization). A `torch.amp.GradScaler` handles loss scaling during the backward pass to prevent underflow in float16 gradients. Evaluation inference also runs under autocast.

**Expected speedup:** 20-40%, with lower GPU memory usage.

### cuDNN auto-tuner

**File:** `engine/runner.py`

`torch.backends.cudnn.benchmark = True` is set before training. This makes cuDNN benchmark different convolution algorithms on the first iteration and cache the fastest one. Since all training inputs have the same spatial size after cropping, the cached algorithm is reused every subsequent iteration.

### Faster gradient zeroing

**File:** `engine/runner.py`

`optimizer.zero_grad(set_to_none=True)` sets gradient tensors to `None` instead of filling them with zeros. This avoids a memset kernel launch per parameter and allows the allocator to reuse the memory.

### Non-blocking GPU transfers

**Files:** `engine/runner.py`, `engine/evaluator.py`

All `.to(device, non_blocking=True)` and `.cuda(non_blocking=True)` calls overlap the CPU-to-GPU data transfer with other work. This is effective because the DataLoader uses `pin_memory=True` (see below), which allocates host tensors in page-locked memory that the GPU can DMA from asynchronously.

### DataLoader optimizations

**File:** `train.py`

Three flags are added to the training DataLoader:

| Flag | Effect |
|---|---|
| `pin_memory=True` | Allocates batch tensors in pinned (page-locked) CPU memory, enabling faster and asynchronous GPU transfers. |
| `persistent_workers=True` | Keeps worker processes alive between epochs, avoiding the overhead of forking and re-initializing workers every epoch. |
| `prefetch_factor=2` | Each worker pre-loads 2 batches ahead, so the next batch is ready when the GPU finishes the current one. |

The validation DataLoader (`engine/evaluator.py`) also uses `pin_memory=True`.

### Reduced logging frequency

**File:** `engine/runner.py`

Training loss is logged every `log_interval` iterations (default: 50) instead of every iteration. This eliminates per-iteration string formatting, implicit `.item()` GPU synchronization for printing, and I/O to the log file. The full loss is still logged at the end of every epoch. Configurable via `train.log_interval` in the YAML config.

### Detached loss accumulation

**File:** `engine/runner.py`

Loss values are `.detach()`-ed before being accumulated into the running sum. This ensures the computation graph is freed immediately after the backward pass, rather than being held alive by the accumulator dictionary.

### Eliminated GPU synchronization in deformable attention

**File:** `models/common_layers.py`

The `multi_scale_deformable_attn_pytorch` function called `.item()` on GPU tensors (spatial shapes) inside a per-level loop, causing a CPU-GPU synchronization point on every attention call. The spatial shapes are now moved to CPU once with `.cpu().tolist()` before the loop, replacing N synchronization points with one.

### Bug fix: random_mirror augmentation

**File:** `datasets/transform.py`

The `random_mirror` function had a bug where it always returned the original `kwargs` dict even when a flip was performed (the flipped `res` dict was computed but discarded). This meant horizontal flip augmentation was never applied during training. The fix returns `res` when a flip occurs, enabling proper data augmentation and improving model generalization.
