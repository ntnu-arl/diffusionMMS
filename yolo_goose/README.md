# YOLO26-seg for GOOSE Challenge

YOLO26-seg (Ultralytics) adaptation for the GOOSE 2D Fine-Grained Semantic Segmentation challenge (CodaBench ICRA 2026).

## Setup

```bash
pip install ultralytics
```

## 1. Convert Dataset

Converts GOOSE semantic masks to YOLO instance segmentation polygon format. Creates symlinks for images and generates `.txt` label files.

```bash
python yolo_goose/convert_dataset.py --goose_root data/goose_dataset --output data/goose_yolo --workers 8
```

## 2. Train

```bash
# Single GPU (large model, competition quality)
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 200 --batch 8 --device 3

# Nano model for quick iteration
python yolo_goose/train.py --model yolo26n-seg.pt --imgsz 640 --epochs 100 --batch 16 --device 3

# Extra-large model
python yolo_goose/train.py --model yolo26x-seg.pt --imgsz 1024 --epochs 200 --batch 4 --device 3

# Multi-GPU
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 200 --batch 8 --device 0,1,2,3

# Resume from checkpoint
python yolo_goose/train.py --resume runs/segment/goose_yolo26l-seg/weights/last.pt
```

## 3. Validate (GOOSE competition metrics)

Computes mIoU_fine (56 classes), mIoU_coarse (11 superclasses), and mIoU_composite.

```bash
python yolo_goose/validate.py --weights runs/segment/goose_yolo26l-seg/weights/best.pt --imgsz 1024 --device 3
```

## 4. Generate Competition Submission

```bash
# Basic inference
python yolo_goose/inference.py \
    --weights runs/segment/goose_yolo26l-seg/weights/best.pt \
    --output submission_yolo/ \
    --imgsz 1024 --device 3

# With test-time augmentation
python yolo_goose/inference.py \
    --weights runs/segment/goose_yolo26l-seg/weights/best.pt \
    --output submission_yolo/ \
    --imgsz 1280 --tta --device 3

# Zip for upload
cd submission_yolo && zip -r ../submission_yolo.zip . && cd ..
```

## Available Model Sizes

| Model | Params | Notes |
|-------|--------|-------|
| `yolo26n-seg.pt` | ~3M | Fast iteration / debugging |
| `yolo26s-seg.pt` | ~11M | Light |
| `yolo26m-seg.pt` | ~22M | Medium |
| `yolo26l-seg.pt` | ~46M | Recommended |
| `yolo26x-seg.pt` | ~86M | Best accuracy, slower |

## Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--imgsz` | 1024 | Training image size |
| `--epochs` | 200 | Training epochs |
| `--batch` | 4 | Batch size |
| `--lr0` | 1e-3 | Initial learning rate |
| `--cos_lr` | off | Cosine LR scheduler (flag) |
| `--patience` | 50 | Early stopping patience (0=disable) |
| `--close_mosaic` | 20 | Disable mosaic for last N epochs |
| `--mask_ratio` | 4 | Mask downsample ratio |

## Hyperparameter Tuning Guide

All parameters below are passed directly to `train.py` (which forwards them to `model.train()`).
The `train.py` script already exposes the most common ones as CLI flags. For any parameter
not exposed, add it to the `model.train(...)` call in `train.py`.

### Priority 1 — Model & Resolution (biggest gains)

| Param | Default | Try | Why |
|-------|---------|-----|-----|
| `--model` | yolo26l-seg.pt | yolo26x-seg.pt | Bigger model = better on 64 classes. Try x if VRAM allows |
| `--imgsz` | 1024 | 1280 | GOOSE images are 2048x1000; higher res preserves small objects/classes |
| `--mask_ratio` | 4 | 1 or 2 | Mask head downsamples by this factor; lower = finer masks (costs VRAM) |

```bash
# Example: max resolution with large model
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1280 --batch 4 --device 3 --mask_ratio 2
```

### Priority 2 — Training Schedule

| Param | Default | Try | Why |
|-------|---------|-----|-----|
| `--epochs` | 200 | 200–300 | 11k images, 64 classes needs longer convergence |
| `--cos_lr` | off | on | Cosine annealing often beats linear decay |
| `--lr0` | 1e-3 | 0.01 or 1e-3 | Default YOLO is 0.01 (SGD); for AdamW try 1e-3 |
| `--lrf` | 0.01 | 0.001–0.01 | Final LR = lr0 * lrf; lower = more aggressive annealing |
| `--optimizer` | AdamW | SGD / AdamW | SGD with momentum is YOLO default; AdamW often better for seg |
| `--warmup_epochs` | 3.0 | 3–5 | Longer warmup stabilizes early training |
| `--close_mosaic` | 20 | 20–30 | Disables mosaic for last N epochs; helps fine-tune details |
| `--patience` | 50 | 0 | Set to 0 to disable early stopping and train full epochs |

```bash
# Example: cosine LR, full 300 epochs, no early stopping
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 300 --batch 8 --device 3 \
    --cos_lr --patience 0 --close_mosaic 30
```

### Priority 3 — Augmentation

| Param | Default in train.py | Try | Why |
|-------|---------------------|-----|-----|
| `mosaic` | 1.0 | 1.0 | Keep on; very effective for multi-class |
| `mixup` | 0.1 | 0.1–0.15 | Blends two images; helps with class overlap |
| `copy_paste` | 0.1 | 0.1–0.2 | Pastes instances from one image to another; helps rare classes |
| `scale` | 0.5 | 0.5–0.9 | Random scale jitter; higher = more size variety |
| `degrees` | 10.0 | 5–15 | Rotation; slight rotation suits driving scenes |
| `erasing` | 0.1 | 0.2–0.4 | Random erasing regularization |
| `flipud` | 0.0 | 0.0 | Keep 0 — vertical flip makes no sense for driving |
| `hsv_h` | 0.015 | 0.015 | Hue jitter — default is fine |
| `hsv_s` | 0.7 | 0.7 | Saturation jitter |
| `hsv_v` | 0.4 | 0.4 | Value/brightness jitter |
| `multi_scale` | 0.0 | 0.5 | Randomly resize each batch +-50%; strong regulariser |

To change augmentation params not exposed as CLI flags, edit `model.train(...)` in `train.py`.

### Priority 4 — Loss Weights

These control the relative importance of each loss component. Edit directly in the
`model.train(...)` call in `train.py`.

| Param | Default | Try | Why |
|-------|---------|-----|-----|
| `box` | 7.5 | 5.0–7.5 | Box loss weight; lower shifts focus toward masks |
| `cls` | 0.5 | 1.0–2.0 | Classification loss; increase for 64-class problem |
| `dfl` | 1.5 | 1.5 | Distribution focal loss; default is fine |

```python
# In train.py, inside model.train(...):
box=5.0,
cls=1.5,
```

### Priority 5 — Regularisation & Other

| Param | Default | Try | Why |
|-------|---------|-----|-----|
| `dropout` | 0.0 | 0.1 | Dropout in head; mild regularisation |
| `weight_decay` | 0.0005 | 0.0005–0.001 | L2 regularisation |
| `freeze` | None | 10 | Freeze first N backbone layers; useful if overfitting |
| `nbs` | 64 | 64 | Nominal batch size for LR scaling; change if your effective batch differs a lot |

### Recommended Experiments (in order)

**Run 1 — Baseline:**
```bash
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 200 --batch 8 --device 3
```

**Run 2 — Cosine LR + lower mask ratio:**
```bash
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 200 --batch 8 --device 3 \
    --cos_lr --mask_ratio 2 --close_mosaic 25 --name goose_yolo26l_cosine
```

**Run 3 — Higher cls loss:**
Edit `train.py` to add `cls=1.5` in `model.train(...)`, then:
```bash
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 200 --batch 8 --device 3 \
    --cos_lr --mask_ratio 2 --name goose_yolo26l_cls15
```

**Run 4 — Bigger model:**
```bash
python yolo_goose/train.py --model yolo26x-seg.pt --imgsz 1024 --epochs 200 --batch 4 --device 3 \
    --cos_lr --mask_ratio 2 --name goose_yolo26x
```

**Run 5 — Higher resolution:**
```bash
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1280 --epochs 200 --batch 4 --device 3 \
    --cos_lr --mask_ratio 2 --name goose_yolo26l_1280
```

**Run 6 — Multi-scale training + stronger augmentation:**
Edit `train.py` to add `multi_scale=0.5, copy_paste=0.2`, then:
```bash
python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 300 --batch 8 --device 3 \
    --cos_lr --patience 0 --name goose_yolo26l_multiscale
```

After each run, validate with competition metrics:
```bash
python yolo_goose/validate.py --weights runs/segment/<name>/weights/best.pt --imgsz 1024 --device 3
```

## Notes

- Class 0 (undefined) is trained as a regular class (not ignored).
- The conversion from semantic masks to YOLO polygons is lossy (contour approximation). This is inherent to using instance segmentation for semantic segmentation.
- Inference converts YOLO instance predictions back to per-pixel semantic masks using a painter's algorithm (higher confidence overwrites lower).
- Competition metric: `mIoU_composite = 50% * mIoU_fine + 50% * mIoU_coarse`.
