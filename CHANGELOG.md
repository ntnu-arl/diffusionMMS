# Changelog

## 2026-04-08

### DINOv2 Backbone
- Added DINOv2 backbone wrapper (`models/backbone/dinov2.py`) with three variants: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`
- Extracts multi-scale features from intermediate transformer blocks, projects and rescales to a 4-level spatial pyramid compatible with the existing FPN neck
- Pretrained weights loaded automatically via `torch.hub`
- Supports `freeze_backbone` option for frozen feature extraction
- Registered in `models/__init__.py`; replaced `isinstance(Single_DAT)` check with a generic `RGB_ONLY_BACKBONES` set in `models/segmentor/diffusionmms.py`

### Cosine Annealing Warm Restarts LR Scheduler
- Added `CosineWarmRestartsLR` to `utils/lr_policy.py` — periodic LR resets to escape loss plateaus
- Configurable cycle length (`cycle_epochs`), cycle multiplier (`cycle_mult`), and minimum LR (`min_lr`)
- Updated `engine/runner.py` to select scheduler via `lr_scheduler` config key (default: `warmup_poly` for backward compat)
- When resuming, cosine cycles start fresh from the resume epoch (not offset by prior training)

### Photometric Augmentations
- Added `PhotoMetricTransform` to `datasets/transform.py` — applies to RGB only, leaves labels untouched
  - Color jitter (brightness, contrast, saturation, hue)
  - Random Gaussian blur
  - Random grayscale
- Registered as `photometric_transform` in `datasets/__init__.py`

### Validation Loss Logging
- `engine/evaluator.py`: `run_inline()` now computes and returns val loss alongside mIoU
- `engine/runner.py`: logs `val_loss` scalar to TensorBoard during eval epochs

### New Configs
- `config/goose/ablation_best_combo_resume90.yaml` — resumes DAT-B from epoch 90 with cosine warm restarts (lr=1e-4, 20-epoch cycles)
- `config/goose/goose_dinov2_b.yaml` — DINOv2-B + photometric augmentation + cosine warm restarts
- `config/goose/goose_dinov2_l.yaml` — DINOv2-L + photometric augmentation + cosine warm restarts
