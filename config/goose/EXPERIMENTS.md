# GOOSE Ablation Experiments

## How to run

```bash
# Phase 1: Run all ablations (one GPU each, or sequentially)
CUDA_VISIBLE_DEVICES=0 python train.py --config config/goose/ablation_baseline_v2.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config config/goose/ablation_crop720.yaml
CUDA_VISIBLE_DEVICES=2 python train.py --config config/goose/ablation_classweight.yaml
CUDA_VISIBLE_DEVICES=3 python train.py --config config/goose/ablation_backbone_b.yaml

# Phase 2: After Phase 1 results, run combos
CUDA_VISIBLE_DEVICES=0 python train.py --config config/goose/ablation_crop720_classweight.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config config/goose/ablation_best_combo.yaml

# Evaluate best checkpoints (eval_last_n_epochs handles this during training)
# For manual eval on a range:
python eval.py --config config/goose/ablation_baseline_v2.yaml --fr 180 --to 200
```

## Experiment Matrix

All Phase 1 experiments change ONE variable from the baseline_v2 (which itself
improves lr_power and epochs over the original).

| Config | Name | What changed | backbone | crop | batch | epochs | lr_power | class_wt |
|--------|------|-------------|----------|------|-------|--------|----------|----------|
| `goose_dat_s_epoch_100.yaml` | original | (baseline) | dat_s | 480x640 | 16 | 100 | 1.0 | no |
| `ablation_baseline_v2.yaml` | baseline_v2 | lr + epochs | dat_s | 480x640 | 16 | 200 | **0.9** | no |
| `ablation_crop720.yaml` | crop720 | crop size | dat_s | **720x960** | **8** | 200 | 0.9 | no |
| `ablation_classweight.yaml` | classweight | loss weights | dat_s | 480x640 | 16 | 200 | 0.9 | **yes** |
| `ablation_backbone_b.yaml` | backbone_b | backbone | **dat_b** | 480x640 | **8** | 200 | 0.9 | no |
| `ablation_crop720_classweight.yaml` | crop720+cw | Phase 2 combo | dat_s | 720x960 | 8 | 200 | 0.9 | yes |
| `ablation_best_combo.yaml` | best_combo | Phase 2 all-in | dat_b | 720x960 | **4** | 200 | 0.9 | yes |

## Results (fill in as experiments complete)

| Experiment | best epoch | val mIoU_fine | val mIoU_coarse | val mIoU_composite | notes |
|------------|-----------|---------------|-----------------|--------------------|----|
| original (ep70) | 70 | - | - | ~48.7% (old metric) | |
| baseline_v2 | | | | | |
| crop720 | | | | | |
| classweight | | | | | |
| backbone_b | | | | | |
| crop720+cw | | | | | |
| best_combo | | | | | |

## Phase 3: Inference tricks (apply to best model)

- [ ] Sweep `--shorter_side` (480, 600, 720, 900) on val set
- [ ] Multi-scale TTA (`--tta --tta_scales 0.75 1.0 1.25`)
- [ ] Checkpoint ensemble (average softmax from top-3 epochs)
