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
