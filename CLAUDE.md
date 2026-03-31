# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffusionMMS — Diffusion-based RGB-D Semantic Segmentation with Deformable Attention Transformer (ICAR 2025). Uses a diffusion process conditioned on image features to refine semantic predictions. Supports three datasets: NYUv2 (40 classes, RGB-D), SunRGBD (37 classes, RGB-D), and GOOSE (64 classes, RGB-only).

## Commands

### Training (single GPU)
```bash
python train.py --config config/goose/goose_dat_s_epoch_100.yaml
```

### Training (multi-GPU with DDP)
```bash
torchrun --nproc_per_node=4 train.py --config config/goose/goose_dat_s_epoch_100.yaml
```

### Evaluation (range of epochs)
```bash
python eval.py --config config/goose/goose_dat_s_epoch_100.yaml --fr 91 --to 100
```

### Single-epoch test with visualization
```bash
python test.py --config config/goose/goose_dat_s_epoch_100.yaml --epoch 100 --show
```

### Compare experiments
```bash
python compare_experiments.py                          # all experiments
python compare_experiments.py --output_dir path/       # specific dir
python compare_experiments.py --sort best_mIoU --csv results.csv
python compare_experiments.py --plot                   # generate visual charts
```

### TensorBoard
```bash
tensorboard --logdir log/
```

### Rank depth images by invalid pixel proportion
```bash
python -m utils.ranking_data --img_dir <path> --file <path>
```

## Environment Setup

```bash
conda create -n cu117 python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
# Install NATTEN separately: https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md
```

Pretrained DAT++ backbones go in `pretrained/` (from https://github.com/LeapLabTHU/DAT-Segmentation). Datasets go in `data/`.

## Architecture

```
train.py → engine/runner.py (Trainer)
                ↓
         engine/evaluator.py (inline eval during last N epochs)

Model pipeline (models/segmentor/diffusionmms.py):
  Backbone (DAT: dual-branch RGB+D or single-branch RGB)
  → Neck (FPN, 256 channels)
  → Decoder (DeformableDETRwithTime, time-conditioned diffusion)
  → Aux Head (FCN, weighted by aux_rate during training)

Loss: CrossEntropy(decode) + aux_rate * CrossEntropy(aux)
Diffusion: DDIM/DDPM sampling, cosine/linear noise schedule
```

### Key module locations
- **Backbones:** `models/backbone/` — DAT variants (dat_s/t/b, single_dat_s/t/b)
- **Decoder:** `models/decode_heads/` — DeformableDETRwithTime
- **Datasets:** `datasets/datasets.py` — NYUv2Dataset, SunRGBDDataset, GooseDataset
- **Transforms:** `datasets/transform.py` — augmentation pipeline
- **LR schedulers:** `utils/lr_policy.py` — WarmUpPolyLR (default), PolyLR, MultiStageLR
- **Deformable attention:** `models/common_layers.py` — multi_scale_deformable_attn_pytorch

### Config system
OmegaConf YAML files in `config/{dataset}/`. Top-level keys: `experiment_type`, `experiment_dataset`, `experiment_name`, `model`, `train`, `val`. Backbone choice (`single_dat_*` vs `dat_*`) determines RGB-only vs RGB-D mode.

### Experiment tracking
After training, `engine/runner.py` saves `experiment_summary.json` (hparams + metrics) and `config.yaml` to `output_dir/{dataset}/{experiment_name}/`. The `compare_experiments.py` script scans these for comparison tables and visualizations.

### Training optimizations (all enabled by default)
AMP (float16 autocast + GradScaler), cuDNN benchmark, `set_to_none=True` gradient zeroing, non-blocking GPU transfers, DataLoader with pin_memory/persistent_workers/prefetch_factor, reduced logging frequency (every 50 iters), detached loss accumulation, eliminated GPU sync in deformable attention.
