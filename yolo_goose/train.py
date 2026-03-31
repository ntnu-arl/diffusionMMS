"""Train YOLO26-seg on GOOSE dataset.

Usage:
    # Single GPU (nano model, fast iteration)
    python yolo_goose/train.py --model yolo26n-seg.pt --imgsz 640 --epochs 100

    # Single GPU (large model, competition quality)
    python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 200 --batch 4

    # Multi-GPU
    python yolo_goose/train.py --model yolo26l-seg.pt --imgsz 1024 --epochs 200 --batch 4 --device 0,1,2,3

    # Resume training
    python yolo_goose/train.py --resume runs/segment/goose_yolo26l/weights/last.pt
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26-seg on GOOSE")
    parser.add_argument("--model", default="yolo26l-seg.pt",
                        help="Model name or path (e.g. yolo26n-seg.pt, yolo26s-seg.pt, "
                             "yolo26m-seg.pt, yolo26l-seg.pt, yolo26x-seg.pt)")
    parser.add_argument("--data", default="yolo_goose/goose.yaml",
                        help="Dataset YAML config path")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Training image size (default: 1024)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--device", default="0",
                        help="Device(s) for training (e.g. '0' or '0,1,2,3')")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--name", default=None,
                        help="Experiment name (default: goose_{model_name})")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--optimizer", default="AdamW",
                        help="Optimizer (default: AdamW)")
    parser.add_argument("--lr0", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate factor (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Weight decay (default: 0.0005)")
    parser.add_argument("--cos_lr", action="store_true",
                        help="Use cosine LR scheduler")
    parser.add_argument("--close_mosaic", type=int, default=20,
                        help="Disable mosaic augmentation for last N epochs (default: 20)")
    parser.add_argument("--mask_ratio", type=int, default=4,
                        help="Mask downsample ratio (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)

    # Derive experiment name
    if args.name is None:
        model_stem = os.path.splitext(os.path.basename(args.model))[0]
        args.name = f"goose_{model_stem}"

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project="runs/segment",
        # Optimiser
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=args.close_mosaic,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.1,
        # Segmentation specific
        mask_ratio=args.mask_ratio,
        overlap_mask=True,
        # Training settings
        patience=args.patience,
        save=True,
        save_period=10,
        val=True,
        amp=True,
        cache=False,
        exist_ok=True,
        verbose=True,
        seed=42,
    )

    print(f"\nTraining complete! Best weights: runs/segment/{args.name}/weights/best.pt")
    print(f"Run inference: python yolo_goose/inference.py --weights runs/segment/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

    