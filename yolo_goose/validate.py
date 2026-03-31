"""Validate YOLO26-seg on GOOSE val set with competition metrics.

Runs YOLO inference on the validation set and computes mIoU_fine, mIoU_coarse,
and mIoU_composite using the official GOOSE evaluation protocol.

Usage:
    python yolo_goose/validate.py \
        --weights runs/segment/goose_yolo26l/weights/best.pt \
        --imgsz 1024
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from engine.metric import goose_compute_scores, GOOSE_COARSE_NAMES, GOOSE_FINE_IDS
from yolo_goose.inference import yolo_results_to_semantic_mask

GOOSE_CLASSES = [
    "undefined", "traffic_cone", "snow", "cobble", "obstacle",
    "leaves", "street_light", "bikeway", "ego_vehicle",
    "pedestrian_crossing", "road_block", "road_marking", "car",
    "bicycle", "person", "bus", "forest", "bush", "moss",
    "traffic_light", "motorcycle", "sidewalk", "curb", "asphalt",
    "gravel", "boom_barrier", "rail_track", "tree_crown",
    "tree_trunk", "debris", "crops", "soil", "rider", "animal",
    "truck", "on_rails", "caravan", "trailer", "building", "wall",
    "rock", "fence", "guard_rail", "bridge", "tunnel", "pole",
    "traffic_sign", "misc_sign", "barrier_tape", "kick_scooter",
    "low_grass", "high_grass", "scenery_vegetation", "sky", "water",
    "wire", "outlier", "heavy_machinery", "container", "hedge",
    "barrel", "pipe", "tree_root", "military_vehicle",
]

GOOSE_RGB_SUFFIXES = [
    "_windshield_vis.png",
    "_front.png",
    "_camera_left.png",
    "_realsense.png",
]

NUM_CLASSES = 64


def scan_val_pairs(goose_root):
    """Scan GOOSE val set and return (img_path, lbl_path) pairs."""
    img_root = os.path.join(goose_root, "images", "val")
    lbl_root = os.path.join(goose_root, "labels", "val")
    pairs = []

    for seq in sorted(os.listdir(img_root)):
        img_dir = os.path.join(img_root, seq)
        lbl_dir = os.path.join(lbl_root, seq)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".png"):
                continue
            base = None
            for suffix in GOOSE_RGB_SUFFIXES:
                if fname.endswith(suffix):
                    base = fname[: -len(suffix)]
                    break
            if base is None:
                continue
            label_fname = base + "_labelids.png"
            label_path = os.path.join(lbl_dir, label_fname)
            if os.path.exists(label_path):
                pairs.append((os.path.join(img_dir, fname), label_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO26-seg on GOOSE")
    parser.add_argument("--weights", required=True, help="Path to trained weights")
    parser.add_argument("--goose_root", default="data/goose_dataset",
                        help="GOOSE dataset root")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", default="0", help="Device")
    parser.add_argument("--tta", action="store_true", help="Enable TTA")
    parser.add_argument("--max_det", type=int, default=1000, help="Max detections")
    args = parser.parse_args()

    model = YOLO(args.weights)
    pairs = scan_val_pairs(args.goose_root)
    print(f"Validating on {len(pairs)} image-label pairs")

    # Accumulate confusion matrix
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for img_path, lbl_path in tqdm(pairs, desc="Validating"):
        # Load ground truth
        gt = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = gt.shape

        # Run YOLO
        results = model.predict(
            img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            augment=args.tta,
            retina_masks=True,
            max_det=args.max_det,
            verbose=False,
        )

        pred = yolo_results_to_semantic_mask(results[0], orig_h, orig_w)

        # Update confusion matrix (exclude undefined class 0 from gt)
        valid = (gt > 0) & (gt < NUM_CLASSES) & (pred < NUM_CLASSES)
        if valid.any():
            hist += np.bincount(
                NUM_CLASSES * gt[valid].astype(int) + pred[valid].astype(int),
                minlength=NUM_CLASSES ** 2
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    # Compute GOOSE metrics
    mIoU_fine, mIoU_coarse, mIoU_composite, fine_ious, coarse_ious = goose_compute_scores(hist)

    print(f"\n{'='*60}")
    print(f"GOOSE Validation Results")
    print(f"{'='*60}")
    print(f"mIoU_fine    (56 classes): {mIoU_fine * 100:.2f}%")
    print(f"mIoU_coarse  (11 classes): {mIoU_coarse * 100:.2f}%")
    print(f"mIoU_composite:            {mIoU_composite * 100:.2f}%")

    # Per-coarse-class breakdown
    print(f"\n{'='*60}")
    print(f"Coarse class IoU:")
    print(f"{'='*60}")
    for i, (name, iou) in enumerate(zip(GOOSE_COARSE_NAMES, coarse_ious)):
        print(f"  {name:20s}: {iou * 100:.2f}%")

    # Per-fine-class breakdown (top/bottom)
    print(f"\n{'='*60}")
    print(f"Fine class IoU (sorted):")
    print(f"{'='*60}")
    fine_results = [(GOOSE_CLASSES[cid], iou) for cid, iou in zip(GOOSE_FINE_IDS, fine_ious)]
    fine_results.sort(key=lambda x: x[1], reverse=True)
    for name, iou in fine_results:
        print(f"  {name:25s}: {iou * 100:.2f}%")


if __name__ == "__main__":
    main()
