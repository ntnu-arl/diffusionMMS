"""Generate GOOSE competition submissions using YOLO26-seg.

Converts YOLO instance segmentation predictions back to semantic segmentation
masks (grayscale PNGs with class IDs) for CodaBench submission.

Usage:
    # Basic inference
    python yolo_goose/inference.py \
        --weights runs/segment/goose_yolo26l/weights/best.pt \
        --output submission_yolo/

    # Higher resolution + TTA
    python yolo_goose/inference.py \
        --weights runs/segment/goose_yolo26l/weights/best.pt \
        --output submission_yolo/ \
        --imgsz 1280 --tta

    # Then zip and upload:
    cd submission_yolo && zip -r ../submission_yolo.zip . && cd ..
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

# Regex to extract timestamp from GOOSE filenames
_GOOSE_TIMESTAMP = re.compile(
    r"_(\d{10,})_(?:windshield_vis|windshield_nir|front|camera_left|realsense|labelids)\.png$"
)

GOOSE_RGB_SUFFIXES = [
    "_windshield_vis.png",
    "_front.png",
    "_camera_left.png",
    "_realsense.png",
]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26-seg inference for GOOSE")
    parser.add_argument("--weights", required=True,
                        help="Path to trained YOLO weights (.pt)")
    parser.add_argument("--output", default="submission_yolo/",
                        help="Output directory for submission labels")
    parser.add_argument("--test_dir", default="data/goose_dataset/images/test",
                        help="Test images directory")
    parser.add_argument("--label_list", default="data/goose_dataset/dev_phase/dev_phase.txt",
                        help="Text file listing expected label filenames (dev_phase.txt)")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Inference image size (default: 1024)")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Confidence threshold (default: 0.01, low to maximize coverage)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation")
    parser.add_argument("--device", default="0",
                        help="Device for inference")
    parser.add_argument("--retina_masks", action="store_true", default=True,
                        help="Use high-resolution retina masks (default: True)")
    parser.add_argument("--max_det", type=int, default=1000,
                        help="Maximum detections per image (default: 1000)")
    return parser.parse_args()


def build_label_lookup(label_list_path):
    """Build {timestamp: label_filename} from a label list file."""
    lookup = {}
    with open(label_list_path) as f:
        for line in f:
            fname = line.strip()
            if not fname:
                continue
            m = _GOOSE_TIMESTAMP.search(fname)
            if m:
                lookup[m.group(1)] = fname
    return lookup


def find_test_images(test_root, label_lookup=None):
    """Walk test root and return (label_filename, abs_img_path) pairs."""
    pairs = []
    for dirpath, _, filenames in os.walk(test_root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(".png"):
                continue
            # Skip NIR images
            if "_nir" in fn.lower():
                continue
            # Only keep RGB camera types
            is_rgb = any(fn.endswith(s) for s in GOOSE_RGB_SUFFIXES)
            if not is_rgb:
                continue

            m = _GOOSE_TIMESTAMP.search(fn)
            if not m:
                continue
            timestamp = m.group(1)
            abs_path = os.path.join(dirpath, fn)

            if label_lookup is not None:
                if timestamp not in label_lookup:
                    continue
                label_fn = label_lookup[timestamp]
            else:
                label_fn = _GOOSE_TIMESTAMP.sub(
                    f"_{timestamp}_labelids.png", fn
                )
            pairs.append((label_fn, abs_path))
    return pairs


def yolo_results_to_semantic_mask(result, orig_h, orig_w):
    """Convert a single YOLO Results object to a semantic segmentation mask.

    Strategy: paint masks in order of ascending confidence, so higher-confidence
    detections overwrite lower-confidence ones (painter's algorithm).
    """
    semantic = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if result.masks is None or len(result.masks) == 0:
        return semantic

    masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
    classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
    confs = result.boxes.conf.cpu().numpy()  # (N,)

    # Sort by confidence ascending (low confidence painted first, high overwrites)
    order = np.argsort(confs)

    for idx in order:
        cls_id = classes[idx]
        mask = masks[idx]  # (mask_h, mask_w) float 0-1

        # Resize mask to original image size if needed
        if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Apply mask where mask > 0.5
        semantic[mask > 0.5] = cls_id

    return semantic


def main():
    args = parse_args()

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Build label lookup
    label_lookup = None
    if args.label_list and os.path.exists(args.label_list):
        label_lookup = build_label_lookup(args.label_list)
        print(f"Label list: {len(label_lookup)} entries from {args.label_list}")

    # Find test images
    image_pairs = find_test_images(args.test_dir, label_lookup)
    print(f"Found {len(image_pairs)} test images to process")

    os.makedirs(args.output, exist_ok=True)

    for label_fn, img_path in tqdm(image_pairs, desc="Inference"):
        # Get original image dimensions
        img = Image.open(img_path)
        orig_w, orig_h = img.size

        # Run YOLO inference
        results = model.predict(
            img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            augment=args.tta,
            retina_masks=args.retina_masks,
            max_det=args.max_det,
            verbose=False,
        )

        # Convert to semantic mask
        result = results[0]
        semantic = yolo_results_to_semantic_mask(result, orig_h, orig_w)

        # Save as grayscale PNG
        out_path = os.path.join(args.output, label_fn)
        Image.fromarray(semantic, mode="L").save(out_path)

    print(f"\nSaved {len(image_pairs)} predictions to {args.output}")
    print(f"To submit: cd {args.output} && zip -r ../submission_yolo.zip . && cd ..")


if __name__ == "__main__":
    main()
