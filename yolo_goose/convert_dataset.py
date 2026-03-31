"""Convert GOOSE semantic segmentation masks to YOLO instance segmentation format.

YOLO expects:
  - images/{split}/ with image files
  - labels/{split}/ with .txt files (same stem as image)
  - Each line in .txt: class_id x1 y1 x2 y2 ... xn yn (normalised 0-1 polygon)

This script creates a flat YOLO dataset directory with:
  - Symlinks to original GOOSE RGB images
  - Generated .txt polygon label files from semantic masks

Usage:
    python yolo_goose/convert_dataset.py \
        --goose_root data/goose_dataset \
        --output data/goose_yolo \
        --workers 8
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

GOOSE_RGB_SUFFIXES = [
    "_windshield_vis.png",
    "_front.png",
    "_camera_left.png",
    "_realsense.png",
]

# No classes are ignored — undefined (class 0) is trained as a regular class
IGNORE_CLASSES = set()

# Minimum contour area in pixels to keep (filters noise)
MIN_CONTOUR_AREA = 50
# Polygon simplification factor (fraction of arc length)
SIMPLIFY_EPS = 0.001


def scan_pairs(img_root, lbl_root):
    """Scan GOOSE directory and return (img_path, lbl_path, flat_stem) tuples."""
    pairs = []
    if not os.path.isdir(img_root):
        return pairs
    seq_dirs = sorted(
        d for d in os.listdir(img_root)
        if os.path.isdir(os.path.join(img_root, d))
    )
    for seq in seq_dirs:
        img_dir = os.path.join(img_root, seq)
        lbl_dir = os.path.join(lbl_root, seq)
        if not os.path.isdir(lbl_dir):
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
                # Flat name: sequence__rest_camera.png
                flat_stem = seq + "__" + fname[len(seq) + 2 : -4]  # strip seq__ prefix and .png
                # Actually, fname already includes the sequence prefix, so just use it directly
                flat_stem = fname[:-4]  # remove .png
                pairs.append((
                    os.path.join(img_dir, fname),
                    label_path,
                    flat_stem,
                ))
    return pairs


def mask_to_polygons(mask_path):
    """Convert a semantic mask to YOLO polygon lines.

    Returns list of strings, each: "class_id x1 y1 x2 y2 ... xn yn"
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    H, W = mask.shape
    lines = []
    unique_classes = np.unique(mask)

    for cls_id in unique_classes:
        cls_id = int(cls_id)
        if cls_id in IGNORE_CLASSES:
            continue

        binary = (mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            # Simplify polygon
            epsilon = SIMPLIFY_EPS * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)

            if len(contour) < 3:
                continue

            # Normalise to [0, 1]
            points = contour.reshape(-1, 2).astype(np.float64)
            points[:, 0] /= W
            points[:, 1] /= H
            # Clip to [0, 1]
            points = np.clip(points, 0.0, 1.0)

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            lines.append(f"{cls_id} {coords}")

    return lines


def process_single(args):
    """Process a single image-label pair. Used by multiprocessing Pool."""
    img_src, lbl_path, flat_stem, out_img_dir, out_lbl_dir = args

    # Create symlink for image
    dst_img = os.path.join(out_img_dir, flat_stem + ".png")
    if not os.path.exists(dst_img):
        os.symlink(os.path.abspath(img_src), dst_img)

    # Convert mask to polygon labels
    lines = mask_to_polygons(lbl_path)

    # Write label file
    dst_lbl = os.path.join(out_lbl_dir, flat_stem + ".txt")
    with open(dst_lbl, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert GOOSE to YOLO segment format")
    parser.add_argument("--goose_root", default="data/goose_dataset",
                        help="Path to GOOSE dataset root")
    parser.add_argument("--output", default="data/goose_yolo",
                        help="Output YOLO dataset directory")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    goose_root = args.goose_root
    out_root = args.output

    for split in ["train", "val"]:
        print(f"\n--- Processing {split} ---")
        img_root = os.path.join(goose_root, "images", split)
        lbl_root = os.path.join(goose_root, "labels", split)

        pairs = scan_pairs(img_root, lbl_root)
        print(f"Found {len(pairs)} image-label pairs")

        out_img_dir = os.path.join(out_root, "images", split)
        out_lbl_dir = os.path.join(out_root, "labels", split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        work_items = [
            (img_src, lbl_path, flat_stem, out_img_dir, out_lbl_dir)
            for img_src, lbl_path, flat_stem in pairs
        ]

        total_polys = 0
        if args.workers > 1:
            with Pool(args.workers) as pool:
                for n in tqdm(pool.imap_unordered(process_single, work_items),
                              total=len(work_items), desc=split):
                    total_polys += n
        else:
            for item in tqdm(work_items, desc=split):
                total_polys += process_single(item)

        print(f"{split}: {len(pairs)} images, {total_polys} total polygons")

    # Also create symlinks for test images (no labels needed)
    print("\n--- Symlinking test images ---")
    test_img_root = os.path.join(goose_root, "images", "test")
    out_test_dir = os.path.join(out_root, "images", "test")
    os.makedirs(out_test_dir, exist_ok=True)
    count = 0
    if os.path.isdir(test_img_root):
        for seq in sorted(os.listdir(test_img_root)):
            seq_dir = os.path.join(test_img_root, seq)
            if not os.path.isdir(seq_dir):
                continue
            for fname in sorted(os.listdir(seq_dir)):
                if not fname.endswith(".png"):
                    continue
                is_rgb = any(fname.endswith(s) for s in GOOSE_RGB_SUFFIXES)
                if not is_rgb:
                    continue
                dst = os.path.join(out_test_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(os.path.join(seq_dir, fname)), dst)
                count += 1
    print(f"Test: {count} RGB images symlinked")

    print(f"\nDone! YOLO dataset at: {out_root}")
    print(f"Next: python yolo_goose/train.py --data {out_root}/goose.yaml")


if __name__ == "__main__":
    main()
