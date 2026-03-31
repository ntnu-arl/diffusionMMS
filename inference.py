"""Generate test set predictions for GOOSE CodaBench competition submission.

Saves per-pixel label predictions as grayscale PNGs at the original image
resolution.  Output filenames follow the CodaBench convention:
    {scene}_{seqnum}_{timestamp}_labelids.png
All predictions are placed in a flat output directory, ready to be zipped
and uploaded.

Image and label files share the same scene prefix and timestamp, but may
differ in sequence number.  A label list file (e.g. dev_phase.txt) is used
to build a timestamp -> expected label filename lookup.

Usage:
    # Development phase (361 images)
    python inference.py --config config/goose/goose_dat_s_epoch_100.yaml \
        --epoch 100 --output submission/ \
        --label_list data/goose_dataset/dev_phase/dev_phase.txt

    # Full test phase (all 1815 images)
    python inference.py --config config/goose/goose_dat_s_epoch_100.yaml \
        --epoch 100 --output submission/

    # Then zip and upload:
    cd submission && zip -r ../submission.zip . && cd ..
"""

import os
import re
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from engine import get_model
from utils.logger import get_root_logger

logger = get_root_logger()

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

# Regex to extract timestamp from any GOOSE filename.
# Works for both MuCAR-3 (double underscore) and GOOSE-Ex / Spot (single underscore):
#   "2022-07-07_campus_no_ptp__0015_1657197601665184211_windshield_vis.png"
#   "alice_scenario01_sequence14_0003_1691670612470132000_camera_left.png"
# Captures: timestamp (long digit string before the suffix)
_GOOSE_TIMESTAMP = re.compile(
    r"_(\d{10,})_(?:windshield_vis|windshield_nir|front|camera_left|realsense|labelids)\.png$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--epoch", type=int, required=True, help="Checkpoint epoch")
    parser.add_argument("--output", default="submission/", help="Output directory")
    parser.add_argument(
        "--test_dir", default=None,
        help="Override test image directory (default: data root / images / test)",
    )
    parser.add_argument(
        "--label_list", default=None,
        help="Text file listing expected label filenames (e.g. dev_phase.txt). "
             "If provided, only those images are processed and output filenames "
             "match the list exactly. If omitted, all test images are processed.",
    )
    parser.add_argument(
        "--shorter_side", type=int, default=480,
        help="Resize shorter side to this before inference (default: 480)",
    )
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


def find_rgb_images(root, label_lookup=None):
    """Walk root and return list of (label_filename, absolute_path) for RGB images.

    If label_lookup is provided, only images whose timestamp exists in the
    lookup are returned, and the label filename comes from the lookup
    (so the sequence number matches what CodaBench expects).
    """
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(".png"):
                continue
            # Skip NIR images
            if "_nir" in fn.lower():
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
                # Derive label filename from image filename
                label_fn = _GOOSE_TIMESTAMP.sub(
                    f"_{timestamp}_labelids.png", fn
                )
            pairs.append((label_fn, abs_path))
    return pairs


def resize_shorter_side(img_pil, shorter_side):
    """Resize PIL image so the shorter side equals shorter_side, preserving aspect ratio."""
    w, h = img_pil.size
    scale = shorter_side / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return img_pil.resize((new_w, new_h), Image.BILINEAR)


def preprocess(img_pil):
    """PIL Image -> (1, 3, H, W) float32 tensor, normalised."""
    img = np.array(img_pil).astype(np.float32) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0).float()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Load model
    model = get_model(config.model.name, eval=True, **config.model.params)
    checkpoint_path = os.path.join(
        "output_dir/",
        config.experiment_dataset,
        config.experiment_name,
        f"checkpoint-{args.epoch}.pth",
    )
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=False)["model"]
    )
    model.cuda()
    model.eval()

    # Locate test images
    if args.test_dir:
        test_root = args.test_dir
    else:
        test_root = os.path.join(config.val.dataset.params.root, "images", "test")
        if not os.path.isdir(test_root):
            test_root = "data/goose_dataset/images/test"
    logger.info(f"Test image root: {test_root}")

    # Build label lookup if a list file is provided
    label_lookup = None
    if args.label_list:
        label_lookup = build_label_lookup(args.label_list)
        logger.info(f"Label list: {len(label_lookup)} entries from {args.label_list}")

    image_pairs = find_rgb_images(test_root, label_lookup)
    logger.info(f"Found {len(image_pairs)} images to process")

    os.makedirs(args.output, exist_ok=True)

    for label_fn, abs_path in tqdm(image_pairs, desc="Inference"):
        img_pil = Image.open(abs_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Resize for inference, then upscale prediction back to original
        img_resized = resize_shorter_side(img_pil, args.shorter_side)
        rgb = preprocess(img_resized).cuda()

        with torch.no_grad(), torch.amp.autocast("cuda"):
            score = model.sampling(rgb, depth=None)

        pred = score.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Upscale prediction to original resolution
        if pred.shape[0] != orig_h or pred.shape[1] != orig_w:
            pred = cv2.resize(
                pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

        # Save as flat grayscale PNG
        out_path = os.path.join(args.output, label_fn)
        Image.fromarray(pred, mode="L").save(out_path)

    logger.info(f"Saved {len(image_pairs)} predictions to {args.output}")
    logger.info(f"To submit: cd {args.output} && zip -r ../submission.zip .")


if __name__ == "__main__":
    torch.manual_seed(1234)
    main()
