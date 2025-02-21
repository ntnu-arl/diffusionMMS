import os
import argparse

from omegaconf import OmegaConf
from engine.evaluator import Evaluator
from engine import get_model
import torch
from utils.helper import convert_depth_to_three_channel_img, get_class_colors
import numpy as np
import copy
import cv2


def visualize(data, pred, num_classes=40):
    NORM_RGB = {
        "mean": np.array([0.485, 0.456, 0.406]),
        "std": np.array([0.229, 0.224, 0.225]),
    }
    # Get color corresponding to each classes
    colors = np.array(get_class_colors(num_classes + 1))

    # Convert data to unit8 numpy type on cpu
    pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    rgb_arr = data["rgb"].squeeze(0).cpu().numpy()
    pred_arr[pred_arr > num_classes] = num_classes
    if data["depth"] is not None:
        depth_arr = data["depth"].squeeze(0).cpu().numpy()
    colored_pred = np.zeros_like(pred_arr)
    colored_pred = np.stack((colored_pred,) * 3, axis=-1)
    colored_pred[:] = colors[pred_arr[:]]

    # Overlay prediction on input images
    rgb_arr = rgb_arr.transpose(1, 2, 0)
    rgb_arr = ((rgb_arr * NORM_RGB["std"] + NORM_RGB["mean"]) * 255).astype(np.uint8)
    rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    depth_arr = (depth_arr.transpose(1, 2, 0) * 255).astype(np.uint8)

    # Concatenate multiple outputs for saving
    output = np.concatenate([rgb_arr, depth_arr, colored_pred], axis=1)

    cv2.imshow("pred", output)
    if cv2.waitKey() == ord("q"):
        exit(0)


def setup_model(cfg_file, device):
    config = OmegaConf.load(cfg_file)
    model = get_model(config.model.name, eval=True, **config.model.params)
    checkpoint_path = os.path.join(
        "output_dir/",
        config.experiment_dataset,
        config.experiment_name,
        "checkpoint-" + str(93) + ".pth",
    )

    model.load_state_dict(torch.load(checkpoint_path)["model"])
    model = model.to(device)
    return model


def preprocess(input_rgb, input_depth):
    depth = copy.copy(input_depth)
    rgb = copy.copy(input_rgb)
    depth[np.isnan(depth)] = 0  # Replace NaN with 0
    depth[np.isinf(depth)] = 0  # Replace Inf with 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb = (rgb / 255.0 - mean) / std

    depth = convert_depth_to_three_channel_img(depth) / 255.0

    rgb = rgb.transpose(2, 0, 1)
    depth = depth.transpose(2, 0, 1)

    rgb = torch.from_numpy(rgb).unsqueeze(0).cuda().float()
    depth = torch.from_numpy(depth).unsqueeze(0).cuda().float()

    output = {"rgb": rgb, "depth": depth}

    return output


def predict(model, rgb, depth):
    data = preprocess(rgb, depth)
    with torch.no_grad():
        score = model.sampling(data["rgb"], data["depth"])
    pred = score.argmax(1)

    return pred


if __name__ == "__main__":
    from PIL import Image

    cfg_file = "config/nyuv2/standard/ddp_dual_dat_s_mmcv_epoch_100.yaml"
    rgb = Image.open("/home/sherlock/Pictures/datasets/NYUDepthv2/image/0.jpg").convert(
        "RGB"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rgb = np.array(rgb)
    depth = np.load("/home/sherlock/Pictures/datasets/NYUDepthv2/depth/0.npy")
    data = preprocess(rgb, depth)

    model = setup_model(cfg_file, device)
    model.eval()
    with torch.no_grad():
        score = model.sampling(data["rgb"], data["depth"])
    pred = score.argmax(1)
    visualize(data, pred)
