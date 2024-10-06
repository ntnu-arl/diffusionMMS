import numpy as np
from utils.helper import convert_depth_to_three_channel_img
import os
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="path to test index file")
parser.add_argument("--img_dir", type=str, help="path to depth images directory")


def calc_invalid_percentage(depth):
    depth = convert_depth_to_three_channel_img(depth)
    numel = np.size(depth)
    num_nonzero = np.count_nonzero(depth)
    return float(numel - num_nonzero) / numel


if __name__ == "__main__":
    args = parser.parse_args()
    depth_dir = args.img_dir

    all_res = {}
    with open(args.file, "r") as f:
        all_file_index = [int(idx) for idx in f.readlines()]

    for file_index in tqdm(sorted(all_file_index)):
        filename = os.path.join(depth_dir, str(file_index).zfill(6) + ".png")
        if os.path.exists(filename):
            depth = np.array(Image.open(filename))
        else:
            filename = os.path.join(depth_dir, str(file_index) + ".npy")
            depth = np.load(filename)
        percentage = calc_invalid_percentage(depth)
        all_res[file_index] = percentage
    top_20 = int(0.2 * len(all_res))
    sorted_res = sorted(all_res.items(), key=lambda x: x[1], reverse=True)

    sorted_res = sorted_res[:top_20]
    with open("most_invalid.txt", "w") as f:
        for file in sorted_res:
            f.write(str(file[0]) + "\n")
