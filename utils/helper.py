import time
import numpy as np
from PIL import Image
import cv2
import os
from utils.logger import get_root_logger

logger = get_root_logger()


class Timer:
    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0
        self.start()

    def start(self):
        self.start_time = time.time()

    def end(self, ms=False, clear=False):
        self.end_time = time.time()
        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.start_time = time.time()

        return duration


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = {}
        for key in keys:
            dataset[key] = float(np.mean(self.data_dic[key]))

        if clear:
            self.clear()

        return dataset

    def clear(self):
        self.data_dic = {key: [] for key in self.keys}


def show_pil_image(window_name: str, img: Image):
    cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, cv2_image)


def convert_depth_to_three_channel_img(depth):
    max_depth = np.max(depth)
    mask = np.where(depth == 0, 0, 1)
    min_depth = np.min(depth[np.nonzero(depth)])
    depth = ((depth - min_depth) / (max_depth - min_depth) * 255.0).astype(np.uint8)
    depth = (depth * mask).astype(np.uint8)
    depth = np.stack((depth,) * 3, axis=-1)
    return depth


def link_file(src, target):
    if os.path.islink(target):
        logger.warning("Delete the previous best model file")
        os.system("rm -rf {}".format(target))
    os.system("ln -s {} {}".format(src, target))


def darken(img, gamma=0.5):
    invgamma = 1 / gamma
    img = np.array(img) / 255.0
    lowlight_image = np.power(img, invgamma)
    lowlight_image = (lowlight_image * 255).astype(np.uint8)
    return lowlight_image


def print_iou(
    iou,
    freq_IoU,
    mean_pixel_acc,
    pixel_acc,
    class_names=None,
    show_no_back=False,
    no_print=False,
):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = "Class %d:" % (i + 1)
        else:
            cls = "%d %s" % (i + 1, class_names[i])
        lines.append("%-8s\t%.3f%%" % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])
    if show_no_back:
        lines.append(
            "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
            % (
                "mean_IoU",
                mean_IoU * 100,
                "mean_IU_no_back",
                mean_IoU_no_back * 100,
                "freq_IoU",
                freq_IoU * 100,
                "mean_pixel_acc",
                mean_pixel_acc * 100,
                "pixel_acc",
                pixel_acc * 100,
            )
        )
    else:
        lines.append(
            "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
            % (
                "mean_IoU",
                mean_IoU * 100,
                "freq_IoU",
                freq_IoU * 100,
                "mean_pixel_acc",
                mean_pixel_acc * 100,
                "pixel_acc",
                pixel_acc * 100,
            )
        )
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line


def get_class_colors(num_class=41):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    cmap = np.zeros((num_class, 3), dtype=np.uint8)
    for i in range(num_class):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    return class_colors


if __name__ == "__main__":
    img = Image.open("data/NYUDepthv2/image/9.jpg").convert("RGB")
    show_pil_image("ori", img)
    img = darken(img)
    show_pil_image("img", img)
    cv2.waitKey()
