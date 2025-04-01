import cv2
import numpy as np
import random


def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]

    tl_h, tl_w = crop_pos
    if (tl_w > w) or (tl_h > h):
        print(f"{tl_h} - {h}")
        print(f"{tl_w} - {w}")
    assert (tl_h < h) and (tl_h >= 0)
    assert (tl_w < w) and (tl_w >= 0)
    crop_h, crop_w = crop_size

    img_crop = img[tl_h : tl_h + crop_h, tl_w : tl_w + crop_w, ...]

    out = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT, pad_label_value)

    return out


def generate_random_crop_pos(ori_size, crop_size):
    h, w = ori_size
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0
    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(
        img, margin[0], margin[1], margin[2], margin[3], border_mode, value=value
    )
    return img


def random_mirror(**kwargs):
    res = {}
    if random.random() >= 0.5:
        for key, value in kwargs.items():
            res[key] = cv2.flip(np.array(value), 1)
        return kwargs
    else:
        return kwargs


def random_scale(scales, **kwargs):
    scale = random.choice(scales)
    res = {}
    for key, value in kwargs.items():
        value = np.array(value)

        sh = int(value.shape[0] * scale)
        sw = int(value.shape[1] * scale)
        if key == "label":
            res[key] = cv2.resize(value, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            res[key] = cv2.resize(value, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return res, (sh, sw)


class SemSegTransform(object):
    def __init__(self, crop_size, train_scale_array):
        self.train_scale_array = train_scale_array
        self.crop_size = crop_size

    def __call__(self, **kwargs):
        kwargs = random_mirror(**kwargs)
        if self.train_scale_array is not None:
            kwargs, input_shape = random_scale(self.train_scale_array, **kwargs)

        crop_size = self.crop_size
        crop_pos = generate_random_crop_pos(input_shape, crop_size)
        res = {}
        for key, value in kwargs.items():
            if key == "label":
                res[key] = random_crop_pad_to_shape(value, crop_pos, crop_size, 255)
            else:
                res[key] = random_crop_pad_to_shape(value, crop_pos, crop_size, 0)

        return res


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):
        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs
