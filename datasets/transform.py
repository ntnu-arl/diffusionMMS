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
    if random.random() >= 0.5:
        res = {}
        for key, value in kwargs.items():
            res[key] = cv2.flip(np.array(value), 1)
        return res
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


class PhotoMetricTransform(object):
    """Photometric augmentations applied only to RGB (numpy uint8 HWC).

    Leaves 'label' (and 'depth') untouched. Should be placed after
    SemSegTransform in common_transforms.

    Args:
        brightness: max absolute delta for brightness (0–255 scale).
        contrast: range [1-contrast, 1+contrast] for contrast factor.
        saturation: range [1-saturation, 1+saturation] for saturation factor.
        hue: max absolute delta for hue shift in degrees (0–180).
        blur_prob: probability of applying Gaussian blur.
        blur_kernel: kernel size for Gaussian blur (must be odd).
        grayscale_prob: probability of converting to grayscale.
    """

    def __init__(
        self,
        brightness=32,
        contrast=0.5,
        saturation=0.5,
        hue=18,
        blur_prob=0.3,
        blur_kernel=5,
        grayscale_prob=0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.blur_prob = blur_prob
        self.blur_kernel = blur_kernel
        self.grayscale_prob = grayscale_prob

    def __call__(self, **kwargs):
        rgb = np.array(kwargs["rgb"], dtype=np.float32)

        # Brightness
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness, self.brightness)
            rgb = np.clip(rgb + delta, 0, 255)

        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = rgb.mean()
            rgb = np.clip((rgb - mean) * factor + mean, 0, 255)

        # Saturation (convert to HSV, scale S channel)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Hue
        if random.random() < 0.5:
            hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            delta = random.uniform(-self.hue, self.hue)
            hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Gaussian blur
        if random.random() < self.blur_prob:
            rgb = cv2.GaussianBlur(
                rgb.astype(np.uint8), (self.blur_kernel, self.blur_kernel), 0
            ).astype(np.float32)

        # Random grayscale
        if random.random() < self.grayscale_prob:
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

        kwargs["rgb"] = rgb.astype(np.uint8)
        return kwargs


class ResizeTransform(object):
    def __init__(self, shorter_side):
        self.shorter_side = shorter_side

    def __call__(self, **kwargs):
        # Determine scale from the first value
        first = np.array(next(iter(kwargs.values())))
        h_orig, w_orig = first.shape[:2]
        scale = self.shorter_side / min(h_orig, w_orig)
        h_new = int(h_orig * scale)
        w_new = int(w_orig * scale)
        res = {}
        for key, value in kwargs.items():
            value = np.array(value)
            if key == "label":
                res[key] = cv2.resize(value, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
            else:
                res[key] = cv2.resize(value, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        return res


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):
        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs
