from .transform import SemSegTransform, ResizeTransform, PhotoMetricTransform, CustomCompose
from omegaconf.dictconfig import DictConfig
from .datasets import NYUv2Dataset, SunRGBDDataset, GooseDataset
from utils.logger import get_root_logger
import torchvision.transforms as T

ALL_TRANSFORM = {
    "resize": T.Resize,
    "to_tensor": T.ToTensor,
    "RandomHorizontalFlip": T.RandomHorizontalFlip,
    "normalize": T.Normalize,
    "to_pil": T.ToPILImage,
    "semseg_transform": SemSegTransform,
    "resize_transform": ResizeTransform,
    "photometric_transform": PhotoMetricTransform,
}
ALL_DATASETS = {
    "nyuv2": NYUv2Dataset,
    "sunrgbd": SunRGBDDataset,
    "goose": GooseDataset,
}
# Datasets that do not have depth modality
RGB_ONLY_DATASETS = {"goose"}
logger = get_root_logger()


def get_dataset(name, cfg):
    if name not in ALL_DATASETS:
        logger.warning(
            "{name} is not supported, please implement it first.".format(name=name)
        )
        return None

    transforms = get_transform(cfg.transforms)
    common_transforms = get_common_transform(cfg.common_transforms)
    if name in RGB_ONLY_DATASETS:
        return ALL_DATASETS[name](
            **cfg.params, transforms=transforms, common_transforms=common_transforms
        )
    else:
        depth_transforms = get_transform(cfg.depth_transforms)
        return ALL_DATASETS[name](
            **cfg.params,
            transforms=transforms,
            depth_transforms=depth_transforms,
            common_transforms=common_transforms
        )


def get_common_transform(transforms: DictConfig):
    transform_list = []
    if transforms is None:
        return None
    for name in transforms.keys():
        assert name in ALL_TRANSFORM, (
            "{T_name} is not supported transform, please implement it and add it to "
            "ALL_TRANSFORM first.".format(T_name=name)
        )
        if transforms[name].params is not None:
            transform_list.append(ALL_TRANSFORM[name](**transforms[name].params))
        else:
            transform_list.append(ALL_TRANSFORM[name]())
    return CustomCompose(transform_list)


def get_transform(transforms: DictConfig):
    transform_list = []
    if transforms is None:
        return None
    for name in transforms.keys():
        assert name in ALL_TRANSFORM, (
            "{T_name} is not supported transform, please implement it and add it to "
            "ALL_TRANSFORM first.".format(T_name=name)
        )
        if transforms[name].params is not None:
            transform_list.append(ALL_TRANSFORM[name](**transforms[name].params))
        else:
            transform_list.append(ALL_TRANSFORM[name]())
    return T.Compose(transform_list)


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        return data
