import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from utils.logger import get_root_logger
from utils.helper import convert_depth_to_three_channel_img, darken

logger = get_root_logger()


class DepthDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        lowlight=False,
        transforms=None,
        depth_transforms=None,
        common_transforms=None,
    ):
        self.CLASSES = []
        self.update_classname()
        self.dataset_name = None
        self.root_dir = root
        self.split = split
        self.lowlight = lowlight
        self.transforms = transforms
        self.depth_transforms = depth_transforms
        self.common_transforms = common_transforms
        if lowlight:
            self.rgb_path = os.path.join(self.root_dir, "dark")
        else:
            self.rgb_path = os.path.join(self.root_dir, "image")
        self.depth_path = os.path.join(self.root_dir, "depth")
        self.label_path = os.path.join(self.root_dir, "seglabel")
        all_index_file = os.path.join(self.root_dir, split + ".txt")

        if not os.path.exists(all_index_file):
            raise Exception(f"File does not exist {all_index_file}")

        with open(all_index_file, "r") as f:
            self.all_index = [int(idx) for idx in f.readlines()]

    def __getitem__(self, index):
        output = {}
        file_index = self.all_index[index]

        if self.dataset_name == "nyuv2":
            file_index = str(file_index)
        elif self.dataset_name == "sunrgbd":
            file_index = str(file_index).zfill(6)
        else:
            raise NotImplementedError
        # Read all necessary types of image
        rgb = Image.open(os.path.join(self.rgb_path, file_index + ".jpg")).convert(
            "RGB"
        )
        if self.lowlight:
            rgb = darken(rgb)

        if self.dataset_name == "nyuv2":
            depth = np.load(os.path.join(self.depth_path, file_index + ".npy"))
        elif self.dataset_name == "sunrgbd":
            depth = np.array(
                Image.open(os.path.join(self.depth_path, file_index + ".png"))
            )

        depth = convert_depth_to_three_channel_img(depth)
        depth = Image.fromarray(depth)
        output["rgb"] = rgb
        output["depth"] = depth

        label = cv2.imread(
            os.path.join(self.label_path, file_index + ".png"), cv2.IMREAD_GRAYSCALE
        )
        label = label - 1
        output["label"] = label

        if self.common_transforms is not None:
            output = self.common_transforms(**output)

        # Transform image
        if self.transforms is not None:
            output["rgb"] = self.transforms(output["rgb"])

        if self.depth_transforms is not None:
            output["depth"] = self.depth_transforms(output["depth"])

        # Return output as a dictionary
        if rgb is None and depth is None:
            logger.error("Receive NoneType")

        return output

    def get_classname(self):
        return self.CLASSES

    def num_classes(self):
        return len(self.CLASSES)

    def update_classname(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.all_index)


class NYUv2Dataset(DepthDataset):
    def __init__(
        self,
        root,
        split,
        lowlight=False,
        transforms=None,
        depth_transforms=None,
        common_transforms=None,
    ):
        super(NYUv2Dataset, self).__init__(
            root,
            split,
            lowlight,
            transforms,
            depth_transforms,
            common_transforms,
        )

        self.dataset_name = "nyuv2"

    def update_classname(self):
        self.CLASSES = [
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "blinds",
            "desk",
            "shelves",
            "curtain",
            "dresser",
            "pillow",
            "mirror",
            "floor mat",
            "clothes",
            "ceiling",
            "books",
            "refridgerator",
            "television",
            "paper",
            "towel",
            "shower curtain",
            "box",
            "whiteboard",
            "person",
            "night stand",
            "toilet",
            "sink",
            "lamp",
            "bathtub",
            "bag",
            "otherstructure",
            "otherfurniture",
            "otherprop",
        ]


class SunRGBDDataset(DepthDataset):
    def __init__(
        self,
        root,
        split,
        lowlight=False,
        transforms=None,
        depth_transforms=None,
        common_transforms=None,
    ):
        super(SunRGBDDataset, self).__init__(
            root,
            split,
            lowlight,
            transforms,
            depth_transforms,
            common_transforms,
        )
        self.dataset_name = "sunrgbd"

    def update_classname(self):
        self.CLASSES = [
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "blinds",
            "desk",
            "shelves",
            "curtain",
            "dresser",
            "pillow",
            "mirror",
            "floor mat",
            "clothes",
            "ceiling",
            "books",
            "fridge",
            "tv",
            "paper",
            "towel",
            "shower curtain",
            "box",
            "whiteboard",
            "person",
            "night stand",
            "toilet",
            "sink",
            "lamp",
            "bathtub",
            "bag",
        ]
