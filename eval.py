import os
import argparse

from omegaconf import OmegaConf
from engine.evaluator import Evaluator
from utils.logger import get_root_logger
from engine import get_model
import torch

logger = get_root_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="path to config file")
parser.add_argument("--show", action="store_true", help="show results")
parser.add_argument("--fr", type=int, help="starting epoch to evaluate")
parser.add_argument("--to", type=int, help="end epoch to evaluate")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(1234)

    config = OmegaConf.load(args.config)
    model = get_model(config.model.name, eval=True, **config.model.params)
    model.cuda()
    model.eval()
    segmentor = Evaluator(config, model, args.show)
    best_iou = 0
    for epoch in range(args.fr, args.to + 1):
        checkpoint_path = os.path.join(
            "output_dir/",
            config.experiment_dataset,
            config.experiment_name,
            "checkpoint-" + str(epoch) + ".pth",
        )
        iou = segmentor.run(checkpoint_path)
        if iou > best_iou:
            best_iou = iou
            best_epoch = epoch
    print(f"Best epoch: {best_epoch}")
