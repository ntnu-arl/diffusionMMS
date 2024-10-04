import os
import argparse

from omegaconf import OmegaConf
from engine.evaluator import Evaluator
from utils.logger import get_root_logger
from engine import get_model
import torch

logger = get_root_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/semseg/nyuv2/dual_dat_small_uper.yaml")
parser.add_argument("--show", action="store_true")
parser.add_argument("--epoch", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(1234)

    config = OmegaConf.load(args.config)
    model = get_model(config.model.name, eval=True, **config.model.params)
    checkpoint_path = os.path.join(
        "output_dir/",
        config.experiment_dataset,
        config.experiment_name,
        "checkpoint-" + str(args.epoch) + ".pth",
    )
    segmentor = Evaluator(config, model, args.show)
    segmentor.run_once(checkpoint_path)
