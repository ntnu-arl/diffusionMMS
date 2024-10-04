import argparse

from omegaconf import OmegaConf
from engine.evaluator import Evaluator
from utils.logger import get_root_logger
from engine import get_model
import torch

logger = get_root_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")
parser.add_argument("--ckpt", type=str, help="Path to checkpoint")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(1234)
    config = OmegaConf.load(args.config)
    model = get_model(config.model.name, eval=True, **config.model.params)
    segmentor = Evaluator(config, model, args.show)
    iou = segmentor.run(args.ckpt)
