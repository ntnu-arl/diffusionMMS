import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from engine import get_model, get_optimizer
from engine.runner import Trainer
from engine.evaluator import Evaluator
from utils.init_func import group_weight
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config file")


def main(config):
    train_cfg = config.train
    train_dataset = get_dataset(config.experiment_dataset, train_cfg.dataset)

    num_workers = train_cfg.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=train_cfg.drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = get_model(model_name=config.model.name, **config.model.params)
    opt_params = group_weight(model, config.model.params.norm_layer, train_cfg.lr)

    optimizer = get_optimizer(
        opt_name=train_cfg.opt_name,
        params=opt_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    evaluator = None
    eval_last_n = getattr(train_cfg, "eval_last_n_epochs", 0)
    if eval_last_n > 0:
        evaluator = Evaluator(config, model, show=False)

    runner = Trainer(config, model, optimizer, train_loader, val_loader=None,
                     evaluator=evaluator)
    runner.train()


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(1234)
    config = OmegaConf.load(args.config)
    main(config=config)
