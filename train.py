import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from datasets import get_dataset
from engine import get_model, get_optimizer
from engine.runner import Trainer
from engine.evaluator import Evaluator
from utils.init_func import group_weight

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config file")


def setup_ddp():
    """Initialize DDP from torchrun environment variables.
    Returns (rank, local_rank, world_size) or (0, 0, 1) for single-GPU."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main(config):
    rank, local_rank, world_size = setup_ddp()
    distributed = world_size > 1

    train_cfg = config.train
    train_dataset = get_dataset(config.experiment_dataset, train_cfg.dataset)

    num_workers = train_cfg.num_workers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=train_cfg.drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model = get_model(model_name=config.model.name, **config.model.params)
    model = model.cuda(local_rank)
    model = torch.compile(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # group_weight needs the unwrapped model
    raw_model = model.module if distributed else model
    opt_params = group_weight(raw_model, config.model.params.norm_layer, train_cfg.lr)

    optimizer = get_optimizer(
        opt_name=train_cfg.opt_name,
        params=opt_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    evaluator = None
    eval_last_n = getattr(train_cfg, "eval_last_n_epochs", 0)
    if eval_last_n > 0 and rank == 0:
        evaluator = Evaluator(config, raw_model, show=False)

    runner = Trainer(config, model, optimizer, train_loader, val_loader=None,
                     evaluator=evaluator, rank=rank, world_size=world_size)
    runner.train()

    cleanup_ddp()


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(1234)
    config = OmegaConf.load(args.config)
    main(config=config)
