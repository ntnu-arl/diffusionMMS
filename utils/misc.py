from pathlib import Path

import torch
from .logger import get_root_logger

logger = get_root_logger()


def load_dual_dat_pretrained_model(model, model_file):
    logger.info(f"Loading pretrained from {model_file}")
    pretrained = torch.load(model_file)
    state_dict = {}
    for key, value in pretrained["state_dict"].items():
        if "backbone." in key:
            new_key = key[9:]
            prefix_depth = new_key.split(".")[0]

            depth_key = prefix_depth + "_d"
            depth_key = depth_key + new_key[len(prefix_depth) :]
            state_dict[new_key] = value
            state_dict[depth_key] = value

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    logger.info("Successfully load DAT++ pretrained model")


def load_model_to_resume(args, model, optimizer):
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        logger.info("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            epoch = checkpoint["epoch"] + 1
            logger.info("With optim & sched!")
            return epoch


def save_model(args, output_dir, epoch, model, optimizer):
    output_dir = Path(output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        torch.save(to_save, checkpoint_path)

    return checkpoint_paths
