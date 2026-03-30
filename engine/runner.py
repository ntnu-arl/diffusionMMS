import os
import json
import torch
from omegaconf import OmegaConf
from utils.logger import get_root_logger
from utils import misc, lr_policy
import time
import datetime
from utils.helper import Timer
from torch.utils.tensorboard import SummaryWriter

logger = get_root_logger()


class Trainer:
    def __init__(self, config, model, optimizer, train_loader, val_loader=None,
                 evaluator=None):
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.cfg = config
        self.train_cfg = config.train
        self.trainer_timer = Timer()
        self.logger = get_root_logger()
        self.evaluator = evaluator

        niters_per_epoch = len(self.train_loader)
        total_iteration = self.train_cfg.num_epochs * niters_per_epoch
        self.scheduler = lr_policy.WarmUpPolyLR(
            self.train_cfg.lr,
            self.train_cfg.lr_power,
            total_iteration,
            self.train_cfg.warmup_iter,
        )

        # Log frequency: every N iterations instead of every iteration
        self.log_interval = getattr(self.train_cfg, "log_interval", 50)

    def train(self):
        train_cfg = self.train_cfg
        self.device = train_cfg.device
        os.makedirs(train_cfg.log_dir, exist_ok=True)
        if train_cfg.log_dir is not None:
            self.log_writer = SummaryWriter(
                log_dir=os.path.join(
                    train_cfg.log_dir,
                    self.cfg.experiment_dataset,
                    self.cfg.experiment_name,
                )
            )

        # cuDNN auto-tuner: finds fastest convolution algorithms for fixed input sizes
        torch.backends.cudnn.benchmark = True

        self.model.to(self.device)

        # AMP: automatic mixed precision
        self.scaler = torch.amp.GradScaler("cuda")

        start_epoch = 1
        if train_cfg.resume is not None:
            logger.info(f"Resume from {train_cfg.resume}")
            start_epoch = misc.load_model_to_resume(
                train_cfg, self.model, optimizer=self.optimizer
            )
        self.model.train()
        start_time = time.time()
        output_dir = os.path.join(
            train_cfg.output_dir, self.cfg.experiment_dataset, self.cfg.experiment_name
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eval_last_n = getattr(train_cfg, "eval_last_n_epochs", 0)
        eval_start_epoch = train_cfg.num_epochs - eval_last_n + 1
        best_iou = 0.0
        best_epoch = -1
        eval_results_path = os.path.join(output_dir, "eval_results.txt")
        self._last_epoch_loss = {}

        for epoch in range(start_epoch, train_cfg.num_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch()

            if (
                train_cfg.output_dir
                and (
                    epoch % train_cfg.saving_interval == 0
                    or epoch == train_cfg.num_epochs
                )
                and epoch >= train_cfg.start_saving_epoch
            ):
                misc.save_model(
                    args=train_cfg,
                    output_dir=output_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                )

            # Evaluate during last N epochs
            if self.evaluator is not None and epoch >= eval_start_epoch:
                self.model.eval()
                result_line, mIoU = self.evaluator.run_inline(epoch)
                self.log_writer.add_scalar("val_mIoU", mIoU, epoch)

                with open(eval_results_path, "a") as f:
                    f.write(f"Epoch {epoch}  mIoU: {mIoU:.4f}\n")
                    f.write(result_line)
                    f.write("\n\n")

                if mIoU > best_iou:
                    best_iou = mIoU
                    best_epoch = epoch

                self.model.train()

        # Create best.pt symlink
        if best_epoch > 0:
            best_ckpt = os.path.join(output_dir, f"checkpoint-{best_epoch}.pth")
            best_link = os.path.join(output_dir, "best.pt")
            if os.path.islink(best_link) or os.path.exists(best_link):
                os.remove(best_link)
            os.symlink(os.path.abspath(best_ckpt), best_link)
            logger.info(
                f"Best epoch: {best_epoch} mIoU: {best_iou:.4f} -> {best_link}"
            )
            with open(eval_results_path, "a") as f:
                f.write(f"Best epoch: {best_epoch}  mIoU: {best_iou:.4f}\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("Training time {}".format(total_time_str))

        # Save experiment summary for tracking
        self._save_experiment_summary(
            output_dir, total_time, best_epoch, best_iou, self._last_epoch_loss
        )

    def _get_hparams(self):
        """Extract key hyperparameters from config as a flat dict."""
        cfg = self.cfg
        train_cfg = self.train_cfg
        crop_size = None
        ct = getattr(
            getattr(train_cfg.dataset, "common_transforms", None),
            "semseg_transform", None,
        )
        if ct is not None:
            crop_size = list(ct.params.crop_size)

        return {
            "backbone": cfg.model.params.backbone.name,
            "decoder": cfg.model.params.decoder.name,
            "num_classes": cfg.model.params.decoder.params.num_classes,
            "pretrained": str(getattr(cfg.model.params, "pretrained", None)),
            "timesteps": cfg.model.params.timesteps,
            "bit_scale": cfg.model.params.bit_scale,
            "accumulation": cfg.model.params.accumulation,
            "aux_rate": cfg.model.params.train_cfg.aux_rate,
            "lr": train_cfg.lr,
            "optimizer": train_cfg.opt_name,
            "weight_decay": train_cfg.weight_decay,
            "batch_size": train_cfg.batch_size,
            "num_epochs": train_cfg.num_epochs,
            "warmup_iter": train_cfg.warmup_iter,
            "lr_power": train_cfg.lr_power,
            "crop_size": crop_size,
            "train_scale_array": list(train_cfg.train_scale_array),
        }

    def _save_experiment_summary(self, output_dir, total_time, best_epoch,
                                 best_iou, last_epoch_loss):
        """Save config snapshot and structured summary to output_dir."""
        # 1. Save full config YAML
        OmegaConf.save(self.cfg, os.path.join(output_dir, "config.yaml"))

        # 2. Build structured summary
        hparams = self._get_hparams()
        final_loss = None
        if last_epoch_loss and "total_loss" in last_epoch_loss:
            v = last_epoch_loss["total_loss"]
            final_loss = float(v.item() if hasattr(v, "item") else v)
            final_loss = round(final_loss / len(self.train_loader), 4)

        summary = {
            "experiment_name": self.cfg.experiment_name,
            "experiment_dataset": self.cfg.experiment_dataset,
            "hparams": hparams,
            "metrics": {
                "best_mIoU": round(best_iou, 4) if best_epoch > 0 else None,
                "best_epoch": best_epoch if best_epoch > 0 else None,
                "final_train_loss": final_loss,
                "training_time_s": round(total_time, 1),
            },
        }
        summary_path = os.path.join(output_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Experiment summary saved to {summary_path}")

        # 3. Log hparams + metrics to TensorBoard
        if hasattr(self, "log_writer"):
            # TensorBoard hparams only accepts scalar values
            tb_hparams = {}
            for k, v in hparams.items():
                if isinstance(v, (int, float, bool)):
                    tb_hparams[k] = v
                elif isinstance(v, str):
                    tb_hparams[k] = v
                elif isinstance(v, list):
                    tb_hparams[k] = str(v)
            tb_metrics = {
                "hparam/best_mIoU": best_iou if best_epoch > 0 else 0.0,
                "hparam/best_epoch": float(best_epoch),
            }
            if final_loss is not None:
                tb_metrics["hparam/final_train_loss"] = final_loss
            self.log_writer.add_hparams(tb_hparams, tb_metrics)

    def train_one_epoch(self):
        cfg = self.train_cfg
        num_iters = len(self.train_loader)
        sum_loss = {}

        for data_iter_step, samples in enumerate(self.train_loader):
            # Non-blocking transfers (works with pin_memory=True)
            rgb = samples["rgb"].to(cfg.device, non_blocking=True)
            depth = samples["depth"].to(cfg.device, non_blocking=True) if "depth" in samples else None
            label = samples["label"].to(cfg.device, non_blocking=True)

            # AMP forward pass
            with torch.amp.autocast("cuda"):
                losses = self.model(rgb, depth, label)

            # Faster than zero_grad(): sets gradients to None instead of zero
            self.optimizer.zero_grad(set_to_none=True)

            # AMP backward + step
            self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            current_step = (self.epoch - 1) * num_iters + data_iter_step
            lr = self.scheduler.get_lr(current_step)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Accumulate losses (detached to avoid holding the graph)
            for key in losses.keys():
                val = losses[key].detach() if hasattr(losses[key], "detach") else losses[key]
                if key not in sum_loss:
                    sum_loss[key] = val
                else:
                    sum_loss[key] += val

            # Log only every N steps to reduce overhead
            if (data_iter_step + 1) % self.log_interval == 0 or (data_iter_step + 1) == num_iters:
                print_loss_str = ""
                for key in sum_loss.keys():
                    print_loss_str += " %s=%.4f" % (
                        key,
                        (sum_loss[key] / (data_iter_step + 1)),
                    )
                print_str = (
                    "Epoch {}/{}".format(self.epoch, cfg.num_epochs)
                    + " Iter {}/{}:".format(data_iter_step + 1, num_iters)
                    + " lr=%.4e" % lr
                    + print_loss_str
                )
                logger.info(print_str)

        self._last_epoch_loss = sum_loss
        self.log_writer.add_scalar(
            "train_loss",
            sum_loss["total_loss"] / num_iters,
            self.epoch,
        )
        self.log_writer.add_scalar("lr", lr, self.epoch)
