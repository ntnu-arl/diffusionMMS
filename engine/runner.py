import os
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

        self.model.to(self.device)

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

    def train_one_epoch(self):
        cfg = self.train_cfg
        for data_iter_step, samples in enumerate(self.train_loader):
            rgb = samples["rgb"].to(cfg.device)
            depth = samples["depth"].to(cfg.device) if "depth" in samples else None
            label = samples["label"].to(cfg.device)

            losses = self.model(rgb, depth, label)
            if data_iter_step == 0:
                sum_loss = dict()
                for key in losses.keys():
                    sum_loss[key] = 0

            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()

            current_step = (self.epoch - 1) * len(self.train_loader) + data_iter_step
            lr = self.scheduler.get_lr(current_step)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            print_loss_str = ""

            for key in losses.keys():
                sum_loss[key] += losses[key]
                print_loss_str += " %s=%.4f" % (
                    key,
                    (sum_loss[key] / (data_iter_step + 1)),
                )

            print_str = (
                "Epoch {}/{}".format(self.epoch, cfg.num_epochs)
                + " Iter {}/{}:".format(data_iter_step + 1, len(self.train_loader))
                + " lr=%.4e" % lr
                + print_loss_str
            )

            del losses
            logger.info(print_str)

        self.log_writer.add_scalar(
            "train_loss",
            sum_loss["total_loss"] / len(self.train_loader),
            self.epoch,
        )
        self.log_writer.add_scalar("lr", lr, self.epoch)
