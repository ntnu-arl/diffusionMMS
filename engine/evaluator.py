import os
import cv2
import numpy as np
from tqdm import tqdm
from datasets import get_dataset

import torch
from torch.utils.data import DataLoader
from utils.logger import get_root_logger
from utils.helper import get_class_colors, print_iou
from .metric import hist_info, compute_score, goose_compute_scores, GOOSE_FINE_IDS, GOOSE_COARSE_NAMES
import time
logger = get_root_logger()


class Evaluator:
    def __init__(self, cfg, model, show=False):
        self.show_image = show
        self.cfg = cfg
        self.val_cfg = cfg.val
        self.model = model

        val_dataset = get_dataset(cfg.experiment_dataset, self.val_cfg.dataset)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.val_cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=getattr(self.val_cfg, "num_workers", 4),
        )
        self.class_names = val_dataset.get_classname()
        self.num_classes = len(self.class_names)
        self.val_logdir = os.path.join(
            self.val_cfg.log_dir,
            self.cfg.experiment_dataset,
            self.cfg.experiment_type,
            self.cfg.experiment_name,
        )
        if "exclude" in self.val_cfg:
            logger.info("ignore some label")
            self.excluded_labels = [
                i
                for i, elem in enumerate(self.class_names)
                if elem in list(self.val_cfg.exclude)
            ]
        else:
            self.excluded_labels = np.array([])

    def run_once(self, model_file, need_load=True):
        if not os.path.exists(self.val_logdir):
            os.makedirs(self.val_logdir)
        with open(os.path.join(self.val_logdir, "result.txt"), "a") as f:
            if not need_load:
                logger.info(f"Evaluate while training")
            else:
                logger.info(f"Loading weight from {model_file}")
                self.model.load_state_dict(torch.load(model_file, weights_only=False)["model"])

        for i, data in enumerate(tqdm(self.val_loader)):
            label = data["label"].squeeze(1)
            pred = self.eval(data)
            self.visualize(label, data, pred, i)

    def test_image(self, rgb, depth, idx):
        data = dict()
        data["rgb"] = rgb
        data["depth"] = depth
        pred = self.eval(data)
        self.visualize(None, data, pred, idx)

    def run(self, model_file, need_load=True):
        if not os.path.exists(self.val_logdir):
            os.makedirs(self.val_logdir)
        with open(os.path.join(self.val_logdir, "result.txt"), "a") as f:
            if not need_load:
                logger.info(f"Evaluate while training")
            else:
                logger.info(f"Loading weight from {model_file}")
                self.model.load_state_dict(torch.load(model_file, weights_only=False)["model"])
            all_results = []

            for _, data in enumerate(tqdm(self.val_loader)):
                label = data["label"].squeeze(1)
                pred = self.eval(data)
                pred_trunc = np.array(pred.cpu())
                pred_trunc[pred_trunc >= self.num_classes] = self.num_classes
                hist_tmp, labeled_tmp, correct_tmp = hist_info(
                    self.num_classes,
                    pred_trunc,
                    np.array(label.cpu()),
                    excluded_labels=self.excluded_labels,
                )
                results_dict = {
                    "hist": hist_tmp,
                    "labeled": labeled_tmp,
                    "correct": correct_tmp,
                }
                self.visualize(label, data, pred)
                all_results.append(results_dict)

            result_line, meanIoU = self.compute_metric(all_results)
            f.write("Model: " + str(model_file) + "\n")
            f.write(result_line)
            f.write("\n")
            f.flush()
        return meanIoU

    def compute_metric(self, results):
        hist = np.zeros((self.num_classes, self.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d["hist"]
            correct += d["correct"]
            labeled += d["labeled"]
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
            hist, correct, labeled
        )
        result_line = print_iou(
            iou,
            freq_IoU,
            mean_pixel_acc,
            pixel_acc,
            self.class_names,
            show_no_back=False,
        )

        # GOOSE competition metrics
        if self.cfg.experiment_dataset == "goose":
            mIoU_fine, mIoU_coarse, mIoU_composite, fine_ious, coarse_ious = \
                goose_compute_scores(hist, self.num_classes)
            goose_lines = [
                "",
                "===== GOOSE Competition Metrics =====",
                "--- mIoU_fine (56 classes) ---",
            ]
            for idx, c in enumerate(GOOSE_FINE_IDS):
                if idx < len(fine_ious):
                    goose_lines.append(
                        "  %d %-25s\t%.3f%%" % (c, self.class_names[c], fine_ious[idx] * 100)
                    )
            goose_lines.append("--- mIoU_coarse (11 superclasses) ---")
            for idx, name in enumerate(GOOSE_COARSE_NAMES):
                if idx < len(coarse_ious):
                    goose_lines.append(
                        "  %-25s\t%.3f%%" % (name, coarse_ious[idx] * 100)
                    )
            goose_lines.append("-------------------------------------")
            goose_lines.append("mIoU_fine:      %.3f%%" % (mIoU_fine * 100))
            goose_lines.append("mIoU_coarse:    %.3f%%" % (mIoU_coarse * 100))
            goose_lines.append("mIoU_composite: %.3f%%" % (mIoU_composite * 100))
            goose_lines.append("=====================================")
            goose_str = "\n".join(goose_lines)
            print(goose_str)
            result_line += goose_str
            return result_line, mIoU_composite

        return result_line, mean_IoU

    def visualize(self, label, data, pred, idx=None):
        NORM_RGB = {
            "mean": np.array([0.485, 0.456, 0.406]),
            "std": np.array([0.229, 0.224, 0.225]),
        }
        # Get color corresponding to each classes
        colors = np.array(get_class_colors(self.num_classes + 1))

        # Convert data to unit8 numpy type on cpu
        pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        rgb_arr = data["rgb"].squeeze(0).cpu().numpy()
        pred_arr[pred_arr > self.num_classes] = self.num_classes

        has_depth = "depth" in data and data["depth"] is not None
        if has_depth:
            depth_arr = data["depth"].squeeze(0).cpu().numpy()
        colored_pred = np.zeros_like(pred_arr)
        colored_pred = np.stack((colored_pred,) * 3, axis=-1)
        colored_pred[:] = colors[pred_arr[:]]

        if label is not None:
            label_arr = label.squeeze(0).cpu().numpy().astype(np.uint8)
            # Convert void classes label from 255 to self.num_classes
            label_arr[label_arr == 255] = self.num_classes
            # Colorize groundtruth and prediction
            colored_label = np.zeros_like(label_arr)
            colored_label = np.stack((colored_label,) * 3, axis=-1)
            colored_label[:] = colors[label_arr[:]]
        else:
            colored_label = colored_pred

        # Overlay prediction on input images
        rgb_arr = rgb_arr.transpose(1, 2, 0)
        rgb_arr = ((rgb_arr * NORM_RGB["std"] + NORM_RGB["mean"]) * 255).astype(
            np.uint8
        )
        rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        # Concatenate multiple outputs for saving
        if has_depth:
            depth_arr = (depth_arr.transpose(1, 2, 0) * 255).astype(np.uint8)
            output = np.concatenate(
                [rgb_arr, depth_arr, colored_label, colored_pred], axis=1
            )
        else:
            output = np.concatenate(
                [rgb_arr, colored_label, colored_pred], axis=1
            )

        # Save results
        if self.val_cfg.save_path is not None:
            if not os.path.exists(self.val_cfg.save_path):
                os.makedirs(self.val_cfg.save_path)
            cv2.imwrite(
                os.path.join(self.val_cfg.save_path, str(idx).zfill(6) + ".png"), output
            )

        # Show results
        if self.show_image:
            cv2.imshow("pred", output)
            if cv2.waitKey() == ord("q"):
                exit(0)

    def run_inline(self, epoch):
        """Evaluate model already in memory (no checkpoint loading).
        Returns (result_line, meanIoU) — for GOOSE, meanIoU is mIoU_composite."""
        all_results = []
        for _, data in enumerate(tqdm(self.val_loader)):
            label = data["label"].squeeze(1)
            pred = self.eval(data)
            pred_trunc = np.array(pred.cpu())
            pred_trunc[pred_trunc >= self.num_classes] = self.num_classes
            hist_tmp, labeled_tmp, correct_tmp = hist_info(
                self.num_classes,
                pred_trunc,
                np.array(label.cpu()),
                excluded_labels=self.excluded_labels,
            )
            all_results.append({
                "hist": hist_tmp,
                "labeled": labeled_tmp,
                "correct": correct_tmp,
            })

        result_line, metric = self.compute_metric(all_results)
        logger.info(f"Epoch {epoch} mIoU: {metric:.4f}")
        return result_line, metric

    def eval_with_loss(self, data):
        """Run inference and compute validation loss. Returns (pred, loss_scalar)."""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda(non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            depth = data.get("depth", None)
            label = data["label"].squeeze(1)
            # Compute loss via the training forward pass
            losses = self.model(data["rgb"], depth, label)
            loss_val = losses["total_loss"].item()
            # Get predictions via sampling (diffusion inference)
            score = self.model.sampling(data["rgb"], depth)
        pred = score.argmax(1)
        return pred, loss_val

    def eval(self, data):
        # model.eval() and .cuda() are assumed to be set by the caller
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda(non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            depth = data.get("depth", None)
            score = self.model.sampling(data["rgb"], depth)
        pred = score.argmax(1)
        return pred
