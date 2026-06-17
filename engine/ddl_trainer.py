import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from data.ddl_dataset_utils import batch_to_labels
from engine.base_trainer import Trainer
from utils.validate import _mask_to_boxes, aggregate_step_metrics, summarize_ddl_metrics, sweep_metric_suffix


class Trainer_DDL(Trainer):
    """Lightning training loop for detection and local mask localization."""

    def __init__(self, opt):
        super().__init__(opt)
        self.validation_step_outputs = []
        self.validation_visuals = []

    @staticmethod
    def _as_list(value, default):
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return list(value)
        return list(value)

    def _bbox_eval_settings(self):
        conf = getattr(self.opt.train, "bbox_eval", None)
        fake_threshold = float(getattr(conf, "fake_threshold", 0.5) if conf is not None else 0.5)
        mask_threshold = float(getattr(conf, "mask_threshold", 0.5) if conf is not None else 0.5)
        min_box_area = int(getattr(conf, "min_box_area", 16) if conf is not None else 16)
        mask_thresholds = self._as_list(
            getattr(conf, "mask_thresholds", None) if conf is not None else None,
            [0.30, 0.40, 0.50, 0.60],
        )
        min_box_areas = self._as_list(
            getattr(conf, "min_box_areas", None) if conf is not None else None,
            [8, 16, 32, 64],
        )
        bbox_sweep = []
        seen = set()
        for sweep_mask_threshold in mask_thresholds:
            for sweep_min_box_area in min_box_areas:
                item = (float(sweep_mask_threshold), int(sweep_min_box_area))
                if item in seen:
                    continue
                seen.add(item)
                bbox_sweep.append(item)
        default_item = (mask_threshold, min_box_area)
        if default_item not in seen:
            bbox_sweep.append(default_item)
        return {
            "fake_threshold": fake_threshold,
            "mask_threshold": mask_threshold,
            "min_box_area": min_box_area,
            "bbox_sweep": bbox_sweep,
        }

    def _shared_step(self, batch, stage: str):
        images = batch["pixel_values"]
        labels = batch_to_labels(batch, images.device)
        outputs = self.model(images, labels=labels)
        loss = outputs["loss"]

        bbox_eval = self._bbox_eval_settings()
        metrics = summarize_ddl_metrics(
            outputs,
            labels,
            fake_threshold=bbox_eval["fake_threshold"],
            mask_threshold=bbox_eval["mask_threshold"],
            min_box_area=bbox_eval["min_box_area"],
            bbox_sweep=bbox_eval["bbox_sweep"] if stage == "val" else None,
        )
        on_step = stage == "train"
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=on_step, on_epoch=True, sync_dist=True)

        loss_dict = outputs.get("loss_dict", {})
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                self.log(f"{stage}_{key}", value.detach(), on_step=on_step, on_epoch=True, sync_dist=True)

        return loss, metrics, outputs

    def _should_collect_visuals(self) -> bool:
        if not getattr(self.trainer, "is_global_zero", False):
            return False
        wandb_conf = getattr(self.opt.train, "wandb", None)
        local_conf = getattr(self.opt.train, "local_val_visuals", None)
        wandb_enabled = bool(getattr(wandb_conf, "enabled", False)) if wandb_conf is not None else False
        local_enabled = bool(getattr(local_conf, "enabled", False)) if local_conf is not None else False
        if not wandb_enabled and not local_enabled:
            return False
        every_n = int(getattr(wandb_conf, "log_val_images_every_n_epochs", 1) or 1)
        if local_enabled:
            every_n = int(getattr(local_conf, "every_n_epochs", every_n) or every_n)
        return (self.current_epoch + 1) % max(1, every_n) == 0

    def _max_visual_items(self) -> int:
        wandb_conf = getattr(self.opt.train, "wandb", None)
        local_conf = getattr(self.opt.train, "local_val_visuals", None)
        wandb_max = int(getattr(wandb_conf, "num_val_images", 4) or 4) if wandb_conf is not None else 4
        local_max = int(getattr(local_conf, "num_val_images", wandb_max) or wandb_max) if local_conf is not None else wandb_max
        return max(wandb_max, local_max)

    @staticmethod
    def _denormalize_image(image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(3, 1, 1)
        return (image * std + mean).clamp(0.0, 1.0)

    @staticmethod
    def _wandb_box_data(boxes, class_id: int, caption: str):
        return [
            {
                "position": {
                    "minX": float(x1),
                    "minY": float(y1),
                    "maxX": float(x2),
                    "maxY": float(y2),
                },
                "class_id": int(class_id),
                "box_caption": caption,
            }
            for x1, y1, x2, y2 in boxes
        ]

    def _collect_validation_visuals(self, batch, outputs) -> None:
        max_items = self._max_visual_items()
        remaining = max_items - len(self.validation_visuals)
        if remaining <= 0:
            return

        images = batch["pixel_values"][:remaining].detach().cpu()
        masks = batch["mask"][:remaining].detach().cpu()
        labels = batch["label"][:remaining].detach().cpu()
        logits = outputs["logits"][:remaining].detach().cpu()
        pred_masks = outputs["pred_mask"][:remaining].detach().cpu()
        patch_score_maps = outputs["patch_score_map"][:remaining].detach().cpu()
        segment_score_maps = outputs["segment_score_map"][:remaining].detach().cpu()
        uids = batch["uid"][:remaining]

        for image, mask, label, logit, pred_mask, patch_score_map, segment_score_map, uid in zip(
            images,
            masks,
            labels,
            logits,
            pred_masks,
            patch_score_maps,
            segment_score_maps,
            uids,
        ):
            self.validation_visuals.append(
                {
                    "uid": str(uid),
                    "image": self._denormalize_image(image),
                    "mask": mask.squeeze(0),
                    "pred_mask": pred_mask.squeeze(0),
                    "patch_score_map": patch_score_map.squeeze(0),
                    "segment_score_map": segment_score_map.squeeze(0),
                    "label": int(label.item()),
                    "pred_prob": float(torch.sigmoid(logit).item()),
                }
            )

    def _find_wandb_logger(self):
        logger = getattr(self, "logger", None)
        if logger is None:
            return None
        if logger.__class__.__name__ == "WandbLogger":
            return logger
        if hasattr(logger, "_logger_iterable"):
            for candidate in logger._logger_iterable:
                if candidate.__class__.__name__ == "WandbLogger":
                    return candidate
        return None

    def _log_validation_visuals(self) -> None:
        if not self.validation_visuals or wandb is None:
            return

        wandb_logger = self._find_wandb_logger()
        if wandb_logger is None or getattr(wandb_logger, "experiment", None) is None:
            return

        images = []
        patch_score_images = []
        segment_score_images = []
        bbox_eval = self._bbox_eval_settings()
        mask_threshold = float(bbox_eval["mask_threshold"])
        min_box_area = int(bbox_eval["min_box_area"])
        fake_threshold = float(bbox_eval["fake_threshold"])
        for item in self.validation_visuals:
            image_np = item["image"].permute(1, 2, 0).numpy()
            gt_mask = (item["mask"] > 0.5).to(torch.int32).numpy()
            pred_mask = (item["pred_mask"] >= mask_threshold).to(torch.int32).numpy()
            gt_boxes = _mask_to_boxes(gt_mask, min_area=1)
            pred_boxes = (
                _mask_to_boxes(pred_mask, min_area=min_box_area)
                if item["pred_prob"] >= fake_threshold
                else []
            )
            caption = (
                f"uid={item['uid']} label={item['label']} "
                f"pred_fake_prob={item['pred_prob']:.4f} "
                f"mask_t={mask_threshold:.2f} min_area={min_box_area}"
            )
            images.append(
                wandb.Image(
                    image_np,
                    caption=caption,
                    masks={
                        "ground_truth": {"mask_data": gt_mask},
                        "prediction": {"mask_data": pred_mask},
                    },
                    boxes={
                        "ground_truth": {
                            "box_data": self._wandb_box_data(gt_boxes, 0, "gt"),
                            "class_labels": {0: "gt"},
                        },
                        "prediction": {
                            "box_data": self._wandb_box_data(pred_boxes, 1, "pred"),
                            "class_labels": {1: "pred"},
                        },
                    },
                )
            )
            patch_score_images.append(
                wandb.Image(
                    item["patch_score_map"].numpy(),
                    caption=f"uid={item['uid']} patch_score_map",
                )
            )
            segment_score_images.append(
                wandb.Image(
                    item["segment_score_map"].numpy(),
                    caption=f"uid={item['uid']} segment_score_map",
                )
            )

        wandb_logger.experiment.log(
            {
                "val/examples": images,
                "val/patch_score_map": patch_score_images,
                "val/segment_score_map": segment_score_images,
                "trainer/global_step": self.global_step,
            },
            step=self.global_step,
        )

    @staticmethod
    def _safe_filename(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"

    @staticmethod
    def _overlay_mask(image: Image.Image, mask, color: tuple[int, int, int], alpha: int) -> Image.Image:
        mask_array = (mask > 0).astype("uint8") * alpha
        color_layer = Image.new("RGBA", image.size, (*color, 0))
        alpha_layer = Image.fromarray(mask_array, mode="L").resize(image.size, resample=Image.Resampling.NEAREST)
        color_layer.putalpha(alpha_layer)
        return Image.alpha_composite(image.convert("RGBA"), color_layer)

    def _local_visual_dir(self) -> Path:
        local_conf = getattr(self.opt.train, "local_val_visuals", None)
        configured_dir = getattr(local_conf, "dir", None) if local_conf is not None else None
        if configured_dir:
            root = Path(str(configured_dir))
        else:
            log_dir = getattr(getattr(self.trainer, "checkpoint_callback", None), "dirpath", None)
            root = Path(log_dir) if log_dir else Path("logs") / str(getattr(self.opt, "name", "ddl"))
            root = root / "val_visuals"
        return root / f"epoch_{int(self.current_epoch):04d}"

    def _save_local_validation_visuals(self) -> None:
        local_conf = getattr(self.opt.train, "local_val_visuals", None)
        if not self.validation_visuals or local_conf is None or not bool(getattr(local_conf, "enabled", False)):
            return
        if not getattr(self.trainer, "is_global_zero", False):
            return

        max_items = int(getattr(local_conf, "num_val_images", self._max_visual_items()) or self._max_visual_items())
        output_dir = self._local_visual_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        bbox_eval = self._bbox_eval_settings()
        mask_threshold = float(bbox_eval["mask_threshold"])
        min_box_area = int(bbox_eval["min_box_area"])
        fake_threshold = float(bbox_eval["fake_threshold"])

        for index, item in enumerate(self.validation_visuals[:max_items]):
            image_np = (item["image"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
            image = Image.fromarray(image_np).convert("RGBA")
            gt_mask = (item["mask"] > 0.5).to(torch.int32).numpy()
            pred_mask = (item["pred_mask"] >= mask_threshold).to(torch.int32).numpy()
            gt_boxes = _mask_to_boxes(gt_mask, min_area=1)
            pred_boxes = (
                _mask_to_boxes(pred_mask, min_area=min_box_area)
                if item["pred_prob"] >= fake_threshold
                else []
            )

            image = self._overlay_mask(image, gt_mask, (30, 180, 80), 70)
            image = self._overlay_mask(image, pred_mask, (230, 70, 60), 85)
            draw = ImageDraw.Draw(image)
            for box in gt_boxes:
                draw.rectangle(box, outline=(30, 220, 80, 255), width=3)
            for box in pred_boxes:
                draw.rectangle(box, outline=(255, 60, 50, 255), width=3)
            draw.text(
                (8, 8),
                (
                    f"uid={item['uid']} label={item['label']} prob={item['pred_prob']:.4f} "
                    f"mask_t={mask_threshold:.2f} min_area={min_box_area}"
                ),
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )

            filename = f"{index:03d}_{self._safe_filename(str(item['uid']))}.png"
            image.convert("RGB").save(output_dir / filename)

    def training_step(self, batch, batch_idx: int):
        loss, metrics, _ = self._shared_step(batch, stage="train")
        self.log(
            "train_acc",
            metrics["acc_sum"] / metrics["num_samples"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        _, metrics, outputs = self._shared_step(batch, stage="val")
        self.validation_step_outputs.append(
            {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in metrics.items()
            }
        )
        if self._should_collect_visuals():
            self._collect_validation_visuals(batch, outputs)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        metrics = aggregate_step_metrics(self.validation_step_outputs)
        bbox_eval = self._bbox_eval_settings()
        loggable = {
            "val_acc_epoch": metrics["acc"],
            "val_mask_iou_epoch": metrics["mask_iou"],
            "val_bbox_iou_epoch": metrics["bbox_iou"],
        }
        best_sweep = None
        for mask_threshold, min_box_area in bbox_eval["bbox_sweep"]:
            suffix = sweep_metric_suffix(mask_threshold, min_box_area)
            metric_key = f"bbox_iou_{suffix}"
            if metric_key not in metrics:
                continue
            metric_value = float(metrics[metric_key])
            loggable[f"val_bbox_sweep/{suffix}"] = metric_value
            if best_sweep is None or metric_value > best_sweep[2]:
                best_sweep = (float(mask_threshold), int(min_box_area), metric_value)

        if best_sweep is not None:
            loggable["val_bbox_iou_best_epoch"] = best_sweep[2]
            loggable["val_bbox_iou_best_mask_threshold"] = best_sweep[0]
            loggable["val_bbox_iou_best_min_box_area"] = float(best_sweep[1])

        for key, value in loggable.items():
            self.log(key, value, prog_bar=(key == "val_acc_epoch"), logger=True, sync_dist=True)

        if getattr(self.trainer, "is_global_zero", False):
            self.print(
                "[val]"
                f"[epoch={self.current_epoch}]"
                f" acc={metrics['acc']:.4f}"
                f" mask_iou={metrics['mask_iou']:.4f}"
                f" bbox_iou={metrics['bbox_iou']:.4f}"
            )
            if best_sweep is not None:
                self.print(
                    "[val_bbox_sweep]"
                    f"[epoch={self.current_epoch}]"
                    f" best_iou={best_sweep[2]:.4f}"
                    f" mask_threshold={best_sweep[0]:.2f}"
                    f" min_box_area={best_sweep[1]}"
                )

        self.validation_step_outputs.clear()
        self._log_validation_visuals()
        self._save_local_validation_visuals()
        self.validation_visuals.clear()
