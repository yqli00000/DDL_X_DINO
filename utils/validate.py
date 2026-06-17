import cv2
import numpy as np
import torch


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if denominator.item() <= 0:
        return numerator.new_tensor(float("nan"))
    return numerator / denominator


def _safe_ratio_or_zero(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    if denominator.item() <= 0:
        return numerator.new_tensor(0.0)
    return numerator / denominator


def sweep_metric_suffix(mask_threshold: float, min_box_area: int) -> str:
    threshold_token = f"{float(mask_threshold):.2f}".replace(".", "p")
    return f"t{threshold_token}_a{int(min_box_area)}"


def _mask_to_boxes(mask: np.ndarray, min_area: int = 16):
    binary = (mask >= 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for component_idx in range(1, num_components):
        x, y, w, h, area = stats[component_idx]
        if int(area) < int(min_area):
            continue
        boxes.append((float(x), float(y), float(x + w - 1), float(y + h - 1)))
    return boxes


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1 + 1.0)
    ih = max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _best_match_bbox_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, min_area: int = 16) -> float:
    pred_boxes = _mask_to_boxes(pred_mask.detach().cpu().numpy(), min_area=min_area)
    gt_boxes = _mask_to_boxes(gt_mask.detach().cpu().numpy(), min_area=1)

    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    per_gt_iou = []
    for gt_box in gt_boxes:
        per_gt_iou.append(max(_box_iou(pred_box, gt_box) for pred_box in pred_boxes))
    return float(sum(per_gt_iou) / len(per_gt_iou))


def _bbox_iou_sum_for_params(
    pred_mask_probs: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_fake: torch.Tensor,
    *,
    mask_threshold: float,
    min_box_area: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    bbox_ious = []
    for sample_pred_fake, sample_pred_mask, sample_gt_mask in zip(pred_fake, pred_mask_probs, gt_masks):
        if int(sample_pred_fake.item()) == 0:
            bbox_ious.append(0.0)
            continue
        bbox_ious.append(
            _best_match_bbox_iou(
                sample_pred_mask >= float(mask_threshold),
                sample_gt_mask,
                min_area=int(min_box_area),
            )
        )
    return (
        torch.tensor(float(sum(bbox_ious)), device=device),
        torch.tensor(float(len(bbox_ious)), device=device),
    )


def summarize_ddl_metrics(
    outputs,
    labels,
    *,
    fake_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_box_area: int = 16,
    bbox_sweep: list[tuple[float, int]] | None = None,
):
    device = outputs["logits"].device
    label = labels["label"].long()
    fake_target = (label != 0).long()
    num_samples = torch.tensor(float(label.numel()), device=device)

    pred_fake = (torch.sigmoid(outputs["logits"]) >= float(fake_threshold)).long()
    fake_acc_sum = (pred_fake == fake_target).sum().to(torch.float32)
    fake_mask = fake_target == 1
    metrics = {
        "acc_sum": fake_acc_sum.detach(),
        "num_samples": num_samples.detach(),
    }
    if fake_mask.any():
        pred_mask = (outputs["pred_mask"][fake_mask] >= float(mask_threshold)).to(torch.float32)
        gt_mask = labels["mask"][fake_mask].to(torch.float32)
        inter = (pred_mask * gt_mask).flatten(start_dim=1).sum(dim=1)
        union = ((pred_mask + gt_mask) > 0).to(torch.float32).flatten(start_dim=1).sum(dim=1)
        per_sample_iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
        mask_iou_sum = per_sample_iou.sum().to(torch.float32)
        mask_count = torch.tensor(float(per_sample_iou.numel()), device=device)

        pred_fake_logits = pred_fake[fake_mask]
        pred_mask_for_boxes = outputs["pred_mask"][fake_mask, 0]
        gt_mask_for_boxes = labels["mask"][fake_mask, 0].to(torch.float32)
        bbox_iou_sum, bbox_count = _bbox_iou_sum_for_params(
            pred_mask_for_boxes,
            gt_mask_for_boxes,
            pred_fake_logits,
            mask_threshold=mask_threshold,
            min_box_area=min_box_area,
            device=device,
        )
        for sweep_mask_threshold, sweep_min_box_area in bbox_sweep or []:
            suffix = sweep_metric_suffix(sweep_mask_threshold, sweep_min_box_area)
            sweep_sum, sweep_count = _bbox_iou_sum_for_params(
                pred_mask_for_boxes,
                gt_mask_for_boxes,
                pred_fake_logits,
                mask_threshold=sweep_mask_threshold,
                min_box_area=sweep_min_box_area,
                device=device,
            )
            metrics[f"bbox_iou_sum_{suffix}"] = sweep_sum.detach()
            metrics[f"bbox_count_{suffix}"] = sweep_count.detach()
    else:
        mask_iou_sum = torch.tensor(0.0, device=device)
        mask_count = torch.tensor(0.0, device=device)
        bbox_iou_sum = torch.tensor(0.0, device=device)
        bbox_count = torch.tensor(0.0, device=device)
        for sweep_mask_threshold, sweep_min_box_area in bbox_sweep or []:
            suffix = sweep_metric_suffix(sweep_mask_threshold, sweep_min_box_area)
            metrics[f"bbox_iou_sum_{suffix}"] = torch.tensor(0.0, device=device)
            metrics[f"bbox_count_{suffix}"] = torch.tensor(0.0, device=device)

    metrics.update(
        {
            "mask_iou_sum": mask_iou_sum.detach(),
            "mask_count": mask_count.detach(),
            "bbox_iou_sum": bbox_iou_sum.detach(),
            "bbox_count": bbox_count.detach(),
        }
    )
    return metrics


def aggregate_step_metrics(step_outputs):
    keys = step_outputs[0].keys()
    totals = {key: sum(item[key] for item in step_outputs) for key in keys}

    metrics = {
        "acc": _safe_ratio(totals["acc_sum"], totals["num_samples"]).item(),
        "mask_iou": _safe_ratio_or_zero(totals["mask_iou_sum"], totals["mask_count"]).item(),
        "bbox_iou": _safe_ratio_or_zero(totals["bbox_iou_sum"], totals["bbox_count"]).item(),
    }
    for key in totals:
        if not key.startswith("bbox_iou_sum_"):
            continue
        suffix = key.removeprefix("bbox_iou_sum_")
        metrics[f"bbox_iou_{suffix}"] = _safe_ratio_or_zero(
            totals[key],
            totals[f"bbox_count_{suffix}"],
        ).item()
    return metrics
