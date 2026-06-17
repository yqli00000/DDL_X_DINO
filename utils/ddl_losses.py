from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from networks.gps_dino_modules import sigmoid_focal_loss
from utils.ddl_config import DDLModelConfig


REAL = 0
def mask_edge_map(mask: torch.Tensor, edge_width: int = 3) -> torch.Tensor:
    """
    Extract soft/differentiable-ish edge map by morphological gradient:
    edge = dilate(mask) - erode(mask)

    mask: [B, 1, H, W], values in [0, 1]
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    edge_width = max(3, int(edge_width))
    if edge_width % 2 == 0:
        edge_width += 1
    pad = edge_width // 2

    dilated = F.max_pool2d(mask, kernel_size=edge_width, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=edge_width, stride=1, padding=pad)
    edge = (dilated - eroded).clamp(0.0, 1.0)
    return edge


def edge_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    edge_width: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Supervise mask decoder on object/forgery boundaries.
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    pred_prob = torch.sigmoid(logits)

    pred_edge = mask_edge_map(pred_prob, edge_width=edge_width)
    target_edge = mask_edge_map(targets.float(), edge_width=edge_width)

    # 如果某些 fake 样本 mask 为空，避免 edge 全 0 导致不稳定
    edge_count = target_edge.flatten(start_dim=1).sum(dim=1)
    valid = edge_count > 0

    if not valid.any():
        return zero_loss_like(logits)

    pred_edge = pred_edge[valid]
    target_edge = target_edge[valid]

    pred_edge = pred_edge.float().clamp(eps, 1.0 - eps)
    target_edge = target_edge.float()
    bce = -(target_edge * pred_edge.log() + (1.0 - target_edge) * (1.0 - pred_edge).log()).mean()

    pred_flat = pred_edge.flatten(start_dim=1)
    target_flat = target_edge.flatten(start_dim=1)
    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()

    return bce + dice


def edge_band_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    edge_width: int = 5,
    band_width: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Supervise predicted edges only around the GT boundary band.

    This keeps edge supervision focused near real object/forgery boundaries instead
    of averaging over the whole mostly-background edge map.
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    edge_width = max(3, int(edge_width))
    if edge_width % 2 == 0:
        edge_width += 1
    band_width = edge_width if band_width is None else max(3, int(band_width))
    if band_width % 2 == 0:
        band_width += 1

    pred_prob = torch.sigmoid(logits)
    target_edge = mask_edge_map(targets.float(), edge_width=edge_width)
    pred_edge = mask_edge_map(pred_prob, edge_width=edge_width)

    band_pad = band_width // 2
    edge_band = F.max_pool2d(target_edge, kernel_size=band_width, stride=1, padding=band_pad) > 0.0
    valid = edge_band.flatten(start_dim=1).any(dim=1)
    if not valid.any():
        return zero_loss_like(logits)

    pred_edge = pred_edge[valid].float().clamp(eps, 1.0 - eps)
    target_edge = target_edge[valid].float()
    edge_band = edge_band[valid]

    pred_values = pred_edge[edge_band]
    target_values = target_edge[edge_band]
    bce = -(target_values * pred_values.log() + (1.0 - target_values) * (1.0 - pred_values).log()).mean()

    pred_band = pred_edge * edge_band.float()
    target_band = target_edge * edge_band.float()
    pred_flat = pred_band.flatten(start_dim=1)
    target_flat = target_band.flatten(start_dim=1)
    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()

    return bce + dice


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def zero_loss_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def mask_to_bbox_filled_mask(mask: torch.Tensor, min_area: int = 1) -> torch.Tensor:
    """Fill the tight foreground bbox for each connected component in each sample mask."""

    mask_2d = mask.squeeze(1) if mask.dim() == 4 else mask
    bbox_mask = torch.zeros_like(mask_2d)
    for sample_idx in range(mask_2d.shape[0]):
        binary = (mask_2d[sample_idx].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_idx in range(1, num_components):
            x, y, w, h, area = stats[component_idx]
            if int(area) < int(min_area):
                continue
            bbox_mask[sample_idx, int(y) : int(y + h), int(x) : int(x + w)] = 1.0
    return bbox_mask.unsqueeze(1) if mask.dim() == 4 else bbox_mask


def mask_to_expanded_bbox_mask(mask: torch.Tensor, margin: int = 3, min_area: int = 1) -> torch.Tensor:
    """Fill a slightly expanded bbox for each connected component in each sample mask."""

    mask_2d = mask.squeeze(1) if mask.dim() == 4 else mask
    bbox_mask = torch.zeros_like(mask_2d)
    height, width = mask_2d.shape[-2:]
    margin = max(0, int(margin))

    for sample_idx in range(mask_2d.shape[0]):
        binary = (mask_2d[sample_idx].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_idx in range(1, num_components):
            x, y, w, h, area = stats[component_idx]
            if int(area) < int(min_area):
                continue
            x1 = max(0, int(x) - margin)
            y1 = max(0, int(y) - margin)
            x2 = min(width, int(x + w) + margin)
            y2 = min(height, int(y + h) + margin)
            bbox_mask[sample_idx, y1:y2, x1:x2] = 1.0
    return bbox_mask.unsqueeze(1) if mask.dim() == 4 else bbox_mask


def bbox_boundary_touch_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    band_width: int = 3,
    topk_ratio: float = 0.1,
    min_area: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage predicted masks to touch each GT bbox boundary without filling the box."""

    pred_prob = torch.sigmoid(logits)
    pred_2d = pred_prob.squeeze(1) if pred_prob.dim() == 4 else pred_prob
    target_2d = targets.squeeze(1) if targets.dim() == 4 else targets
    losses = []
    height, width = pred_2d.shape[-2:]
    band_width = max(1, int(band_width))

    for sample_idx in range(target_2d.shape[0]):
        binary = (target_2d[sample_idx].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        sample_pred = pred_2d[sample_idx]

        for component_idx in range(1, num_components):
            x, y, w, h, area = stats[component_idx]
            if int(area) < int(min_area):
                continue

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(width - 1, int(x + w - 1))
            y2 = min(height - 1, int(y + h - 1))
            if x2 < x1 or y2 < y1:
                continue

            strips = [
                sample_pred[y1 : min(height, y1 + band_width), x1 : x2 + 1],
                sample_pred[max(0, y2 - band_width + 1) : y2 + 1, x1 : x2 + 1],
                sample_pred[y1 : y2 + 1, x1 : min(width, x1 + band_width)],
                sample_pred[y1 : y2 + 1, max(0, x2 - band_width + 1) : x2 + 1],
            ]

            for strip in strips:
                values = strip.flatten()
                if values.numel() == 0:
                    continue
                k = max(1, int(round(values.numel() * float(topk_ratio))))
                score = values.topk(min(k, values.numel())).values.mean()
                losses.append(-torch.log(score.clamp_min(eps)))

    if not losses:
        return zero_loss_like(logits)
    return torch.stack(losses).mean()


def bbox_outside_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: int = 3,
    min_area: int = 1,
) -> torch.Tensor:
    """Penalize predicted probability outside slightly expanded GT bboxes."""

    keep_mask = mask_to_expanded_bbox_mask(targets, margin=margin, min_area=min_area)
    outside_mask = 1.0 - keep_mask
    outside_count = outside_mask.sum()
    if outside_count.item() <= 0:
        return zero_loss_like(logits)
    pred_prob = torch.sigmoid(logits)
    return (pred_prob * outside_mask).sum() / outside_count.clamp_min(1.0)


def _classification_loss(logits: torch.Tensor, targets: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "focal":
        return sigmoid_focal_loss(logits, targets, alpha=0.5, reduction="mean").mean()
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)
    raise ValueError(f"Unsupported classification_loss: {mode}")


def compute_ddl_losses(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    cfg: DDLModelConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    w = cfg.loss_weights
    label = labels["label"].long()
    fake_label = (label != REAL).float()
    fake_mask = label != REAL

    loss_main = _classification_loss(outputs["logits"], fake_label, cfg.classification_loss)
    loss_global = _classification_loss(outputs["global_logits"], fake_label, cfg.classification_loss)
    loss_patch = _classification_loss(outputs["patch_logits"], fake_label, cfg.classification_loss)
    loss_segment = _classification_loss(outputs["segment_logits"], fake_label, cfg.classification_loss)

    cluster_terms = []
    if "patch_mask_loss" in outputs:
        cluster_terms.append(outputs["patch_mask_loss"])
    if "segment_mask_loss" in outputs:
        cluster_terms.append(outputs["segment_mask_loss"])
    if cluster_terms:
        loss_cluster_mask = sum(cluster_terms) / len(cluster_terms)
    else:
        loss_cluster_mask = zero_loss_like(outputs["logits"])

    if "mask" in labels and fake_mask.any():
        gt_mask = labels["mask"].float()
        pred_fake = outputs["pred_mask_logits"][fake_mask]
        gt_fake = gt_mask[fake_mask]
        loss_decoder_mask = F.binary_cross_entropy_with_logits(pred_fake, gt_fake) + dice_loss_with_logits(
            pred_fake,
            gt_fake,
        )
        if float(w.edge) > 0.0:
            # loss_edge = edge_loss_with_logits(
            #     pred_fake,
            #     gt_fake,
            #     edge_width=getattr(w, "edge_width", 3),
            # )
            # Optional narrow-band variant. Keep this commented to preserve the
            # current loss behavior; uncomment to focus edge supervision around
            # GT boundaries only.
            loss_edge = edge_band_loss_with_logits(
                pred_fake,
                gt_fake,
                edge_width=getattr(w, "edge_width", 5),
            )
        else:
            loss_edge = zero_loss_like(outputs["pred_mask_logits"])
        if float(w.bbox_mask) > 0.0:
            gt_bbox_fake = mask_to_bbox_filled_mask(gt_fake)
            loss_bbox_mask = F.binary_cross_entropy_with_logits(pred_fake, gt_bbox_fake) + dice_loss_with_logits(
                pred_fake,
                gt_bbox_fake,
            )
        else:
            loss_bbox_mask = zero_loss_like(outputs["pred_mask_logits"])
        if float(w.bbox_boundary) > 0.0:
            loss_bbox_boundary = bbox_boundary_touch_loss_with_logits(pred_fake, gt_fake)
        else:
            loss_bbox_boundary = zero_loss_like(outputs["pred_mask_logits"])
        if float(w.bbox_outside) > 0.0:
            loss_bbox_outside = bbox_outside_loss_with_logits(pred_fake, gt_fake)
        else:
            loss_bbox_outside = zero_loss_like(outputs["pred_mask_logits"])
    else:
        loss_decoder_mask = zero_loss_like(outputs["pred_mask_logits"])
        loss_bbox_mask = zero_loss_like(outputs["pred_mask_logits"])
        loss_bbox_boundary = zero_loss_like(outputs["pred_mask_logits"])
        loss_bbox_outside = zero_loss_like(outputs["pred_mask_logits"])
        loss_edge = zero_loss_like(outputs["pred_mask_logits"])

    loss = (
        w.main * loss_main
        + w.global_branch * loss_global
        + w.patch_branch * loss_patch
        + w.segment_branch * loss_segment
        + w.cluster_mask * loss_cluster_mask
        + w.decoder_mask * loss_decoder_mask
        + w.bbox_mask * loss_bbox_mask
        + w.bbox_boundary * loss_bbox_boundary
        + w.bbox_outside * loss_bbox_outside
        + w.edge * loss_edge
    )

    loss_dict = {
        "loss_main": loss_main.detach(),
        "loss_global": loss_global.detach(),
        "loss_patch": loss_patch.detach(),
        "loss_segment": loss_segment.detach(),
        "loss_cluster_mask": loss_cluster_mask.detach(),
        "loss_decoder_mask": loss_decoder_mask.detach(),
        "loss_bbox_mask": loss_bbox_mask.detach(),
        "loss_bbox_boundary": loss_bbox_boundary.detach(),
        "loss_bbox_outside": loss_bbox_outside.detach(),
        "loss_total": loss.detach(),
        "loss_edge": loss_edge.detach(),
    }
    return loss, loss_dict
