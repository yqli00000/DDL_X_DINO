from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma_pos: float = 2.0,
    gamma_neg: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()

    probs = torch.sigmoid(inputs)
    pos_loss = -((1.0 - probs) ** gamma_pos) * torch.log(probs.clamp(min=1e-8))
    neg_loss = -(probs**gamma_neg) * torch.log((1.0 - probs).clamp(min=1e-8))
    loss = targets * pos_loss + (1.0 - targets) * neg_loss

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        if loss.ndim == 1:
            return loss
        return loss.mean(dim=tuple(range(1, loss.ndim)))
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


class FlexibleMLP(nn.Module):
    """GPS-DINO style configurable MLP head."""

    def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int, drop_rates: Optional[list[float]] = None):
        super().__init__()
        if drop_rates is None:
            drop_rates = [0.0] * len(hidden_sizes)
        if len(drop_rates) != len(hidden_sizes):
            raise ValueError("drop_rates must have the same length as hidden_sizes.")

        layers = []
        prev_size = input_size
        for hidden_size, drop_rate in zip(hidden_sizes, drop_rates):
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.Dropout(drop_rate),
                    nn.ReLU(),
                ]
            )
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchClassifierReducer(nn.Module):
    """GPS-DINO patch MIL reducer, adapted to DDL interfaces."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, temperature: float = 0.07, topk_ratio: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.temperature = float(temperature)
        self.topk_ratio = float(topk_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def mask_to_patch(mask: torch.Tensor, patch_size: int = 16, threshold: float = 0.5) -> torch.Tensor:
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.ndim != 3:
            raise ValueError(f"Expected mask with shape [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
        patch_mask = F.avg_pool2d(mask.unsqueeze(1).float(), kernel_size=patch_size, stride=patch_size)
        return (patch_mask.squeeze(1).flatten(1) > threshold).float()

    def forward(
        self,
        x: torch.Tensor,
        *,
        gt_mask: Optional[torch.Tensor] = None,
        patch_size: int = 16,
        mask_threshold: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.norm(x)
        instance_logits = torch.clamp(self.mlp(x).squeeze(-1), min=-10.0, max=10.0)
        weights = F.softmax(instance_logits / self.temperature, dim=-1)
        aggregated = torch.sum(weights.unsqueeze(-1) * x, dim=1)

        batch_size, num_tokens = instance_logits.shape
        k = max(1, min(num_tokens, int(self.topk_ratio * num_tokens)))
        topk_vals, topk_idx = torch.topk(instance_logits, k, dim=1)

        rest_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=x.device)
        rest_mask.scatter_(1, topk_idx, False)
        rest_logits = instance_logits.masked_fill(~rest_mask, 0.0).sum(dim=1) / max(1, num_tokens - k)
        topk_logits = topk_vals.mean(dim=1)

        mask_loss = None
        if gt_mask is not None:
            patch_targets = self.mask_to_patch(gt_mask, patch_size=patch_size, threshold=mask_threshold)
            mask_loss = sigmoid_focal_loss(instance_logits, patch_targets, alpha=0.9, reduction="mean")

        return aggregated, topk_logits, rest_logits, instance_logits, mask_loss


class SegmentClassifierReducer(nn.Module):
    """GPS-DINO segment MIL reducer with mask-to-segment supervision support."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, temperature: float = 0.07, topk_ratio: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.temperature = float(temperature)
        self.topk_ratio = float(topk_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def mask_to_patch(mask: torch.Tensor, patch_size: int = 16, threshold: float = 0.5) -> torch.Tensor:
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.ndim != 3:
            raise ValueError(f"Expected mask with shape [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
        patch_mask = F.avg_pool2d(mask.unsqueeze(1).float(), kernel_size=patch_size, stride=patch_size)
        return (patch_mask.squeeze(1).flatten(1) > threshold).float()

    def mask_to_segment(
        self,
        gt_mask: torch.Tensor,
        cluster_labels: torch.Tensor,
        *,
        patch_size: int = 16,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        patch_targets = self.mask_to_patch(gt_mask, patch_size=patch_size, threshold=threshold).squeeze(0)
        num_segments = int(cluster_labels.max().item()) + 1
        segment_targets = []
        for segment_idx in range(num_segments):
            member_mask = cluster_labels == segment_idx
            if member_mask.any():
                segment_target = (patch_targets[member_mask].float().mean() > threshold).float()
            else:
                segment_target = patch_targets.new_tensor(0.0)
            segment_targets.append(segment_target)
        return torch.stack(segment_targets, dim=0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cluster_labels: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        patch_size: int = 16,
        mask_threshold: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = self.norm(x)
        instance_logits = torch.clamp(self.mlp(x).squeeze(-1), min=-10.0, max=10.0)
        weights = F.softmax(instance_logits / self.temperature, dim=-1)
        aggregated = torch.sum(weights.unsqueeze(-1) * x, dim=1)

        batch_size, num_segments = instance_logits.shape
        k = max(1, min(num_segments, int(self.topk_ratio * num_segments)))
        topk_vals, topk_idx = torch.topk(instance_logits, k, dim=1)

        rest_mask = torch.ones(batch_size, num_segments, dtype=torch.bool, device=x.device)
        rest_mask.scatter_(1, topk_idx, False)
        rest_logits = instance_logits.masked_fill(~rest_mask, 0.0).sum(dim=1) / max(1, num_segments - k)
        topk_logits = topk_vals.mean(dim=1)

        mask_loss = None
        if gt_mask is not None:
            if cluster_labels is None:
                raise ValueError("cluster_labels is required when gt_mask is provided for segment reduction.")
            segment_targets = self.mask_to_segment(
                gt_mask,
                cluster_labels=cluster_labels,
                patch_size=patch_size,
                threshold=mask_threshold,
            )
            mask_loss = sigmoid_focal_loss(instance_logits, segment_targets, alpha=0.9, reduction="mean")

        return aggregated, topk_logits, rest_logits, instance_logits, mask_loss
