from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-layer perceptron reused by the DDL heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimplePatchBackbone(nn.Module):
    """Small CNN backbone used for smoke tests and interface validation."""

    def __init__(self, feature_dim: int = 384, patch_size: int = 16) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(3, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.cls_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, images: torch.Tensor) -> Dict[str, object]:
        patch_map = self.proj(images)
        _, _, hp, wp = patch_map.shape
        patch_tokens = patch_map.flatten(2).transpose(1, 2)
        cls_token = self.cls_proj(patch_tokens.mean(dim=1))
        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "patch_shape": (hp, wp),
            "multi_layer_patch_tokens": [patch_tokens, patch_tokens, patch_tokens, patch_tokens],
        }


class SimpleMaskDecoder(nn.Module):
    """Lightweight decoder that upsamples patch features into a dense fake mask."""

    def __init__(self, in_channels: int, hidden_channels: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        x = self.proj(x)
        logits_lowres = self.decode(x)
        return F.interpolate(logits_lowres, size=output_size, mode="bilinear", align_corners=False)


class SegFormerStyleDecoder(nn.Module):
    """Lightweight multi-layer fusion decoder inspired by SegFormer heads."""

    def __init__(
        self,
        in_channels_list: list[int],
        score_channels: int = 2,
        embed_dim: int = 256,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        if not in_channels_list:
            raise ValueError("in_channels_list must not be empty.")

        self.feature_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim, kernel_size=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                )
                for in_channels in in_channels_list
            ]
        )
        self.score_proj = nn.Sequential(
            nn.Conv2d(score_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        fusion_in = embed_dim * (len(in_channels_list) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_in, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        score_maps: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(feature_maps) != len(self.feature_projs):
            raise ValueError("Number of feature maps does not match decoder projections.")

        target_hw = feature_maps[-1].shape[-2:]
        fused_features = []
        for proj, feature_map in zip(self.feature_projs, feature_maps):
            projected = proj(feature_map)
            if projected.shape[-2:] != target_hw:
                projected = F.interpolate(projected, size=target_hw, mode="bilinear", align_corners=False)
            fused_features.append(projected)

        projected_scores = self.score_proj(score_maps)
        if projected_scores.shape[-2:] != target_hw:
            projected_scores = F.interpolate(projected_scores, size=target_hw, mode="bilinear", align_corners=False)
        fused_features.append(projected_scores)

        logits_lowres = self.fuse(torch.cat(fused_features, dim=1))
        return F.interpolate(logits_lowres, size=output_size, mode="bilinear", align_corners=False)


class UpsampleRefineMaskDecoder(nn.Module):
    """Multi-layer fusion decoder with three feature-space upsample refinement stages."""

    def __init__(
        self,
        in_channels_list: list[int],
        score_channels: int = 2,
        embed_dim: int = 256,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        if not in_channels_list:
            raise ValueError("in_channels_list must not be empty.")

        refine_channels_1 = max(hidden_channels // 2, 32)
        refine_channels_2 = max(hidden_channels // 4, 32)
        refine_channels_3 = max(hidden_channels // 8, 32)

        self.feature_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim, kernel_size=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                )
                for in_channels in in_channels_list
            ]
        )
        self.score_proj = nn.Sequential(
            nn.Conv2d(score_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        fusion_in = embed_dim * (len(in_channels_list) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_in, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.refine1 = self._make_refine_block(hidden_channels, refine_channels_1)
        self.refine2 = self._make_refine_block(refine_channels_1, refine_channels_2)
        self.refine3 = self._make_refine_block(refine_channels_2, refine_channels_3)
        self.pred = nn.Conv2d(refine_channels_3, 1, kernel_size=1)

    @staticmethod
    def _make_refine_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        score_maps: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(feature_maps) != len(self.feature_projs):
            raise ValueError("Number of feature maps does not match decoder projections.")

        target_hw = feature_maps[-1].shape[-2:]
        fused_features = []
        for proj, feature_map in zip(self.feature_projs, feature_maps):
            projected = proj(feature_map)
            if projected.shape[-2:] != target_hw:
                projected = F.interpolate(projected, size=target_hw, mode="bilinear", align_corners=False)
            fused_features.append(projected)

        projected_scores = self.score_proj(score_maps)
        if projected_scores.shape[-2:] != target_hw:
            projected_scores = F.interpolate(projected_scores, size=target_hw, mode="bilinear", align_corners=False)
        fused_features.append(projected_scores)

        x = self.fuse(torch.cat(fused_features, dim=1))
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.refine1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.refine2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.refine3(x)

        logits = self.pred(x)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
