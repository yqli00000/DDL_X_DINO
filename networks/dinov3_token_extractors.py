from __future__ import annotations

import math
from typing import Tuple

import torch

from networks.rine_dinov3_detector import RINEDINOv3Model
from networks.rine_dinov3_lora_detector import RINEDINOv3LoRAModel


def resolve_patch_grid(height: int, width: int, num_patches: int, patch_size: int) -> Tuple[int, int]:
    hp = max(1, height // patch_size)
    wp = max(1, width // patch_size)
    if hp * wp == num_patches:
        return hp, wp

    square = int(math.isqrt(num_patches))
    if square * square == num_patches:
        return square, square

    for h_candidate in range(square, 0, -1):
        if num_patches % h_candidate == 0:
            return h_candidate, num_patches // h_candidate
    raise ValueError(f"Unable to infer patch grid from num_patches={num_patches}.")


class DINOv3TokenFeatureExtractor(RINEDINOv3Model):
    """Expose DINOv3 outputs in cls-token + patch-token form."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_conf = kwargs.get("model", kwargs)
        self.feature_dim = int(model_conf.get("backbone1", getattr(self.backbone.config, "hidden_size", 0)))
        self.patch_size = int(model_conf.get("patch_size", getattr(self.backbone.config, "patch_size", 16) or 16))
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)

        for attr_name in ("proj1", "proj2", "head", "alpha"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def forward(self, x, use_lora: bool = True):
        with self._backbone_context():
            outputs = self.backbone(pixel_values=x, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        layer_outputs = [hidden_states[idx + 1].float() for idx in self.layer_indices]
        hidden_state = layer_outputs[-1]
        cls_token = hidden_state[:, 0, :]
        patch_tokens = hidden_state[:, 1 + self.num_register_tokens :, :]
        patch_shape = resolve_patch_grid(x.shape[-2], x.shape[-1], patch_tokens.shape[1], self.patch_size)
        multi_layer_patch_tokens = [state[:, 1 + self.num_register_tokens :, :] for state in layer_outputs]
        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "patch_shape": patch_shape,
            "multi_layer_patch_tokens": multi_layer_patch_tokens,
        }


class DINOv3LoRATokenFeatureExtractor(RINEDINOv3LoRAModel):
    """Token extractor with LoRA injected into DINOv3 blocks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_conf = kwargs.get("model", kwargs)
        self.feature_dim = int(model_conf.get("backbone1", getattr(self.backbone.config, "hidden_size", 0)))
        self.patch_size = int(model_conf.get("patch_size", getattr(self.backbone.config, "patch_size", 16) or 16))
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)

        for attr_name in ("proj1", "proj2", "head", "alpha"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def forward(self, x, use_lora: bool = True):
        self.set_lora_enabled(use_lora)
        context = self._backbone_context() if use_lora else torch.no_grad()
        with context:
            outputs = self.backbone(pixel_values=x, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        layer_outputs = [hidden_states[idx + 1].float() for idx in self.layer_indices]
        hidden_state = layer_outputs[-1]
        cls_token = hidden_state[:, 0, :]
        patch_tokens = hidden_state[:, 1 + self.num_register_tokens :, :]
        patch_shape = resolve_patch_grid(x.shape[-2], x.shape[-1], patch_tokens.shape[1], self.patch_size)
        multi_layer_patch_tokens = [state[:, 1 + self.num_register_tokens :, :] for state in layer_outputs]
        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "patch_shape": patch_shape,
            "multi_layer_patch_tokens": multi_layer_patch_tokens,
        }
