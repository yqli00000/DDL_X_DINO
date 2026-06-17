import os
from collections import Counter

import torch
import torch.nn as nn
from omegaconf import ListConfig

from networks.dinov3_lora_layers import DINOv3LinearLoRA
from networks.rine_dinov3_detector import RINEDINOv3Model
from utils.registry import MODELS


@MODELS.register_module(name="RINEDINOv3LoRAModel")
class RINEDINOv3LoRAModel(RINEDINOv3Model):
    """RINE detector variant that inserts LoRA adapters into DINOv3 Linear layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_conf = kwargs.get("model", kwargs)
        lora_conf = model_conf.get("lora", None)
        if lora_conf is None:
            raise ValueError("RINEDINOv3LoRAModel requires a model.lora config section")

        enabled = bool(lora_conf.get("enabled", True))
        if not enabled:
            raise ValueError("model.lora.enabled must be true for RINEDINOv3LoRAModel")

        self.resume_strict = False
        self.lora_rank = int(lora_conf.get("rank", 8) or 8)
        self.lora_alpha = int(lora_conf.get("alpha", self.lora_rank) or self.lora_rank)
        self.lora_dropout = float(lora_conf.get("dropout", 0.0) or 0.0)
        self.lora_train_bias = bool(lora_conf.get("train_bias", False))
        self.lora_verbose = bool(lora_conf.get("verbose", False))
        self.lora_print_trainable_params = bool(lora_conf.get("print_trainable_params", False))
        self.lora_target_modules = self._normalize_patterns(
            lora_conf.get(
                "target_modules",
                [
                    "attention.attention.query",
                    "attention.attention.value",
                    "q_proj",
                    "v_proj",
                    "query",
                    "value",
                ],
            )
        )
        self.lora_target_blocks = self._resolve_lora_target_blocks(lora_conf.get("target_blocks", None))
        self.lora_applied_modules = self._apply_lora_to_backbone()

        if not self.lora_applied_modules:
            raise RuntimeError(
                "LoRA was enabled but no matching Linear layers were found. "
                f"Target blocks={list(self.lora_target_blocks)}, target_modules={self.lora_target_modules}"
            )

        print(
            "Applied LoRA to "
            f"{len(self.lora_applied_modules)} modules across {len(self.lora_target_blocks)} backbone blocks."
        )
        self._maybe_log_lora_details()

    def _normalize_patterns(self, value):
        if value is None:
            return tuple()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple, ListConfig)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise ValueError("model.lora.target_modules must be a string, list, tuple, or ListConfig")

    def _resolve_lora_target_blocks(self, value):
        num_layers = len(self.encoder_layers)
        if value is None:
            return tuple(range(num_layers))
        if isinstance(value, str):
            value = value.strip()
            if value.lower() == "all":
                return tuple(range(num_layers))
            parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
        elif isinstance(value, int):
            if value <= 0:
                raise ValueError("model.lora.target_blocks must be positive when provided as an integer")
            parsed = list(range(max(0, num_layers - value), num_layers))
        elif isinstance(value, (list, tuple, ListConfig)):
            parsed = [int(item) for item in value]
        else:
            raise ValueError("model.lora.target_blocks must be a string, int, list, tuple, or ListConfig")

        normalized = []
        for idx in parsed:
            normalized_idx = idx if idx >= 0 else num_layers + idx
            if normalized_idx < 0 or normalized_idx >= num_layers:
                raise ValueError(f"LoRA target block {idx} is out of range for {num_layers} layers")
            normalized.append(normalized_idx)
        return tuple(sorted(set(normalized)))

    def _matches_lora_pattern(self, qualified_name, child_name):
        for pattern in self.lora_target_modules:
            if qualified_name.endswith(pattern) or child_name == pattern:
                return True
        return False

    def _wrap_linear_lora(self, linear_module):
        return DINOv3LinearLoRA(
            linear_module,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            dropout_rate=self.lora_dropout,
            train_bias=self.lora_train_bias,
        )

    def _apply_lora_recursive(self, module, prefix=""):
        applied = []
        for child_name, child_module in module.named_children():
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child_module, nn.Linear) and not isinstance(child_module, DINOv3LinearLoRA):
                if self._matches_lora_pattern(qualified_name, child_name):
                    setattr(module, child_name, self._wrap_linear_lora(child_module))
                    applied.append(qualified_name)
                    continue
            applied.extend(self._apply_lora_recursive(child_module, qualified_name))
        return applied

    def _apply_lora_to_backbone(self):
        applied = []
        for block_idx in self.lora_target_blocks:
            block = self.encoder_layers[block_idx]
            block_prefix = f"encoder_layers.{block_idx}"
            applied.extend(self._apply_lora_recursive(block, prefix=block_prefix))
        return tuple(applied)

    def _backbone_context(self):
        if self.lora_applied_modules:
            return torch.enable_grad()
        return super()._backbone_context()

    def set_lora_enabled(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, DINOv3LinearLoRA):
                module.use_lora = bool(enabled)

    def _maybe_log_lora_details(self):
        if not self.lora_verbose:
            return

        if os.getenv("LOCAL_RANK", "0") != "0":
            return

        print("LoRA target blocks:", list(self.lora_target_blocks))
        print("LoRA target module patterns:", list(self.lora_target_modules))

        block_counter = Counter()
        for module_name in self.lora_applied_modules:
            parts = module_name.split(".")
            if len(parts) > 1 and parts[0] == "encoder_layers":
                block_counter[parts[1]] += 1
            else:
                block_counter["other"] += 1

        print("LoRA applied modules:")
        for module_name in self.lora_applied_modules:
            print(f"  - {module_name}")

        print("LoRA block hit summary:")
        for block_idx in sorted(block_counter, key=lambda item: (item == "other", int(item) if item.isdigit() else 10**9)):
            print(f"  - block {block_idx}: {block_counter[block_idx]} modules")

        if not self.lora_print_trainable_params:
            return

        print("LoRA trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  - {name}: {tuple(param.shape)}")
