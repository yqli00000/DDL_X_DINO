from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn


def _unwrap_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _map_legacy_key(key: str) -> str:
    """Map older GPS-DINO checkpoint names to the current DDL module names."""

    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]

    if key.startswith("dinov3."):
        key = "backbone." + key[len("dinov3.") :]

    replacements = {
        "patch_classifier_reducer.": "patch_reducer.",
        "segment_classifier_reducer.": "segment_reducer.",
    }
    for old, new in replacements.items():
        if key.startswith(old):
            key = new + key[len(old) :]
            break

    # Older FlexibleMLP used `layers/output_layer`; the current version stores
    # the same Linear layers inside a Sequential named `net`.
    mlp_prefixes = {
        "main_classifier.": {"layers.0.": "net.0.", "layers.1.": "net.3.", "output_layer.": "net.6."},
        "global_classifier.": {"layers.0.": "net.0.", "output_layer.": "net.3."},
        "patch_classifier.": {"layers.0.": "net.0.", "output_layer.": "net.3."},
        "segment_classifier.": {"layers.0.": "net.0.", "output_layer.": "net.3."},
    }
    for prefix, layer_map in mlp_prefixes.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        for old, new in layer_map.items():
            if suffix.startswith(old):
                return prefix + new + suffix[len(old) :]
    return key


def load_partial_pretrained(model: nn.Module, checkpoint_path: str, *, map_location: str = "cpu") -> dict[str, object]:
    """Load all shape-compatible parameters from a raw or Lightning checkpoint.

    This intentionally leaves newly-added modules, such as the dense mask decoder,
    at their default initialization when the pretrained file does not contain
    matching weights.
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    source_state = _unwrap_state_dict(checkpoint)
    target_state = model.state_dict()

    loadable = OrderedDict()
    skipped = []
    mapped_names = {}
    for source_key, value in source_state.items():
        if not torch.is_tensor(value):
            skipped.append((source_key, "not_tensor"))
            continue
        target_key = _map_legacy_key(source_key)
        mapped_names[source_key] = target_key
        target_value = target_state.get(target_key)
        if target_value is None:
            skipped.append((source_key, "missing_target"))
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            skipped.append((source_key, f"shape {tuple(value.shape)} != {tuple(target_value.shape)}"))
            continue
        loadable[target_key] = value

    missing_before_load = [key for key in target_state.keys() if key not in loadable]
    incompatible = model.load_state_dict(loadable, strict=False)
    return {
        "loaded": len(loadable),
        "skipped": skipped,
        "missing": missing_before_load,
        "unexpected": list(incompatible.unexpected_keys),
        "mapped_names": mapped_names,
    }
