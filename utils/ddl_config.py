from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional


@dataclass
class DDLLossWeights:
    main: float = 1.0
    global_branch: float = 1.0
    patch_branch: float = 1.0
    segment_branch: float = 1.0
    cluster_mask: float = 1.0
    decoder_mask: float = 1.0
    bbox_mask: float = 0.0
    bbox_boundary: float = 0.0
    bbox_outside: float = 0.0
    edge: float = 0.2
    edge_width: int = 3


@dataclass
class DDLModelConfig:
    feature_dim: int = 384
    agglomerative_num_clusters: int = 32
    mask_hidden_dim: int = 256
    dropout: float = 0.1
    cluster_tau: float = 0.9
    reducer_hidden_dim: int = 256
    reducer_temperature: float = 0.07
    topk_ratio: float = 0.05
    mask_target_threshold: float = 0.1
    classification_loss: str = "focal"

    loss_weights: DDLLossWeights = field(default_factory=DDLLossWeights)
    simple_patch_size: Optional[int] = 16


def build_ddl_config(raw_cfg: Mapping[str, Any] | DDLModelConfig) -> DDLModelConfig:
    if isinstance(raw_cfg, DDLModelConfig):
        return raw_cfg

    cfg_dict = dict(raw_cfg)
    loss_weights = cfg_dict.get("loss_weights", None)
    if isinstance(loss_weights, Mapping):
        cfg_dict["loss_weights"] = DDLLossWeights(**dict(loss_weights))

    valid_field_names = {field.name for field in fields(DDLModelConfig)}
    filtered_cfg = {key: value for key, value in cfg_dict.items() if key in valid_field_names}
    return DDLModelConfig(**filtered_cfg)
