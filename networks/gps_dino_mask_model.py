from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from networks.attention_blocks import SegFormerStyleDecoder, SimplePatchBackbone, UpsampleRefineMaskDecoder
from networks.dinov3_lora_layers import DINOv3LinearLoRA
from networks.dinov3_token_extractors import (
    DINOv3LoRATokenFeatureExtractor,
    DINOv3TokenFeatureExtractor,
    resolve_patch_grid,
)
from networks.gps_dino_modules import FlexibleMLP, PatchClassifierReducer, SegmentClassifierReducer
from utils.ddl_config import DDLModelConfig, build_ddl_config
from utils.ddl_losses import compute_ddl_losses
from utils.registry import MODELS

try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:  # pragma: no cover - optional dependency
    AgglomerativeClustering = None
@MODELS.register_module(name=["GPSDINOMaskModel", "GPSDINOModel"])
class GPSDINOMaskModel(nn.Module):
    """GPS-DINO clustering branches plus a dense decoder head for pixel-level mask prediction."""

    def __init__(self, cfg: Optional[DDLModelConfig] = None, backbone: Optional[nn.Module] = None, **kwargs) -> None:
        super().__init__()

        model_conf = kwargs.get("model", kwargs)
        self.cfg = build_ddl_config(cfg if cfg is not None else model_conf)
        self.backbone = backbone or self._build_backbone(model_conf)

        self.feature_dim = int(getattr(self.backbone, "feature_dim", self.cfg.feature_dim) or self.cfg.feature_dim)
        if self.feature_dim != self.cfg.feature_dim:
            self.cfg.feature_dim = self.feature_dim

        self.patch_size = int(
            model_conf.get("patch_size", getattr(self.backbone, "patch_size", self.cfg.simple_patch_size or 16)) or 16
        )
        self.decoder_num_layers = 4

        self.main_classifier = FlexibleMLP(
            input_size=self.feature_dim * 3,
            hidden_sizes=[self.feature_dim * 2, self.feature_dim],
            num_classes=1,
            drop_rates=[0.1, 0.2],
        )
        self.global_classifier = FlexibleMLP(
            input_size=self.feature_dim,
            hidden_sizes=[max(self.feature_dim // 4, 64)],
            num_classes=1,
            drop_rates=[0.1],
        )
        self.patch_classifier = FlexibleMLP(
            input_size=self.feature_dim,
            hidden_sizes=[max(self.feature_dim // 4, 64)],
            num_classes=1,
            drop_rates=[0.1],
        )
        self.segment_classifier = FlexibleMLP(
            input_size=self.feature_dim,
            hidden_sizes=[max(self.feature_dim // 4, 64)],
            num_classes=1,
            drop_rates=[0.1],
        )

        self.patch_reducer = PatchClassifierReducer(
            input_dim=self.feature_dim,
            hidden_dim=self.cfg.reducer_hidden_dim,
            temperature=self.cfg.reducer_temperature,
            topk_ratio=self.cfg.topk_ratio,
        )
        self.segment_reducer = SegmentClassifierReducer(
            input_dim=self.feature_dim,
            hidden_dim=self.cfg.reducer_hidden_dim,
            temperature=self.cfg.reducer_temperature,
            topk_ratio=self.cfg.topk_ratio,
        )

        self.norm_global = nn.LayerNorm(self.feature_dim)
        self.norm_patch = nn.LayerNorm(self.feature_dim)
        self.norm_segment = nn.LayerNorm(self.feature_dim)

        self.mask_decoder = SegFormerStyleDecoder(
            in_channels_list=[self.feature_dim] * 4,
            score_channels=2,
            embed_dim=self.cfg.mask_hidden_dim,
            hidden_channels=self.cfg.mask_hidden_dim,
        )
        # Optional stronger decoder for 16x16 patch grids. Keep this commented to
        # preserve current checkpoint/runtime behavior; uncomment to switch.
        # self.mask_decoder = UpsampleRefineMaskDecoder(
        #     in_channels_list=[self.feature_dim] * 4,
        #     score_channels=2,
        #     embed_dim=self.cfg.mask_hidden_dim,
        #     hidden_channels=self.cfg.mask_hidden_dim,
        # )
        self.resume_strict = False

    def _build_backbone(self, model_conf):
        if model_conf.get("backbone0", None):
            if model_conf.get("lora", None):
                return DINOv3LoRATokenFeatureExtractor(**model_conf)
            return DINOv3TokenFeatureExtractor(**model_conf)
        return SimplePatchBackbone(feature_dim=self.cfg.feature_dim, patch_size=self.cfg.simple_patch_size or 16)

    def export_config_dict(self):
        return asdict(self.cfg)

    def build_optimizer_param_groups(self, train_conf, *, base_lr: float, base_weight_decay: float):
        head_lr_ratio = float(getattr(train_conf, "head_lr_ratio", 10.0) or 10.0)
        token_head_lr_ratio = float(getattr(train_conf, "token_head_lr_ratio", 0.1) or 0.1)
        zero_wd_for_norm = bool(getattr(train_conf, "zero_wd_for_norm", True))

        lora_params = []
        head_params = []
        token_head_params = []
        head_norm_params = []
        reducer_norm_params = []
        assigned = set()

        for module in self.modules():
            if isinstance(module, DINOv3LinearLoRA):
                for param in module.parameters():
                    if param.requires_grad and id(param) not in assigned:
                        lora_params.append(param)
                        assigned.add(id(param))

        head_module_names = {"main_classifier", "global_classifier"}
        token_module_names = {"patch_classifier", "segment_classifier", "patch_reducer", "segment_reducer", "mask_decoder"}
        for module_name, module in self.named_modules():
            target_list = None
            if module_name in head_module_names:
                target_list = head_params
            elif module_name in token_module_names:
                target_list = token_head_params
            elif module_name.startswith("norm_"):
                target_list = head_norm_params
            elif "reducer" in module_name and module_name.endswith("norm"):
                target_list = reducer_norm_params

            if target_list is None:
                continue

            for param in module.parameters(recurse=False):
                if param.requires_grad and id(param) not in assigned:
                    target_list.append(param)
                    assigned.add(id(param))

        remaining_params = []
        for _, param in self.named_parameters():
            if param.requires_grad and id(param) not in assigned:
                remaining_params.append(param)
                assigned.add(id(param))
        if remaining_params:
            token_head_params.extend(remaining_params)

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": base_lr, "weight_decay": base_weight_decay})
        if head_params:
            param_groups.append({"params": head_params, "lr": base_lr * head_lr_ratio, "weight_decay": base_weight_decay})
        if token_head_params:
            param_groups.append(
                {"params": token_head_params, "lr": base_lr * token_head_lr_ratio, "weight_decay": base_weight_decay}
            )
        if head_norm_params:
            param_groups.append(
                {
                    "params": head_norm_params,
                    "lr": base_lr * head_lr_ratio,
                    "weight_decay": 0.0 if zero_wd_for_norm else base_weight_decay,
                }
            )
        if reducer_norm_params:
            param_groups.append(
                {
                    "params": reducer_norm_params,
                    "lr": base_lr * token_head_lr_ratio,
                    "weight_decay": 0.0 if zero_wd_for_norm else base_weight_decay,
                }
            )
        return param_groups

    def forward(self, images: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor | Tuple[int, int]]:
        bsz, _, height, width = images.shape
        backbone_out = self.backbone(images)
        cls_tokens = backbone_out["cls_token"]
        patch_tokens = backbone_out["patch_tokens"]
        hp, wp = backbone_out.get("patch_shape", resolve_patch_grid(height, width, patch_tokens.shape[1], self.patch_size))
        multi_layer_patch_tokens = backbone_out.get("multi_layer_patch_tokens", [patch_tokens])
        multi_layer_patch_tokens = self._select_decoder_layers(multi_layer_patch_tokens)
        decoder_feature_maps = [
            layer_tokens.reshape(bsz, hp, wp, self.feature_dim).permute(0, 3, 1, 2).contiguous()
            for layer_tokens in multi_layer_patch_tokens
        ]

        gt_mask = labels.get("mask", None) if labels is not None else None
        cluster_patch_tokens = patch_tokens
        if hasattr(self.backbone, "set_lora_enabled"):
            frozen_backbone_out = self.backbone(images, use_lora=False)
            cluster_patch_tokens = frozen_backbone_out["patch_tokens"]

        aggregated_patch, weak_patch_logits, rest_patch_logits, patch_instance_logits, patch_mask_loss = self.patch_reducer(
            patch_tokens,
            gt_mask=gt_mask,
            patch_size=self.patch_size,
            mask_threshold=self.cfg.mask_target_threshold,
        )
        patch_logits = self.patch_classifier(self.norm_patch(aggregated_patch)).squeeze(-1)
        global_logits = self.global_classifier(self.norm_global(cls_tokens)).squeeze(-1)

        aggregated_segment_tokens = []
        weak_segment_logits = []
        rest_segment_logits = []
        segment_logits = []
        segment_instance_maps = []
        cluster_assignments = []
        segment_mask_losses = []

        for batch_idx in range(bsz):
            cluster_labels, cluster_prototypes = self.cluster_patch_tokens(
                cluster_patch_tokens[batch_idx],
                train_patch_tokens=patch_tokens[batch_idx],
            )
            cluster_assignments.append(cluster_labels)

            sample_mask = gt_mask[batch_idx : batch_idx + 1] if gt_mask is not None else None
            aggregated_segment, weak_segment_logit, rest_segment_logit, segment_instance_logits, segment_mask_loss = self.segment_reducer(
                cluster_prototypes.unsqueeze(0),
                cluster_labels=cluster_labels,
                gt_mask=sample_mask,
                patch_size=self.patch_size,
                mask_threshold=self.cfg.mask_target_threshold,
            )

            aggregated_segment_tokens.append(aggregated_segment.squeeze(0))
            weak_segment_logits.append(weak_segment_logit.squeeze(0))
            rest_segment_logits.append(rest_segment_logit.squeeze(0))
            segment_logits.append(self.segment_classifier(self.norm_segment(aggregated_segment)).squeeze(0).squeeze(-1))

            segment_patch_logits = segment_instance_logits.squeeze(0)[cluster_labels]
            segment_instance_maps.append(segment_patch_logits.reshape(hp, wp))

            if segment_mask_loss is not None:
                segment_mask_losses.append(segment_mask_loss.squeeze(0))

        aggregated_segment_tokens = torch.stack(aggregated_segment_tokens, dim=0)
        weak_segment_logits = torch.stack(weak_segment_logits, dim=0)
        rest_segment_logits = torch.stack(rest_segment_logits, dim=0)
        segment_logits = torch.stack(segment_logits, dim=0)
        segment_patch_map_logits = torch.stack(segment_instance_maps, dim=0).unsqueeze(1)
        region_assignments = torch.stack(cluster_assignments, dim=0).reshape(bsz, hp, wp)

        main_input = torch.cat(
            [
                self.norm_global(cls_tokens),
                self.norm_patch(aggregated_patch),
                self.norm_segment(aggregated_segment_tokens),
            ],
            dim=-1,
        )
        logits = self.main_classifier(main_input).squeeze(-1)

        patch_score_map = torch.sigmoid(patch_instance_logits).reshape(bsz, 1, hp, wp).contiguous()
        segment_score_map = torch.sigmoid(segment_patch_map_logits).contiguous()
        score_maps = torch.cat([patch_score_map, segment_score_map], dim=1)
        pred_mask_logits = self.mask_decoder(decoder_feature_maps, score_maps, output_size=(height, width))
        pred_mask = torch.sigmoid(pred_mask_logits)

        outputs: Dict[str, torch.Tensor | Tuple[int, int]] = {
            "logits": logits,
            "global_logits": global_logits,
            "patch_logits": patch_logits,
            "segment_logits": segment_logits,
            "weak_patch_logits": weak_patch_logits,
            "weak_segment_logits": weak_segment_logits,
            "rest_patch_logits": rest_patch_logits,
            "rest_segment_logits": rest_segment_logits,
            "patch_instance_logits": patch_instance_logits,
            "segment_patch_map_logits": segment_patch_map_logits,
            "pred_mask_logits": pred_mask_logits,
            "pred_mask": pred_mask,
            "patch_score_map": patch_score_map,
            "segment_score_map": segment_score_map,
            "region_assignments": region_assignments,
            "patch_shape": (hp, wp),
        }

        if patch_mask_loss is not None:
            outputs["patch_mask_loss"] = patch_mask_loss.mean()
        if segment_mask_losses:
            outputs["segment_mask_loss"] = torch.stack(segment_mask_losses).mean()

        if labels is not None:
            loss, loss_dict = compute_ddl_losses(outputs, labels, self.cfg)  # type: ignore[arg-type]
            outputs["loss"] = loss
            outputs["loss_dict"] = loss_dict  # type: ignore[assignment]
        return outputs

    def debug_forward_paths(self, images: torch.Tensor) -> Dict[str, object]:
        """Return intermediate tensors and shape summaries for debugging the dual-path forward."""

        with torch.no_grad():
            backbone_out = self.backbone(images)
            train_patch_tokens = backbone_out["patch_tokens"]
            cls_tokens = backbone_out["cls_token"]
            hp, wp = backbone_out["patch_shape"]
            multi_layer_patch_tokens = self._select_decoder_layers(backbone_out.get("multi_layer_patch_tokens", [train_patch_tokens]))

            cluster_patch_tokens = train_patch_tokens
            lora_path_split = False
            if hasattr(self.backbone, "set_lora_enabled"):
                frozen_backbone_out = self.backbone(images, use_lora=False)
                cluster_patch_tokens = frozen_backbone_out["patch_tokens"]
                lora_path_split = True

            cluster_labels, cluster_prototypes = self.cluster_patch_tokens(
                cluster_patch_tokens[0],
                train_patch_tokens=train_patch_tokens[0],
            )
            _, _, _, patch_instance_logits, _ = self.patch_reducer(train_patch_tokens)
            segment_patch_logits = self.segment_reducer(
                cluster_prototypes.unsqueeze(0),
                cluster_labels=cluster_labels,
                gt_mask=None,
                patch_size=self.patch_size,
                mask_threshold=self.cfg.mask_target_threshold,
            )[3]

            patch_score_map = torch.sigmoid(patch_instance_logits).reshape(images.shape[0], 1, hp, wp)
            segment_score_map = torch.sigmoid(segment_patch_logits.squeeze(0)[cluster_labels].reshape(hp, wp)).unsqueeze(0).unsqueeze(0)
            decoder_feature_maps = [
                layer_tokens.reshape(images.shape[0], hp, wp, self.feature_dim).permute(0, 3, 1, 2)
                for layer_tokens in multi_layer_patch_tokens
            ]
            pred_mask_logits = self.mask_decoder(
                decoder_feature_maps,
                torch.cat([patch_score_map[:1], segment_score_map], dim=1),
                output_size=(images.shape[-2], images.shape[-1]),
            )

            train_vs_frozen_l1 = (train_patch_tokens - cluster_patch_tokens).abs().mean().item()
            return {
                "lora_path_split": lora_path_split,
                "train_patch_tokens_shape": tuple(train_patch_tokens.shape),
                "cluster_patch_tokens_shape": tuple(cluster_patch_tokens.shape),
                "train_vs_frozen_l1_mean": float(train_vs_frozen_l1),
                "multi_layer_patch_token_shapes": [tuple(tokens.shape) for tokens in multi_layer_patch_tokens],
                "decoder_feature_map_shapes": [tuple(feature_map.shape) for feature_map in decoder_feature_maps],
                "cls_tokens_shape": tuple(cls_tokens.shape),
                "patch_grid": (int(hp), int(wp)),
                "cluster_label_count": int(cluster_labels.max().item()) + 1,
                "cluster_labels_shape": tuple(cluster_labels.shape),
                "cluster_prototypes_shape": tuple(cluster_prototypes.shape),
                "patch_instance_logits_shape": tuple(patch_instance_logits.shape),
                "patch_score_map_shape": tuple(patch_score_map.shape),
                "segment_score_map_shape": tuple(segment_score_map.shape),
                "pred_mask_logits_shape": tuple(pred_mask_logits.shape),
            }

    def cluster_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        train_patch_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_tokens = patch_tokens
        target_tokens = train_patch_tokens if train_patch_tokens is not None else patch_tokens
        if AgglomerativeClustering is None:
            assignments = self._fallback_sequential_clusters(source_tokens.shape[0], source_tokens.device)
            prototypes = self._cluster_means(target_tokens, assignments)
            return assignments, prototypes

        normalized = F.normalize(source_tokens.float(), p=2, dim=-1)
        similarity = torch.matmul(normalized, normalized.T).clamp(-1.0, 1.0)
        distance = (1.0 - similarity).detach().cpu().numpy()
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - self.cfg.cluster_tau,
        )
        cluster_labels = torch.from_numpy(clusterer.fit_predict(distance)).to(device=source_tokens.device, dtype=torch.long)
        prototypes = self._cluster_means(target_tokens, cluster_labels)
        return cluster_labels, prototypes

    @staticmethod
    def _cluster_means(patch_tokens: torch.Tensor, cluster_labels: torch.Tensor) -> torch.Tensor:
        num_clusters = int(cluster_labels.max().item()) + 1
        prototypes = []
        for cluster_idx in range(num_clusters):
            member_mask = cluster_labels == cluster_idx
            if member_mask.any():
                prototype = patch_tokens[member_mask].mean(dim=0)
            else:
                prototype = patch_tokens.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes, dim=0)

    def _fallback_sequential_clusters(self, num_patches: int, device: torch.device) -> torch.Tensor:
        target_clusters = max(1, min(int(self.cfg.agglomerative_num_clusters), num_patches))
        if target_clusters >= num_patches:
            return torch.arange(num_patches, device=device, dtype=torch.long)
        boundaries = torch.linspace(0, num_patches, target_clusters + 1, device=device)
        labels = torch.empty(num_patches, device=device, dtype=torch.long)
        for cluster_idx in range(target_clusters):
            start = int(boundaries[cluster_idx].item())
            end = int(boundaries[cluster_idx + 1].item())
            labels[start:end] = cluster_idx
        return labels

    def _select_decoder_layers(self, multi_layer_patch_tokens):
        if len(multi_layer_patch_tokens) == self.decoder_num_layers:
            return multi_layer_patch_tokens
        if len(multi_layer_patch_tokens) < self.decoder_num_layers:
            last = multi_layer_patch_tokens[-1]
            while len(multi_layer_patch_tokens) < self.decoder_num_layers:
                multi_layer_patch_tokens.append(last)
            return multi_layer_patch_tokens

        selected = []
        last_index = len(multi_layer_patch_tokens) - 1
        for idx in range(self.decoder_num_layers):
            position = round(idx * last_index / max(1, self.decoder_num_layers - 1))
            selected.append(multi_layer_patch_tokens[position])
        return selected
