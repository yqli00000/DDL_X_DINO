from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import ListConfig

from utils.registry import MODELS


@MODELS.register_module(name="RINEDINOv3Model")
class RINEDINOv3Model(nn.Module):
    """Baseline DINOv3 classifier that pools selected layer cls tokens."""

    def __init__(self, **kwargs):
        super().__init__()

        model_conf = kwargs.get("model", kwargs)
        backbone_name = model_conf.get("backbone0")
        if backbone_name is None:
            raise ValueError("RINEDINOv3Model requires 'backbone0' in model config")

        nproj = model_conf.get("nproj")
        proj_dim = model_conf.get("proj_dim")
        if nproj is None:
            raise ValueError("RINEDINOv3Model requires 'nproj' parameter")
        if proj_dim is None:
            raise ValueError("RINEDINOv3Model requires 'proj_dim' parameter")

        cache_dir = model_conf.get("hf_cache_dir", None)
        local_files_only = bool(model_conf.get("hf_local_files_only", False))
        hf_token = model_conf.get("hf_token", None)
        trust_remote_code = bool(model_conf.get("trust_remote_code", False))
        self.backbone = self._load_backbone(
            backbone_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.trainable_backbone_blocks = int(model_conf.get("trainable_backbone_blocks", 0) or 0)
        self._enable_partial_backbone_training(self.trainable_backbone_blocks)

        backbone_dim = int(getattr(self.backbone.config, "hidden_size", 0) or 0)
        config_backbone_dim = model_conf.get("backbone1", None)
        if config_backbone_dim is not None:
            config_backbone_dim = int(config_backbone_dim)
            if backbone_dim and config_backbone_dim != backbone_dim:
                raise ValueError(
                    f"Configured backbone1={config_backbone_dim} does not match model hidden size {backbone_dim}."
                )
            backbone_dim = config_backbone_dim
        if backbone_dim <= 0:
            raise ValueError("Unable to infer DINOv3 hidden size; set model.backbone1 explicitly.")

        self.encoder_layers = self._get_encoder_layers()
        self.layer_indices = self._resolve_layer_indices(model_conf, len(self.encoder_layers))

        proj1_layers = [nn.Dropout()]
        for idx in range(int(nproj)):
            proj1_layers.extend(
                [
                    nn.Linear(backbone_dim if idx == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)

        self.alpha = nn.Parameter(torch.zeros(1, len(self.layer_indices), proj_dim))

        proj2_layers = [nn.Dropout()]
        for _ in range(int(nproj)):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)

        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, 1),
        )

    def _load_backbone(self, backbone_name, cache_dir=None, local_files_only=False, hf_token=None, trust_remote_code=False):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "RINEDINOv3Model requires transformers>=4.56.0. Install it with `pip install 'transformers>=4.56.0'`."
            ) from exc

        model_source = str(backbone_name)
        expanded_path = Path(model_source).expanduser()
        if expanded_path.exists():
            model_source = str(expanded_path.resolve())

        return AutoModel.from_pretrained(
            model_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

    def _get_encoder_layers(self):
        layers = getattr(self.backbone, "layer", None)
        if layers is not None:
            return layers

        layers = getattr(self.backbone, "layers", None)
        if layers is not None:
            return layers

        model = getattr(self.backbone, "model", None)
        if model is not None:
            for name in ("layer", "layers", "blocks"):
                layers = getattr(model, name, None)
                if layers is not None:
                    return layers

        encoder = getattr(self.backbone, "encoder", None)
        if encoder is not None:
            for name in ("layer", "layers", "blocks"):
                layers = getattr(encoder, name, None)
                if layers is not None:
                    return layers

        layers = getattr(self.backbone, "blocks", None)
        if layers is not None:
            return layers

        raise ValueError(f"DINOv3 backbone structure not recognized: {type(self.backbone)}")

    def _resolve_layer_indices(self, model_conf, num_layers):
        layer_indices = model_conf.get("layer_indices", None)
        if layer_indices is not None:
            if isinstance(layer_indices, str):
                parsed = [int(item.strip()) for item in layer_indices.split(",") if item.strip()]
            elif isinstance(layer_indices, (list, tuple, ListConfig)):
                parsed = [int(item) for item in layer_indices]
            else:
                raise ValueError("model.layer_indices must be a string, list, tuple, or ListConfig.")
        else:
            num_last_layers = model_conf.get("num_last_layers", None)
            if num_last_layers is None:
                parsed = list(range(num_layers))
            else:
                num_last_layers = int(num_last_layers)
                if num_last_layers <= 0:
                    raise ValueError("model.num_last_layers must be positive when provided.")
                start = max(0, num_layers - num_last_layers)
                parsed = list(range(start, num_layers))

        normalized = []
        for idx in parsed:
            normalized_idx = idx if idx >= 0 else num_layers + idx
            if normalized_idx < 0 or normalized_idx >= num_layers:
                raise ValueError(f"Layer index {idx} is out of range for {num_layers} backbone layers.")
            normalized.append(normalized_idx)

        if not normalized:
            raise ValueError("RINEDINOv3Model requires at least one selected backbone layer.")
        return tuple(normalized)

    def _enable_partial_backbone_training(self, trainable_blocks):
        if trainable_blocks <= 0:
            return

        layers = self._get_encoder_layers()
        start_idx = max(0, len(layers) - trainable_blocks)
        for idx in range(start_idx, len(layers)):
            for param in layers[idx].parameters():
                param.requires_grad = True

        for attr_name in ["layernorm", "post_layernorm", "ln_f"]:
            module = getattr(self.backbone, attr_name, None)
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = True

        module = getattr(self.backbone, "norm", None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True

    def _backbone_context(self):
        backbone_has_trainable_params = any(param.requires_grad for param in self.backbone.parameters())
        return nullcontext() if backbone_has_trainable_params else torch.no_grad()

    def _collect_layer_tokens(self, hidden_states, batch_size):
        if hidden_states is None:
            raise RuntimeError("DINOv3 forward pass did not return hidden states.")

        tokens = []
        for idx in self.layer_indices:
            output = hidden_states[idx + 1]
            if not torch.is_tensor(output) or output.dim() != 3:
                raise RuntimeError(f"Unexpected hidden state shape for layer {idx}.")
            if output.shape[0] != batch_size:
                raise RuntimeError(
                    f"Unexpected batch dimension for layer {idx}: expected {batch_size}, got {output.shape[0]}."
                )
            tokens.append(output[:, 0, :])
        return torch.stack(tokens, dim=1)

    def forward(self, x):
        with self._backbone_context():
            outputs = self.backbone(pixel_values=x, output_hidden_states=True)
            g = self._collect_layer_tokens(outputs.hidden_states, x.shape[0]).float()

        g = self.proj1(g)
        z = torch.softmax(self.alpha, dim=1) * g
        z_pre = torch.sum(z, dim=1)
        z = self.proj2(z_pre)

        p = self.head(z)
        return {"logits": p, "z": z, "z_pre": z_pre}
