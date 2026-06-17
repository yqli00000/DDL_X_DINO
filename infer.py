from __future__ import annotations

import torch
from omegaconf import OmegaConf

import networks
from utils.resume_tools import _torch_load_compat, resume_lightning


def choose_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _override_backbone_path(conf, backbone_path: str):
    if hasattr(conf, "model") and hasattr(conf.model, "backbone0"):
        old = conf.model.backbone0
        print(f"[infer] override conf.model.backbone0: {old} -> {backbone_path}")
        conf.model.backbone0 = backbone_path
    else:
        print("[infer] warning: conf.model.backbone0 not found")
    return conf

def _load_model_from_checkpoint(checkpoint_path: str, device: torch.device, backbone_path=None):
    checkpoint = _torch_load_compat(checkpoint_path, map_location=device)

    if "hyper_parameters" in checkpoint and "opt" in checkpoint["hyper_parameters"]:
        saved_opt = checkpoint["hyper_parameters"]["opt"]
        if not isinstance(saved_opt, dict):
            raise ValueError("Checkpoint hyper_parameters.opt must be a dict-like object.")

        # Some training checkpoints serialize optimizer/scheduler factories as
        # Python partials, which OmegaConf.create cannot reconstruct. For model
        # inference we only need the architecture name and model kwargs.
        minimal_opt = {
            "arch": saved_opt["arch"],
            "model": saved_opt.get("model", {}),
        }
        conf = OmegaConf.create(minimal_opt)
        print("[infer] conf before override:")
        print(OmegaConf.to_yaml(conf))
        if backbone_path is not None:
            conf = _override_backbone_path(conf, backbone_path)

        print("[infer] conf after override:")
        print(OmegaConf.to_yaml(conf))
        from utils.network_factory import get_model

        model = get_model(conf).to(device)
        resume_lightning(model, checkpoint_path)
        return model
    raise ValueError("Checkpoint must contain Lightning hyper_parameters.opt for model reconstruction.")
