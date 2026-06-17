import torch


def _torch_load_compat(path, map_location="cpu"):
    try:
        import inspect

        if "weights_only" in inspect.signature(torch.load).parameters:
            return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        pass
    return torch.load(path, map_location=map_location)


def resume_lightning(model, weight_path):
    state_dict = _torch_load_compat(weight_path, map_location="cpu")["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key[6:]] = value
        else:
            new_state_dict[key] = value
    strict = bool(getattr(model, "resume_strict", True))
    model.load_state_dict(new_state_dict, strict=strict)
