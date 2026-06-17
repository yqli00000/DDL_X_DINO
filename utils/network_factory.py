from omegaconf import OmegaConf

from utils.registry import MODELS


def get_model(conf):
    if not hasattr(conf, "arch"):
        raise AttributeError("Config must define `arch`.")

    arch = conf.arch
    if not MODELS.has(arch):
        MODELS.get(arch)

    kwargs = {}
    if hasattr(conf, "model"):
        try:
            kwargs = OmegaConf.to_container(conf.model, resolve=True)
        except Exception:
            kwargs = dict(conf.model)
    return MODELS.build(arch, **kwargs)
