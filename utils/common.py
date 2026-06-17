import os
import random
from typing import Any, Iterable
from omegaconf import ListConfig, OmegaConf
import shutil
import tempfile
import numpy as np
import torch


def seed_everything(torch_seed):
    random.seed(torch_seed)
    os.environ["PYTHONHASHSEED"] = str(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_config_undefined(cfg):
    itr: Iterable[Any] = range(len(cfg)) if isinstance(cfg, ListConfig) else cfg

    undefined_keys = []
    for key in itr:
        if cfg._get_child(key) == "---":
            undefined_keys.append(key)
        elif OmegaConf.is_config(cfg[key]):
            remove_config_undefined(cfg[key])
    for key in undefined_keys:
        del cfg[key]
    return cfg


def load_config(path, remove_undefined=True):
    cfg = OmegaConf.load(path)
    if "_base_" in cfg:
        for base in cfg["_base_"]:
            cfg = OmegaConf.merge(load_config(base, remove_undefined=False), cfg)
        del cfg["_base_"]
    if remove_undefined:
        cfg = remove_config_undefined(cfg)
    return cfg


def load_config_with_cli(path, args_list=None, remove_undefined=True):
    cfg = load_config(path, remove_undefined=False)
    cfg_cli = OmegaConf.from_cli(args_list)
    cfg = OmegaConf.merge(cfg, cfg_cli)
    if remove_undefined:
        cfg = remove_config_undefined(cfg)
    return cfg


def archive_files(log_name, exclude_dirs):
    log_dir = os.path.join("./logs", log_name)
    os.makedirs(log_dir, exist_ok=True)
    archive_name = os.path.join(log_dir, log_name)

    files_and_dirs = [f for f in os.listdir(".") if f not in exclude_dirs]
    excluded_suffixes = (
        ".pth",
        ".pyc",
        ".npy",
        ".pt",
        ".ckpt",
        ".gz",
        ".tar",
        ".zip",
        ".pkl",
        ".csv",
        ".jsonl",
    )

    def _ignore_filtered_files(_, names):
        ignored = []
        for name in names:
            if name.endswith(excluded_suffixes):
                ignored.append(name)
        return ignored

    with tempfile.TemporaryDirectory(dir=log_dir) as tmpdirname:
        for f in files_and_dirs:
            f_path = os.path.join(".", f)
            tmp_f_path = os.path.join(tmpdirname, f)
            if os.path.isdir(f_path):
                shutil.copytree(f_path, tmp_f_path, ignore=_ignore_filtered_files)
            else:
                if f_path.endswith(excluded_suffixes):
                    continue
                shutil.copy2(f_path, tmp_f_path)

        for root, _, files in os.walk(tmpdirname):
            for file in files:
                if file.endswith(excluded_suffixes):
                    os.remove(os.path.join(root, file))

        shutil.make_archive(archive_name, "tar", tmpdirname)
