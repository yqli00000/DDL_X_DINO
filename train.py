import argparse
import datetime
import importlib
import inspect
import os

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:  # pragma: no cover - optional dependency
    WandbLogger = None
from omegaconf import ListConfig
from torch.utils.data import DataLoader

import data
import engine
import networks
from utils.common import archive_files, load_config_with_cli, seed_everything


def _maybe_configure_mp_sharing(conf):
    try:
        import torch.multiprocessing as mp
    except Exception:
        return

    world_size = int(os.getenv("WORLD_SIZE", "1") or 1)
    user_strategy = getattr(conf.train, "mp_sharing_strategy", None)

    strategy = user_strategy
    if strategy is None and world_size > 1:
        strategy = "file_system"

    if strategy:
        try:
            mp.set_sharing_strategy(strategy)
        except Exception:
            pass


def _resolve_callable(target):
    module_name, attr_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _build_loggers(conf, run_name: str, run_dir: str):
    loggers = [CSVLogger(save_dir="logs", name=run_name, version="")]

    wandb_conf = getattr(conf.train, "wandb", None)
    wandb_enabled = bool(getattr(wandb_conf, "enabled", False)) if wandb_conf is not None else False
    if not wandb_enabled:
        return loggers

    if WandbLogger is None:
        raise ImportError(
            "Weights & Biases logging was enabled, but WandbLogger is unavailable. "
            "Install `wandb` in the active environment first."
        )

    project = getattr(wandb_conf, "project", None)
    if not project:
        raise ValueError("train.wandb.project must be set when train.wandb.enabled=true.")

    loggers.append(
        WandbLogger(
            project=project,
            entity=getattr(wandb_conf, "entity", None),
            name=getattr(wandb_conf, "name", None) or run_name,
            save_dir=run_dir,
            mode=getattr(wandb_conf, "mode", "online"),
            log_model=bool(getattr(wandb_conf, "log_model", False)),
            tags=list(getattr(wandb_conf, "tags", []) or []),
            group=getattr(wandb_conf, "group", None),
            notes=getattr(wandb_conf, "notes", None),
        )
    )
    return loggers


def _build_dataset(dataset_conf, train: bool):
    target = dataset_conf.get("target", None)
    if target is None:
        raise ValueError("Dataset config must define `target`.")
    dataset_cls = _resolve_callable(target)
    kwargs = {key: dataset_conf[key] for key in dataset_conf.keys() if key not in {"target", "batch_size", "loader_workers", "pin_memory", "persistent_workers", "prefetch_factor"}}
    kwargs["train"] = train
    return dataset_cls(**kwargs)


def build_dataloader(conf):
    train_dataset = _build_dataset(conf.datasets.train, train=True)
    val_dataset = _build_dataset(conf.datasets.val, train=False)

    train_workers = int(getattr(conf.datasets.train, "loader_workers", 0) or 0)
    val_workers = int(getattr(conf.datasets.val, "loader_workers", 0) or 0)
    world_size = int(os.getenv("WORLD_SIZE", "1") or 1)
    default_prefetch = 1 if world_size > 1 else 2

    train_loader_kwargs = dict(
        batch_size=int(conf.datasets.train.batch_size),
        shuffle=True,
        num_workers=train_workers,
        pin_memory=bool(getattr(conf.datasets.train, "pin_memory", True)),
        persistent_workers=(train_workers > 0) and bool(getattr(conf.datasets.train, "persistent_workers", True)),
        drop_last=False,
    )
    train_collate = getattr(train_dataset, "collate_fn", None)
    if callable(train_collate):
        train_loader_kwargs["collate_fn"] = train_collate
    if train_workers > 0:
        train_loader_kwargs["prefetch_factor"] = int(
            getattr(conf.datasets.train, "prefetch_factor", default_prefetch) or default_prefetch
        )

    val_loader_kwargs = dict(
        batch_size=int(conf.datasets.val.batch_size),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=bool(getattr(conf.datasets.val, "pin_memory", True)),
        persistent_workers=(val_workers > 0) and bool(getattr(conf.datasets.val, "persistent_workers", True)),
        drop_last=False,
    )
    val_collate = getattr(val_dataset, "collate_fn", None)
    if callable(val_collate):
        val_loader_kwargs["collate_fn"] = val_collate
    if val_workers > 0:
        val_loader_kwargs["prefetch_factor"] = int(
            getattr(conf.datasets.val, "prefetch_factor", default_prefetch) or default_prefetch
        )

    return DataLoader(train_dataset, **train_loader_kwargs), DataLoader(val_dataset, **val_loader_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDL training with Lightning")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    _maybe_configure_mp_sharing(conf)
    seed_everything(int(conf.train.seed))

    train_loader, val_loader = build_dataloader(conf)

    resume_ckpt = args.resume
    if isinstance(resume_ckpt, str) and resume_ckpt.strip().lower() in {"", "none", "null"}:
        resume_ckpt = None

    if args.logdir is not None:
        run_name = args.logdir
        run_dir = os.path.join("logs", run_name)
    elif resume_ckpt is not None:
        run_dir = os.path.dirname(resume_ckpt)
        run_name = os.path.basename(run_dir)
    else:
        run_name = conf.name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        run_dir = os.path.join("logs", run_name)

    os.makedirs(run_dir, exist_ok=True)
    loggers = _build_loggers(conf, run_name, run_dir)

    world_size = int(os.getenv("WORLD_SIZE", "1") or 1)
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1") or 1)
    rank = os.getenv("RANK")
    launched_with_torchrun = world_size > 1 and os.getenv("LOCAL_RANK") is not None and rank is not None

    if os.getenv("LOCAL_RANK", "0") == "0" and not launched_with_torchrun:
        archive_files(run_name, exclude_dirs=["logs", ".git", "__pycache__", "outputs"])

    monitor = getattr(conf.train, "monitor", "val_acc_epoch")
    monitor_mode = getattr(conf.train, "monitor_mode", "max")
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=run_dir,
        filename="{epoch:02d}-{" + monitor + ":.4f}",
        save_top_k=1,
        save_last=True,
        mode=monitor_mode,
    )

    model = eval(conf.train.pipeline)(opt=conf)
    torch.set_float32_matmul_precision("high")

    if launched_with_torchrun:
        devices = local_world_size
        num_nodes = max(1, world_size // max(1, local_world_size))
        strategy = "ddp"
    else:
        gpu_ids = getattr(conf.train, "gpu_ids", 1)
        if isinstance(gpu_ids, (list, tuple, ListConfig)):
            devices = len(gpu_ids)
        else:
            devices = int(gpu_ids)
        num_nodes = 1
        strategy = "ddp" if devices > 1 else "auto"

    accelerator = getattr(conf.train, "accelerator", "gpu")
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    trainer = L.Trainer(
        logger=loggers,
        max_epochs=int(conf.train.train_epochs),
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=int(conf.train.check_val_every_n_epoch),
        precision=conf.train.get("precision", "16"),
        log_every_n_steps=int(getattr(conf.train, "log_every_n_steps", 10) or 10),
        accumulate_grad_batches=int(getattr(conf.train, "accumulation_steps", 1) or 1),
        gradient_clip_val=float(getattr(conf.train, "gradient_clip_val", 0.0) or 0.0),
        gradient_clip_algorithm=str(getattr(conf.train, "gradient_clip_algorithm", "norm")),
    )

    fit_kwargs = dict(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)
    if resume_ckpt is not None and "weights_only" in inspect.signature(trainer.fit).parameters:
        fit_kwargs["weights_only"] = False

    trainer.fit(**fit_kwargs)
    trainer.save_checkpoint(os.path.join(run_dir, "last.ckpt"))
