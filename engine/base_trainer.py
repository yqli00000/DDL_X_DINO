import os
from functools import partial

import lightning as L
import torch
from omegaconf import OmegaConf

from utils.network_factory import get_model
from utils.lr_schedulers import WarmupCosineScheduler
from utils.pretrained_loader import load_partial_pretrained


class Trainer(L.LightningModule):
    """Shared Lightning wrapper that builds the configured model and optimizer."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        pretrained_path = getattr(getattr(self.opt, "train", None), "pretrained", None)
        if pretrained_path:
            report = load_partial_pretrained(self.model, str(pretrained_path))
            if os.getenv("LOCAL_RANK", "0") == "0":
                print(
                    "[pretrained]"
                    f" loaded={report['loaded']}"
                    f" missing={len(report['missing'])}"
                    f" skipped={len(report['skipped'])}"
                    f" path={pretrained_path}"
                )
        self.debug_unused_params = bool(getattr(getattr(self.opt, "train", None), "debug_unused_params", False))
        self._debug_unused_params_logged = False

        try:
            serialized_opt = OmegaConf.to_container(opt, resolve=True)
        except Exception:
            serialized_opt = None
        if serialized_opt is not None:
            self.save_hyperparameters({"opt": serialized_opt})

    def configure_optimizers(self):
        optimizer_factory = self.opt.train.optimizer
        optimizer_keywords = dict(getattr(optimizer_factory, "keywords", {}) or {})
        base_lr = float(optimizer_keywords.get("lr", 1e-4))
        base_weight_decay = float(optimizer_keywords.get("weight_decay", 0.0))

        if hasattr(self.model, "build_optimizer_param_groups"):
            optparams = self.model.build_optimizer_param_groups(
                self.opt.train,
                base_lr=base_lr,
                base_weight_decay=base_weight_decay,
            )
        else:
            optparams = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = optimizer_factory(optparams)

        scheduler_factory = getattr(self.opt.train, "scheduler", None)
        if scheduler_factory is None:
            return optimizer

        scheduler = self._build_scheduler(optimizer, scheduler_factory)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": getattr(self.opt.train, "scheduler_interval", "epoch"),
                "frequency": int(getattr(self.opt.train, "scheduler_frequency", 1) or 1),
            },
        }

    def _build_scheduler(self, optimizer, scheduler_factory):
        factory_func = getattr(scheduler_factory, "func", None)
        if factory_func is WarmupCosineScheduler:
            interval = str(getattr(self.opt.train, "scheduler_interval", "epoch"))
            train_epochs = int(getattr(self.opt.train, "train_epochs", 1) or 1)
            warmup_epochs = int(getattr(self.opt.train, "warmup_epochs", 0) or 0)
            eta_min = float(getattr(self.opt.train, "eta_min", 1e-9) or 1e-9)

            if interval == "step":
                total_steps = int(getattr(self.trainer, "estimated_stepping_batches", train_epochs) or train_epochs)
                steps_per_epoch = max(1, total_steps // max(1, train_epochs))
                warmup_steps = warmup_epochs * steps_per_epoch
            else:
                total_steps = train_epochs
                warmup_steps = warmup_epochs

            return scheduler_factory(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                eta_min=eta_min,
            )

        return scheduler_factory(optimizer)

    def on_after_backward(self):
        if not self.debug_unused_params or self._debug_unused_params_logged:
            return

        if os.getenv("LOCAL_RANK", "0") != "0":
            return

        unused = []
        zero_grad = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                unused.append(name)
            elif torch.count_nonzero(param.grad).item() == 0:
                zero_grad.append(name)

        print("[debug_unused_params] parameters with grad=None:")
        for name in unused:
            print(f"  - {name}")

        print("[debug_unused_params] parameters with all-zero grads:")
        for name in zero_grad:
            print(f"  - {name}")

        self._debug_unused_params_logged = True
