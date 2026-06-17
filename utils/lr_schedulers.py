from __future__ import annotations

import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-9,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * float(step + 1) / float(self.warmup_steps + 1)
            else:
                progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                lr = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
