from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class OptimizerConfig:
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 2000
    total_steps: int = 200_000
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1


def create_adamw(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Warmup -> cosine decay -> floor at lr_min."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float,
        lr_min: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.lr_max = lr_max
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch + 1
        if t <= self.warmup_steps:
            lr = self.lr_max * (t / self.warmup_steps)
        elif t <= self.total_steps:
            p = (t - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(math.pi * p))
        else:
            lr = self.lr_min
        return [lr for _ in self.optimizer.param_groups]


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimizerConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    return CosineWithWarmup(
        optimizer=optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
        lr_max=cfg.lr_max,
        lr_min=cfg.lr_min,
    )
