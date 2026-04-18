from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Dict, Iterable

import torch
from torch.cuda.amp import autocast

from ray import tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_model
from src.model import LLM
from src.training.objectives import TrainingLossConfig, compute_language_model_loss
from src.training.optim import OptimizerConfig, create_adamw, create_scheduler


def _build_synthetic_dataloader(config: Dict) -> list[Dict[str, torch.Tensor]]:
    """Create tiny random token batches for smoke-level tuning runs."""
    vocab_size = int(config.get("vocab_size", 128))
    batch_size = int(config.get("batch_size", 8))
    seq_len = int(config.get("seq_len", 16))
    steps_per_epoch = int(config.get("steps_per_epoch", 8))

    batches: list[Dict[str, torch.Tensor]] = []
    for _ in range(steps_per_epoch):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        labels = input_ids.roll(-1, dims=1)
        batches.append({"input_ids": input_ids, "labels": labels})
    return batches


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    loss_cfg: TrainingLossConfig,
) -> Dict[str, float]:
    model.train()
    total_ce = 0.0
    total_route = 0.0
    total_aux = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        amp_ctx = autocast(dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            loss, parts = compute_language_model_loss(model, input_ids, labels, loss_cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_ce += float(parts["loss_ce"].item())
        total_route += float(parts["loss_route"].item())
        total_aux += float(parts["loss_aux"].item())
        steps += 1

    steps = max(1, steps)
    return {
        "loss_ce": total_ce / steps,
        "loss_route": total_route / steps,
        "loss_aux": total_aux / steps,
    }


def train_llm_ray(config: Dict):
    device = get_device()

    model = LLM(
        vocab_size=config.get("vocab_size", 128),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_experts=config["n_experts"],
        d_c=config["d_c"],
        max_seq_len=config.get("max_seq_len", 128),
        d_rope=config.get("d_rope", 8),
    ).to(device)
    model = prepare_model(model)

    optim_cfg = OptimizerConfig(
        lr_max=config["lr"],
        lr_min=config.get("lr_min", config["lr"] * 0.1),
        warmup_steps=config["warmup_steps"],
        total_steps=config.get("total_steps", 200_000),
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
    )
    loss_cfg = TrainingLossConfig(lambda_route=0.01, lambda_aux=0.001)

    optimizer = create_adamw(model, optim_cfg)
    scheduler = create_scheduler(optimizer, optim_cfg)

    dataloader = config.get("dataloader") or _build_synthetic_dataloader(config)
    for epoch in range(config.get("epochs", 10)):
        metrics = train_one_epoch(model, dataloader, optimizer, scheduler, device, loss_cfg)
        ppl = math.exp(metrics["loss_ce"])
        tune.report(
            {
                "epoch": epoch,
                "loss": metrics["loss_ce"],
                "perplexity": ppl,
                "loss_route": metrics["loss_route"],
                "loss_aux": metrics["loss_aux"],
            }
        )


def build_tuner(
    search_space: Dict,
    num_workers: int = 64,
    use_gpu: bool = True,
    num_samples: int = 50,
    max_t: int = 100,
    grace_period: int = 10,
):
    trainer = TorchTrainer(
        train_llm_ray,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )
    return tune.Tuner(
        trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=tune.schedulers.ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=max_t,
                grace_period=grace_period,
            ),
        ),
    )


def run_quick_tune_local(num_samples: int = 4):
    """Run a tiny local HPO sweep (single worker) for quick feedback."""
    search_space = {
        "lr": tune.loguniform(1e-4, 5e-3),
        "batch_size": tune.choice([4, 8]),
        "seq_len": tune.choice([8, 16]),
        "warmup_steps": tune.choice([5, 10]),
        "total_steps": tune.choice([100]),
        "d_model": tune.choice([64, 96]),
        "n_heads": tune.choice([4]),
        "n_layers": tune.choice([1, 2]),
        "n_experts": tune.choice([8]),
        "d_c": tune.choice([16, 24]),
        "vocab_size": 128,
        "max_seq_len": 128,
        "d_rope": 8,
        "epochs": 2,
        "steps_per_epoch": 4,
    }

    def _local_trainable(cfg: Dict):
        device = torch.device("cpu")
        model = LLM(
            vocab_size=cfg.get("vocab_size", 128),
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            n_experts=cfg["n_experts"],
            d_c=cfg["d_c"],
            max_seq_len=cfg.get("max_seq_len", 128),
            d_rope=cfg.get("d_rope", 8),
        ).to(device)

        optim_cfg = OptimizerConfig(
            lr_max=cfg["lr"],
            lr_min=cfg.get("lr_min", cfg["lr"] * 0.1),
            warmup_steps=cfg["warmup_steps"],
            total_steps=cfg.get("total_steps", 100),
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
        )
        loss_cfg = TrainingLossConfig(lambda_route=0.01, lambda_aux=0.001)

        optimizer = create_adamw(model, optim_cfg)
        scheduler = create_scheduler(optimizer, optim_cfg)
        dataloader = _build_synthetic_dataloader(cfg)

        for epoch in range(cfg.get("epochs", 2)):
            metrics = train_one_epoch(model, dataloader, optimizer, scheduler, device, loss_cfg)
            ppl = math.exp(metrics["loss_ce"])
            tune.report(
                {
                    "epoch": epoch,
                    "loss": metrics["loss_ce"],
                    "perplexity": ppl,
                    "loss_route": metrics["loss_route"],
                    "loss_aux": metrics["loss_aux"],
                }
            )

    tuner = tune.Tuner(
        tune.with_resources(_local_trainable, resources={"cpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=tune.schedulers.ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=2,
                grace_period=1,
            ),
        ),
    )
    results = tuner.fit()
    return results.get_best_result(metric="loss", mode="min")
