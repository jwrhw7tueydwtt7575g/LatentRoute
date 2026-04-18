from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TrainingLossConfig:
    lambda_route: float = 0.01
    lambda_aux: float = 0.001
    ignore_index: int = -100


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "model", model)


def _collect_route_loss(core_model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    loss = torch.zeros((), device=device)
    for layer in getattr(core_model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "aux_loss"):
            aux = getattr(mlp, "aux_loss")
            if isinstance(aux, torch.Tensor):
                loss = loss + aux.to(device=device)
    return loss


def _collect_latent_aux_loss(core_model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    loss = torch.zeros((), device=device)
    for layer in getattr(core_model, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "last_recon_loss"):
            rec = getattr(attn, "last_recon_loss")
            if isinstance(rec, torch.Tensor):
                loss = loss + rec.to(device=device)
    return loss


def compute_language_model_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainingLossConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute L_total = L_CE + lambda_1*L_route + lambda_2*L_aux."""
    logits = model(input_ids)
    if isinstance(logits, tuple):
        logits = logits[0]

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=cfg.ignore_index,
    )

    core_model = _unwrap_model(model)
    route = _collect_route_loss(core_model, device=ce.device)
    aux = _collect_latent_aux_loss(core_model, device=ce.device)

    total = ce + cfg.lambda_route * route + cfg.lambda_aux * aux
    parts = {
        "loss_total": total.detach(),
        "loss_ce": ce.detach(),
        "loss_route": route.detach(),
        "loss_aux": aux.detach(),
    }
    return total, parts
