from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.model import LLM
from src.training.objectives import TrainingLossConfig, compute_language_model_loss
from src.training.pipeline import recheck_pipeline_connections


def _tiny_model() -> LLM:
    return LLM(
        vocab_size=128,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_c=16,
        n_experts=8,
        max_seq_len=64,
        d_rope=8,
    )


def test_forward_shape() -> None:
    model = _tiny_model()
    model.eval()

    bsz, seq_len = 2, 12
    input_ids = torch.randint(0, 128, (bsz, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (bsz, seq_len, 128)
    assert torch.isfinite(logits).all()


def test_total_loss_components_finite() -> None:
    model = _tiny_model()
    model.train()

    bsz, seq_len = 2, 10
    input_ids = torch.randint(0, 128, (bsz, seq_len))
    labels = torch.randint(0, 128, (bsz, seq_len))

    cfg = TrainingLossConfig(lambda_route=0.01, lambda_aux=0.001)
    loss, parts = compute_language_model_loss(model, input_ids, labels, cfg)

    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"loss_total", "loss_ce", "loss_route", "loss_aux"}
    assert torch.isfinite(parts["loss_total"])
    assert torch.isfinite(parts["loss_ce"])
    assert torch.isfinite(parts["loss_route"])
    assert torch.isfinite(parts["loss_aux"])


def test_cache_path() -> None:
    model = _tiny_model()
    model.eval()

    bsz = 2
    with torch.no_grad():
        first_logits, caches = model(torch.randint(0, 128, (bsz, 4)), use_cache=True)
        second_logits, caches = model(torch.randint(0, 128, (bsz, 2)), use_cache=True, caches=caches)

    assert first_logits.shape == (bsz, 4, 128)
    assert second_logits.shape == (bsz, 2, 128)
    assert len(caches) == 1
    assert getattr(caches[0], "seq_len", 0) >= 6


def test_pipeline_recheck_ready() -> None:
    model = _tiny_model()
    report = recheck_pipeline_connections(model)

    assert report["innovation_1"] is True
    assert report["innovation_2"] is True
    assert report["innovation_3"] is True
    assert report["innovation_4"] is True
    assert report["innovation_5"] is True
    assert report["overall_ready"] is True
