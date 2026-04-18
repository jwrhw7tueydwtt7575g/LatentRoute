from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch

        from src.model import LLM
        from src.training.objectives import TrainingLossConfig, compute_language_model_loss
        from src.training.pipeline import recheck_pipeline_connections
    except Exception as exc:  # pragma: no cover
        print(f"Import/setup failure: {exc}")
        return 1

    torch.manual_seed(42)

    model = LLM(
        vocab_size=128,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_c=16,
        n_experts=8,
        max_seq_len=64,
        d_rope=8,
    )

    bsz, seq_len = 2, 12
    input_ids = torch.randint(0, 128, (bsz, seq_len))
    labels = torch.randint(0, 128, (bsz, seq_len))

    # forward
    with torch.no_grad():
        logits = model(input_ids)
    if logits.shape != (bsz, seq_len, 128):
        print(f"FAIL: bad logits shape {logits.shape}")
        return 2

    # total loss
    cfg = TrainingLossConfig(lambda_route=0.01, lambda_aux=0.001)
    loss, parts = compute_language_model_loss(model, input_ids, labels, cfg)
    if not torch.isfinite(loss):
        print("FAIL: non-finite loss")
        return 3

    # cache path
    with torch.no_grad():
        _, caches = model(torch.randint(0, 128, (bsz, 4)), use_cache=True)
        next_logits, caches = model(torch.randint(0, 128, (bsz, 2)), use_cache=True, caches=caches)

    if next_logits.shape != (bsz, 2, 128):
        print(f"FAIL: bad cached logits shape {next_logits.shape}")
        return 4

    # innovation wiring
    report = recheck_pipeline_connections(model)

    print("Smoke test passed")
    print("loss_total:", float(parts["loss_total"]))
    print("loss_ce:", float(parts["loss_ce"]))
    print("loss_route:", float(parts["loss_route"]))
    print("loss_aux:", float(parts["loss_aux"]))
    print("overall_ready:", report["overall_ready"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
