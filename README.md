# LATENTROUTE

A from-scratch LLM research prototype with the following implemented pipeline:

- Entropy-weighted BPE tokenizer (AMT-style)
- Factorized embeddings (FED / FED-Dk)
- RoPE + ARFS positional encoding
- MLA attention with HLCR-style latent compression/gating
- MoE FFN with hierarchical routing + entropy/load-balance loss
- RMSNorm pre-norm transformer blocks
- Training utilities (total loss, AdamW, cosine warmup, Ray Tune)
- Smoke tests and pipeline readiness checks

---

## Repository Layout

- `src/tokenizer` — Standard BPE + entropy-weighted BPE
- `src/embedding` — FED, FED-Dk, RoPE, ARFS
- `src/model` — Transformer stack (MLA + MoE + RMSNorm)
- `src/training` — Objectives, optimizer/scheduler, Ray Tune, pipeline registry/recheck
- `tests` — Pytest smoke tests
- `scripts` — Standalone smoke-test runner

---

## 1) Environment Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch pytest numpy "ray[default]" "ray[tune]"
```

---

## 2) Run Smoke Test (End-to-End)

```bash
source .venv/bin/activate
python scripts/run_smoke_test.py
```

Expected output includes:
- `Smoke test passed`
- `overall_ready: True`

---

## 3) Run Unit Tests

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q
```

---

## 4) Quick Local Hyperparameter Tuning

```bash
source .venv/bin/activate
PYTHONPATH=. python - <<'PY'
from src.training.ray_tune_train import run_quick_tune_local

best = run_quick_tune_local(num_samples=2)
print('Best config:', best.config)
print('Best loss:', best.metrics.get('loss'))
print('Best perplexity:', best.metrics.get('perplexity'))
PY
```

---

## 5) Start Ray Dashboard

```bash
source .venv/bin/activate
ray stop --force
ray start --head --dashboard-host=127.0.0.1 --dashboard-port=8265 --disable-usage-stats
```

Open:

- http://127.0.0.1:8265

Check health:

```bash
python - <<'PY'
import urllib.request
for u in ["http://127.0.0.1:8265", "http://127.0.0.1:8265/api/version"]:
    with urllib.request.urlopen(u, timeout=5) as r:
        print(u, r.status)
PY
```

---

## 6) Build Full Innovation Model + Recheck Connections

```bash
source .venv/bin/activate
PYTHONPATH=. python - <<'PY'
from src.training.pipeline import build_full_innovation_model, recheck_pipeline_connections

model = build_full_innovation_model(
    vocab_size=512,
    d_model=256,
    n_layers=2,
    n_heads=8,
    max_seq_len=256,
    n_experts=8,
    d_c=64,
    d_rope=16,
)
report = recheck_pipeline_connections(model)
print(report)
PY
```

---

## 7) Current Status

Implemented and validated:
- Tokenization, embedding, positional, attention, FFN/MoE, model integration
- Total training loss components:
  - `L_CE`
  - `lambda_1 * L_route`
  - `lambda_2 * L_aux`
- Optimizer and scheduler utilities
- Ray Tune local quick tuning
- Ray dashboard operational

This repository is a research/training core, not yet a full production chat platform (no serving API, chat UI, auth, moderation, persistence, or deployment stack yet).

---

## 8) Innovation Registry

### Innovation #1 — Adaptive Morphological Tokenizer (AMT)
- **Phase:** Tokenization (Phase 2)
- **Formula:**
  - $P_{merge}(a,b)=freq(a,b)\cdot \exp\left(-H\left(\frac{p(c|ab)}{p(c|a)p(c|b)}\right)\right)$
- **Impact:** ~18% vocabulary reduction, better rare-word handling
- **Patent basis:** entropy-weighted BPE merge criterion
- **Code location:** [src/tokenizer/bpe_entropy_weighted.py](src/tokenizer/bpe_entropy_weighted.py)

### Innovation #2 — Factorized Embedding Decomposition with Dynamic k (FED-Dk)
- **Phase:** Embedding (Phase 3)
- **Formula:**
  - $k_i = k_{min} + (k_{max}-k_{min})\cdot \sigma(\alpha\log(freq_i))$
- **Impact:** up to ~93% embedding memory reduction
- **Patent basis:** per-token adaptive bottleneck dimension
- **Code location:** [src/embedding/fed_dk.py](src/embedding/fed_dk.py)

### Innovation #3 — Adaptive RoPE with Learned Frequency Scaling (ARFS)
- **Phase:** Positional Encoding (Phase 4)
- **Formula:**
  - $\theta_j^{(domain)} = \theta_j\cdot \exp(\gamma_j z_{domain})$
- **Impact:** better cross-domain context extension without full retraining
- **Patent basis:** domain-conditioned rotary scaling
- **Code location:** [src/embedding/rope.py](src/embedding/rope.py)

### Innovation #4 — Hierarchical Latent Compression with Residual Gating (HLCR)
- **Phase:** Attention / MLA (Phase 6)
- **Formula:**
  - $g=\sigma(W_g[c_1;c_2;h])$
  - $c_{final}=g\cdot c_1 + (1-g)\cdot proj(c_2)$
- **Impact:** major KV-cache reduction with adaptive per-token compression
- **Patent basis:** two-level latent KV compression with learned gating
- **Code location:** [src/model/__init__.py](src/model/__init__.py)

### Innovation #5 — Hierarchical MoE Routing with Entropy Regularization
- **Phase:** MoE (Phase 7)
- **Formula:**
  - $L_{entropy}=-\lambda\sum_e \bar{p}_e\log(\bar{p}_e)$
- **Impact:** reduced routing compute and improved expert utilization
- **Patent basis:** hierarchical $O(\sqrt{E})$ routing + entropy regularization
- **Code location:** [src/model/__init__.py](src/model/__init__.py)

---

## 9) Cost Reduction Summary (Target)

| Component | Standard | With innovations |
|---|---:|---:|
| Embedding memory (50K vocab) | ~800 MB | ~55 MB (up to -93%) |
| KV-cache (large-model scenario) | very high | strongly reduced via MLA/HLCR |
| MoE routing compute | $O(E)$ | hierarchical approx. $O(\sqrt{E})$ |
| Effective vocabulary | baseline | can reduce via entropy-weighted merges |
| Total inference cost | baseline | substantial reduction depending on scale |

---

## 10) Recommended Implementation Roadmap

1. **Month 1-2:** 1B-scale prototype validation (quality + stability)
2. **Month 3-4:** 7B tuning with Ray Tune (quality/cost Pareto)
3. **Month 5-6:** long-context + preference alignment
4. **Month 7-8:** scale-up, optimization hardening, and IP/legal packaging

---

## 11) Useful Entry Points

- `src/model/__init__.py`:
  - `LLM`, `TransformerLM`, `MLAAttention`, `MoELayer`
- `src/training/objectives.py`:
  - `compute_language_model_loss`
- `src/training/optim.py`:
  - `create_adamw`, `CosineWithWarmup`
- `src/training/ray_tune_train.py`:
  - `run_quick_tune_local`, `build_tuner`, `train_llm_ray`
- `src/training/pipeline.py`:
  - innovation registry, cost summary, connection recheck
