from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embedding.factory import create_embedding_module
from ..embedding.rope import ARFSRoPEEmbedding, RoPEEmbedding, apply_rope
from ..embedding.token_embedding import PositionalEmbedding
from ..embedding.config import EmbeddingConfig


@dataclass
class ModelConfig:
    vocab_size: int = 50_000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    max_seq_length: int = 2048
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    norm_eps: float = 1e-5
    use_bias: bool = False
    padding_idx: int = 0
    embedding_mode: str = "plain"
    pos_embedding_type: str = "learned"
    rope_base: float = 10000.0
    rope_n_domains: int = 4
    k: int = 256
    k_min: int = 64
    k_max: int = 512
    alpha: float = 1.0

    attention_type: str = "mha"  # "mha" or "mla"
    mla_d_c: int = 512
    mla_d_rope: int = 64
    use_hlcr: bool = False
    hlcr_c1: int = 1024
    hlcr_c2: int = 256

    ffn_type: str = "gelu"  # "gelu", "swiglu", or "moe"
    moe_num_experts: int = 64
    moe_top_k: int = 2
    moe_d_expert: int = 2048
    moe_aux_loss_weight: float = 0.01
    moe_entropy_weight: float = 0.01
    moe_hierarchical: bool = False
    moe_num_groups: int = 8
    moe_group_top_k: int = 1

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, and n_heads must be positive")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1), got {self.attn_dropout}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got d_model={self.d_model}, n_heads={self.n_heads}")
        if self.embedding_mode not in {"plain", "fed", "fed_dk"}:
            raise ValueError(f"embedding_mode must be one of {{'plain', 'fed', 'fed_dk'}}, got {self.embedding_mode}")
        if self.pos_embedding_type not in {"learned", "rope", "arfs"}:
            raise ValueError(f"pos_embedding_type must be one of {{'learned', 'rope', 'arfs'}}, got {self.pos_embedding_type}")
        if self.embedding_mode in {"fed", "fed_dk"}:
            if self.k <= 0:
                raise ValueError(f"k must be positive, got {self.k}")
            if self.k_min <= 0 or self.k_max <= 0 or self.k_min > self.k_max:
                raise ValueError(f"k_min/k_max invalid: k_min={self.k_min}, k_max={self.k_max}")
        if self.pos_embedding_type in {"rope", "arfs"}:
            head_dim = self.d_model // self.n_heads
            if head_dim % 2 != 0:
                raise ValueError(f"head_dim must be even for RoPE/ARFS, got {head_dim}")
        if self.rope_n_domains <= 0:
            raise ValueError(f"rope_n_domains must be positive, got {self.rope_n_domains}")
        if self.padding_idx < 0 or self.padding_idx >= self.vocab_size:
            raise ValueError(f"padding_idx must be in [0, vocab_size), got {self.padding_idx}")
        if self.attention_type not in {"mha", "mla"}:
            raise ValueError(f"attention_type must be one of {{'mha', 'mla'}}, got {self.attention_type}")
        if self.mla_d_c <= 0:
            raise ValueError(f"mla_d_c must be positive, got {self.mla_d_c}")
        if self.mla_d_rope < 0:
            raise ValueError(f"mla_d_rope must be non-negative, got {self.mla_d_rope}")
        if self.mla_d_rope > 0 and self.mla_d_rope % 2 != 0:
            raise ValueError(f"mla_d_rope must be even, got {self.mla_d_rope}")
        if self.use_hlcr:
            if self.hlcr_c1 <= 0 or self.hlcr_c2 <= 0:
                raise ValueError(f"hlcr_c1 and hlcr_c2 must be positive, got hlcr_c1={self.hlcr_c1}, hlcr_c2={self.hlcr_c2}")
        if self.ffn_type not in {"gelu", "swiglu", "moe"}:
            raise ValueError(f"ffn_type must be one of {{'gelu', 'swiglu', 'moe'}}, got {self.ffn_type}")
        if self.moe_num_experts <= 0:
            raise ValueError(f"moe_num_experts must be positive, got {self.moe_num_experts}")
        if self.moe_top_k <= 0 or self.moe_top_k > self.moe_num_experts:
            raise ValueError(f"moe_top_k must be in [1, moe_num_experts], got moe_top_k={self.moe_top_k}")
        if self.moe_d_expert <= 0:
            raise ValueError(f"moe_d_expert must be positive, got {self.moe_d_expert}")
        if self.moe_num_groups <= 0:
            raise ValueError(f"moe_num_groups must be positive, got {self.moe_num_groups}")
        if self.moe_hierarchical and self.moe_num_experts % self.moe_num_groups != 0:
            raise ValueError("moe_num_experts must be divisible by moe_num_groups for hierarchical routing")
        if self.moe_hierarchical:
            experts_per_group = self.moe_num_experts // self.moe_num_groups
            if self.moe_top_k > experts_per_group:
                raise ValueError(
                    "moe_top_k must be <= experts_per_group for hierarchical routing, "
                    f"got moe_top_k={self.moe_top_k}, experts_per_group={experts_per_group}"
                )
        if self.moe_group_top_k != 1:
            raise ValueError("moe_group_top_k currently supports only 1")

    def __post_init__(self) -> None:
        self.validate()

    def to_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_length=self.max_seq_length,
            dropout=self.dropout,
            padding_idx=self.padding_idx,
            mode=self.embedding_mode,
            k=self.k,
            k_min=self.k_min,
            k_max=self.k_max,
            alpha=self.alpha,
            pos_embedding_type=self.pos_embedding_type,
            rope_base=self.rope_base,
            rope_n_domains=self.rope_n_domains,
        )


@dataclass
class KVCache:
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    seq_len: int = 0

    def is_empty(self) -> bool:
        return self.key is None or self.value is None or self.seq_len == 0

    def append(self, key: torch.Tensor, value: torch.Tensor) -> KVCache:
        if self.is_empty():
            self.key = key
            self.value = value
        else:
            self.key = torch.cat([self.key, key], dim=2)
            self.value = torch.cat([self.value, value], dim=2)
        self.seq_len = int(self.key.shape[2])
        return self

    def get(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.is_empty():
            return None, None
        return self.key, self.value

    def reset(self) -> KVCache:
        self.key = None
        self.value = None
        self.seq_len = 0
        return self

    def to(self, *args, **kwargs) -> KVCache:
        if self.key is not None:
            self.key = self.key.to(*args, **kwargs)
        if self.value is not None:
            self.value = self.value.to(*args, **kwargs)
        return self


@dataclass
class LatentKVCache:
    c_kv: Optional[torch.Tensor] = None
    seq_len: int = 0

    def is_empty(self) -> bool:
        return self.c_kv is None or self.seq_len == 0

    def append(self, c_kv: torch.Tensor) -> LatentKVCache:
        if self.is_empty():
            self.c_kv = c_kv
        else:
            self.c_kv = torch.cat([self.c_kv, c_kv], dim=1)
        self.seq_len = int(self.c_kv.shape[1])
        return self

    def get(self) -> Optional[torch.Tensor]:
        if self.is_empty():
            return None
        return self.c_kv

    def reset(self) -> LatentKVCache:
        self.c_kv = None
        self.seq_len = 0
        return self

    def to(self, *args, **kwargs) -> LatentKVCache:
        if self.c_kv is not None:
            self.c_kv = self.c_kv.to(*args, **kwargs)
        return self


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    is_causal: bool = True,
    causal_offset: int = 0,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        q_len = query.size(-2)
        k_len = key.size(-2)
        query_positions = torch.arange(q_len, device=query.device).unsqueeze(-1) + causal_offset
        key_positions = torch.arange(k_len, device=query.device).unsqueeze(0)
        causal_mask = key_positions <= query_positions
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

    if attn_mask is not None:
        mask = attn_mask
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    attn = torch.softmax(scores.float(), dim=-1).to(value.dtype)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    return torch.matmul(attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        pos_embedding_type: str = "learned",
        max_seq_length: int = 2048,
        rope_base: float = 10000.0,
        rope_n_domains: int = 4,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got d_model={d_model}, n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.pos_embedding_type = pos_embedding_type

        if pos_embedding_type in {"rope", "arfs"} and self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE/ARFS, got {self.head_dim}")

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if pos_embedding_type == "rope":
            self.rope: Optional[nn.Module] = RoPEEmbedding(
                d_model=self.head_dim,
                max_seq_len=max_seq_length,
                base=rope_base,
            )
        elif pos_embedding_type == "arfs":
            self.rope = ARFSRoPEEmbedding(
                d_model=self.head_dim,
                max_seq_len=max_seq_length,
                base=rope_base,
                n_domains=rope_n_domains,
            )
        else:
            self.rope = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _apply_rope(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        domain_id: int = 0,
    ) -> torch.Tensor:
        if self.rope is None:
            return x

        seq_len = x.size(-2)
        if isinstance(self.rope, RoPEEmbedding):
            freqs_cos = self.rope.freqs_cos[position_offset : position_offset + seq_len].to(device=x.device, dtype=x.dtype)
            freqs_sin = self.rope.freqs_sin[position_offset : position_offset + seq_len].to(device=x.device, dtype=x.dtype)
        elif isinstance(self.rope, ARFSRoPEEmbedding):
            domain_tensor = torch.tensor(domain_id, device=self.rope.gamma.device, dtype=torch.long)
            domain_embed = self.rope.domain_embed(domain_tensor)
            scaling = torch.exp(self.rope.gamma * domain_embed)
            freqs_cos = (self.rope.freqs_cos_base[position_offset : position_offset + seq_len] * scaling.unsqueeze(0)).to(
                device=x.device,
                dtype=x.dtype,
            )
            freqs_sin = (self.rope.freqs_sin_base[position_offset : position_offset + seq_len] * scaling.unsqueeze(0)).to(
                device=x.device,
                dtype=x.dtype,
            )
        else:
            return x

        x = x.transpose(1, 2)
        x = apply_rope(x, freqs_cos, freqs_sin)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        domain_id: int = 0,
        position_offset: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        past_len = 0 if cache is None else cache.seq_len
        if position_offset is None:
            position_offset = past_len

        query = self._shape(self.q_proj(x))
        key = self._shape(self.k_proj(x))
        value = self._shape(self.v_proj(x))

        query = self._apply_rope(query, position_offset=position_offset, domain_id=domain_id)
        key = self._apply_rope(key, position_offset=position_offset, domain_id=domain_id)

        if cache is not None:
            cache.append(key, value)
            key, value = cache.get()
        else:
            key, value = key, value

        output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
            is_causal=True,
            causal_offset=position_offset,
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA) with optional HLCR gating."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_c: int = 512,
        d_rope: int = 64,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_length: int = 2048,
        rope_base: float = 10000.0,
        use_hlcr: bool = False,
        hlcr_c1: int = 1024,
        hlcr_c2: int = 256,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got d_model={d_model}, n_heads={n_heads}")
        if d_rope % 2 != 0:
            raise ValueError(f"d_rope must be even for RoPE, got {d_rope}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_c = d_c
        self.d_rope = d_rope
        self.dropout = dropout
        self.use_hlcr = use_hlcr

        if use_hlcr:
            self.W_DKV_1 = nn.Linear(d_model, hlcr_c1, bias=False)
            self.W_DKV_2 = nn.Linear(d_model, hlcr_c2, bias=False)
            self.W_C2_PROJ = nn.Linear(hlcr_c2, hlcr_c1, bias=False)
            self.W_G = nn.Linear(d_model + hlcr_c1 + hlcr_c2, hlcr_c1, bias=True)
            self.W_MERGE = nn.Linear(hlcr_c1, d_c, bias=False)
        else:
            self.W_DKV = nn.Linear(d_model, d_c, bias=False)

        self.W_UK = nn.Linear(d_c, n_heads * self.d_head, bias=False)
        self.W_UV = nn.Linear(d_c, n_heads * self.d_head, bias=False)
        self.W_DQ = nn.Linear(d_model, d_c, bias=False)
        self.W_UQ = nn.Linear(d_c, n_heads * self.d_head, bias=False)
        self.W_REC = nn.Linear(d_c, d_model, bias=False)
        self.last_recon_loss = torch.tensor(0.0)

        self.W_QR = nn.Linear(d_model, n_heads * d_rope, bias=False) if d_rope > 0 else None
        self.W_KR = nn.Linear(d_c, d_rope, bias=False) if d_rope > 0 else None
        self.rope = RoPEEmbedding(d_model=d_rope, max_seq_len=max_seq_length, base=rope_base) if d_rope > 0 else None

        self.W_O = nn.Linear(n_heads * self.d_head, d_model, bias=bias)

    def _compress_kv(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_hlcr:
            return self.W_DKV(x)

        c1 = self.W_DKV_1(x)
        c2 = self.W_DKV_2(x)
        c2_proj = self.W_C2_PROJ(c2)
        g_in = torch.cat([c1, c2, x], dim=-1)
        g = torch.sigmoid(self.W_G(g_in))
        c_final = g * c1 + (1.0 - g) * c2_proj
        return self.W_MERGE(c_final)

    def _shape_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, head_dim)

    def _apply_rope(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        if self.rope is None or self.d_rope == 0:
            return x
        seq_len = x.size(1)
        freqs_cos = self.rope.freqs_cos[position_offset : position_offset + seq_len].to(device=x.device, dtype=x.dtype)
        freqs_sin = self.rope.freqs_sin[position_offset : position_offset + seq_len].to(device=x.device, dtype=x.dtype)
        return apply_rope(x, freqs_cos, freqs_sin)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[LatentKVCache] = None,
        domain_id: int = 0,
        position_offset: Optional[int] = None,
    ) -> torch.Tensor:
        del domain_id
        bsz, seq_len, _ = x.shape
        past_len = 0 if cache is None else cache.seq_len
        if position_offset is None:
            position_offset = past_len

        c_kv_new = self._compress_kv(x)
        recon = self.W_REC(c_kv_new)
        self.last_recon_loss = F.mse_loss(recon, x)
        c_q = self.W_DQ(x)

        if cache is not None:
            cache.append(c_kv_new)
            c_kv_all = cache.get()
            assert c_kv_all is not None
        else:
            c_kv_all = c_kv_new

        k_content = self.W_UK(c_kv_all).view(bsz, c_kv_all.size(1), self.n_heads, self.d_head)
        v = self.W_UV(c_kv_all).view(bsz, c_kv_all.size(1), self.n_heads, self.d_head)
        q_content = self.W_UQ(c_q).view(bsz, seq_len, self.n_heads, self.d_head)

        if self.d_rope > 0 and self.W_QR is not None and self.W_KR is not None:
            q_r = self._shape_heads(self.W_QR(x), self.d_rope)
            k_r = self.W_KR(c_kv_all).view(bsz, c_kv_all.size(1), 1, self.d_rope).expand(-1, -1, self.n_heads, -1)

            q_r = self._apply_rope(q_r, position_offset=position_offset)
            k_r = self._apply_rope(k_r, position_offset=0)

            q = torch.cat([q_content, q_r], dim=-1)
            k = torch.cat([k_content, k_r], dim=-1)
        else:
            q = q_content
            k = k_content

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
            is_causal=True,
            causal_offset=position_offset,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.d_head)
        return self.W_O(out)


class SwiGLUFFN(nn.Module):
    """Position-wise FFN using SwiGLU: (SiLU(xW_gate) ⊙ xW_up)W_down."""

    def __init__(self, d_model: int, d_ffn: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=bias)
        self.w_up = nn.Linear(d_model, d_ffn, bias=bias)
        self.w_down = nn.Linear(d_ffn, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x_flat).float()
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        return top_k_probs, top_k_idx, probs


class HierarchicalRouter(nn.Module):
    """Coarse group routing, then fine routing among experts inside the selected group."""

    def __init__(self, d_model: int, n_experts: int, top_k: int, n_groups: int):
        super().__init__()
        if n_experts % n_groups != 0:
            raise ValueError("n_experts must be divisible by n_groups")

        self.n_experts = n_experts
        self.top_k = top_k
        self.n_groups = n_groups
        self.experts_per_group = n_experts // n_groups

        self.group_router = nn.Linear(d_model, n_groups, bias=False)
        self.expert_routers = nn.ModuleList(
            [nn.Linear(d_model, self.experts_per_group, bias=False) for _ in range(n_groups)]
        )

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = x_flat.size(0)
        k = min(self.top_k, self.experts_per_group)
        group_logits = self.group_router(x_flat).float()
        group_probs = torch.softmax(group_logits, dim=-1)
        selected_group = torch.argmax(group_probs, dim=-1)

        top_k_idx = torch.zeros(num_tokens, k, device=x_flat.device, dtype=torch.long)
        top_k_probs = torch.zeros(num_tokens, k, device=x_flat.device, dtype=x_flat.dtype)
        full_probs = torch.zeros(num_tokens, self.n_experts, device=x_flat.device, dtype=x_flat.dtype)

        for g in range(self.n_groups):
            token_mask = selected_group == g
            if not token_mask.any():
                continue

            token_ids = token_mask.nonzero(as_tuple=False).squeeze(-1)
            x_g = x_flat[token_ids]

            expert_logits = self.expert_routers[g](x_g).float()
            expert_probs_local = torch.softmax(expert_logits, dim=-1)
            weighted_local = expert_probs_local * group_probs[token_ids, g].unsqueeze(-1)

            top_local_probs, top_local_idx = torch.topk(weighted_local, k, dim=-1)
            top_local_probs = top_local_probs / (top_local_probs.sum(dim=-1, keepdim=True) + 1e-9)
            global_idx = top_local_idx + g * self.experts_per_group

            top_k_idx[token_ids] = global_idx
            top_k_probs[token_ids] = top_local_probs.to(top_k_probs.dtype)
            full_probs[token_ids, g * self.experts_per_group : (g + 1) * self.experts_per_group] = weighted_local.to(full_probs.dtype)

        full_probs = full_probs / (full_probs.sum(dim=-1, keepdim=True) + 1e-9)
        return top_k_probs, top_k_idx, full_probs


class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int = 64,
        n_active: int = 2,
        d_expert: int = 2048,
        bias: bool = True,
        dropout: float = 0.0,
        hierarchical: bool = False,
        n_groups: int = 8,
        aux_loss_weight: float = 0.01,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.aux_loss_weight = aux_loss_weight
        self.entropy_weight = entropy_weight

        if hierarchical:
            self.router = HierarchicalRouter(d_model=d_model, n_experts=n_experts, top_k=n_active, n_groups=n_groups)
        else:
            self.router = TopKRouter(d_model=d_model, n_experts=n_experts, top_k=n_active)

        self.experts = nn.ModuleList(
            [SwiGLUFFN(d_model=d_model, d_ffn=d_expert, bias=bias, dropout=dropout) for _ in range(n_experts)]
        )

        self.load_balance_loss = torch.tensor(0.0)
        self.entropy_loss = torch.tensor(0.0)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)

        top_k_probs, top_k_idx, router_probs = self.router(x_flat)
        top_k_probs = top_k_probs.to(x_flat.dtype)

        output = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            assigned = (top_k_idx == i).any(dim=-1)
            if not assigned.any():
                continue

            tokens_i = x_flat[assigned]
            expert_out = self.experts[i](tokens_i)
            gate_w = (top_k_probs * (top_k_idx == i).float()).sum(dim=-1, keepdim=True)
            output[assigned] += gate_w[assigned] * expert_out

        with torch.no_grad():
            token_expert_count = F.one_hot(top_k_idx, num_classes=self.n_experts).float().sum(dim=1)
            f_i = token_expert_count.sum(dim=0) / max(float(x_flat.size(0) * self.n_active), 1.0)
        p_i = router_probs.mean(dim=0)

        self.load_balance_loss = self.n_experts * torch.sum(f_i * p_i)
        self.entropy_loss = -(p_i * torch.log(p_i + 1e-8)).sum()
        self.aux_loss = self.aux_loss_weight * self.load_balance_loss - self.entropy_weight * self.entropy_loss

        out = output.view(bsz, seq_len, dim)
        if return_aux:
            return out, self.aux_loss
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        norm_eps: float = 1e-5,
        bias: bool = True,
        pos_embedding_type: str = "learned",
        max_seq_length: int = 2048,
        rope_base: float = 10000.0,
        rope_n_domains: int = 4,
        attention_type: str = "mha",
        mla_d_c: int = 512,
        mla_d_rope: int = 64,
        use_hlcr: bool = False,
        hlcr_c1: int = 1024,
        hlcr_c2: int = 256,
        ffn_type: str = "gelu",
        moe_num_experts: int = 64,
        moe_top_k: int = 2,
        moe_d_expert: int = 2048,
        moe_aux_loss_weight: float = 0.01,
        moe_entropy_weight: float = 0.01,
        moe_hierarchical: bool = False,
        moe_num_groups: int = 8,
    ):
        super().__init__()
        hidden_dim = int(round(d_model * mlp_ratio))
        if hidden_dim <= 0:
            raise ValueError(f"mlp_ratio produced invalid hidden_dim={hidden_dim}")

        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        if attention_type == "mla":
            self.attn = MLAAttention(
                d_model=d_model,
                n_heads=n_heads,
                d_c=mla_d_c,
                d_rope=mla_d_rope,
                dropout=attn_dropout,
                bias=bias,
                max_seq_length=max_seq_length,
                rope_base=rope_base,
                use_hlcr=use_hlcr,
                hlcr_c1=hlcr_c1,
                hlcr_c2=hlcr_c2,
            )
        else:
            self.attn = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=attn_dropout,
                bias=bias,
                pos_embedding_type=pos_embedding_type,
                max_seq_length=max_seq_length,
                rope_base=rope_base,
                rope_n_domains=rope_n_domains,
            )
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn_type = ffn_type
        if ffn_type == "swiglu":
            self.mlp = SwiGLUFFN(d_model=d_model, d_ffn=hidden_dim, bias=bias, dropout=dropout)
        elif ffn_type == "moe":
            self.mlp = MoELayer(
                d_model=d_model,
                n_experts=moe_num_experts,
                n_active=moe_top_k,
                d_expert=moe_d_expert,
                bias=bias,
                dropout=dropout,
                hierarchical=moe_hierarchical,
                n_groups=moe_num_groups,
                aux_loss_weight=moe_aux_loss_weight,
                entropy_weight=moe_entropy_weight,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model, bias=bias),
                nn.Dropout(dropout),
            )

        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[Union[KVCache, LatentKVCache]] = None,
        domain_id: int = 0,
        position_offset: Optional[int] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, cache=cache, domain_id=domain_id, position_offset=position_offset)
        ffn_in = self.norm2(x)

        aux_loss = x.new_zeros(())
        if isinstance(self.mlp, MoELayer):
            ffn_out, aux_loss = self.mlp(ffn_in, return_aux=True)
            self.last_aux_loss = aux_loss
        else:
            ffn_out = self.mlp(ffn_in)
            self.last_aux_loss = None

        x = x + ffn_out
        if return_aux:
            return x, aux_loss
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = create_embedding_module(config.to_embedding_config())
        self.positional_embedding = PositionalEmbedding(config.d_model, config.max_seq_length) if config.pos_embedding_type == "learned" else None
        self.emb_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    norm_eps=config.norm_eps,
                    bias=config.use_bias,
                    pos_embedding_type=config.pos_embedding_type,
                    max_seq_length=config.max_seq_length,
                    rope_base=config.rope_base,
                    rope_n_domains=config.rope_n_domains,
                    attention_type=config.attention_type,
                    mla_d_c=config.mla_d_c,
                    mla_d_rope=config.mla_d_rope,
                    use_hlcr=config.use_hlcr,
                    hlcr_c1=config.hlcr_c1,
                    hlcr_c2=config.hlcr_c2,
                    ffn_type=config.ffn_type,
                    moe_num_experts=config.moe_num_experts,
                    moe_top_k=config.moe_top_k,
                    moe_d_expert=config.moe_d_expert,
                    moe_aux_loss_weight=config.moe_aux_loss_weight,
                    moe_entropy_weight=config.moe_entropy_weight,
                    moe_hierarchical=config.moe_hierarchical,
                    moe_num_groups=config.moe_num_groups,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        caches: Optional[Sequence[Union[KVCache, LatentKVCache]]] = None,
        domain_id: int = 0,
        use_cache: bool = False,
        return_aux: bool = False,
    ) -> Union[
        torch.Tensor,
        tuple[torch.Tensor, Sequence[Union[KVCache, LatentKVCache]]],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, Sequence[Union[KVCache, LatentKVCache]], torch.Tensor],
    ]:
        x = self.token_embedding(input_ids)
        if self.positional_embedding is not None:
            cache_offset = 0
            if caches is not None and len(caches) > 0 and caches[0] is not None:
                cache_offset = caches[0].seq_len
            pos = self.positional_embedding(cache_offset + input_ids.size(1)).to(device=x.device, dtype=x.dtype)
            pos = pos[cache_offset:cache_offset + input_ids.size(1)]
            x = x + pos.unsqueeze(0)
        x = self.emb_dropout(x)

        cache_cls = LatentKVCache if self.config.attention_type == "mla" else KVCache
        if caches is None:
            caches = [cache_cls() if use_cache else None for _ in range(len(self.layers))]
        elif len(caches) != len(self.layers):
            raise ValueError(f"caches must have length {len(self.layers)}, got {len(caches)}")
        elif use_cache:
            caches = [cache if cache is not None else cache_cls() for cache in caches]

        aux_total = x.new_zeros(())
        for layer, cache in zip(self.layers, caches):
            if return_aux:
                x, layer_aux = layer(x, attn_mask=attn_mask, cache=cache, domain_id=domain_id, return_aux=True)
                aux_total = aux_total + layer_aux
            else:
                x = layer(x, attn_mask=attn_mask, cache=cache, domain_id=domain_id)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if use_cache and return_aux:
            return logits, caches, aux_total
        if use_cache:
            return logits, caches
        if return_aux:
            return logits, aux_total
        return logits


class TransformerBlock(DecoderBlock):
    """Alias wrapper for DecoderBlock."""


class LLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50_000,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        d_c: int = 512,
        n_experts: int = 64,
        max_seq_len: int = 8192,
        d_rope: int = 64,
        embedding_mode: str = "fed_dk",
        tie_weights: bool = False,
    ):
        super().__init__()
        moe_num_groups = max(1, min(8, n_experts // 2))
        config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_length=max_seq_len,
            attention_type="mla",
            mla_d_c=d_c,
            mla_d_rope=d_rope,
            use_hlcr=True,
            ffn_type="moe",
            moe_num_experts=n_experts,
            moe_hierarchical=True,
            moe_num_groups=moe_num_groups,
            pos_embedding_type="arfs",
            embedding_mode=embedding_mode,
        )
        self.model = TransformerLM(config)

        token_embed_weight = None
        if hasattr(self.model.token_embedding, "embed") and hasattr(self.model.token_embedding.embed, "weight"):
            token_embed_weight = self.model.token_embedding.embed.weight
        if tie_weights and token_embed_weight is not None and self.model.lm_head.weight.shape == token_embed_weight.shape:
            self.model.lm_head.weight = token_embed_weight

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


__all__ = [
    "ModelConfig",
    "KVCache",
    "LatentKVCache",
    "RMSNorm",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "MLAAttention",
    "SwiGLUFFN",
    "TopKRouter",
    "HierarchicalRouter",
    "MoELayer",
    "DecoderBlock",
    "TransformerBlock",
    "TransformerLM",
    "LLM",
]
