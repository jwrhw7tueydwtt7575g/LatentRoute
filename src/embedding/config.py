from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Configuration for token and positional embeddings."""

    vocab_size: int = 50_000
    d_model: int = 4096
    max_seq_length: int = 2048
    dropout: float = 0.1
    padding_idx: int = 0
    mode: str = "plain"  # "plain", "fed", or "fed_dk"

    # FED / FED-Dk parameters
    k: int = 256  # Fixed bottleneck for FED
    k_min: int = 64  # Min bottleneck for FED-Dk
    k_max: int = 512  # Max bottleneck for FED-Dk
    alpha: float = 1.0  # Sigmoid slope for FED-Dk frequency scaling

    # Positional embedding parameters
    pos_embedding_type: str = "learned"  # "learned", "rope", or "arfs"
    rope_base: float = 10000.0  # Base for RoPE frequency
    rope_n_domains: int = 4  # Number of domains for ARFS

    # Special tokens
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def validate(self) -> None:
        if self.mode not in {"plain", "fed", "fed_dk"}:
            raise ValueError(f"mode must be in {{'plain', 'fed', 'fed_dk'}}, got {self.mode}")
        if self.k <= 0 or self.d_model <= 0:
            raise ValueError(f"k and d_model must be positive, got k={self.k}, d_model={self.d_model}")
        if self.k_min <= 0 or self.k_max <= 0 or self.k_min > self.k_max:
            raise ValueError(f"k_min/k_max invalid: k_min={self.k_min}, k_max={self.k_max}")
        if self.pos_embedding_type not in {"learned", "rope", "arfs"}:
            raise ValueError(f"pos_embedding_type must be in {{'learned', 'rope', 'arfs'}}, got {self.pos_embedding_type}")
        if self.d_model % 2 != 0 and self.pos_embedding_type in {"rope", "arfs"}:
            raise ValueError(f"d_model must be even for RoPE/ARFS, got {self.d_model}")
