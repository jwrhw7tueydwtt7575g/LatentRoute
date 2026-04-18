"""Embedding modules: plain, FED, FED-Dk, RoPE, and ARFS implementations."""

from .config import EmbeddingConfig
from .factory import (
    create_embedding_module,
    create_positional_embedding,
    create_vocab_adapter,
)
from .fed import FEDEmbedding
from .fed_dk import FEDDkEmbedding
from .rope import ARFSRoPEEmbedding, RoPEEmbedding, apply_rope, precompute_rope_freqs
from .token_embedding import PositionalEmbedding, TokenEmbedding, TokenizerVocabAdapter

__all__ = [
    "EmbeddingConfig",
    "FEDEmbedding",
    "FEDDkEmbedding",
    "TokenEmbedding",
    "PositionalEmbedding",
    "TokenizerVocabAdapter",
    "RoPEEmbedding",
    "ARFSRoPEEmbedding",
    "precompute_rope_freqs",
    "apply_rope",
    "create_embedding_module",
    "create_vocab_adapter",
    "create_positional_embedding",
]
