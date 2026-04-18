from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn

from ..tokenizer.bpe_standard import BPETokenizer
from .config import EmbeddingConfig
from .fed import FEDEmbedding
from .fed_dk import FEDDkEmbedding
from .rope import ARFSRoPEEmbedding, RoPEEmbedding
from .token_embedding import PositionalEmbedding, TokenEmbedding, TokenizerVocabAdapter


def create_embedding_module(
    config: EmbeddingConfig,
    tokenizer: Optional[BPETokenizer] = None,
    token_freqs: Optional[Dict[int, float]] = None,
) -> nn.Module:
    """Factory function to create embedding module based on config.
    
    Args:
        config: EmbeddingConfig specifying vocab_size, d_model, mode, etc.
        tokenizer: Optional BPETokenizer for vocabulary; if None, uses config.vocab_size.
        token_freqs: Optional dict of token_id -> frequency for FED-Dk.
    
    Returns:
        Embedding module (plain, FED, or FED-Dk).
    
    Raises:
        ValueError: if config is invalid or mode is unsupported.
    """
    config.validate()

    if config.mode == "plain":
        return TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.padding_idx,
        )

    elif config.mode == "fed":
        return FEDEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            k=config.k,
            padding_idx=config.padding_idx,
        )

    elif config.mode == "fed_dk":
        return FEDDkEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            k_min=config.k_min,
            k_max=config.k_max,
            alpha=config.alpha,
            token_freqs=token_freqs,
            padding_idx=config.padding_idx,
        )

    else:
        raise ValueError(f"Unsupported embedding mode: {config.mode}")


def create_vocab_adapter(
    tokenizer: BPETokenizer,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> TokenizerVocabAdapter:
    """Create vocabulary adapter from tokenizer.
    
    Args:
        tokenizer: BPETokenizer instance with trained vocab.
        pad_token: Special padding token.
        unk_token: Special unknown token.
    
    Returns:
        TokenizerVocabAdapter for token <-> id conversion.
    """
    return TokenizerVocabAdapter(
        tokenizer=tokenizer,
        pad_token=pad_token,
        unk_token=unk_token,
    )


def create_positional_embedding(
    d_model: int,
    max_seq_length: int = 2048,
    pos_type: str = "learned",
    rope_base: float = 10000.0,
    rope_n_domains: int = 4,
) -> nn.Module:
    """Create positional embedding module (learned, RoPE, or ARFS).
    
    Args:
        d_model: Model dimension.
        max_seq_length: Maximum sequence length to support.
        pos_type: Type of positional embedding: "learned", "rope", or "arfs".
        rope_base: Base frequency for RoPE/ARFS.
        rope_n_domains: Number of domains for ARFS.
    
    Returns:
        Positional embedding module.
    """
    if pos_type == "learned":
        return PositionalEmbedding(d_model=d_model, max_seq_length=max_seq_length)
    elif pos_type == "rope":
        return RoPEEmbedding(d_model=d_model, max_seq_len=max_seq_length, base=rope_base)
    elif pos_type == "arfs":
        return ARFSRoPEEmbedding(
            d_model=d_model,
            max_seq_len=max_seq_length,
            base=rope_base,
            n_domains=rope_n_domains,
        )
    else:
        raise ValueError(f"Unknown positional embedding type: {pos_type}")
