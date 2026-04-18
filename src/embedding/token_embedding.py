from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..tokenizer.bpe_standard import BPETokenizer


class TokenizerVocabAdapter:
    """Bridges tokenizer vocabulary to integer token IDs."""

    def __init__(
        self,
        tokenizer: BPETokenizer,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab  # token_str -> id
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Ensure special tokens are in vocab
        self._ensure_special_tokens()

    def _ensure_special_tokens(self) -> None:
        """Register special tokens in vocab if missing."""
        specials = [
            (self.pad_token, 0),
            (self.unk_token, 1),
        ]
        if self.bos_token:
            specials.append((self.bos_token, 2))
        if self.eos_token:
            specials.append((self.eos_token, 3 if self.bos_token else 2))

        for token, fallback_id in specials:
            if token not in self.vocab:
                new_id = max(self.vocab.values()) + 1 if self.vocab else fallback_id
                self.vocab[token] = new_id
                self.reverse_vocab[new_id] = token

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token, 1))

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab.get(token_id, self.unk_token)

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token(token_id) for token_id in token_ids]

    def get_special_token_ids(self) -> Dict[str, int]:
        return {
            "pad": self.vocab.get(self.pad_token, 0),
            "unk": self.vocab.get(self.unk_token, 1),
            "bos": self.vocab.get(self.bos_token, None) if self.bos_token else None,
            "eos": self.vocab.get(self.eos_token, None) if self.eos_token else None,
        }


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, d_model: int, max_seq_length: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pos_embed = nn.Embedding(max_seq_length, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional embeddings for sequence length.
        
        Args:
            seq_len: sequence length
        
        Returns:
            pos_embed: shape [seq_len, d_model]
        """
        positions = torch.arange(seq_len, dtype=torch.long, device=self.pos_embed.weight.device)
        return self.pos_embed(positions)


class TokenEmbedding(nn.Module):
    """Simple token embedding without factorization."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = (d_model ** 0.5)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(token_ids) * self.scale
