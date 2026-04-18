from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FEDEmbedding(nn.Module):
    """Factorized Embedding Decomposition (FED): E = A @ B.
    
    Reduces memory from (vocab_size * d_model) to (vocab_size * k) + (k * d_model).
    With k=256, d_model=4096, vocab_size=50000: 800MB -> 55MB (93% reduction).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        k: int = 256,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k = k

        # A: vocab_size -> k (embedding projection)
        self.A = nn.Embedding(vocab_size, k, padding_idx=padding_idx)
        
        # B: k -> d_model (expansion to model dimension)
        self.B = nn.Linear(k, d_model, bias=False)

        # Scaling factor for stable gradients
        self.scale = math.sqrt(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize A and B with scaled uniform distribution."""
        nn.init.uniform_(self.A.weight, -0.05, 0.05)
        nn.init.uniform_(self.B.weight, -0.05, 0.05)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: token_ids -> A projection -> B expansion -> scaled output.
        
        Args:
            token_ids: shape [batch_size, seq_len] or [batch_size]
        
        Returns:
            embeddings: shape [..., d_model]
        """
        # A: [batch, seq, k]
        a_out = self.A(token_ids)
        
        # B: [batch, seq, d_model]
        b_out = self.B(a_out)
        
        # Scale for stable training
        return b_out * self.scale

    def get_embedding_matrix(self) -> torch.Tensor:
        """Materialize full embedding matrix E = A @ B for inspection.
        
        Returns:
            E: shape [vocab_size, d_model]
        """
        # Temporarily disable padding_idx to get full matrix
        a_weight = self.A.weight  # [vocab_size, k]
        b_weight = self.B.weight  # [d_model, k] -> transpose to [k, d_model]
        E = torch.mm(a_weight, b_weight.t())  # [vocab_size, d_model]
        return E * self.scale
