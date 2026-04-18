from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class FEDDkEmbedding(nn.Module):
    """Factorized Embedding with Dynamic k (FED-Dk).
    
    Per-token bottleneck dimension k_i based on token frequency:
        k_i = k_min + (k_max - k_min) * sigmoid(alpha * log(freq_i))
    
    High-frequency tokens get larger k (richer representation).
    Rare tokens get smaller k (compressed, regularized).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        k_min: int = 64,
        k_max: int = 512,
        alpha: float = 1.0,
        token_freqs: Optional[Dict[int, float]] = None,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k_min = k_min
        self.k_max = k_max
        self.alpha = alpha
        self.padding_idx = padding_idx

        # Use uniform k if frequencies not provided
        if token_freqs is None:
            token_freqs = {i: 1.0 for i in range(vocab_size)}

        # Compute k_i for each token
        self.register_buffer("k_per_token", self._compute_k_per_token(token_freqs))

        # Global A and B matrices (padded to k_max for simplicity)
        self.A = nn.Embedding(vocab_size, k_max, padding_idx=padding_idx)
        self.B = nn.Linear(k_max, d_model, bias=False)

        # Mask to zero out unused dimensions for tokens with k_i < k_max
        self.register_buffer("dim_mask", self._compute_dim_mask())

        self.scale = math.sqrt(d_model)

        self._init_weights()

    def _compute_k_per_token(self, token_freqs: Dict[int, float]) -> torch.Tensor:
        """Compute k_i for each token based on frequency."""
        k_per_token = torch.zeros(self.vocab_size, dtype=torch.float32)

        for token_id, freq in token_freqs.items():
            if token_id < self.vocab_size:
                # Avoid log(0)
                log_freq = math.log(max(freq, 1e-8))
                sigmoid_val = 1.0 / (1.0 + math.exp(-self.alpha * log_freq))
                k_i = self.k_min + (self.k_max - self.k_min) * sigmoid_val
                k_per_token[token_id] = k_i

        return k_per_token.long()

    def _compute_dim_mask(self) -> torch.Tensor:
        """Compute [vocab_size, k_max] mask for per-token k_i masking."""
        mask = torch.zeros(self.vocab_size, self.k_max, dtype=torch.bool)
        for token_id in range(self.vocab_size):
            k_i = self.k_per_token[token_id].item()
            mask[token_id, :k_i] = True
        return mask

    def _init_weights(self) -> None:
        nn.init.uniform_(self.A.weight, -0.05, 0.05)
        nn.init.uniform_(self.B.weight, -0.05, 0.05)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-token bottleneck dimension masking.
        
        Args:
            token_ids: shape [batch_size, seq_len] or [batch_size]
        
        Returns:
            embeddings: shape [..., d_model]
        """
        # A: [batch, seq, k_max]
        a_out = self.A(token_ids)

        # Mask: [batch, seq, k_max] — select only valid dims per token
        batch_shape = token_ids.shape
        token_ids_flat = token_ids.view(-1)
        mask_selected = self.dim_mask[token_ids_flat]  # [batch*seq, k_max]
        mask_selected = mask_selected.view(*batch_shape, self.k_max)

        # Zero out inactive dimensions
        a_out_masked = a_out * mask_selected.float()

        # B: [batch, seq, d_model]
        b_out = self.B(a_out_masked)

        return b_out * self.scale

    def get_embedding_matrix(self) -> torch.Tensor:
        """Materialize full embedding matrix with per-token k masking.
        
        Returns:
            E: shape [vocab_size, d_model]
        """
        a_weight = self.A.weight  # [vocab_size, k_max]
        mask_all = self.dim_mask.float()  # [vocab_size, k_max]
        a_masked = a_weight * mask_all

        b_weight = self.B.weight  # [d_model, k_max]
        E = torch.mm(a_masked, b_weight.t())  # [vocab_size, d_model]
        return E * self.scale
