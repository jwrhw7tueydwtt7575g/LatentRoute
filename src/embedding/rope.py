"""Rotary Position Embeddings (RoPE) and Adaptive RoPE with Frequency Scaling (ARFS)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def precompute_rope_freqs(
    d_model: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE frequency components.
    
    Standard RoPE: theta_j = base^(-2j / d_model) for j = 0, 1, ..., d_model/2
    Then compute position * theta for all positions and take cos/sin.
    
    Args:
        d_model: Model dimension (must be even).
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation (default 10000).
        device: Device for tensors.
    
    Returns:
        (freqs_cos, freqs_sin): shape [max_seq_len, d_model//2] each.
    """
    # j = 0, 2, 4, ..., d_model-2
    j = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    
    # theta_j = base^(-2j / d_model)
    theta = 1.0 / (base ** (j / d_model))  # shape: [d_model//2]
    
    # positions: [max_seq_len]
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    
    # freqs: [max_seq_len, d_model//2] = outer product of positions and theta
    freqs = torch.outer(positions, theta)
    
    # Precompute cos and sin
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    
    return freqs_cos, freqs_sin


def apply_rope(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation to query or key vectors.
    
    Args:
        x: [batch, seq_len, n_heads, d_head] or [batch, seq_len, d_model]
        freqs_cos: [seq_len, d_head//2]
        freqs_sin: [seq_len, d_head//2]
    
    Returns:
        rotated: same shape as x
    """
    # Handle different input shapes
    original_shape = x.shape
    
    # Flatten to [..., seq_len, d] for processing
    if x.dim() == 4:
        batch, seq_len, n_heads, d_head = x.shape
        x = x.reshape(batch * n_heads, seq_len, d_head)
    else:
        seq_len = x.shape[-2]
        d_model = x.shape[-1]
    
    # Extract even and odd dimensions
    x1 = x[..., 0::2]  # dims 0, 2, 4, ...
    x2 = x[..., 1::2]  # dims 1, 3, 5, ...
    
    # Slice freqs to actual sequence length (in case x is shorter)
    freqs_cos_seq = freqs_cos[:seq_len]
    freqs_sin_seq = freqs_sin[:seq_len]
    
    # Broadcast freqs to match batch/n_heads dims
    freqs_cos_seq = freqs_cos_seq.unsqueeze(0)  # [1, seq_len, d//2]
    freqs_sin_seq = freqs_sin_seq.unsqueeze(0)
    
    # Apply rotation: [x1, x2] @ [[cos, -sin], [sin, cos]]^T
    rotated_x1 = x1 * freqs_cos_seq - x2 * freqs_sin_seq
    rotated_x2 = x1 * freqs_sin_seq + x2 * freqs_cos_seq
    
    # Interleave back: [r1_0, r2_0, r1_1, r2_1, ...]
    rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    
    # Restore original shape
    rotated = rotated.reshape(original_shape)
    return rotated


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (standard, fixed frequencies)."""
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        # Precompute and cache
        freqs_cos, freqs_sin = precompute_rope_freqs(
            d_model=d_model,
            max_seq_len=max_seq_len,
            base=base,
            device="cpu",
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply RoPE to x.
        
        Args:
            x: [batch, seq_len, n_heads, d_head] or [batch, seq_len, d_model]
            seq_len: Optional override for sequence length.
        
        Returns:
            rotated: same shape as x
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        return apply_rope(x, self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])


class ARFSRoPEEmbedding(nn.Module):
    """Adaptive RoPE with Frequency Scaling (ARFS).
    
    Learns per-dimension frequency scalars gamma_j conditioned on domain embedding.
    theta_j^(domain) = theta_j * exp(gamma_j * z_domain)
    
    Where z_domain is a learned domain embedding (e.g., for code vs. math).
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        n_domains: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.n_domains = n_domains
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        # Precompute base frequencies
        freqs_cos_base, freqs_sin_base = precompute_rope_freqs(
            d_model=d_model,
            max_seq_len=max_seq_len,
            base=base,
            device="cpu",
        )
        self.register_buffer("freqs_cos_base", freqs_cos_base)
        self.register_buffer("freqs_sin_base", freqs_sin_base)
        
        # Domain embeddings: [n_domains, d_model//2]
        self.domain_embed = nn.Embedding(n_domains, d_model // 2)
        
        # Per-dimension frequency scalars: [d_model//2]
        self.gamma = nn.Parameter(torch.zeros(d_model // 2))
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.normal_(self.domain_embed.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gamma)
    
    def forward(
        self,
        x: torch.Tensor,
        domain_id: int = 0,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply adaptive RoPE with domain-specific frequency scaling.
        
        Args:
            x: [batch, seq_len, n_heads, d_head] or [batch, seq_len, d_model]
            domain_id: Domain index (0, 1, ..., n_domains-1)
            seq_len: Optional override for sequence length.
        
        Returns:
            rotated: same shape as x
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Get domain embedding: [d_model//2]
        z_domain = self.domain_embed(torch.tensor(domain_id, device=self.gamma.device))
        
        # Compute adaptive frequency scaling: exp(gamma_j * z_domain)
        # gamma: [d_model//2], z_domain: [d_model//2]
        scaling = torch.exp(self.gamma * z_domain)  # [d_model//2]
        
        # Scale the precomputed frequencies
        freqs_cos_scaled = self.freqs_cos_base[:seq_len] * scaling.unsqueeze(0)
        freqs_sin_scaled = self.freqs_sin_base[:seq_len] * scaling.unsqueeze(0)
        
        return apply_rope(x, freqs_cos_scaled, freqs_sin_scaled)
