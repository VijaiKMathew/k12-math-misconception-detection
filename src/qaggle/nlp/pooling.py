"""
Custom Pooling Layers for Transformer Encoders
==============================================
Kaggle Grandmaster Best Practice (Dieter / Chris Deotte / Yauhen Babakhin).
Provides expressive pooling alternatives to standard [CLS] token pooling:
  - MeanPooling: attention-mask weighted average across sequence tokens
  - GeMPooling: Generalized Mean pooling with learnable p parameter
  - MultiHeadAttentionPooling: trainable query attention pooling
"""

from __future__ import annotations

import torch
from torch import nn


class MeanPooling(nn.Module):
    """Attention-mask weighted mean pooling over token embeddings."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class GeMPooling(nn.Module):
    """Generalized Mean (GeM) pooling with trainable p exponent."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Clamp embeddings to avoid NaN under exponentiation
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        clamped = torch.clamp(last_hidden_state, min=self.eps) * mask_expanded
        power_sum = torch.sum(clamped.pow(self.p), dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_power = power_sum / sum_mask
        return torch.clamp(mean_power, min=self.eps).pow(1.0 / self.p)


class MultiHeadAttentionPooling(nn.Module):
    """Trainable query multi-head attention pooling over token sequences."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = last_hidden_state.size(0)
        query = self.query.expand(batch_size, -1, -1)
        # key_padding_mask is True for padded positions in PyTorch MHA
        key_padding_mask = attention_mask == 0
        pooled, _ = self.mha(query, last_hidden_state, last_hidden_state, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)
