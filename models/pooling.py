"""
pooling.py - Adaptive Context Pooling для Этапа 3.

Вместо простого mean/max pooling берём последние K реальных шагов (по маске)
и считаем attention-взвешенную сумму: свежим точкам больше веса, но есть
доступ к недалёкому прошлому; корректно обрабатывает padding.

Перенесено из архивной реализации (не надо/models/pooling.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class AdaptiveContextPooling(nn.Module):
    """Attention-pooling по последним K реальным шагам."""

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 k: int = config.CONTEXT_POOL_K,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: (B, L, d_model), mask: (B, L) -> context: (B, d_model)."""
        B, L, D = h.shape
        K = min(self.k, L)
        device = h.device

        lengths = mask.sum(dim=1).long().clamp(min=1)          # (B,)
        arange_k = torch.arange(K, device=device).unsqueeze(0)  # (1, K)
        start = (lengths - K).clamp(min=0).unsqueeze(1)         # (B, 1)
        idx = start + arange_k                                  # (B, K)
        idx = torch.where(idx < lengths.unsqueeze(1), idx,
                          (lengths - 1).unsqueeze(1))
        idx = idx.clamp(min=0, max=L - 1)

        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)           # (B, K, D)
        h_last = torch.gather(h, dim=1, index=idx_exp)          # (B, K, D)
        last_mask = (idx <= (lengths - 1).unsqueeze(1)).float()  # (B, K)

        scores = self.scorer(h_last).squeeze(-1)                # (B, K)
        scores = scores.masked_fill(last_mask < 0.5, float("-inf"))
        weights = F.softmax(scores, dim=-1)                     # (B, K)

        if torch.isnan(weights).any():
            weights = torch.nan_to_num(weights, nan=0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        context = (weights.unsqueeze(-1) * h_last).sum(dim=1)   # (B, D)
        return context
