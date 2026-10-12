"""
eta_head.py - Этап 5. MLP-голова оценки времени до цели (ETA).

Вход - конкатенация контекста траектории и target-aware вектора. Softplus на
выходе гарантирует неотрицательность ETA (секунды).

Перенесено из архивной реализации (не надо/models/heads.py).
"""

import torch
import torch.nn as nn

import config


class ETAHead(nn.Module):
    """context+target_aware -> ETA (сек), неотрицательный."""

    def __init__(self, in_dim: int,
                 hidden_dim: int = config.ETA_HEAD_HIDDEN,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)      # (B,)
