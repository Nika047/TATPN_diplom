"""
models/heads.py — Дополнительные MLP-головы TATPN.

Используются для side-task'ов:
  - ETAHead: регрессия времени до конечной точки (секунды).

Классификация цели реализована внутри Cross-Attention Target Head
(см. attention.py), поэтому отдельная голова не нужна.
"""

import torch
import torch.nn as nn

import config


class ETAHead(nn.Module):
    """
    MLP-голова для регрессии ETA (estimated time of arrival).

    Args:
        in_dim:     размерность входа (context + target_aware, конкатенация)
        hidden_dim: размерность скрытого слоя
        dropout:    вероятность dropout
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = config.ETA_HEAD_HIDDEN,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            # ETA >= 0 → softplus (мягкий ReLU) после линейного выхода
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)

        Returns:
            eta: (B,) — оценка ETA в секундах
        """
        return self.net(x).squeeze(-1)
