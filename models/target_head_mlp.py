"""
target_head_mlp.py - Этап 4. MLP-голова классификации цели.

Для каждой цели формируем вход [контекст траектории, геометрия к этой цели]
и общим MLP считаем скалярный score. Softmax по целям (в loss) даёт
распределение вероятностей P(цель_i | траектория).

Простой ("наивный") вариант target head: геометрия подаётся явными признаками.
На Этапе 5 её заменит Cross-Attention с обучаемыми эмбеддингами целей.
"""

import torch
import torch.nn as nn

import config
from features.geometry import GEO_DIM


class TargetHeadMLP(nn.Module):
    """context (B,D) + geo (B,K,G) -> логиты целей (B,K)."""

    def __init__(self,
                 context_dim: int,
                 geo_dim: int = GEO_DIM,
                 hidden: int = config.TARGET_HEAD_HIDDEN,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + geo_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, context: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, D)     - контекст траектории
            geo:     (B, K, G)  - геометрия к каждой цели

        Returns:
            logits: (B, K)
        """
        B, K, _ = geo.shape
        ctx = context.unsqueeze(1).expand(-1, K, -1)     # (B, K, D)
        x = torch.cat([ctx, geo], dim=-1)                # (B, K, D+G)
        score = self.mlp(x).squeeze(-1)                  # (B, K)
        return score
