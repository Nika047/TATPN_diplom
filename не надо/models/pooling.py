"""
models/pooling.py — Adaptive Context Pooling.

Идея:
  Вместо простого mean/max pooling по всем шагам последовательности
  берём последние K шагов (CONTEXT_POOL_K = 64) и считаем
  attention-взвешенную сумму. Это:
    - учитывает что для прогноза важнее свежие точки;
    - но позволяет модели "оглянуться" в недалёкое прошлое;
    - устойчиво к padding (через маску).

Алгоритм:
  1) Выбираем последние K реальных шагов (по маске).
  2) Считаем скоры через small MLP (h_t → score_t).
  3) Softmax по K.
  4) Взвешенная сумма даёт контекстный вектор.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class AdaptiveContextPooling(nn.Module):
    """
    Attention-pooling по последним K реальным шагам.

    Args:
        d_model: размерность входа (= MAMBA_D_MODEL)
        k:       количество последних шагов
        dropout: вероятность dropout
    """

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 k: int = config.CONTEXT_POOL_K,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.k = k

        # Attention-скоринг: (d_model) → (1)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:    (B, L, d_model)   — выход Mamba-энкодера
            mask: (B, L)            — 1 для реальных точек, 0 для padding

        Returns:
            context: (B, d_model)   — контекстный вектор для голов
        """
        B, L, D = h.shape
        K = min(self.k, L)

        # Для каждого батча берём последние K реальных точек.
        # Поскольку padding идёт ПОСЛЕ реальных точек, нам нужны
        # реальные индексы [n_real - K, ..., n_real - 1].
        # Реализуем через gather по индексам.

        # Длина реальной части каждого сэмпла
        lengths = mask.sum(dim=1).long().clamp(min=1)   # (B,)

        # Индексы последних K реальных точек для каждого сэмпла
        # idx[b, i] = lengths[b] - K + i,  при i in [0, K)
        # Если lengths[b] < K, добиваем повторением последней точки.
        device = h.device
        arange_k = torch.arange(K, device=device).unsqueeze(0)       # (1, K)
        start = (lengths - K).clamp(min=0).unsqueeze(1)               # (B, 1)
        idx = start + arange_k                                        # (B, K)
        # Если lengths < K, повторяем первый реальный шаг (idx=0)
        idx = torch.where(idx < lengths.unsqueeze(1), idx,
                          (lengths - 1).unsqueeze(1))
        idx = idx.clamp(min=0, max=L - 1)

        # Собираем последние K шагов
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)                 # (B, K, D)
        h_last = torch.gather(h, dim=1, index=idx_exp)                # (B, K, D)

        # Маска для собранных шагов (1 если соответствующий idx — реальный)
        last_mask = (idx < lengths.unsqueeze(1)).float()              # (B, K)

        # Если у короткой траектории lengths<K, мы продублировали последний
        # реальный — он валиден. Поэтому переписываем:
        last_mask = (idx <= (lengths - 1).unsqueeze(1)).float()       # (B, K)

        # Attention scores
        scores = self.scorer(h_last).squeeze(-1)                       # (B, K)
        scores = scores.masked_fill(last_mask < 0.5, float("-inf"))
        weights = F.softmax(scores, dim=-1)                            # (B, K)

        # Если все позиции маскированы (короткая траектория) — на всякий случай
        # заменяем NaN на 0 и считаем mean по реальным
        if torch.isnan(weights).any():
            weights = torch.nan_to_num(weights, nan=0.0)
            wsum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            weights = weights / wsum

        # Взвешенная сумма
        context = (weights.unsqueeze(-1) * h_last).sum(dim=1)          # (B, D)
        return context
