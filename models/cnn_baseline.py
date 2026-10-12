"""
cnn_baseline.py - Этап 2. Простая свёрточная сеть прогноза траектории.

Только координаты (без цели и ETA). Простейшая обучаемая модель, baseline
для последующих этапов.

Архитектура:
  observed (B, L, 15) -> Conv1d стек (15->64->128->256) + BN + ReLU
  -> masked global average pool -> (B, 256)
  -> FC-голова -> (B, H, 3) ДЕЛЬТЫ от последней наблюдаемой точки

Прогноз: pred_k = last_point + delta_k, k = 1..H.
Сеть предсказывает будущее САМА, без встроенной физической экстраполяции:
всё движение (скорость, повороты) она должна извлечь из наблюдаемых
кинематических признаков. Это делает вклад архитектуры измеримым и позволяет
честно сравнивать этапы (без prior'а, который маскировал бы обучение).

Последний слой инициализируется нулём: на старте delta = 0, то есть модель
стартует как бейзлайн "стоять на месте" (constant_position), а затем в ходе
обучения учится реальному смещению. Так кривая обучения интерпретируема:
val ADE стартует у уровня constant_position и должен падать.
"""

import torch
import torch.nn as nn

import config


class ConvTrajectoryNet(nn.Module):
    """Свёрточный энкодер + FC-голова, прогноз дельт от последней точки."""

    def __init__(self,
                 in_features: int = config.INPUT_FEATURES,
                 pred_horizon: int = config.PRED_HORIZON,
                 hidden: int = 256,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.pred_horizon = pred_horizon

        # --- Свёрточный стек (padding сохраняет длину) ---
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )

        # --- FC-голова: контекст -> дельты (H*3) ---
        self.head = nn.Sequential(
            nn.Linear(256, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, pred_horizon * 3),
        )
        # Нулевая инициализация последнего слоя: старт delta = 0
        # (модель стартует как "стоять на месте", затем учится смещению).
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.head[-1].weight)

    @staticmethod
    def _masked_mean_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, L), mask: (B, L) -> (B, C) среднее только по валидным точкам.
        """
        m = mask.unsqueeze(1)                       # (B, 1, L)
        summed = (feat * m).sum(dim=2)              # (B, C)
        denom = m.sum(dim=2).clamp(min=1.0)         # (B, 1)
        return summed / denom

    def forward(self,
                observed: torch.Tensor,
                mask: torch.Tensor,
                last_point: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observed:   (B, L, 15) нормализованные признаки
            mask:       (B, L)
            last_point: (B, 3) последняя наблюдаемая точка (м)

        Returns:
            pred: (B, H, 3) прогноз будущих точек (м)
        """
        x = observed.transpose(1, 2)                # (B, 15, L)
        feat = self.conv(x)                         # (B, 256, L)
        ctx = self._masked_mean_pool(feat, mask)    # (B, 256)

        delta = self.head(ctx)                      # (B, H*3)
        delta = delta.view(-1, self.pred_horizon, 3)  # (B, H, 3)

        return last_point.unsqueeze(1) + delta      # (B, H, 3)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
