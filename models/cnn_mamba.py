"""
cnn_mamba.py - Этап 3. CNN + Mamba энкодер (только координаты).

Добавляем к свёрточному фронту временное моделирование (Mamba), сохраняя
формулировку задачи Этапа 2 (прогноз дельт от последней точки, без физического
prior'а). Это изолирует вклад Mamba: всё остальное - как в CNN-бейзлайне.

Схема:
  observed (B, L, 15)
    -> Multi-Scale CNN (ядра 3/7/15, без MaxPool) -> (B, L, d_model)
    -> Mamba Encoder (селективный SSM) -> (B, L, d_model)
    -> Adaptive Context Pooling (attention по последним K) -> (B, d_model)
    -> FC-голова -> (B, H, 3) дельты
  pred_k = last_point + delta_k

Последний слой инициализируется нулём: старт delta=0 (= "стоять на месте"),
затем модель учится смещению - как на Этапе 2, для честного сравнения.
"""

import torch
import torch.nn as nn

import config
from models.multiscale_cnn import MultiScaleCNN
from models.mamba import MambaEncoder
from models.pooling import AdaptiveContextPooling


class ConvMambaTrajectory(nn.Module):
    """Multi-Scale CNN + Mamba энкодер + adaptive pooling + FC-голова дельт."""

    def __init__(self,
                 in_features: int = config.INPUT_FEATURES,
                 d_model: int = config.MAMBA_D_MODEL,
                 pred_horizon: int = config.PRED_HORIZON,
                 hidden: int = 256,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.pred_horizon = pred_horizon

        self.cnn = MultiScaleCNN(in_features=in_features, out_dim=d_model,
                                 dropout=dropout)
        self.encoder = MambaEncoder(d_model=d_model, dropout=dropout)
        self.pool = AdaptiveContextPooling(d_model=d_model, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, pred_horizon * 3),
        )
        # Нулевая инициализация последнего слоя: старт delta = 0.
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.head[-1].weight)

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
        x = self.cnn(observed, mask)            # (B, L, d_model)
        x = self.encoder(x, mask)               # (B, L, d_model)
        ctx = self.pool(x, mask)                # (B, d_model)
        delta = self.head(ctx).view(-1, self.pred_horizon, 3)   # (B, H, 3)
        return last_point.unsqueeze(1) + delta

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
