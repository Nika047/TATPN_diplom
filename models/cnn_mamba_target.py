"""
cnn_mamba_target.py - Этап 4. CNN + Mamba + Target Head (multi-task).

К модели Этапа 3 (CNN + Mamba + adaptive pooling -> прогноз траектории)
добавляется вторая голова - классификация цели. Общий энкодер даёт контекст,
из которого работают обе головы:
  - траекторная голова: context -> дельты -> прогноз точек (как на Этапе 3);
  - target-голова: [context, геометрия к целям] -> логиты целей.

Геометрия к целям (расстояние, углы) считается внутри forward из последней
наблюдаемой точки и скорости - никаких данных телеметрии дрона не требуется.
"""

import numpy as np
import torch
import torch.nn as nn

import config
from models.multiscale_cnn import MultiScaleCNN
from models.mamba import MambaEncoder
from models.pooling import AdaptiveContextPooling
from models.target_head_mlp import TargetHeadMLP
from features.geometry import compute_geometric_features


class ConvMambaTarget(nn.Module):
    """CNN + Mamba энкодер + две головы: траектория и классификация цели."""

    def __init__(self,
                 target_coords: np.ndarray,
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

        # Траекторная голова (как на Этапе 3), дельты от последней точки
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, pred_horizon * 3),
        )
        nn.init.zeros_(self.traj_head[-1].bias)
        nn.init.zeros_(self.traj_head[-1].weight)

        # Target-голова
        self.target_head = TargetHeadMLP(context_dim=d_model, dropout=dropout)

        # Координаты целей (E, N) - фиксированный буфер (K, 2)
        self.register_buffer(
            "target_coords",
            torch.tensor(np.asarray(target_coords, dtype=np.float32)))

    def forward(self,
                observed: torch.Tensor,
                mask: torch.Tensor,
                last_point: torch.Tensor,
                last_velocity: torch.Tensor) -> dict:
        """
        Returns:
            dict:
              pred_future:   (B, H, 3) прогноз точек (м)
              target_logits: (B, K) логиты целей
        """
        x = self.cnn(observed, mask)
        x = self.encoder(x, mask)
        ctx = self.pool(x, mask)                     # (B, d_model)

        delta = self.traj_head(ctx).view(-1, self.pred_horizon, 3)
        pred_future = last_point.unsqueeze(1) + delta

        geo = compute_geometric_features(last_point, last_velocity,
                                         self.target_coords)     # (B, K, G)
        target_logits = self.target_head(ctx, geo)               # (B, K)

        return {"pred_future": pred_future, "target_logits": target_logits}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
