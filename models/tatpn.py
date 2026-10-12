"""
tatpn.py - Этап 5. Полная модель TATPN (Target-Aware Trajectory Prediction Network).

Собирает всё вместе:
  observed -> Multi-Scale CNN -> Mamba энкодер -> Adaptive Pooling -> контекст
    ├─► Cross-Attention с эмбеддингами целей -> attention-логиты, target_aware
    │       + геометрия (расстояние/углы) -> комбинированные логиты цели
    ├─► авторегрессионный GRU-декодер (context, target_aware) -> прогноз точек
    └─► ETA-голова (context, target_aware) -> время до цели (сек)

Отличия от Этапа 4:
  - target head: обучаемое внимание к эмбеддингам целей ПЛЮС явная геометрия
    (внимание учит представление целей, геометрия даёт сильную прямую подсказку);
  - траектория: авторегрессионный декодер вместо FC-головы;
  - добавлена ETA (третья задача).
"""

import numpy as np
import torch
import torch.nn as nn

import config
from models.multiscale_cnn import MultiScaleCNN
from models.mamba import MambaEncoder
from models.pooling import AdaptiveContextPooling
from models.cross_attention import TargetEmbedding, CrossAttentionTargetHead
from models.autoregressive_decoder import TrajectoryDecoder
from models.eta_head import ETAHead
from features.geometry import compute_geometric_features, GEO_DIM

# Масштаб нормировки координат целей для эмбеддинга (м)
TARGET_COORD_NORM = 1.0e5


class TATPN(nn.Module):
    """Полная целевая архитектура: encoder + cross-attention + геометрия + декодер + ETA."""

    def __init__(self,
                 target_coords: np.ndarray,
                 target_types: np.ndarray,
                 d_model: int = config.MAMBA_D_MODEL,
                 dropout: float = config.DROPOUT):
        super().__init__()

        self.cnn = MultiScaleCNN(out_dim=d_model, dropout=dropout)
        self.encoder = MambaEncoder(d_model=d_model, dropout=dropout)
        self.pool = AdaptiveContextPooling(d_model=d_model, dropout=dropout)

        self.target_embed = TargetEmbedding()
        self.cross_attn = CrossAttentionTargetHead(d_model=d_model)
        # Комбинатор: [полный контекст, attention-логит, геометрия] -> логит цели.
        # Полный контекст (а не только attention-скаляр) даёт голове столько же
        # информации о траектории, сколько было на Этапе 4, плюс attention.
        combiner_in = d_model + 1 + GEO_DIM
        self.target_combiner = nn.Sequential(
            nn.Linear(combiner_in, config.TARGET_HEAD_HIDDEN), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(config.TARGET_HEAD_HIDDEN, config.TARGET_HEAD_HIDDEN // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.TARGET_HEAD_HIDDEN // 2, 1),
        )

        self.decoder = TrajectoryDecoder(context_dim=d_model)
        self.eta_head = ETAHead(in_dim=2 * d_model)

        coords_m = np.asarray(target_coords, dtype=np.float32)               # (K, 2) метры
        self.register_buffer("target_coords_meters", torch.tensor(coords_m))
        self.register_buffer("target_coords_norm",
                             torch.tensor(coords_m / TARGET_COORD_NORM))
        self.register_buffer("target_types",
                             torch.tensor(np.asarray(target_types, dtype=np.int64)))

    def forward(self,
                observed: torch.Tensor,
                mask: torch.Tensor,
                last_point: torch.Tensor,
                last_velocity: torch.Tensor,
                future_points: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.0) -> dict:
        x = self.cnn(observed, mask)
        x = self.encoder(x, mask)
        ctx = self.pool(x, mask)                                  # (B, D)

        # Cross-Attention: обучаемые эмбеддинги целей
        tembeds = self.target_embed(self.target_types, self.target_coords_norm)  # (K, E)
        target_aware, attn_logits = self.cross_attn(ctx, tembeds)               # (B,D), (B,K)

        # Геометрия к целям (сильная прямая подсказка)
        geo = compute_geometric_features(last_point, last_velocity,
                                         self.target_coords_meters)             # (B, K, G)

        # Финальные логиты = комбинация полного контекста, внимания и геометрии
        K = attn_logits.shape[1]
        ctx_b = ctx.unsqueeze(1).expand(-1, K, -1)                            # (B, K, D)
        combo = torch.cat([ctx_b, attn_logits.unsqueeze(-1), geo], dim=-1)     # (B, K, D+1+G)
        target_logits = self.target_combiner(combo).squeeze(-1)                # (B, K)

        pred_future = self.decoder(ctx, target_aware, last_point, last_velocity,
                                   future_truth=future_points,
                                   teacher_forcing_ratio=teacher_forcing_ratio)

        eta = self.eta_head(torch.cat([ctx, target_aware], dim=-1))            # (B,)

        return {"pred_future": pred_future, "target_logits": target_logits, "eta": eta}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
