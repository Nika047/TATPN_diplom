"""
models/tatpn.py — Главный класс TATPN
(Target-Aware Trajectory Prediction Network).

Архитектура (порядок обработки):

  observed (B, L, F=15)
      │
      ▼
  Multi-Scale CNN  (Conv1d k=3,7,15 → BN → GELU → проекция в d_model)
      │
      ▼  (B, L, d_model)
  Mamba Encoder   (4 слоя selective SSM)
      │
      ▼  (B, L, d_model)
  Adaptive Context Pooling  (attention по последним K шагам)
      │
      ▼  (B, d_model)  — контекстный вектор траектории
      │
      ├──► Cross-Attention Target Head
      │       (Query=context, Key/Value=target_embeds)
      │       │
      │       ├──► target_logits     (B, T)        ← классификация цели
      │       └──► target_aware      (B, d_model)  ← target-aware вектор
      │
      ├──► Trajectory Decoder (GRU)
      │       input: context + target_aware + last_point
      │       output: pred_future    (B, H, 3)     ← прогноз дельт
      │
      └──► ETA Head (MLP)
              input: [context, target_aware]
              output: eta            (B,)          ← время до цели

Multi-task loss комбинирует все три выхода.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np

import config
from .cnn import MultiScaleCNN
from .mamba import MambaEncoder
from .pooling import AdaptiveContextPooling
from .attention import CrossAttentionTargetHead, TargetEmbedding
from .decoder import TrajectoryDecoder
from .heads import ETAHead


def _load_targets(targets_json: str = config.TARGETS_JSON) -> tuple:
    """
    Загружает базу целей из JSON и возвращает:
        target_types:  (T,)   long
        target_coords: (T, 3) float (нормализованные координаты,
                       делим на масштаб, чтобы MLP получал значения ~O(1))

    JSON-формат ожидается как массив словарей с полями:
        {"id": "...", "name": "...", "type": "<категория>", "x": ..., "y": ..., "z": ...}
    """
    if not os.path.exists(targets_json):
        # Заглушка — если файла нет, создаём случайные данные нужного размера
        T = config.NUM_TARGETS
        return (torch.zeros(T, dtype=torch.long),
                torch.zeros(T, 3, dtype=torch.float32))

    with open(targets_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON может быть либо списком, либо словарём {"targets": [...]}
    if isinstance(data, dict) and "targets" in data:
        data = data["targets"]

    # Маппинг типов целей → индексы
    type_names = sorted(set(t.get("type", "unknown") for t in data))
    type_to_idx = {name: i for i, name in enumerate(type_names)}

    types = []
    coords = []
    for t in data:
        types.append(type_to_idx.get(t.get("type", "unknown"), 0))
        coords.append([t.get("x", 0.0), t.get("y", 0.0), t.get("z", 0.0)])

    types_t = torch.tensor(types, dtype=torch.long)
    coords_t = torch.tensor(coords, dtype=torch.float32)
    # Нормализация координат на масштаб 100 км → значения примерно [-5, 5]
    coords_t = coords_t / 1e5

    return types_t, coords_t


class TATPN(nn.Module):
    """
    Target-Aware Trajectory Prediction Network.

    Args:
        input_features: число входных кинематических признаков
        d_model:        размерность представления Mamba
        num_targets:    количество целей в базе
        pred_horizon:   горизонт прогноза (шагов)
        targets_json:   путь к JSON с базой целей (для type/coord)
    """

    def __init__(self,
                 input_features: int = config.INPUT_FEATURES,
                 d_model: int = config.MAMBA_D_MODEL,
                 num_targets: int = config.NUM_TARGETS,
                 pred_horizon: int = config.PRED_HORIZON,
                 targets_json: str = config.TARGETS_JSON):
        super().__init__()
        self.d_model = d_model
        self.num_targets = num_targets
        self.pred_horizon = pred_horizon

        # ---- 1. Multi-Scale CNN ----
        self.cnn = MultiScaleCNN(
            in_features=input_features,
            out_dim=d_model,
        )

        # ---- 2. Mamba Encoder ----
        self.encoder = MambaEncoder(d_model=d_model)

        # ---- 3. Adaptive Context Pooling ----
        self.pool = AdaptiveContextPooling(d_model=d_model)

        # ---- 4. Target Embedding + Cross-Attention ----
        self.target_embed = TargetEmbedding(
            num_targets=num_targets,
            num_types=config.NUM_TARGET_TYPES,
        )
        self.target_head = CrossAttentionTargetHead(
            d_model=d_model,
            target_embed_dim=config.TARGET_EMBED_DIM,
        )

        # Регистрация типов и координат целей как буферов
        target_types, target_coords = _load_targets(targets_json)
        # Приводим к ожидаемому размеру num_targets (паддим/обрезаем)
        if target_types.shape[0] < num_targets:
            pad = num_targets - target_types.shape[0]
            target_types = torch.cat(
                [target_types, torch.zeros(pad, dtype=torch.long)], dim=0
            )
            target_coords = torch.cat(
                [target_coords, torch.zeros(pad, 3, dtype=torch.float32)], dim=0
            )
        elif target_types.shape[0] > num_targets:
            target_types = target_types[:num_targets]
            target_coords = target_coords[:num_targets]

        self.register_buffer("target_types_buf", target_types)
        self.register_buffer("target_coords_buf", target_coords)

        # ---- 5. Trajectory Decoder ----
        self.decoder = TrajectoryDecoder(
            context_dim=d_model,
            pred_horizon=pred_horizon,
        )

        # ---- 6. ETA Head ----
        self.eta_head = ETAHead(in_dim=2 * d_model)

    def get_target_embeddings(self) -> torch.Tensor:
        """Возвращает эмбеддинги всех целей (T, TARGET_EMBED_DIM)."""
        return self.target_embed(self.target_types_buf, self.target_coords_buf)

    def forward(self,
                observed: torch.Tensor,
                mask: torch.Tensor,
                last_point: torch.Tensor,
                last_velocity: torch.Tensor = None,
                future_points: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> dict:
        """
        Args:
            observed:      (B, L, F)        — нормализованные кинематические признаки
            mask:          (B, L)           — 1 для реальных точек, 0 для padding
            last_point:    (B, 3)           — последняя НАБЛЮДАЕМАЯ точка
                                              в исходных координатах (для декодера)
            future_points: (B, H, 3)        — истинные будущие точки (для teacher forcing).
                                              Может быть None при инференсе.
            teacher_forcing_ratio: вероятность teacher forcing в декодере.

        Returns:
            dict с полями:
                "pred_future":   (B, H, 3) — прогноз будущих точек
                "target_logits": (B, T)    — logits классификации цели
                "eta":           (B,)      — оценка времени до цели
                "context":       (B, D)    — контекстный вектор (для диагностики)
        """
        # 1. CNN: (B, L, F) → (B, L, D)
        h = self.cnn(observed, mask=mask)

        # 2. Mamba: (B, L, D) → (B, L, D)
        h = self.encoder(h, mask=mask)

        # 3. Adaptive Pooling: (B, L, D) → (B, D)
        context = self.pool(h, mask)

        # 4. Cross-Attention с целями
        target_embeds = self.get_target_embeddings()        # (T, TARGET_EMBED_DIM)
        target_aware, target_logits = self.target_head(context, target_embeds)

        # 5. Декодер
        pred_future = self.decoder(
            context=context,
            target_aware=target_aware,
            last_point=last_point,
            last_velocity=last_velocity,
            future_truth=future_points,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # 6. ETA
        eta_input = torch.cat([context, target_aware], dim=-1)  # (B, 2*D)
        eta = self.eta_head(eta_input)

        return {
            "pred_future": pred_future,
            "target_logits": target_logits,
            "eta": eta,
            "context": context,
        }

    def num_parameters(self) -> int:
        """Подсчёт обучаемых параметров."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
