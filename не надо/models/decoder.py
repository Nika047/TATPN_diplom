"""
models/decoder.py — Авторегрессивный GRU-декодер траектории.

Идея:
  - Декодер прогнозирует PRED_HORIZON шагов вперёд.
  - На вход каждого шага подаётся:
      • эмбеддинг последней предсказанной точки (или последней наблюдаемой
        точки на первом шаге),
      • target-aware вектор (от Cross-Attention),
      • контекстный вектор траектории.
  - Выход GRU → MLP → (dx, dy, dz) — дельта координат.
  - Накапливаем дельты, чтобы получить абсолютные точки.

При обучении используется teacher-forcing: вместо предсказанной точки на
вход подаётся истинная (с вероятностью teacher_forcing_ratio). При оценке
работает в автореграсивном режиме.
"""

import torch
import torch.nn as nn

import config


class TrajectoryDecoder(nn.Module):
    """
    GRU-декодер: генерирует PRED_HORIZON будущих точек как дельты от предыдущих.

    Args:
        context_dim:    размерность контекстных векторов (= MAMBA_D_MODEL)
        hidden_dim:     размерность скрытого состояния GRU
        num_layers:     количество слоёв GRU
        output_dim:     размерность выхода (3 = dx, dy, dz)
        pred_horizon:   количество шагов прогноза
        dropout:        вероятность dropout
    """

    def __init__(self,
                 context_dim: int = config.MAMBA_D_MODEL,
                 hidden_dim: int = config.DECODER_HIDDEN_DIM,
                 num_layers: int = config.DECODER_NUM_LAYERS,
                 output_dim: int = config.DECODER_OUTPUT_DIM,
                 pred_horizon: int = config.PRED_HORIZON,
                 dropout: float = config.DROPOUT,
                 max_delta_correction: float = config.MAX_DELTA_CORRECTION_PER_STEP,
                 step_dt: float = config.DECODER_STEP_DT):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        # Декодер использует velocity-prior: next = prev + last_velocity * dt + correction,
        # где correction = tanh(raw) * max_delta_correction. Без prior'а декодер
        # учится "выдавать 0" (что при teacher forcing даёт нулевой loss
        # благодаря prev=истина) и в итоге на val/test стоит на месте.
        self.max_delta_correction = max_delta_correction
        self.step_dt = step_dt

        # Входной вектор GRU на каждом шаге =
        #   [последняя точка (3), последняя скорость (3),
        #    context (D), target_aware (D)]
        # Подача скорости на каждом шаге позволяет декодеру не только
        # использовать её как baseline (это делается напрямую сложением),
        # но и учитывать её при выработке коррекции
        # (например, замедляться по мере приближения к цели).
        self.input_dim = output_dim + 3 + 2 * context_dim

        # Инициализация скрытого состояния из контекста
        self.init_hidden = nn.Linear(context_dim, hidden_dim * num_layers)

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Голова: hidden → (dx, dy, dz)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def init_state(self, context: torch.Tensor) -> torch.Tensor:
        """Инициализация скрытого состояния GRU из контекста."""
        B = context.shape[0]
        h0 = self.init_hidden(context)                          # (B, H*L)
        h0 = h0.view(B, self.num_layers, self.hidden_dim)       # (B, L, H)
        h0 = h0.permute(1, 0, 2).contiguous()                   # (L, B, H)
        return torch.tanh(h0)

    def forward(self,
                context: torch.Tensor,
                target_aware: torch.Tensor,
                last_point: torch.Tensor,
                last_velocity: torch.Tensor = None,
                future_truth: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            context:        (B, D)  — контекстный вектор траектории
            target_aware:   (B, D)  — target-aware вектор
            last_point:     (B, 3)  — последняя НАБЛЮДАЕМАЯ точка
                                      (в исходных, не нормализованных координатах)
            last_velocity:  (B, 3)  — последняя НАБЛЮДАЕМАЯ скорость (м/с).
                                      Используется как baseline линейной
                                      экстраполяции. Если None — считаем 0.
            future_truth:   (B, H, 3) — истинные будущие точки (для teacher forcing).
            teacher_forcing_ratio: вероятность teacher forcing.

        Returns:
            pred_future: (B, H, 3) — предсказанные будущие точки (накопленные).
        """
        B = context.shape[0]
        device = context.device

        if last_velocity is None:
            last_velocity = torch.zeros_like(last_point)

        h = self.init_state(context)                            # (L, B, H)
        prev = last_point                                        # (B, 3)
        preds = []

        # baseline-смещение на один шаг (м), считается один раз —
        # для всех 50 шагов прогноза используется одна и та же скорость.
        # Модель учится КОРРЕКЦИЯМ к этой линейной экстраполяции.
        v_dt = last_velocity * self.step_dt                      # (B, 3)

        for t in range(self.pred_horizon):
            # Формируем вход: prev + last_velocity + context + target_aware
            inp = torch.cat([prev, last_velocity, context, target_aware], dim=-1)
            inp = inp.unsqueeze(1)                               # (B, 1, input_dim)

            out, h = self.gru(inp, h)                            # out: (B, 1, H)
            delta_raw = self.head(out.squeeze(1))                # (B, 3)
            # Коррекция к baseline-у: tanh ограничивает физически
            # разумным диапазоном (max_delta_correction м/шаг).
            correction = torch.tanh(delta_raw) * self.max_delta_correction

            # ИТОГ: baseline (линейная экстраполяция) + выученная коррекция
            next_point = prev + v_dt + correction                # (B, 3)
            preds.append(next_point)

            # Teacher forcing
            if (self.training and future_truth is not None
                    and t < future_truth.shape[1]
                    and torch.rand(1, device=device).item() < teacher_forcing_ratio):
                prev = future_truth[:, t]
            else:
                prev = next_point

        pred_future = torch.stack(preds, dim=1)                  # (B, H, 3)
        return pred_future
