"""
autoregressive_decoder.py - Этап 5. Авторегрессионный GRU-декодер траектории.

Прогнозирует будущие точки ПО ОДНОЙ: каждая следующая зависит от предыдущей,
поэтому декодер "продолжает" начатый манёвр (в отличие от FC-головы, которая
выдаёт все точки независимо).

На каждом шаге:
  next = prev + last_velocity * dt + correction,
  correction = tanh(GRU_head(...)) * MAX_DELTA_CORRECTION_PER_STEP.
Velocity-prior (продолжай движение с текущей скоростью) стартует от разумного
прогноза и устраняет вырожденный минимум "выдавать 0". Модель учит только
поправки на манёвры.

Teacher forcing: при обучении с вероятностью tf_ratio на вход следующего шага
подаётся истинная точка, иначе - собственный прогноз. При оценке tf выключен.

Перенесено из архивной реализации (не надо/models/decoder.py).
"""

import torch
import torch.nn as nn

import config


class TrajectoryDecoder(nn.Module):
    """GRU-декодер: PRED_HORIZON будущих точек как коррекции к velocity-prior."""

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
        self.max_delta_correction = max_delta_correction
        self.step_dt = step_dt

        # Вход GRU на шаге: [prev(3), last_velocity(3), context(D), target_aware(D)]
        self.input_dim = output_dim + 3 + 2 * context_dim
        self.init_hidden = nn.Linear(context_dim, hidden_dim * num_layers)
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def init_state(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        h0 = self.init_hidden(context).view(B, self.num_layers, self.hidden_dim)
        return torch.tanh(h0.permute(1, 0, 2).contiguous())     # (L, B, H)

    def forward(self,
                context: torch.Tensor,
                target_aware: torch.Tensor,
                last_point: torch.Tensor,
                last_velocity: torch.Tensor = None,
                future_truth: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        Args:
            context:      (B, D)
            target_aware: (B, D)
            last_point:   (B, 3)  - последняя наблюдаемая точка (м)
            last_velocity:(B, 3)  - последняя скорость (м/с); None -> 0
            future_truth: (B, H, 3) - истинные точки (для teacher forcing)
            teacher_forcing_ratio: вероятность teacher forcing

        Returns:
            pred_future: (B, H, 3)
        """
        device = context.device
        if last_velocity is None:
            last_velocity = torch.zeros_like(last_point)

        h = self.init_state(context)
        prev = last_point
        v_dt = last_velocity * self.step_dt                     # (B, 3)
        preds = []
        for t in range(self.pred_horizon):
            inp = torch.cat([prev, last_velocity, context, target_aware], dim=-1)
            out, h = self.gru(inp.unsqueeze(1), h)              # (B, 1, H)
            correction = torch.tanh(self.head(out.squeeze(1))) * self.max_delta_correction
            next_point = prev + v_dt + correction
            preds.append(next_point)

            if (self.training and future_truth is not None
                    and t < future_truth.shape[1]
                    and torch.rand(1, device=device).item() < teacher_forcing_ratio):
                prev = future_truth[:, t]
            else:
                prev = next_point

        return torch.stack(preds, dim=1)                        # (B, H, 3)
