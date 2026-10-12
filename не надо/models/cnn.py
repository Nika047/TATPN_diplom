"""
models/cnn.py — Multi-Scale CNN front-end для TATPN.

Идея:
  - На вход подаётся последовательность кинематических признаков
    формы (B, L, F=INPUT_FEATURES).
  - Несколько параллельных Conv1d с разными размерами ядер
    (CNN_KERNEL_SIZES = [3, 7, 15]) извлекают признаки разных масштабов:
       3  — мгновенные изменения (резкий манёвр)
       7  — короткие манёвры (поворот, набор высоты)
       15 — крупные паттерны движения (этап маршрута)
  - MaxPool не используется — длина последовательности сохраняется,
    чтобы Mamba-блоки получили все шаги без потери информации.
  - Выходы веток конкатенируются по каналам:
       (B, 3 * CNN_CHANNELS, L) → проекция в d_model для Mamba.

Все свёртки делаются с padding='same', чтобы длина L не менялась.
"""

import torch
import torch.nn as nn

import config


class MultiScaleCNN(nn.Module):
    """
    Параллельные Conv1d ветки разных масштабов + проекция в d_model.

    Args:
        in_features:  число входных каналов (по умолчанию INPUT_FEATURES = 15)
        channels:     число выходных каналов каждой ветки (CNN_CHANNELS = 64)
        kernel_sizes: список размеров ядер (CNN_KERNEL_SIZES = [3, 7, 15])
        out_dim:      целевая размерность (MAMBA_D_MODEL = 256)
        dropout:      вероятность dropout
    """

    def __init__(self,
                 in_features: int = config.INPUT_FEATURES,
                 channels: int = config.CNN_CHANNELS,
                 kernel_sizes=None,
                 out_dim: int = config.MAMBA_D_MODEL,
                 dropout: float = config.DROPOUT):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = config.CNN_KERNEL_SIZES

        self.kernel_sizes = list(kernel_sizes)
        self.channels = channels

        # Параллельные ветки. padding = k // 2 → 'same' (для нечётных ядер).
        self.branches = nn.ModuleList()
        for k in self.kernel_sizes:
            assert k % 2 == 1, f"Ядро {k} должно быть нечётным для 'same' padding"
            branch = nn.Sequential(
                nn.Conv1d(in_features, channels, kernel_size=k, padding=k // 2),
                nn.GELU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout),
            )
            self.branches.append(branch)

        # После конкатенации: channels * len(kernel_sizes)
        concat_dim = channels * len(self.kernel_sizes)

        # Проекция в d_model для Mamba (1x1-свёртка == линейный слой по каналам)
        self.project = nn.Sequential(
            nn.Conv1d(concat_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (B, L, F) — батч кинематических признаков
            mask: (B, L)   — 1 для реальных точек, 0 для padding (необязательно)

        Returns:
            (B, L, out_dim) — признаки после Multi-Scale CNN, длина L сохранена.
        """
        # Conv1d ожидает (B, C, L)
        x = x.transpose(1, 2)  # (B, F, L)

        # Применяем маску ко входу, чтобы padding не попал в свёртки
        # (умножаем на маску по канальной оси)
        if mask is not None:
            x = x * mask.unsqueeze(1)  # (B, F, L) * (B, 1, L)

        # Параллельные ветки
        outs = [branch(x) for branch in self.branches]  # каждый: (B, C, L)
        out = torch.cat(outs, dim=1)  # (B, C*K, L)

        # Проекция в d_model
        out = self.project(out)  # (B, d_model, L)

        # Возвращаем в формате (B, L, d_model)
        out = out.transpose(1, 2)

        # Снова обнуляем padding-позиции после свёрток
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out

    def extra_repr(self) -> str:
        return (f"kernels={self.kernel_sizes}, channels={self.channels}, "
                f"concat={self.channels * len(self.kernel_sizes)}")
