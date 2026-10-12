"""
multiscale_cnn.py - Multi-Scale CNN фронт-энд для Этапа 3.

Параллельные Conv1d с ядрами разных масштабов (3/7/15) без MaxPool -
длина последовательности сохраняется, чтобы Mamba получил все шаги.
Малое ядро - мгновенные манёвры, среднее - короткие манёвры, крупное -
общий курс. Выходы веток конкатенируются и проецируются 1x1-свёрткой в
d_model. Маска зануляет padding до и после свёрток.

Перенесено из архивной реализации (не надо/models/cnn.py).
"""

import torch
import torch.nn as nn

import config


class MultiScaleCNN(nn.Module):
    """Параллельные Conv1d ветки разных масштабов + проекция в d_model."""

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

        self.branches = nn.ModuleList()
        for k in self.kernel_sizes:
            assert k % 2 == 1, f"Ядро {k} должно быть нечётным для 'same' padding"
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_features, channels, kernel_size=k, padding=k // 2),
                nn.GELU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout),
            ))

        concat_dim = channels * len(self.kernel_sizes)
        self.project = nn.Sequential(
            nn.Conv1d(concat_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """x: (B, L, F) -> (B, L, out_dim), длина L сохранена."""
        x = x.transpose(1, 2)                    # (B, F, L)
        if mask is not None:
            x = x * mask.unsqueeze(1)
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)             # (B, C*K, L)
        out = self.project(out)                  # (B, d_model, L)
        out = out.transpose(1, 2)                # (B, L, d_model)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out
