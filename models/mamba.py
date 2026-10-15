"""
mamba.py - Mamba (Selective State Space Model) на чистом PyTorch.

Перенесено из архивной реализации (не надо/models/mamba.py) с небольшой
оптимизацией scan-цикла под CPU: умножение на C вынесено из цикла
(векторизовано по L), в цикле остаётся только рекуррентное обновление
состояния h. Математика не изменена.

Источник: A. Gu, T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces", 2023 (arXiv:2312.00752).

Selective SSM:
    Δ, B, C зависят от входа x; A - обучаемая (логарифмическая параметризация).
    Дискретизация: A_bar = exp(Δ·A), B_bar = Δ·B.
    Рекуррентность: h_t = A_bar_t · h_{t-1} + B_bar_t · x_t;  y_t = C_t · h_t + D·x_t.
Так как Δ, B, C зависят от входа, рекуррентность применяется явным циклом по L.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SelectiveSSM(nn.Module):
    """Ядро Mamba - selective state space model."""

    def __init__(self,
                 d_inner: int,
                 d_state: int = config.MAMBA_D_STATE,
                 dt_rank: int = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_inner / 16)

        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        dt_min, dt_max = 1e-3, 1e-1
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))     # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner) -> y: (B, L, d_inner)."""
        B_size, L, d_inner = x.shape

        A = -torch.exp(self.A_log)                  # (d_inner, d_state)

        x_dbl = self.x_proj(x)                       # (B, L, dt_rank + 2*d_state)
        dt_raw, B_in, C_in = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt_raw))        # (B, L, d_inner)

        # Дискретизация: A_bar = exp(Δ·A), B_bar·x = Δ·B·x.
        # Тензоры полного размера (B, L, d_inner, d_state) строим один раз.
        A_bar = torch.exp(dt.unsqueeze(-1) * A)       # (B, L, d_inner, d_state)
        Bx = (dt * x).unsqueeze(-1) * B_in.unsqueeze(2)   # (B, L, d_inner, d_state)

        # Разрезаем по времени ОДИН раз до цикла. Срез A_bar[:, t] внутри цикла
        # при обратном проходе создаёт на каждом шаге тензор полного размера
        # (select_backward) и суммирует их: работа растёт как L^2 и на
        # длине 192 даёт ~5 с на шаг вместо долей секунды. unbind при обратном
        # проходе делает одну сборку (stack) и стоит линейно.
        A_steps = A_bar.unbind(dim=1)                 # L тензоров (B, d_inner, d_state)
        Bx_steps = Bx.unbind(dim=1)
        C_steps = C_in.unbind(dim=1)                  # L тензоров (B, d_state)

        # Рекуррентность по L (единственный последовательный шаг).
        # y_t = sum_n C_t[n] · h_t[..., n] считаем сразу в цикле: так не
        # нужно хранить стек состояний (B, L, d_inner, d_state).
        h = torch.zeros(B_size, d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = A_steps[t] * h + Bx_steps[t]          # (B, d_inner, d_state)
            ys.append((h * C_steps[t].unsqueeze(1)).sum(dim=-1))   # (B, d_inner)
        y = torch.stack(ys, dim=1)                    # (B, L, d_inner)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)  # skip через D
        return y


class MambaBlock(nn.Module):
    """linear -> causal conv1d -> SiLU -> SSM -> gate -> linear -> residual."""

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 d_state: int = config.MAMBA_D_STATE,
                 d_conv: int = config.MAMBA_D_CONV,
                 expand: int = config.MAMBA_EXPAND,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_in, gate = xz.chunk(2, dim=-1)

        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_in.shape[1]]   # причинный срез
        x_conv = x_conv.transpose(1, 2)

        y = self.ssm(F.silu(x_conv))
        y = y * F.silu(gate)
        out = self.dropout(self.out_proj(y))
        return residual + out


class MambaEncoder(nn.Module):
    """Стек Mamba-блоков + финальный LayerNorm. Маска зануляет padding."""

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 num_layers: int = config.MAMBA_NUM_LAYERS,
                 d_state: int = config.MAMBA_D_STATE,
                 d_conv: int = config.MAMBA_D_CONV,
                 expand: int = config.MAMBA_EXPAND,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask.unsqueeze(-1)
        x = self.norm_f(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
