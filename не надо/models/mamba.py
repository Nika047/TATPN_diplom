"""
models/mamba.py — Кастомная реализация Mamba (Selective State Space Model)
на чистом PyTorch. Не требует CUDA-расширений, работает на CPU и GPU.

Источник: A. Gu, T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces", 2023 (arXiv:2312.00752).

Структура Mamba-блока (упрощённая, но математически корректная):

    x ──► Linear (d_model → 2 * d_inner) ──► split → (x_inner, gate)
    x_inner ──► Conv1d (causal, ширина d_conv) ──► SiLU ──► Selective SSM ──► out
    out * SiLU(gate) ──► Linear (d_inner → d_model) ──► +residual

Selective SSM:
    Δ, B, C — это функции от входа x (а не фиксированные параметры).
    A — обучаемая матрица (логарифмическая параметризация).
    Дискретизация:
        A_bar = exp(Δ · A)
        B_bar = Δ · B
    Рекуррентность:
        h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        y_t = C_t * h_t + D * x_t

Так как Δ, B, C зависят от входа, рекуррентность не сводится к одной свёртке
и применяется явным циклом по длине последовательности.
Для типичных L ≤ 512 (наш MAX_SEQ_LEN) — это допустимо.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model — ядро Mamba.

    Args:
        d_inner: внутренняя размерность (= d_model * expand)
        d_state: размерность скрытого состояния (N в статье)
        dt_rank: ранг проекции Δ (по умолчанию ceil(d_inner/16))
    """

    def __init__(self,
                 d_inner: int,
                 d_state: int = config.MAMBA_D_STATE,
                 dt_rank: int = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_inner / 16)

        # x → (Δ_raw, B, C)
        # Δ_raw: d_inner (после проекции через dt_rank)
        # B: d_state
        # C: d_state
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Δ_raw (dt_rank) → Δ (d_inner), с инициализацией bias так, чтобы
        # Δ через softplus имел небольшое положительное значение
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        # Инициализация согласно статье (dt_min, dt_max)
        dt_min, dt_max = 1e-3, 1e-1
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # bias инициализируется так, чтобы softplus(bias) ~ U[dt_min, dt_max]
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        # обратное преобразование softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A в логарифмической параметризации: A = -exp(A_log)
        # Изначально A_log = log(1..d_state), т.е. A = -1..-d_state (стабильно)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # D — skip-connection (обучаемая)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_inner)

        Returns:
            y: (B, L, d_inner)
        """
        B_size, L, d_inner = x.shape

        # A: (d_inner, d_state), отрицательная и стабильная
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Получаем Δ_raw, B, C из x
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt_raw, B_in, C_in = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ: (B, L, d_inner), positive через softplus
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        # Дискретизация:
        #   A_bar = exp(Δ * A)                        (B, L, d_inner, d_state)
        #   B_bar = Δ * B                             (B, L, d_inner, d_state)
        # Размерности:
        #   dt:   (B, L, d_inner, 1)
        #   A:    (1, 1, d_inner, d_state)
        #   B_in: (B, L, 1, d_state)
        dt_ = dt.unsqueeze(-1)                  # (B, L, d_inner, 1)
        A_  = A.unsqueeze(0).unsqueeze(0)       # (1, 1, d_inner, d_state)
        B_  = B_in.unsqueeze(2)                  # (B, L, 1, d_state)

        A_bar = torch.exp(dt_ * A_)              # (B, L, d_inner, d_state)
        B_bar = dt_ * B_                          # (B, L, d_inner, d_state)

        # Рекуррентное вычисление состояния
        # h: (B, d_inner, d_state)
        h = torch.zeros(B_size, d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        # x: (B, L, d_inner)
        # x_t * B_bar_t : (B, d_inner, 1) * (B, d_inner, d_state) = (B, d_inner, d_state)
        for t in range(L):
            A_t = A_bar[:, t]                    # (B, d_inner, d_state)
            B_t = B_bar[:, t]                    # (B, d_inner, d_state)
            x_t = x[:, t].unsqueeze(-1)          # (B, d_inner, 1)
            h = A_t * h + B_t * x_t              # (B, d_inner, d_state)

            C_t = C_in[:, t].unsqueeze(1)         # (B, 1, d_state)
            # y_t = sum_n C_t[n] * h[..., n]
            y_t = (C_t * h).sum(dim=-1)           # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                # (B, L, d_inner)

        # Skip connection через D
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y


class MambaBlock(nn.Module):
    """
    Полный Mamba-блок: linear → 1D-conv (causal) → SSM → gate → linear → residual.

    Args:
        d_model: размерность входа/выхода
        d_state: размерность скрытого состояния SSM
        d_conv:  ширина каузальной свёртки
        expand:  множитель внутренней размерности
        dropout: вероятность dropout
    """

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 d_state: int = config.MAMBA_D_STATE,
                 d_conv: int = config.MAMBA_D_CONV,
                 expand: int = config.MAMBA_EXPAND,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # in_proj: d_model → 2 * d_inner (для x_inner и gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal Conv1d по последовательности (groups = d_inner — depth-wise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)

        # out_proj: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Нормализация перед блоком (pre-norm)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # in_proj и split на x_inner / gate
        xz = self.in_proj(x)                 # (B, L, 2*d_inner)
        x_in, gate = xz.chunk(2, dim=-1)     # каждый (B, L, d_inner)

        # Causal conv: (B, L, d_inner) → (B, d_inner, L) → conv → срез до L
        x_conv = x_in.transpose(1, 2)        # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :x_in.shape[1]]  # причинный срез
        x_conv = x_conv.transpose(1, 2)      # (B, L, d_inner)

        # SiLU + Selective SSM
        x_act = F.silu(x_conv)
        y = self.ssm(x_act)                  # (B, L, d_inner)

        # Gating
        y = y * F.silu(gate)

        # out_proj + residual
        out = self.out_proj(y)               # (B, L, d_model)
        out = self.dropout(out)
        return residual + out


class MambaEncoder(nn.Module):
    """
    Стек из нескольких Mamba-блоков + финальный LayerNorm.

    Args:
        d_model:    размерность представления
        num_layers: количество Mamba-блоков (MAMBA_NUM_LAYERS = 4)
        d_state:    размерность скрытого состояния SSM
        d_conv:     ширина каузальной свёртки
        expand:     множитель внутренней размерности
        dropout:    вероятность dropout
    """

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 num_layers: int = config.MAMBA_NUM_LAYERS,
                 d_state: int = config.MAMBA_D_STATE,
                 d_conv: int = config.MAMBA_D_CONV,
                 expand: int = config.MAMBA_EXPAND,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state,
                       d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (B, L, d_model)
            mask: (B, L) — 1 для реальных точек, 0 для padding

        Returns:
            (B, L, d_model)
        """
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask.unsqueeze(-1)
        x = self.norm_f(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
