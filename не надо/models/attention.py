"""
models/attention.py — Cross-Attention Target Head.

Назначение:
  Сравнивает контекстный вектор траектории (Query) с эмбеддингами всех целей
  из базы (Key/Value). Это даёт:
    - logits для классификации цели (через скоринг attention);
    - target-aware вектор (взвешенная сумма value), который вкладывается
      в декодер и помогает прогнозу учитывать цель.

Архитектура:
  Q = Linear(context)            # (B, D)
  K = Linear(target_embeds)      # (T, D)
  V = Linear(target_embeds)      # (T, D)

  attn_logits = Q · K^T / sqrt(D)        # (B, T) — это и есть классификационные logits
  attn_weights = softmax(attn_logits)
  target_aware = attn_weights · V         # (B, D)

  Реализовано как multi-head attention для большей выразительности.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class CrossAttentionTargetHead(nn.Module):
    """
    Multi-head cross-attention: контекст траектории ↔ эмбеддинги целей.

    Args:
        d_model:       размерность контекстного вектора (= MAMBA_D_MODEL)
        target_embed_dim: размерность эмбеддинга цели (= TARGET_EMBED_DIM)
        num_heads:     число голов внимания
        d_attn:        внутренняя размерность Q/K/V (CROSS_ATTN_DIM = 256)
        dropout:       вероятность dropout
    """

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 target_embed_dim: int = config.TARGET_EMBED_DIM,
                 num_heads: int = config.CROSS_ATTN_HEADS,
                 d_attn: int = config.CROSS_ATTN_DIM,
                 dropout: float = config.DROPOUT):
        super().__init__()
        assert d_attn % num_heads == 0, \
            f"d_attn ({d_attn}) должен делиться на num_heads ({num_heads})"

        self.num_heads = num_heads
        self.d_attn = d_attn
        self.d_head = d_attn // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Query из контекста траектории
        self.q_proj = nn.Linear(d_model, d_attn, bias=False)
        # Key/Value из эмбеддингов целей
        self.k_proj = nn.Linear(target_embed_dim, d_attn, bias=False)
        self.v_proj = nn.Linear(target_embed_dim, d_attn, bias=False)

        # Выходная проекция (для target_aware вектора)
        self.out_proj = nn.Linear(d_attn, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Классификационные logits — это среднее по головам attention-скоров.
        # Чтобы получить чистые logits размерности (B, T), мы используем
        # отдельную линейную голову поверх среднего attention-скора.
        # Альтернатива: суммировать скоры по головам. Используем сумму.

    def forward(self, context: torch.Tensor,
                target_embeds: torch.Tensor) -> tuple:
        """
        Args:
            context:       (B, d_model)        — контекстный вектор траектории
            target_embeds: (T, target_embed_dim) — эмбеддинги всех целей

        Returns:
            target_aware: (B, d_model) — target-aware вектор (для декодера)
            target_logits: (B, T)      — классификационные logits
        """
        B = context.shape[0]
        T = target_embeds.shape[0]

        # Q: (B, d_attn) → (B, H, 1, d_head)
        Q = self.q_proj(context).view(B, self.num_heads, 1, self.d_head)

        # K, V: (T, d_attn) → (1, H, T, d_head)
        K = self.k_proj(target_embeds).view(T, self.num_heads, self.d_head)
        K = K.permute(1, 0, 2).unsqueeze(0)  # (1, H, T, d_head)

        V = self.v_proj(target_embeds).view(T, self.num_heads, self.d_head)
        V = V.permute(1, 0, 2).unsqueeze(0)  # (1, H, T, d_head)

        # Attention scores: (B, H, 1, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Softmax → веса
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Target-aware: (B, H, 1, d_head)
        target_aware = torch.matmul(attn_weights, V)
        # → (B, d_attn) → (B, d_model)
        target_aware = target_aware.squeeze(2).reshape(B, self.d_attn)
        target_aware = self.out_proj(target_aware)

        # Классификационные logits = сумма скоров по головам
        target_logits = scores.squeeze(2).sum(dim=1)  # (B, T)

        return target_aware, target_logits


class TargetEmbedding(nn.Module):
    """
    Эмбеддинг базы целей: для каждой цели формируем вектор из:
      - type-embedding (категориальный) → TARGET_TYPE_EMBED_DIM
      - MLP по нормализованным координатам (x, y, z) → TARGET_COORD_EMBED_DIM
    Конкатенация → TARGET_EMBED_DIM.

    Args:
        num_targets:      количество целей (= NUM_TARGETS)
        num_types:        количество типов (= NUM_TARGET_TYPES)
        type_embed_dim:   размерность type-эмбеддинга
        coord_embed_dim:  размерность coord-эмбеддинга
    """

    def __init__(self,
                 num_targets: int = config.NUM_TARGETS,
                 num_types: int = config.NUM_TARGET_TYPES,
                 type_embed_dim: int = config.TARGET_TYPE_EMBED_DIM,
                 coord_embed_dim: int = config.TARGET_COORD_EMBED_DIM):
        super().__init__()
        self.num_targets = num_targets

        self.type_embed = nn.Embedding(num_types, type_embed_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, coord_embed_dim),
        )

    def forward(self, target_types: torch.Tensor,
                target_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_types:  (T,)   — long, индексы типов
            target_coords: (T, 3) — float, нормализованные координаты

        Returns:
            (T, TARGET_EMBED_DIM)
        """
        type_e = self.type_embed(target_types)        # (T, type_embed_dim)
        coord_e = self.coord_mlp(target_coords)        # (T, coord_embed_dim)
        return torch.cat([type_e, coord_e], dim=-1)    # (T, TARGET_EMBED_DIM)
