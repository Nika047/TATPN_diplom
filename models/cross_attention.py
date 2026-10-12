"""
cross_attention.py - Этап 5. Cross-Attention Target Head + эмбеддинги целей.

Заменяет наивную геометрию Этапа 4 обучаемым механизмом внимания:
  - каждая цель кодируется эмбеддингом = TypeEmbedding(тип) + MLP(координаты);
  - контекст траектории (Query) сопоставляется с эмбеддингами целей (Key/Value);
  - attention-скоры дают логиты классификации цели, а взвешенная сумма Value -
    target-aware вектор, который идёт в декодер и ETA-голову.

Преимущество перед Этапом 4: модель сама учит представление целей и их типов
(обобщение "похожие цели -> похожий отклик"), а не опирается только на
вручную заданные расстояние/угол.

Перенесено из архивной реализации (не надо/models/attention.py), координаты
целей адаптированы под 2D (E, N).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class TargetEmbedding(nn.Module):
    """Эмбеддинг целей: TypeEmbedding(тип) + MLP(координаты E,N)."""

    def __init__(self,
                 num_types: int = config.NUM_TARGET_TYPES,
                 type_embed_dim: int = config.TARGET_TYPE_EMBED_DIM,
                 coord_embed_dim: int = config.TARGET_COORD_EMBED_DIM):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, type_embed_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),
            nn.Linear(64, coord_embed_dim),
        )

    def forward(self, target_types: torch.Tensor,
                target_coords_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_types:       (K,)   long - индексы типов
            target_coords_norm: (K, 2) float - нормированные координаты (E, N)

        Returns:
            (K, TARGET_EMBED_DIM)
        """
        type_e = self.type_embed(target_types)          # (K, type_embed_dim)
        coord_e = self.coord_mlp(target_coords_norm)     # (K, coord_embed_dim)
        return torch.cat([type_e, coord_e], dim=-1)      # (K, TARGET_EMBED_DIM)


class CrossAttentionTargetHead(nn.Module):
    """Multi-head cross-attention: контекст траектории (Q) <-> эмбеддинги целей (K/V)."""

    def __init__(self,
                 d_model: int = config.MAMBA_D_MODEL,
                 target_embed_dim: int = config.TARGET_EMBED_DIM,
                 num_heads: int = config.CROSS_ATTN_HEADS,
                 d_attn: int = config.CROSS_ATTN_DIM,
                 dropout: float = config.DROPOUT):
        super().__init__()
        assert d_attn % num_heads == 0
        self.num_heads = num_heads
        self.d_attn = d_attn
        self.d_head = d_attn // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_attn, bias=False)
        self.k_proj = nn.Linear(target_embed_dim, d_attn, bias=False)
        self.v_proj = nn.Linear(target_embed_dim, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context: torch.Tensor,
                target_embeds: torch.Tensor) -> tuple:
        """
        Args:
            context:       (B, d_model)
            target_embeds: (K, target_embed_dim)

        Returns:
            target_aware:  (B, d_model) - взвешенная сумма Value
            target_logits: (B, K)       - логиты классификации цели
        """
        B = context.shape[0]
        K = target_embeds.shape[0]

        Q = self.q_proj(context).view(B, self.num_heads, 1, self.d_head)
        Kt = self.k_proj(target_embeds).view(K, self.num_heads, self.d_head)
        Kt = Kt.permute(1, 0, 2).unsqueeze(0)            # (1, H, K, d_head)
        Vt = self.v_proj(target_embeds).view(K, self.num_heads, self.d_head)
        Vt = Vt.permute(1, 0, 2).unsqueeze(0)            # (1, H, K, d_head)

        scores = torch.matmul(Q, Kt.transpose(-2, -1)) * self.scale   # (B, H, 1, K)
        attn = self.dropout(F.softmax(scores, dim=-1))
        target_aware = torch.matmul(attn, Vt)            # (B, H, 1, d_head)
        target_aware = target_aware.squeeze(2).reshape(B, self.d_attn)
        target_aware = self.out_proj(target_aware)       # (B, d_model)

        target_logits = scores.squeeze(2).sum(dim=1)     # (B, K)
        return target_aware, target_logits
