"""
losses.py - функции потерь для прогноза траектории.

masked_huber_traj - Huber (smooth L1) между предсказанными и истинными
координатами с учётом маски валидности (паддинг не штрафуется). Координаты
делятся на COORD_SCALE, чтобы ошибки в сотни метров попадали в разумный
диапазон и градиенты не были огромными.
"""

import torch
import torch.nn.functional as F

import config


def masked_huber_traj(pred: torch.Tensor,
                      target: torch.Tensor,
                      mask: torch.Tensor,
                      delta: float = config.HUBER_DELTA,
                      scale: float = config.COORD_SCALE) -> torch.Tensor:
    """
    Args:
        pred:   (B, H, 3) прогноз (м)
        target: (B, H, 3) истина (м)
        mask:   (B, H) 1 для валидных шагов
        delta:  порог Huber (в масштабированных единицах)
        scale:  делитель координат (м)

    Returns:
        скаляр - средняя потеря по валидным шагам.
    """
    p = pred / scale
    t = target / scale
    # smooth L1 поэлементно, затем сумма по осям координат
    per_elem = F.smooth_l1_loss(p, t, beta=delta, reduction="none")  # (B, H, 3)
    per_step = per_elem.sum(dim=-1)                                   # (B, H)
    denom = mask.sum().clamp(min=1.0)
    return (per_step * mask).sum() / denom
