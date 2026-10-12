"""
geometry.py - геометрические признаки "траектория -> цель" для Target Head.

Для каждой пары (образец, цель i) считаем подсказки, из которых модели проще
понять, к какой цели направлен объект:
  - dist_norm  - расстояние от последней наблюдаемой точки до цели i
                 (нормировано на 100 км);
  - cos_angle  - косинус угла между курсом объекта и направлением на цель i
                 (+1 = летит прямо на цель, -1 = от неё);
  - sin_angle  - синус того же угла со знаком (лево/право от курса).

Работает с torch-тензорами (вызывается внутри model.forward). Координаты -
горизонтальные (E, N).
"""

import torch

# Масштаб нормировки расстояния (м). Цели в задаче удалены на десятки-сотни км.
DIST_SCALE = 1.0e5   # 100 км
GEO_DIM = 3          # число геометрических признаков на цель


def compute_geometric_features(last_point: torch.Tensor,
                               last_velocity: torch.Tensor,
                               target_coords: torch.Tensor) -> torch.Tensor:
    """
    Args:
        last_point:    (B, 3) - последняя наблюдаемая точка (E, N, U), м
        last_velocity: (B, 3) - последняя скорость (vE, vN, vU), м/с
        target_coords: (K, 2) - координаты целей (E, N), м

    Returns:
        geo: (B, K, GEO_DIM) - [dist_norm, cos_angle, sin_angle]
    """
    pos = last_point[:, :2]                                  # (B, 2)
    vel = last_velocity[:, :2]                               # (B, 2)

    # Вектор и расстояние до каждой цели
    d = target_coords.unsqueeze(0) - pos.unsqueeze(1)        # (B, K, 2)
    dist = torch.linalg.norm(d, dim=-1).clamp(min=1e-6)      # (B, K)
    dir_ = d / dist.unsqueeze(-1)                            # (B, K, 2)

    # Единичный курс
    speed = torch.linalg.norm(vel, dim=-1).clamp(min=1e-6)   # (B,)
    head = (vel / speed.unsqueeze(-1)).unsqueeze(1)          # (B, 1, 2)

    cos_a = (head * dir_).sum(dim=-1)                        # (B, K)
    sin_a = head[..., 0] * dir_[..., 1] - head[..., 1] * dir_[..., 0]  # (B, K)
    dist_norm = dist / DIST_SCALE                            # (B, K)

    return torch.stack([dist_norm, cos_a, sin_a], dim=-1)    # (B, K, GEO_DIM)
