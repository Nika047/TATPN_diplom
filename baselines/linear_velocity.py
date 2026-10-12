"""
linear_velocity.py - бейзлайн линейной экстраполяции по последней скорости.

Прогноз: p_k = last_point + last_velocity * dt * k, k = 1..H.
Это "продолжай движение с текущей скоростью" - физически разумный опорный
прогноз и главная нижняя планка для обучаемых моделей.
"""

import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def predict(last_point: np.ndarray,
            last_velocity: np.ndarray,
            horizon: int,
            dt: float = config.RESAMPLE_DT) -> np.ndarray:
    """
    Args:
        last_point:    (B, 3) - последняя наблюдаемая точка (метры).
        last_velocity: (B, 3) - последняя наблюдаемая скорость (м/с).
        horizon:       число будущих шагов H.
        dt:            шаг времени между точками (сек).

    Returns:
        (B, H, 3) - прогноз линейной экстраполяции.
    """
    last_point = np.asarray(last_point, dtype=np.float32)       # (B, 3)
    last_velocity = np.asarray(last_velocity, dtype=np.float32)  # (B, 3)
    steps = np.arange(1, horizon + 1, dtype=np.float32)         # (H,)
    # (B, 1, 3) + (B, 1, 3) * (1, H, 1) -> (B, H, 3)
    disp = last_velocity[:, None, :] * (steps[None, :, None] * dt)
    return last_point[:, None, :] + disp
