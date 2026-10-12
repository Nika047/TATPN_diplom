"""
constant_position.py - бейзлайн "стоять на месте".

Прогноз всех H будущих точек = последняя наблюдаемая точка. Самый наивный
опорный вариант; показывает, какую ошибку даёт полное игнорирование движения.
"""

import numpy as np


def predict(last_point: np.ndarray, horizon: int) -> np.ndarray:
    """
    Args:
        last_point: (B, 3) - последняя наблюдаемая точка (метры).
        horizon:    число будущих шагов H.

    Returns:
        (B, H, 3) - прогноз (повтор last_point).
    """
    last_point = np.asarray(last_point, dtype=np.float32)   # (B, 3)
    B = last_point.shape[0]
    return np.repeat(last_point[:, None, :], horizon, axis=1)  # (B, H, 3)
