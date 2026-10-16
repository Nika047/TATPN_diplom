"""
kinematics.py — Вычисление кинематических признаков из траектории (x, y, z, dt).

ПРИЧИННОСТЬ: признаки точки i зависят только от точек 0..i (см. compute_kinematics).

Из 4 базовых координат вычисляет 11 дополнительных признаков:
  vx, vy, vz    — компоненты скорости (м/с)
  |v|            — модуль скорости (м/с)
  ax, ay, az     — компоненты ускорения (м/с^2)
  heading        — курс в горизонтальной плоскости (рад, 0 = North, CW)
  climb_angle    — угол набора высоты (рад)
  curvature      — кривизна траектории (1/м)
  heading_rate   — скорость изменения курса (рад/с)

Итого: 4 + 11 = 15 признаков на точку.
"""

import numpy as np
from typing import Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _causal_mean(arr: np.ndarray, window: int, start: int = 0) -> np.ndarray:
    """
    Скользящее среднее по последним `window` значениям (причинное).

    Значение в точке i зависит только от arr[start:i+1], в начале окно
    сжимается. Элементы до `start` не учитываются и остаются нулями.
    """
    out = np.zeros(len(arr), dtype=np.float64)
    seg = np.asarray(arr[start:], dtype=np.float64)
    if seg.size == 0:
        return out
    if window <= 1:
        out[start:] = seg
        return out
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    idx = np.arange(1, seg.size + 1)
    lo = np.maximum(idx - window, 0)
    out[start:] = (cum[idx] - cum[lo]) / (idx - lo)
    return out


def compute_kinematics(trajectory: np.ndarray,
                       smoothing_window: int = config.SMOOTHING_WINDOW
                       ) -> np.ndarray:
    """
    Вычисляет кинематические признаки для траектории (ПРИЧИННО).

    Признаки точки i зависят только от точек 0..i. Это обязательное условие:
    система в момент i знает только прошлое, а признаки считаются один раз по
    всей траектории и затем режутся на окна. Любая центральная разность или
    симметричное сглаживание подмешали бы в последние наблюдаемые точки
    информацию о будущих (раньше так и было: затрагивались 5 последних точек).

    Скорость - разность назад (p[i] - p[i-1]) / dt, затем причинное сглаживание
    по последним `smoothing_window` значениям. Ускорение - разность назад
    сглаженной скорости. В первой точке скорость неизвестна (0), ускорение и
    скорость поворота определены с третьей точки.

    Args:
        trajectory: (N, 4) массив [x, y, z, dt], dt - время от начала, с
        smoothing_window: размер причинного окна сглаживания

    Returns:
        (N, 15) массив [x, y, z, dt, vx, vy, vz, |v|, ax, ay, az,
                         heading, climb_angle, curvature, heading_rate]
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    N = traj.shape[0]
    x, y, z, t = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

    feats = np.zeros((N, 15), dtype=np.float64)
    feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3] = x, y, z, t
    if N < 2:
        return feats.astype(np.float32)

    step = np.maximum(np.diff(t), 1e-6)                 # step[i-1] = t[i] - t[i-1]

    # Скорость: разность назад, затем причинное сглаживание
    v = np.zeros((N, 3))
    v[1:, 0] = np.diff(x) / step
    v[1:, 1] = np.diff(y) / step
    v[1:, 2] = np.diff(z) / step
    for j in range(3):
        v[:, j] = _causal_mean(v[:, j], smoothing_window, start=1)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    # Ускорение: разность назад сглаженной скорости (с третьей точки)
    a = np.zeros((N, 3))
    if N > 2:
        a[2:] = (v[2:] - v[1:-1]) / step[1:, None]
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]

    # Курс (0 = север, по часовой), угол набора высоты
    heading = np.arctan2(vx, vy)
    horizontal = np.hypot(vx, vy)
    climb_angle = np.arctan2(vz, np.maximum(horizontal, 1e-6))

    curvature = _compute_curvature(vx, vy, ax, ay)

    # Скорость изменения курса (с третьей точки), с обработкой перехода через ±pi
    heading_rate = np.zeros(N)
    if N > 2:
        dh = np.diff(heading)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        heading_rate[2:] = dh[1:] / step[1:]
    heading_rate = _causal_mean(heading_rate, smoothing_window, start=2)

    feats[:, 4], feats[:, 5], feats[:, 6], feats[:, 7] = vx, vy, vz, speed
    feats[:, 8], feats[:, 9], feats[:, 10] = ax, ay, az
    feats[:, 11], feats[:, 12] = heading, climb_angle
    feats[:, 13], feats[:, 14] = curvature, heading_rate
    return feats.astype(np.float32)


def _compute_curvature(vx: np.ndarray, vy: np.ndarray,
                       ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    """
    Кривизна плоской кривой:
    κ = |vx*ay - vy*ax| / (vx^2 + vy^2)^(3/2)
    """
    numerator = np.abs(vx * ay - vy * ax)
    denominator = (vx**2 + vy**2) ** 1.5
    denominator = np.maximum(denominator, 1e-10)
    curvature = numerator / denominator

    # Ограничиваем выбросы (макс. кривизна = поворот на 1м)
    curvature = np.clip(curvature, 0, 1.0)

    return curvature.astype(np.float32)


# ============================================================================
# Нормализация признаков
# ============================================================================

class FeatureNormalizer:
    """
    Нормализация признаков (z-score или min-max).
    Сохраняет статистики для обратного преобразования.
    """

    def __init__(self, method: str = config.NORMALIZATION_METHOD):
        self.method = method
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        self.fitted = False

    def fit(self, data: np.ndarray):
        """
        Вычисляет статистики по данным.

        Args:
            data: (N_total, F) — все точки всех траекторий
        """
        if self.method == "standard":
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.std = np.maximum(self.std, 1e-8)  # Защита от деления на 0
        elif self.method == "minmax":
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
            range_val = self.max_val - self.min_val
            range_val = np.maximum(range_val, 1e-8)
            self.max_val = self.min_val + range_val  # Обновляем с защитой

        self.fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Нормализует данные."""
        assert self.fitted, "Сначала вызовите fit()"

        if self.method == "standard":
            return (data - self.mean) / self.std
        elif self.method == "minmax":
            return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Обратная нормализация."""
        assert self.fitted, "Сначала вызовите fit()"

        if self.method == "standard":
            return data * self.std + self.mean
        elif self.method == "minmax":
            return data * (self.max_val - self.min_val) + self.min_val

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """fit + transform."""
        self.fit(data)
        return self.transform(data)

    def save(self, filepath: str):
        """Сохраняет статистики в файл."""
        np.savez(filepath,
                 method=self.method,
                 mean=self.mean, std=self.std,
                 min_val=self.min_val, max_val=self.max_val)

    def load(self, filepath: str):
        """Загружает статистики из файла."""
        data = np.load(filepath, allow_pickle=True)
        self.method = str(data["method"])
        self.mean = data["mean"]
        self.std = data["std"]
        self.min_val = data["min_val"]
        self.max_val = data["max_val"]
        self.fitted = True


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    # Создаём тестовую траекторию: прямая линия с шумом
    N = 100
    t = np.arange(N, dtype=np.float32)
    x = 100.0 * t + np.random.randn(N) * 5
    y = 50.0 * t + np.random.randn(N) * 5
    z = 200.0 + np.random.randn(N) * 2
    dt = t * 1.0

    trajectory = np.stack([x, y, z, dt], axis=1)
    print(f"Входная траектория: {trajectory.shape}")

    features = compute_kinematics(trajectory)
    print(f"Выходные признаки: {features.shape}")

    # Имена признаков
    names = ["x", "y", "z", "dt", "vx", "vy", "vz", "|v|",
             "ax", "ay", "az", "heading", "climb", "curvature", "heading_rate"]

    print("\nСтатистика признаков (медиана):")
    for i, name in enumerate(names):
        vals = features[:, i]
        print(f"  {name:14s}: median={np.median(vals):10.3f}, "
              f"std={np.std(vals):10.3f}")
