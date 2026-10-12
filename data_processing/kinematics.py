"""
kinematics.py — Вычисление кинематических признаков из траектории (x, y, z, dt).

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


def compute_kinematics(trajectory: np.ndarray,
                       smoothing_window: int = config.SMOOTHING_WINDOW
                       ) -> np.ndarray:
    """
    Вычисляет кинематические признаки для траектории.

    Args:
        trajectory: (N, 4) массив [x, y, z, dt]
        smoothing_window: размер окна сглаживания

    Returns:
        (N, 15) массив [x, y, z, dt, vx, vy, vz, |v|, ax, ay, az,
                         heading, climb_angle, curvature, heading_rate]
    """
    N = trajectory.shape[0]
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    dt = trajectory[:, 3]

    # Вычисляем промежутки времени между точками
    delta_t = np.diff(dt)
    delta_t = np.maximum(delta_t, 1e-6)  # Защита от деления на 0

    # === Скорости (центральные разности для внутренних точек) ===
    vx = np.zeros(N, dtype=np.float32)
    vy = np.zeros(N, dtype=np.float32)
    vz = np.zeros(N, dtype=np.float32)

    # Внутренние точки: центральная разность
    if N > 2:
        dt_central = dt[2:] - dt[:-2]
        dt_central = np.maximum(dt_central, 1e-6)
        vx[1:-1] = (x[2:] - x[:-2]) / dt_central
        vy[1:-1] = (y[2:] - y[:-2]) / dt_central
        vz[1:-1] = (z[2:] - z[:-2]) / dt_central

    # Граничные точки: односторонние разности
    if N > 1:
        vx[0] = (x[1] - x[0]) / delta_t[0]
        vy[0] = (y[1] - y[0]) / delta_t[0]
        vz[0] = (z[1] - z[0]) / delta_t[0]
        vx[-1] = (x[-1] - x[-2]) / delta_t[-1]
        vy[-1] = (y[-1] - y[-2]) / delta_t[-1]
        vz[-1] = (z[-1] - z[-2]) / delta_t[-1]

    # Сглаживание скоростей
    if smoothing_window > 1 and N > smoothing_window:
        vx = _moving_average(vx, smoothing_window)
        vy = _moving_average(vy, smoothing_window)
        vz = _moving_average(vz, smoothing_window)

    # Модуль скорости
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # === Ускорения ===
    ax = np.zeros(N, dtype=np.float32)
    ay = np.zeros(N, dtype=np.float32)
    az = np.zeros(N, dtype=np.float32)

    if N > 2:
        dt_central = dt[2:] - dt[:-2]
        dt_central = np.maximum(dt_central, 1e-6)
        ax[1:-1] = (vx[2:] - vx[:-2]) / dt_central
        ay[1:-1] = (vy[2:] - vy[:-2]) / dt_central
        az[1:-1] = (vz[2:] - vz[:-2]) / dt_central

    if N > 1:
        ax[0] = ax[1] if N > 1 else 0
        ay[0] = ay[1] if N > 1 else 0
        az[0] = az[1] if N > 1 else 0
        ax[-1] = ax[-2]
        ay[-1] = ay[-2]
        az[-1] = az[-2]

    # === Курс (heading) ===
    # atan2(east_velocity, north_velocity) → 0=North, CW
    heading = np.arctan2(vx, vy)  # x=East, y=North

    # === Угол набора высоты (climb angle) ===
    horizontal_speed = np.sqrt(vx**2 + vy**2)
    horizontal_speed_safe = np.maximum(horizontal_speed, 1e-6)
    climb_angle = np.arctan2(vz, horizontal_speed_safe)

    # === Кривизна траектории ===
    curvature = _compute_curvature(vx, vy, ax, ay)

    # === Скорость изменения курса (heading rate) ===
    heading_rate = np.zeros(N, dtype=np.float32)
    if N > 2:
        dh = np.diff(heading)
        # Обработка перехода через ±pi
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        heading_rate[1:] = dh / delta_t
        # Сглаживание
        if smoothing_window > 1 and N > smoothing_window:
            heading_rate = _moving_average(heading_rate, smoothing_window)

    # Собираем все признаки: (N, 15)
    features = np.stack([
        x, y, z, dt,                    # 0-3:  базовые координаты
        vx, vy, vz, speed,              # 4-7:  скорости
        ax, ay, az,                      # 8-10: ускорения
        heading, climb_angle,            # 11-12: углы
        curvature, heading_rate,         # 13-14: кривизна и поворот
    ], axis=1).astype(np.float32)

    return features


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


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее с сохранением длины массива."""
    if window <= 1 or len(arr) < window:
        return arr

    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='same')

    return smoothed.astype(arr.dtype)


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
