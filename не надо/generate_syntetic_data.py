# Генерация синтетических данных
# Генерирует синтетические траектории в кубе 0…100 м со случайными манёврами
# Структура точки: [x, y, z, dt] (координаты + временной шаг)

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline

import globals


class TrajectoryDataset(Dataset):
    def __init__(self, n_samples=2000, seq_len=100, pred_horizon=20, dt=0.1, seed=None):
        self.seq_len, self.pred_horizon, self.dt = seq_len, pred_horizon, dt

        if seed is not None:
            # собственный seed сплита: даёт воспроизводимость и различие между train/val/test
            self.rng = np.random.RandomState(seed)
            self.rand = random.Random(seed)
        else:
            self.rng = np.random
            self.rand = random

        self.data = [self._make_sample() for _ in range(n_samples)]

    def _make_sample(self):
        # Случайные ключевые точки
        n_knots = self.rand.randint(4, 8)
        knots_x = np.linspace(0, 100, n_knots)
        knots_y = self.rng.uniform(0, 100, n_knots)
        knots_z = self.rng.uniform(0, 50, n_knots)

        # Сплайны на нормированном времени
        u = np.linspace(0, 1, n_knots)
        cs_x = CubicSpline(u, knots_x)
        cs_y = CubicSpline(u, knots_y)
        cs_z = CubicSpline(u, knots_z)

        # Дискретизация траектории
        t = np.linspace(0, 1, self.seq_len + self.pred_horizon)   # t: 120 моментов времени (100+20) от 0 до 1
        traj = np.column_stack([
            cs_x(t),                                              # cs_x(t): Вычисление координат в этих точках
            cs_y(t),
            cs_z(t),
            np.full(len(t), self.dt)                              # dt: Фиксированный временной шаг (0.1 сек) добавляется как 4-я компонента
        ])

        return torch.tensor(traj, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]                        # (seq_len+pred_horizon, 4)
        inp  = sample[: self.seq_len]                  # (seq_len, 4)
        tgt  = sample[self.seq_len:][:, :3]            # (pred_horizon, 3)
        return inp, tgt

    # def _make_sample(self):
    #     x, y, z = (random.uniform(0, 100) for _ in range(3))
    #     vx = vy = vz = 0
    #     traj = []

    #     for _ in range(self.seq_len + self.pred_horizon):
    #         # Симуляция "винеровского процесса" для скорости
    #         vx += random.uniform(-0.3, 0.3)
    #         vy += random.uniform(-0.3, 0.3)
    #         vz += random.uniform(-0.1, 0.1)

    #         # Ограничения скорости
    #         vx = np.clip(vx, -3, 3)
    #         vy = np.clip(vy, -3, 3)
    #         vz = np.clip(vz, -1, 1)

    #         # Обновление позиции
    #         x += vx * self.dt
    #         y += vy * self.dt
    #         z += vz * self.dt

    #         # Границы пространства
    #         x, y, z = [max(0, min(100, v)) for v in (x, y, z)]

    #         traj.append([x, y, z, self.dt])

    #     return torch.tensor(traj).float()

def create_datasets():
    train_ds = TrajectoryDataset(globals.n_samples_train, globals.seq_len, globals.pred_horizon, globals.dt, seed=123)
    val_ds = TrajectoryDataset(globals.n_samples_val, globals.seq_len, globals.pred_horizon, globals.dt, seed=456)
    test_ds = TrajectoryDataset(globals.n_samples_test, globals.seq_len, globals.pred_horizon, globals.dt, seed=789)

    return train_ds, val_ds, test_ds
