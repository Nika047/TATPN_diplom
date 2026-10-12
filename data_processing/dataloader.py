"""
dataloader.py — DataLoader для обучения TATPN.

Формирует обучающие пары:
  Вход:  первые X% траектории (наблюдаемый участок) + маска паддинга
  Метка: {target_id, future_points, eta}

Ключевые особенности:
  - Padding + mask: все последовательности дополняются до MAX_SEQ_LEN
  - Варьирование X%: модель видит 20-80% траектории при каждом обращении
  - Кинематические признаки вычисляются для наблюдаемого участка
  - Нормализация признаков (z-score)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import os

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Заглушки для работы без torch (тестирование)
    class Dataset:
        pass
    class DataLoader:
        pass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.kinematics import compute_kinematics, FeatureNormalizer


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset для обучения TATPN.

    Каждый элемент содержит:
    - observed: (MAX_SEQ_LEN, INPUT_FEATURES) — наблюдаемый участок + padding
    - mask: (MAX_SEQ_LEN,) — 1 для реальных точек, 0 для padding
    - target_id: int — индекс цели (0..K-1)
    - future_points: (PRED_HORIZON, 3) — будущие точки (x, y, z)
    - eta: float — время до конечной точки (секунды)
    - observe_ratio: float — доля наблюдаемого (для анализа)
    """

    def __init__(self, trajectories: List[Dict],
                 normalizer: Optional[FeatureNormalizer] = None,
                 is_train: bool = True):
        """
        Args:
            trajectories: список словарей с полями:
                - "trajectory": np.array (N, 4) [x, y, z, dt]
                - "target_id": int
            normalizer: FeatureNormalizer (для train — создаётся новый, для val/test — передаётся)
            is_train: если True — X% варьируется случайно, иначе — фиксированно 50%
        """
        self.is_train = is_train
        self.samples = []

        # Вычисляем кинематические признаки для каждой траектории
        all_features = []
        for item in trajectories:
            traj = item["trajectory"]  # (N, 4)
            if len(traj) < config.MIN_TRAJECTORY_POINTS:
                continue

            features = compute_kinematics(traj)  # (N, 15)
            all_features.append(features)

            self.samples.append({
                "features": features,
                "trajectory_xyz": traj[:, :3],  # (N, 3) — для future_points
                "trajectory_dt": traj[:, 3],     # (N,) — для ETA
                "target_id": item["target_id"],
                "n_points": len(traj),
            })

        # Нормализация
        if normalizer is None and len(all_features) > 0:
            # Fit на train данных
            all_data = np.concatenate(all_features, axis=0)
            self.normalizer = FeatureNormalizer()
            self.normalizer.fit(all_data)
        else:
            self.normalizer = normalizer

        # Нормализуем признаки
        if self.normalizer is not None and self.normalizer.fitted:
            for sample in self.samples:
                sample["features_norm"] = self.normalizer.transform(sample["features"])
        else:
            for sample in self.samples:
                sample["features_norm"] = sample["features"]

        print(f"[DataLoader] Создан dataset: {len(self.samples)} сэмплов, "
              f"train={is_train}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        n_points = sample["n_points"]
        features = sample["features_norm"]  # (N, 15)
        xyz = sample["trajectory_xyz"]       # (N, 3)
        dt = sample["trajectory_dt"]         # (N,)

        # Определяем процент наблюдения
        if self.is_train:
            observe_ratio = np.random.uniform(*config.OBSERVE_RATIO_RANGE)
        else:
            observe_ratio = 0.5  # Для val/test — фиксированные 50%

        # Разбиваем на наблюдаемый и будущий участки
        split_idx = max(int(n_points * observe_ratio), config.MIN_TRAJECTORY_POINTS)
        split_idx = min(split_idx, n_points - config.PRED_HORIZON)

        if split_idx < config.MIN_TRAJECTORY_POINTS:
            split_idx = config.MIN_TRAJECTORY_POINTS
        if split_idx >= n_points:
            split_idx = n_points - 1

        # Наблюдаемый участок
        observed = features[:split_idx]  # (split_idx, 15)

        # Будущие точки (до PRED_HORIZON)
        future_end = min(split_idx + config.PRED_HORIZON, n_points)
        future = xyz[split_idx:future_end]  # (actual_future, 3)

        # ETA: время от точки наблюдения до конца траектории
        eta = dt[-1] - dt[split_idx - 1]

        # === Padding ===
        seq_len = min(len(observed), config.MAX_SEQ_LEN)

        # Observed: pad до MAX_SEQ_LEN
        observed_padded = np.zeros((config.MAX_SEQ_LEN, config.INPUT_FEATURES), dtype=np.float32)
        observed_padded[:seq_len] = observed[:seq_len]

        # Mask: 1 для реальных точек, 0 для padding
        mask = np.zeros(config.MAX_SEQ_LEN, dtype=np.float32)
        mask[:seq_len] = 1.0

        # Future: pad до PRED_HORIZON
        future_padded = np.zeros((config.PRED_HORIZON, 3), dtype=np.float32)
        actual_future_len = min(len(future), config.PRED_HORIZON)
        if actual_future_len > 0:
            future_padded[:actual_future_len] = future[:actual_future_len]

        # Future mask
        future_mask = np.zeros(config.PRED_HORIZON, dtype=np.float32)
        future_mask[:actual_future_len] = 1.0

        return {
            "observed": torch.tensor(observed_padded, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "target_id": torch.tensor(sample["target_id"], dtype=torch.long),
            "future_points": torch.tensor(future_padded, dtype=torch.float32),
            "future_mask": torch.tensor(future_mask, dtype=torch.float32),
            "eta": torch.tensor(eta, dtype=torch.float32),
            "observe_ratio": torch.tensor(observe_ratio, dtype=torch.float32),
        }


# ============================================================================
# Функции для создания DataLoader-ов
# ============================================================================

def create_dataloaders(all_trajectories: List[Dict],
                       batch_size: int = config.BATCH_SIZE
                       ) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureNormalizer]:
    """
    Создаёт train/val/test DataLoader-ы из общего списка траекторий.

    Разбивает данные на train/val/test и создаёт DataLoader для каждого.
    Нормализатор обучается ТОЛЬКО на train данных.

    Args:
        all_trajectories: список всех траекторий (все 3 слоя)
        batch_size: размер батча

    Returns:
        train_loader, val_loader, test_loader, normalizer
    """
    np.random.seed(config.RANDOM_SEED)

    # Перемешиваем
    indices = np.random.permutation(len(all_trajectories))

    n_train = int(len(indices) * config.TRAIN_RATIO)
    n_val = int(len(indices) * config.VAL_RATIO)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_data = [all_trajectories[i] for i in train_indices]
    val_data = [all_trajectories[i] for i in val_indices]
    test_data = [all_trajectories[i] for i in test_indices]

    print(f"\n[DataLoader] Разбиение: train={len(train_data)}, "
          f"val={len(val_data)}, test={len(test_data)}")

    # Train dataset (с нормализатором)
    train_dataset = TrajectoryDataset(train_data, normalizer=None, is_train=True)
    normalizer = train_dataset.normalizer

    # Val/Test — используем нормализатор от train
    val_dataset = TrajectoryDataset(val_data, normalizer=normalizer, is_train=False)
    test_dataset = TrajectoryDataset(test_data, normalizer=normalizer, is_train=False)

    # pin_memory имеет смысл только при наличии CUDA, иначе PyTorch ругается
    pin = bool(HAS_TORCH and torch.cuda.is_available())

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, pin_memory=pin
    )

    return train_loader, val_loader, test_loader, normalizer


def save_dataloaders(train_data: List[Dict], val_data: List[Dict],
                     test_data: List[Dict], normalizer: FeatureNormalizer,
                     save_dir: str = config.PROCESSED_DATA_DIR):
    """Сохраняет данные и нормализатор на диск."""
    os.makedirs(save_dir, exist_ok=True)

    # Сохраняем траектории
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        trajectories = [d["trajectory"] for d in data]
        target_ids = [d["target_id"] for d in data]
        path = os.path.join(save_dir, f"{name}_trajectories.npz")
        np.savez(path,
                 trajectories=np.array(trajectories, dtype=object),
                 target_ids=np.array(target_ids))
        print(f"[Save] {name}: {len(data)} траекторий → {path}")

    # Сохраняем нормализатор
    norm_path = os.path.join(save_dir, "normalizer.npz")
    normalizer.save(norm_path)
    print(f"[Save] Normalizer → {norm_path}")


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    print("Тестирование DataLoader...")

    # Генерируем тестовые данные
    rng = np.random.RandomState(42)
    fake_trajectories = []
    for i in range(50):
        n = rng.randint(100, 500)
        t = np.arange(n, dtype=np.float32)
        x = np.cumsum(rng.randn(n) * 100)
        y = np.cumsum(rng.randn(n) * 100)
        z = rng.uniform(50, 500, n)
        dt = t * 1.0
        traj = np.stack([x, y, z, dt], axis=1).astype(np.float32)
        fake_trajectories.append({
            "trajectory": traj,
            "target_id": rng.randint(0, config.NUM_TARGETS),
        })

    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        fake_trajectories, batch_size=8
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Проверяем один батч
    batch = next(iter(train_loader))
    print(f"\nБатч:")
    for key, val in batch.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
