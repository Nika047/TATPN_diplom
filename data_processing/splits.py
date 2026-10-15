"""
splits.py - разбиение на train/val/test без утечки источников.

Проблема прежнего разбиения: траектории scale, stitch и warp производны от
реальных логов, а делились случайно по траекториям - производные от одного
полёта попадали и в обучающую, и в тестовую выборки.

Решение (по построению, без пост-обработки):
  1. реальные полёты делятся на train/val/test случайно (seed);
  2. расширенные траектории каждой выборки порождаются ТОЛЬКО из реальных
     полётов этой же выборки (run_stage0.py вызывает аугментор по выборкам);
  3. синтетические траектории независимы от реальных и делятся случайно.
Метка выборки сохраняется в all_trajectories.npz (массив `splits`), и все
этапы читают её оттуда, а не пересчитывают перестановку.

Метки: 0 - train, 1 - val, 2 - test.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.kinematics import FeatureNormalizer, compute_kinematics

SPLIT_NAMES = ("train", "val", "test")


def random_labels(n: int, seed: int) -> np.ndarray:
    """Случайные метки выборки (0/1/2) для n элементов в пропорции TRAIN/VAL/TEST."""
    perm = np.random.RandomState(seed).permutation(n)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    labels = np.empty(n, dtype=np.int8)
    labels[perm[:n_train]] = 0
    labels[perm[n_train:n_train + n_val]] = 1
    labels[perm[n_train + n_val:]] = 2
    return labels


def split_counts(total: int) -> Tuple[int, int, int]:
    """Разделить число total по выборкам в той же пропорции (остаток - в test)."""
    n_train = int(total * config.TRAIN_RATIO)
    n_val = int(total * config.VAL_RATIO)
    return n_train, n_val, total - n_train - n_val


def fit_normalizer(trajectories: List[np.ndarray]) -> FeatureNormalizer:
    """
    z-score нормализатор по признакам обучающих траекторий.

    Статистики копятся по траекториям (float64), без склейки всех признаков в
    один массив: признаки всех траекторий занимают сотни мегабайт.
    """
    n = 0
    total = None
    total_sq = None
    for traj in trajectories:
        feats = compute_kinematics(np.asarray(traj, dtype=np.float32)).astype(np.float64)
        if total is None:
            total = np.zeros(feats.shape[1])
            total_sq = np.zeros(feats.shape[1])
        n += feats.shape[0]
        total += feats.sum(axis=0)
        total_sq += (feats ** 2).sum(axis=0)
    if n == 0:
        raise ValueError("Нет траекторий для обучения нормализатора")
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))

    norm = FeatureNormalizer()
    norm.mean = mean.astype(np.float32)
    norm.std = np.maximum(std, 1e-8).astype(np.float32)
    norm.fitted = True
    return norm


def load_splits(npz_path: str = None) -> Dict:
    """
    Загружает траектории и метки выборок, созданные run_stage0.py.

    Returns:
        {"train": [...], "val": [...], "test": [...], "n_all": int}, где элемент -
        {"trajectory": (N, 4), "target_id": int, "type": str}.
    """
    npz_path = npz_path or os.path.join(config.PROCESSED_DATA_DIR,
                                        "all_trajectories.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Не найден {npz_path}. Сначала выполните Этап 0: python run_stage0.py")

    data = np.load(npz_path, allow_pickle=True)
    if "splits" not in data.files:
        raise RuntimeError(
            "Датасет создан прежней версией Этапа 0 и не содержит меток выборок "
            "(в нём производные траектории могли попасть и в train, и в test). "
            "Пересоздайте его: python run_stage0.py")

    trajectories = data["trajectories"]
    target_ids = data["target_ids"]
    types = data["trajectory_types"]
    labels = data["splits"]

    out = {name: [] for name in SPLIT_NAMES}
    for i in range(len(trajectories)):
        out[SPLIT_NAMES[int(labels[i])]].append({
            "trajectory": trajectories[i],
            "target_id": int(target_ids[i]),
            "type": str(types[i]),
        })
    out["n_all"] = len(trajectories)
    return out
