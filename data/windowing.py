"""
windowing.py - корректная нарезка траектории на наблюдаемый и будущий участки.

Главное отличие от прежнего data_processing/dataloader.py:
  - наблюдаемое окно берётся ХВОСТОМ перед точкой разреза: observed[-L:],
    а не головой observed[:L]. Для длинных траекторий (94% длиннее
    MAX_SEQ_LEN) это гарантирует, что последняя наблюдаемая точка и начало
    future пространственно смежны (расстояние ~ один шаг dt, а не десятки км).
  - future[0] всегда идёт сразу за последней наблюдаемой точкой.

Модуль не зависит от torch: возвращает numpy-массивы. На этапах с обучением
поверх него делается тонкая обёртка для torch DataLoader.

Формат признаков (см. data_processing/kinematics.py), 15 каналов:
  0-3   x, y, z, dt
  4-6   vx, vy, vz        (м/с)
  7     |v|
  8-10  ax, ay, az
  11    heading
  12    climb_angle
  13    curvature
  14    heading_rate
"""

import numpy as np
from typing import List, Dict, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.kinematics import compute_kinematics, FeatureNormalizer

# Индексы каналов скорости (vx, vy, vz) в массиве признаков
VEL_CHANNELS = slice(4, 7)
# Минимальное число наблюдаемых точек в окне
MIN_OBSERVED_POINTS = 2


def compute_split_index(n_points: int,
                        observe_ratio: float,
                        min_observed: int = MIN_OBSERVED_POINTS) -> int:
    """
    Индекс разреза траектории на observed[:split] и future[split:].

    Гарантирует: min_observed <= split <= n_points - 1
    (то есть в наблюдении хотя бы min_observed точек, а в будущем - хотя бы 1).
    """
    split = int(round(n_points * observe_ratio))
    split = max(split, min_observed)
    split = min(split, n_points - 1)
    return split


class WindowedDataset:
    """
    Формирует окна (observed/future) из списка траекторий.

    Каждая траектория - словарь с полями:
      - "trajectory": np.ndarray (N, 4) = (x, y, z, dt) в ENU-метрах;
      - "target_id":  int, индекс цели.

    Признаки (15 каналов) вычисляются один раз при инициализации и
    нормализуются переданным FeatureNormalizer. last_point и last_velocity
    возвращаются в ИСХОДНЫХ единицах (метры, м/с) - они нужны декодеру и
    бейзлайнам без обратной нормализации.
    """

    def __init__(self,
                 trajectories: List[Dict],
                 normalizer: FeatureNormalizer,
                 mode: str = "fixed",
                 fixed_ratio: float = 0.5,
                 max_seq_len: int = None,
                 pred_horizon: int = None,
                 min_trajectory_points: int = None,
                 seed: int = config.RANDOM_SEED):
        """
        Args:
            trajectories: список словарей траекторий.
            normalizer:   обученный FeatureNormalizer (fit на train).
            mode:         "fixed" - доля наблюдения = fixed_ratio;
                          "random" - доля наблюдения из OBSERVE_RATIO_RANGE.
            fixed_ratio:  доля наблюдения для mode="fixed" (val/test = 0.5).
            max_seq_len:  максимум точек в наблюдаемом окне (по умолч. из config).
            pred_horizon: число будущих точек прогноза (по умолч. из config).
        """
        self.mode = mode
        self.fixed_ratio = fixed_ratio
        self.max_seq_len = max_seq_len or config.MAX_SEQ_LEN
        self.pred_horizon = pred_horizon or config.PRED_HORIZON
        self.min_traj = min_trajectory_points or config.MIN_TRAJECTORY_POINTS
        self.rng = np.random.RandomState(seed)

        assert normalizer is not None and normalizer.fitted, \
            "WindowedDataset требует обученный FeatureNormalizer"
        self.normalizer = normalizer

        # Храним ТОЛЬКО сырые признаки (15 каналов); нормализация окна
        # выполняется на лету в make(). Это экономит память на больших
        # выборках (для train ~2600 траекторий хранение и raw, и norm
        # разом дало бы ~1.6 ГБ). xyz и dt берём из feats_raw (каналы 0-2 и 3).
        self.samples = []
        skipped = 0
        for item in trajectories:
            traj = np.asarray(item["trajectory"], dtype=np.float32)  # (N, 4)
            n = traj.shape[0]
            if n < self.min_traj:
                skipped += 1
                continue
            feats_raw = compute_kinematics(traj)                     # (N, 15)
            self.samples.append({
                "feats_raw":  feats_raw,                             # (N, 15)
                "target_id":  int(item["target_id"]),
                "n_points":   n,
            })
        print(f"[windowing] окон: {len(self.samples)} "
              f"(пропущено коротких: {skipped}), mode={mode}, "
              f"L={self.max_seq_len}, H={self.pred_horizon}")

    def __len__(self):
        return len(self.samples)

    def _observe_ratio(self) -> float:
        if self.mode == "random":
            lo, hi = config.OBSERVE_RATIO_RANGE
            return float(self.rng.uniform(lo, hi))
        return self.fixed_ratio

    def make(self, idx: int, observe_ratio: Optional[float] = None) -> Dict:
        """Строит одно окно для образца idx (numpy)."""
        s = self.samples[idx]
        feats_raw = s["feats_raw"]                               # (N, 15)
        n = s["n_points"]
        xyz = feats_raw[:, :3]                                   # (N, 3)
        ratio = observe_ratio if observe_ratio is not None else self._observe_ratio()

        split = compute_split_index(n, ratio)

        # --- Наблюдаемое окно = ХВОСТ перед разрезом; нормализуем на лету ---
        start = max(0, split - self.max_seq_len)
        observed = self.normalizer.transform(feats_raw[start:split]).astype(np.float32)
        seq_len = observed.shape[0]

        observed_padded = np.zeros((self.max_seq_len, config.INPUT_FEATURES),
                                   dtype=np.float32)
        observed_padded[:seq_len] = observed
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        # --- Будущее = сразу за точкой разреза ---
        fut_end = min(split + self.pred_horizon, n)
        future = xyz[split:fut_end]                              # (h, 3)
        fh = future.shape[0]
        future_padded = np.zeros((self.pred_horizon, 3), dtype=np.float32)
        future_padded[:fh] = future
        future_mask = np.zeros(self.pred_horizon, dtype=np.float32)
        future_mask[:fh] = 1.0

        # --- Последняя наблюдаемая точка и скорость (исходные единицы) ---
        last_point = xyz[split - 1].astype(np.float32).copy()          # (3,)
        last_velocity = feats_raw[split - 1, VEL_CHANNELS].astype(np.float32).copy()  # (3,)

        # --- ETA: время от точки наблюдения до конца траектории (сек) ---
        eta = float(feats_raw[-1, 3] - feats_raw[split - 1, 3])

        return {
            "observed":      observed_padded,     # (L, 15) норм.
            "mask":          mask,                # (L,)
            "last_point":    last_point,          # (3,) м
            "last_velocity": last_velocity,       # (3,) м/с
            "future_points": future_padded,       # (H, 3) м
            "future_mask":   future_mask,         # (H,)
            "target_id":     s["target_id"],
            "eta":           eta,
            "observe_ratio": float(ratio),
            "n_points":      n,
        }

    def build_all(self, observe_ratio: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Строит окна для всех образцов и складывает в батч-массивы.
        Удобно для не-нейросетевой оценки (Этап 1).

        Returns:
            dict со стеками numpy-массивов по всем образцам.
        """
        keys_stack = ["observed", "mask", "last_point", "last_velocity",
                      "future_points", "future_mask"]
        keys_scalar = ["target_id", "eta", "observe_ratio", "n_points"]

        acc = {k: [] for k in keys_stack + keys_scalar}
        for i in range(len(self.samples)):
            w = self.make(i, observe_ratio=observe_ratio)
            for k in keys_stack:
                acc[k].append(w[k])
            for k in keys_scalar:
                acc[k].append(w[k])

        out = {k: np.stack(acc[k], axis=0) for k in keys_stack}
        out["target_id"]     = np.asarray(acc["target_id"], dtype=np.int64)
        out["eta"]           = np.asarray(acc["eta"], dtype=np.float32)
        out["observe_ratio"] = np.asarray(acc["observe_ratio"], dtype=np.float32)
        out["n_points"]      = np.asarray(acc["n_points"], dtype=np.int64)
        return out
