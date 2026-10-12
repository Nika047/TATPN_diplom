"""
augmentor.py — Расширение реальных данных до боевого масштаба (Слой 2).

Три метода расширения:
  A. Scale & Translate — масштабирование координат в 10-50 раз
  B. Stitch — сшивка 3-5 реальных траекторий в один длинный маршрут
  C. Warp to Target — деформация траектории к конкретной цели

Все методы сохраняют реалистичную кинематику движения БПЛА,
но увеличивают масштаб до десятков-сотен километров.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.target_database import TargetDatabase
from data_processing.parser import resample_trajectory


def _resample_aug_trajectory(traj: np.ndarray,
                              target_dt: float = config.RESAMPLE_DT,
                              max_points: int = 1800) -> Optional[np.ndarray]:
    """
    Привести аугментированную траекторию к единому временному шагу.

    КРИТИЧНО: без этого вызова между соседними точками траектории может быть
    не 1 секунда, а 10-50 секунд (из-за умножения dt на scale в методах aug).
    Это ломает обучение модели последовательностей: PRED_HORIZON шагов
    физически означает разное время для разных траекторий, и velocity-prior
    работает с неправильным интервалом.

    Дополнительно длина траектории обрезается до max_points последних точек
    (≈30 мин при шаге 1 с) - это нужно как от переполнения памяти на длинных
    50-кратно растянутых траекториях, так и для физической реалистичности
    (БПЛА редко летят более получаса в исследуемом сценарии).

    Args:
        traj: (N, 4) - [x, y, z, dt]
        target_dt: целевой шаг между точками (с)
        max_points: максимум точек в результате

    Returns:
        (M, 4) или None, если траектория стала слишком короткой
    """
    if traj is None or len(traj) < 2:
        return None
    x, y, z, t = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    new_x, new_y, new_z, new_t = resample_trajectory(x, y, z, t, target_dt=target_dt)
    if len(new_t) < config.MIN_TRAJECTORY_POINTS:
        return None
    # Обрезка до max_points последних точек (сохраняем конец трека -
    # он содержит подход к цели, который важен для меток).
    if len(new_t) > max_points:
        new_x = new_x[-max_points:]
        new_y = new_y[-max_points:]
        new_z = new_z[-max_points:]
        new_t = new_t[-max_points:] - new_t[-max_points]  # перенумеровать с 0
    return np.stack([new_x, new_y, new_z, new_t], axis=1).astype(np.float32)


class TrajectoryAugmentor:
    """
    Расширяет реальные траектории до боевого масштаба.

    Входные данные: реальные траектории (N, 4) = [x, y, z, dt] в ENU
    Выходные данные: расширенные траектории того же формата
    """

    def __init__(self, target_db: Optional[TargetDatabase] = None,
                 seed: int = config.RANDOM_SEED):
        self.rng = np.random.RandomState(seed + 100)

        if target_db is None:
            self.target_db = TargetDatabase()
        else:
            self.target_db = target_db

        self.target_coords = self.target_db.get_enu_coords()

    def augment_all(self, real_trajectories: List[np.ndarray],
                    total_count: int = config.AUGMENTED_COUNT) -> List[Dict]:
        """
        Создаёт расширенные траектории из реальных.

        Args:
            real_trajectories: список реальных траекторий (N, 4)
            total_count: итоговое количество расширенных траекторий

        Returns:
            Список словарей с расширенными траекториями
        """
        if len(real_trajectories) == 0:
            print("[Augmentor] Нет реальных траекторий для расширения!")
            return []

        results = []

        # Распределение по методам: 40% scale, 30% stitch, 30% warp
        n_scale = int(total_count * 0.4)
        n_stitch = int(total_count * 0.3)
        n_warp = total_count - n_scale - n_stitch

        # Метод A: Scale & Translate
        print(f"[Augmentor] Scale & Translate: {n_scale} траекторий...")
        for i in range(n_scale):
            src = real_trajectories[self.rng.randint(len(real_trajectories))]
            target_idx = self.rng.randint(self.target_db.num_targets)
            traj = self._scale_and_translate(src, target_idx)
            if traj is not None:
                results.append({
                    "trajectory": traj,
                    "target_id": target_idx,
                    "augmentation_type": "scale",
                })

        # Метод B: Stitch
        print(f"[Augmentor] Stitch: {n_stitch} траекторий...")
        for i in range(n_stitch):
            n_segments = self.rng.randint(*config.STITCH_NUM_SEGMENTS)
            indices = self.rng.choice(len(real_trajectories), size=n_segments, replace=True)
            segments = [real_trajectories[idx] for idx in indices]
            target_idx = self.rng.randint(self.target_db.num_targets)
            traj = self._stitch(segments, target_idx)
            if traj is not None:
                results.append({
                    "trajectory": traj,
                    "target_id": target_idx,
                    "augmentation_type": "stitch",
                })

        # Метод C: Warp to Target
        print(f"[Augmentor] Warp to Target: {n_warp} траекторий...")
        for i in range(n_warp):
            src = real_trajectories[self.rng.randint(len(real_trajectories))]
            target_idx = self.rng.randint(self.target_db.num_targets)
            traj = self._warp_to_target(src, target_idx)
            if traj is not None:
                results.append({
                    "trajectory": traj,
                    "target_id": target_idx,
                    "augmentation_type": "warp",
                })

        print(f"[Augmentor] Всего: {len(results)} расширенных траекторий")
        return results

    # ========================================================================
    # Метод A: Scale & Translate
    # ========================================================================

    def _scale_and_translate(self, trajectory: np.ndarray,
                             target_idx: int) -> Optional[np.ndarray]:
        """
        Масштабирует реальную траекторию и сдвигает к цели.

        Берём реальную траекторию (кружение в радиусе 5 км)
        → Масштабируем координаты в 10-50 раз
        → Получаем «полёт» в радиусе 50-250 км
        → Разворачиваем и сдвигаем так, чтобы конец был у цели

        Кинематика остаётся реалистичной, но масштаб боевой.
        """
        if len(trajectory) < config.MIN_TRAJECTORY_POINTS:
            return None

        x = trajectory[:, 0].copy()
        y = trajectory[:, 1].copy()
        z = trajectory[:, 2].copy()
        dt = trajectory[:, 3].copy()

        # Масштабный коэффициент
        scale = self.rng.uniform(*config.SCALE_FACTORS)

        # Масштабируем XY (от центра траектории)
        cx, cy = np.mean(x), np.mean(y)
        x = (x - cx) * scale
        y = (y - cy) * scale

        # Случайный поворот
        angle = self.rng.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        # Сдвигаем конец к цели
        target_xy = self.target_coords[target_idx]
        end_x, end_y = x_rot[-1], y_rot[-1]
        x_rot += target_xy[0] - end_x
        y_rot += target_xy[1] - end_y

        # Масштабируем высоту (более скромно, 2-5x)
        z_scale = self.rng.uniform(2, 5)
        z = z * z_scale
        z = np.clip(z, 0, 3000)

        # Масштабируем время пропорционально расстоянию и скорости
        # Скорость остаётся той же → время растёт пропорционально масштабу
        dt = dt * scale

        result = np.stack([x_rot, y_rot, z, dt], axis=1).astype(np.float32)
        # Ресемплинг к шагу RESAMPLE_DT, иначе шаг между точками
        # = scale секунд, а не 1 с (что ломает обучение sequence-модели).
        return _resample_aug_trajectory(result)

    # ========================================================================
    # Метод B: Stitch (сшивка)
    # ========================================================================

    def _stitch(self, segments: List[np.ndarray],
                target_idx: int) -> Optional[np.ndarray]:
        """
        Сшивает несколько реальных траекторий в один длинный маршрут.

        1. Из каждой траектории выбираем «прямолинейный» участок
        2. Масштабируем каждый сегмент
        3. Соединяем сплайном в один длинный маршрут
        4. Направляем от случайной точки старта к цели
        """
        if len(segments) < 2:
            return None

        # Извлекаем и масштабируем сегменты
        processed_segments = []
        for seg in segments:
            if len(seg) < 10:
                continue

            # Берём наиболее прямолинейный участок (60-100% длины)
            frac = self.rng.uniform(0.6, 1.0)
            n = max(int(len(seg) * frac), 10)
            start_idx = self.rng.randint(0, max(len(seg) - n, 1))
            subseg = seg[start_idx:start_idx + n].copy()

            # Масштабируем
            scale = self.rng.uniform(5, 30)
            subseg[:, 0] = (subseg[:, 0] - subseg[0, 0]) * scale
            subseg[:, 1] = (subseg[:, 1] - subseg[0, 1]) * scale

            processed_segments.append(subseg)

        if len(processed_segments) < 2:
            return None

        # Направление: от случайной точки к цели
        target_xy = self.target_coords[target_idx]
        distance = self.rng.uniform(50000, 300000)  # 50-300 км
        angle = self.rng.uniform(0, 2 * np.pi)
        start_xy = target_xy + distance * np.array([np.cos(angle), np.sin(angle)])

        # Разделяем общий вектор на сегменты
        total_direction = target_xy - start_xy
        n_segs = len(processed_segments)

        all_points = []
        current_pos = start_xy.copy()
        time_offset = 0.0

        for i, seg in enumerate(processed_segments):
            # Вычисляем направление для этого сегмента
            seg_direction = total_direction / n_segs
            target_len = np.linalg.norm(seg_direction)

            # ВАЖНО: масштаб считаем по ГАБАРИТАМ сегмента,
            # а не по вектору start->end (тот же баг, что был в _warp_to_target:
            # если БПЛА кружил, start->end вектор маленький, а размах сегмента -
            # большой; деление давало scale в тысячи раз и выбрасывало точки
            # на сотни тысяч километров).
            seg_dx = seg[:, 0] - seg[0, 0]
            seg_dy = seg[:, 1] - seg[0, 1]
            seg_vec = np.array([seg_dx[-1], seg_dy[-1]])
            seg_len = np.linalg.norm(seg_vec)
            seg_extent = float(np.max(np.hypot(seg_dx, seg_dy)))

            if seg_extent < 1.0 or target_len < 1e-6:
                continue

            scale_factor = target_len / seg_extent
            # Защита от вырожденных случаев (см. комментарий в _warp_to_target).
            # При корректной формуле max координата всегда = target_len,
            # лимит нужен только чтобы отсечь патологически короткие сегменты.
            if not (0.1 <= scale_factor <= 5000.0):
                continue

            # Поворот - в направлении сегмента, если он есть,
            # иначе - в направлении точки максимального удаления
            if seg_len < 1.0:
                far_idx = int(np.argmax(np.hypot(seg_dx, seg_dy)))
                seg_angle = np.arctan2(seg_dy[far_idx], seg_dx[far_idx])
            else:
                seg_angle = np.arctan2(seg_vec[1], seg_vec[0])
            target_angle = np.arctan2(seg_direction[1], seg_direction[0])
            rot_angle = target_angle - seg_angle

            cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)

            dx = seg_dx * scale_factor
            dy = seg_dy * scale_factor

            x_rot = dx * cos_r - dy * sin_r + current_pos[0]
            y_rot = dx * sin_r + dy * cos_r + current_pos[1]

            z = seg[:, 2].copy()
            z = np.clip(z * self.rng.uniform(2, 5), 0, 3000)

            dt = seg[:, 3] - seg[0, 3] + time_offset

            segment_data = np.stack([x_rot, y_rot, z, dt], axis=1)
            all_points.append(segment_data)

            current_pos = np.array([x_rot[-1], y_rot[-1]])
            time_offset = dt[-1] + config.SYNTHETIC_DT

        if not all_points:
            return None

        trajectory = np.concatenate(all_points, axis=0).astype(np.float32)

        if len(trajectory) < config.MIN_TRAJECTORY_POINTS:
            return None

        # Ресемплинг к равномерному шагу 1 с (см. _resample_aug_trajectory).
        return _resample_aug_trajectory(trajectory)

    # ========================================================================
    # Метод C: Warp to Target
    # ========================================================================

    def _warp_to_target(self, trajectory: np.ndarray,
                        target_idx: int) -> Optional[np.ndarray]:
        """
        Деформирует реальную траекторию к конкретной цели.

        1. Масштабируем траекторию
        2. Вычисляем генеральный курс (от первой к последней точке)
        3. Поворачиваем и растягиваем, чтобы конец совпал с целью
        4. Добавляем участок подхода к цели (последние 5 км — более прямолинейно)
        """
        if len(trajectory) < config.MIN_TRAJECTORY_POINTS:
            return None

        x = trajectory[:, 0].copy()
        y = trajectory[:, 1].copy()
        z = trajectory[:, 2].copy()
        dt = trajectory[:, 3].copy()

        # Цель
        target_xy = self.target_coords[target_idx]

        # Случайная точка старта (50-300 км от цели)
        distance = self.rng.uniform(50000, 300000)
        angle = self.rng.uniform(0, 2 * np.pi)
        start_xy = target_xy + distance * np.array([np.cos(angle), np.sin(angle)])

        # Вектор маршрута
        route_vec = target_xy - start_xy
        route_len = np.linalg.norm(route_vec)

        # Вектор исходной траектории (от первой к последней точке)
        traj_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
        traj_len = np.linalg.norm(traj_vec)

        if route_len < 1e-6:
            return None

        # ВАЖНО: scale считаем по ГАБАРИТАМ траектории, а не по
        # вектору start->end. Многие реальные траектории БПЛА -
        # это кружение: |x[-1]-x[0]| может быть 50 м, а размах
        # самой траектории - 5 км. Использование traj_len в
        # знаменателе давало гигантские scale (тысячи) и координаты
        # уходили на десятки тысяч километров.
        src_extent = float(np.max(np.hypot(x - x[0], y - y[0])))
        if src_extent < 1.0:
            return None

        approach_fraction = 0.9
        scale = (route_len * approach_fraction) / src_extent

        # Защита от вырожденных случаев. При корректной формуле
        # максимальная координата после масштабирования всегда равна
        # route_len * approach_fraction (constant), независимо от scale,
        # поэтому верхний предел нужен только чтобы отсеять патологически
        # короткие траектории (< 10 м размах), которые при таком scale
        # дают неестественный шум.
        if not (0.1 <= scale <= 5000.0):
            return None

        # Угол поворота - берём по тому направлению, в котором двигалась
        # исходная траектория. Если конец почти совпадает с началом
        # (кружение), используем направление к точке максимального удаления.
        if traj_len < 1.0:
            far_idx = int(np.argmax(np.hypot(x - x[0], y - y[0])))
            traj_angle = np.arctan2(y[far_idx] - y[0], x[far_idx] - x[0])
        else:
            traj_angle = np.arctan2(traj_vec[1], traj_vec[0])
        route_angle = np.arctan2(route_vec[1], route_vec[0])
        rot_angle = route_angle - traj_angle

        # Применяем поворот и масштаб
        dx = (x - x[0]) * scale
        dy = (y - y[0]) * scale

        cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
        x_new = dx * cos_r - dy * sin_r + start_xy[0]
        y_new = dx * sin_r + dy * cos_r + start_xy[1]

        # Масштабируем высоту
        z_scale = self.rng.uniform(2, 5)
        z_new = np.clip(z * z_scale, 0, 3000)

        # Масштабируем время пропорционально новой длине маршрута
        # (исходная формула содержала лишние сокращающиеся множители).
        # Базовая скорость ~30 м/с -> новая длительность = новая длина / 30.
        new_path_len = route_len * approach_fraction
        approach_speed_ref = 30.0  # м/с, типичная крейсерская скорость БПЛА
        new_duration = new_path_len / approach_speed_ref
        # Раскладываем равномерно по точкам исходной траектории
        n_pts = len(dt)
        dt_new = np.linspace(0.0, new_duration, n_pts, dtype=np.float32)

        # Добавляем участок подхода к цели (прямолинейный)
        last_point = np.array([x_new[-1], y_new[-1]])
        approach_dist = np.linalg.norm(target_xy - last_point)
        approach_speed = self.rng.uniform(25, 50)
        approach_duration = approach_dist / approach_speed
        approach_points = max(int(approach_duration / config.SYNTHETIC_DT), 5)

        t_approach = np.linspace(0, 1, approach_points)
        x_approach = last_point[0] + t_approach * (target_xy[0] - last_point[0])
        y_approach = last_point[1] + t_approach * (target_xy[1] - last_point[1])

        # Высота: плавное снижение на подходе
        z_approach = np.linspace(z_new[-1], self.rng.uniform(30, 100), approach_points)

        dt_approach = dt_new[-1] + np.arange(1, approach_points + 1) * config.SYNTHETIC_DT

        # GPS-шум на подходе
        x_approach += self.rng.randn(approach_points) * config.GPS_NOISE_SIGMA
        y_approach += self.rng.randn(approach_points) * config.GPS_NOISE_SIGMA

        # Объединяем
        x_full = np.concatenate([x_new, x_approach])
        y_full = np.concatenate([y_new, y_approach])
        z_full = np.concatenate([z_new, z_approach])
        dt_full = np.concatenate([dt_new, dt_approach])

        trajectory = np.stack([x_full, y_full, z_full, dt_full], axis=1).astype(np.float32)

        if len(trajectory) < config.MIN_TRAJECTORY_POINTS:
            return None

        # Ресемплинг к равномерному шагу 1 с (см. _resample_aug_trajectory).
        return _resample_aug_trajectory(trajectory)


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    print("Тестовая аугментация...")

    # Создаём тестовые "реальные" траектории
    rng = np.random.RandomState(42)
    fake_real = []
    for i in range(10):
        n = rng.randint(50, 200)
        t = np.arange(n, dtype=np.float32)
        x = rng.uniform(-1000, 1000) + np.cumsum(rng.randn(n) * 10)
        y = rng.uniform(-1000, 1000) + np.cumsum(rng.randn(n) * 10)
        z = rng.uniform(50, 300, n)
        dt = t * 2.0  # dt=2 сек
        fake_real.append(np.stack([x, y, z, dt], axis=1).astype(np.float32))

    augmentor = TrajectoryAugmentor(seed=42)

    # Генерируем 20 расширенных (для теста)
    results = augmentor.augment_all(fake_real, total_count=20)

    for aug_type in ["scale", "stitch", "warp"]:
        subset = [r for r in results if r["augmentation_type"] == aug_type]
        if subset:
            lengths = [len(r["trajectory"]) for r in subset]
            print(f"\n{aug_type}: {len(subset)} траекторий, "
                  f"длина {np.min(lengths)}-{np.max(lengths)} точек")
