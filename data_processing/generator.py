"""
generator.py — Генерация синтетических боевых траекторий БПЛА (Слой 3).

Генерирует 4 типа маршрутов:
  A. Прямой маршрут (60%) — старт → цель с GPS-шумом и ветровым сносом
  B. Маршрут с уклонением (20%) — через 1-3 промежуточных waypoint
  C. Маршрут с барражированием (10%) — кружение + заход на цель
  D. Ложный маршрут (10%) — смена курса в середине

Параметры генерации задаются в config.py.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, List, Dict, Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.target_database import TargetDatabase


class TrajectoryGenerator:
    """
    Генератор синтетических боевых траекторий БПЛА.

    Использует базу целей для создания реалистичных маршрутов
    от случайной точки старта к конкретной цели.
    """

    def __init__(self, target_db: Optional[TargetDatabase] = None,
                 seed: int = config.RANDOM_SEED):
        """
        Args:
            target_db: база целей (загрузится автоматически если None)
            seed: зерно ГСЧ для воспроизводимости
        """
        self.rng = np.random.RandomState(seed)

        if target_db is None:
            self.target_db = TargetDatabase()
        else:
            self.target_db = target_db

        self.target_coords = self.target_db.get_enu_coords()  # (K, 2)

    def generate_all(self) -> List[Dict]:
        """
        Генерирует полный набор синтетических траекторий.

        Returns:
            Список словарей:
            [
                {
                    "trajectory": np.array (N, 4),  # [x, y, z, dt]
                    "target_id": int,                # индекс цели (0..K-1)
                    "trajectory_type": str,           # "direct"/"evasion"/"loitering"/"deceptive"
                    "start_enu": (float, float),
                    "target_enu": (float, float),
                    "speed_ms": float,
                    "distance_km": float,
                },
                ...
            ]
        """
        all_trajectories = []

        for traj_type, count in config.SYNTHETIC_COUNTS.items():
            print(f"[Generator] Генерация {count} траекторий типа '{traj_type}'...")

            for i in range(count):
                # Случайный выбор цели
                target_idx = self.rng.randint(0, self.target_db.num_targets)
                target_xy = self.target_coords[target_idx]

                # Случайный выбор параметров
                speed_type = self.rng.choice(list(config.SPEED_RANGES.keys()))
                speed = self.rng.uniform(*config.SPEED_RANGES[speed_type])

                alt_type = self.rng.choice(list(config.ALTITUDE_RANGES.keys()))
                alt_range = config.ALTITUDE_RANGES[alt_type]

                dist_type = self.rng.choice(list(config.DISTANCE_RANGES.keys()))
                distance_km = self.rng.uniform(*config.DISTANCE_RANGES[dist_type])

                # Генерация точки старта (случайное направление от цели)
                angle = self.rng.uniform(0, 2 * np.pi)
                distance_m = distance_km * 1000
                start_xy = target_xy + distance_m * np.array([np.cos(angle), np.sin(angle)])

                # Генерация траектории по типу
                if traj_type == "direct":
                    traj = self._generate_direct(start_xy, target_xy, speed, alt_range)
                elif traj_type == "evasion":
                    traj = self._generate_evasion(start_xy, target_xy, speed, alt_range)
                elif traj_type == "loitering":
                    traj = self._generate_loitering(start_xy, target_xy, speed, alt_range)
                elif traj_type == "deceptive":
                    # Для ложного маршрута выбираем вторую (реальную) цель
                    real_target_idx = self.rng.randint(0, self.target_db.num_targets)
                    while real_target_idx == target_idx:
                        real_target_idx = self.rng.randint(0, self.target_db.num_targets)
                    real_target_xy = self.target_coords[real_target_idx]
                    traj = self._generate_deceptive(start_xy, target_xy, real_target_xy,
                                                    speed, alt_range)
                    target_idx = real_target_idx  # Реальная цель — вторая
                else:
                    continue

                if traj is not None and len(traj) >= config.MIN_TRAJECTORY_POINTS:
                    all_trajectories.append({
                        "trajectory": traj,
                        "target_id": target_idx,
                        "trajectory_type": traj_type,
                        "start_enu": (float(start_xy[0]), float(start_xy[1])),
                        "target_enu": (float(self.target_coords[target_idx][0]),
                                       float(self.target_coords[target_idx][1])),
                        "speed_ms": float(speed),
                        "distance_km": float(distance_km),
                    })

        print(f"[Generator] Всего сгенерировано: {len(all_trajectories)} траекторий")
        return all_trajectories

    # ========================================================================
    # Тип A: Прямой маршрут
    # ========================================================================

    def _generate_direct(self, start: np.ndarray, target: np.ndarray,
                         speed: float, alt_range: Tuple[float, float]
                         ) -> Optional[np.ndarray]:
        """
        Прямой маршрут: старт → цель с GPS-шумом и ветровым сносом.

        Генеральный курс на цель с добавлением:
        1. GPS-шума (высокочастотный)
        2. Ветрового сноса (низкочастотный синусоидальный)
        3. Вариации скорости ±15%
        4. Плавного профиля высоты (набор → крейсер → снижение)
        """
        distance = np.linalg.norm(target - start)
        if distance < 100:
            return None

        duration = distance / speed
        n_points = int(duration / config.SYNTHETIC_DT)
        if n_points < config.MIN_TRAJECTORY_POINTS:
            return None

        t = np.linspace(0, 1, n_points)

        # Базовый прямой маршрут
        direction = (target - start) / distance
        perpendicular = np.array([-direction[1], direction[0]])

        # Позиция вдоль маршрута (с вариацией скорости)
        speed_profile = 1.0 + config.SPEED_VARIATION * np.sin(
            2 * np.pi * self.rng.uniform(1, 5) * t + self.rng.uniform(0, 2 * np.pi)
        )
        # Кумулятивный прогресс (нормируем чтобы закончить в 1.0)
        progress = np.cumsum(speed_profile)
        progress = progress / progress[-1]

        base_xy = start[np.newaxis, :] + progress[:, np.newaxis] * (target - start)[np.newaxis, :]

        # Ветровой снос (низкочастотный)
        n_waves = self.rng.uniform(1, 4)
        wind_amp = self.rng.uniform(*config.WIND_DRIFT_AMPLITUDE)
        wind = wind_amp * np.sin(2 * np.pi * n_waves * t + self.rng.uniform(0, 2 * np.pi))
        wind_offset = wind[:, np.newaxis] * perpendicular[np.newaxis, :]

        # GPS-шум
        gps_noise = self.rng.randn(n_points, 2) * config.GPS_NOISE_SIGMA

        # Итоговые XY
        xy = base_xy + wind_offset + gps_noise

        # Высота: профиль набор→крейсер→снижение
        z = self._generate_altitude_profile(n_points, alt_range)

        # Время
        dt = np.arange(n_points, dtype=np.float32) * config.SYNTHETIC_DT

        trajectory = np.stack([xy[:, 0], xy[:, 1], z, dt], axis=1).astype(np.float32)
        return trajectory

    # ========================================================================
    # Тип B: Маршрут с уклонением
    # ========================================================================

    def _generate_evasion(self, start: np.ndarray, target: np.ndarray,
                          speed: float, alt_range: Tuple[float, float]
                          ) -> Optional[np.ndarray]:
        """
        Маршрут с уклонением: старт → waypoint1 → ... → waypointN → цель.

        1-3 промежуточных waypoint смещены от прямой линии.
        Между ними — сглаженные кубические сплайны.
        Финальный участок (последние 10-20 км) — более прямой.
        """
        distance = np.linalg.norm(target - start)
        if distance < 100:
            return None

        direction = (target - start) / distance
        perpendicular = np.array([-direction[1], direction[0]])

        # Генерируем 1-3 промежуточных waypoints
        num_waypoints = self.rng.randint(1, 4)
        waypoints = [start]

        for i in range(num_waypoints):
            # Прогресс вдоль маршрута (не в самом начале и не в самом конце)
            progress = (i + 1) / (num_waypoints + 1)
            base_point = start + progress * (target - start)

            # Случайное смещение перпендикулярно маршруту
            lateral_offset = self.rng.uniform(-0.3, 0.3) * distance
            waypoint = base_point + lateral_offset * perpendicular

            waypoints.append(waypoint)

        waypoints.append(target)
        waypoints = np.array(waypoints)

        # Сплайн через waypoints
        num_wp = len(waypoints)
        wp_t = np.linspace(0, 1, num_wp)

        duration = distance / speed
        n_points = int(duration / config.SYNTHETIC_DT)
        if n_points < config.MIN_TRAJECTORY_POINTS:
            return None

        t = np.linspace(0, 1, n_points)

        cs_x = CubicSpline(wp_t, waypoints[:, 0])
        cs_y = CubicSpline(wp_t, waypoints[:, 1])

        xy = np.stack([cs_x(t), cs_y(t)], axis=1)

        # GPS-шум
        xy += self.rng.randn(n_points, 2) * config.GPS_NOISE_SIGMA

        z = self._generate_altitude_profile(n_points, alt_range)
        dt = np.arange(n_points, dtype=np.float32) * config.SYNTHETIC_DT

        trajectory = np.stack([xy[:, 0], xy[:, 1], z, dt], axis=1).astype(np.float32)
        return trajectory

    # ========================================================================
    # Тип C: Маршрут с барражированием
    # ========================================================================

    def _generate_loitering(self, start: np.ndarray, target: np.ndarray,
                            speed: float, alt_range: Tuple[float, float]
                            ) -> Optional[np.ndarray]:
        """
        Маршрут с барражированием:
        1. Полёт к зоне барражирования (промежуточная точка)
        2. Кружение над зоной (1-3 круга)
        3. Прямой заход на цель
        """
        distance = np.linalg.norm(target - start)
        if distance < 100:
            return None

        direction = (target - start) / distance

        # Зона барражирования: 30-70% пути до цели
        loiter_progress = self.rng.uniform(0.3, 0.7)
        loiter_center = start + loiter_progress * (target - start)

        # Смещение зоны барражирования от прямой линии
        perpendicular = np.array([-direction[1], direction[0]])
        loiter_center += self.rng.uniform(-0.15, 0.15) * distance * perpendicular

        # Радиус кружения: 500-3000 м
        loiter_radius = self.rng.uniform(500, 3000)
        num_circles = self.rng.uniform(1, 3)

        # Фаза 1: полёт к зоне барражирования
        dist_to_loiter = np.linalg.norm(loiter_center - start)
        phase1_duration = dist_to_loiter / speed
        phase1_points = max(int(phase1_duration / config.SYNTHETIC_DT), 10)

        t1 = np.linspace(0, 1, phase1_points)
        phase1_xy = start[np.newaxis, :] + t1[:, np.newaxis] * (loiter_center - start)[np.newaxis, :]

        # Фаза 2: кружение
        loiter_duration = 2 * np.pi * loiter_radius * num_circles / speed
        phase2_points = max(int(loiter_duration / config.SYNTHETIC_DT), 20)

        angles = np.linspace(0, 2 * np.pi * num_circles, phase2_points)
        phase2_xy = loiter_center[np.newaxis, :] + loiter_radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

        # Фаза 3: заход на цель
        dist_to_target = np.linalg.norm(target - loiter_center)
        phase3_duration = dist_to_target / speed
        phase3_points = max(int(phase3_duration / config.SYNTHETIC_DT), 10)

        # Начало фазы 3 — последняя точка кружения
        loiter_exit = phase2_xy[-1]
        t3 = np.linspace(0, 1, phase3_points)
        phase3_xy = loiter_exit[np.newaxis, :] + t3[:, np.newaxis] * (target - loiter_exit)[np.newaxis, :]

        # Объединяем фазы
        xy = np.concatenate([phase1_xy, phase2_xy, phase3_xy], axis=0)
        n_points = len(xy)

        # GPS-шум
        xy += self.rng.randn(n_points, 2) * config.GPS_NOISE_SIGMA

        z = self._generate_altitude_profile(n_points, alt_range)
        dt = np.arange(n_points, dtype=np.float32) * config.SYNTHETIC_DT

        trajectory = np.stack([xy[:, 0], xy[:, 1], z, dt], axis=1).astype(np.float32)
        return trajectory

    # ========================================================================
    # Тип D: Ложный маршрут
    # ========================================================================

    def _generate_deceptive(self, start: np.ndarray, fake_target: np.ndarray,
                            real_target: np.ndarray,
                            speed: float, alt_range: Tuple[float, float]
                            ) -> Optional[np.ndarray]:
        """
        Ложный маршрут:
        1. Начало движения к ложной цели (30-60% пути)
        2. Плавная смена курса
        3. Движение к реальной цели

        Метка = реальная цель (куда дрон в итоге прилетит).
        """
        dist_to_fake = np.linalg.norm(fake_target - start)
        if dist_to_fake < 100:
            return None

        # Момент смены курса: 30-60% пути к ложной цели
        switch_progress = self.rng.uniform(0.3, 0.6)
        switch_point = start + switch_progress * (fake_target - start)

        # Фаза 1: к ложной цели
        dist_phase1 = np.linalg.norm(switch_point - start)
        phase1_duration = dist_phase1 / speed
        phase1_points = max(int(phase1_duration / config.SYNTHETIC_DT), 10)

        t1 = np.linspace(0, 1, phase1_points)
        phase1_xy = start[np.newaxis, :] + t1[:, np.newaxis] * (switch_point - start)[np.newaxis, :]

        # Фаза 2: к реальной цели (через сплайн для плавности)
        dist_phase2 = np.linalg.norm(real_target - switch_point)
        phase2_duration = dist_phase2 / speed
        phase2_points = max(int(phase2_duration / config.SYNTHETIC_DT), 10)

        # Промежуточная точка для сплайна (плавный поворот)
        mid_point = switch_point + 0.3 * (real_target - switch_point)
        perpendicular = np.array([-(real_target[1] - switch_point[1]),
                                   real_target[0] - switch_point[0]])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        mid_point += self.rng.uniform(-0.1, 0.1) * dist_phase2 * perpendicular

        wp = np.array([switch_point, mid_point, real_target])
        wp_t = np.array([0, 0.3, 1.0])

        t2 = np.linspace(0, 1, phase2_points)
        cs_x = CubicSpline(wp_t, wp[:, 0])
        cs_y = CubicSpline(wp_t, wp[:, 1])
        phase2_xy = np.stack([cs_x(t2), cs_y(t2)], axis=1)

        # Объединяем
        xy = np.concatenate([phase1_xy, phase2_xy], axis=0)
        n_points = len(xy)

        # GPS-шум
        xy += self.rng.randn(n_points, 2) * config.GPS_NOISE_SIGMA

        z = self._generate_altitude_profile(n_points, alt_range)
        dt = np.arange(n_points, dtype=np.float32) * config.SYNTHETIC_DT

        trajectory = np.stack([xy[:, 0], xy[:, 1], z, dt], axis=1).astype(np.float32)
        return trajectory

    # ========================================================================
    # Вспомогательные функции
    # ========================================================================

    def _generate_altitude_profile(self, n_points: int,
                                   alt_range: Tuple[float, float]) -> np.ndarray:
        """
        Генерирует реалистичный профиль высоты:
        Набор высоты (10%) → крейсерский полёт (70%) → снижение (20%)
        + небольшие вариации.
        """
        t = np.linspace(0, 1, n_points)

        min_alt, max_alt = alt_range
        cruise_alt = self.rng.uniform(min_alt, max_alt)
        start_alt = self.rng.uniform(0, min_alt)
        end_alt = self.rng.uniform(0, min_alt * 0.5)

        # Три фазы
        phase1_end = 0.1   # набор высоты
        phase2_end = 0.8   # крейсерский
        # phase3: снижение

        z = np.zeros(n_points, dtype=np.float32)
        for i, ti in enumerate(t):
            if ti < phase1_end:
                # Набор: синусоидальный профиль (плавный)
                progress = ti / phase1_end
                z[i] = start_alt + (cruise_alt - start_alt) * (1 - np.cos(np.pi * progress)) / 2
            elif ti < phase2_end:
                # Крейсер: стабильная высота с небольшими вариациями
                z[i] = cruise_alt
            else:
                # Снижение
                progress = (ti - phase2_end) / (1 - phase2_end)
                z[i] = cruise_alt + (end_alt - cruise_alt) * (1 - np.cos(np.pi * progress)) / 2

        # Добавляем небольшие вариации высоты
        z += self.rng.randn(n_points) * 5.0  # σ = 5 м

        # Ограничиваем отрицательные высоты
        z = np.maximum(z, 0)

        return z


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    print("Тестовая генерация...")

    gen = TrajectoryGenerator(seed=42)

    # Генерируем небольшой тестовый набор
    # Переопределяем счётчики для теста
    original_counts = config.SYNTHETIC_COUNTS.copy()
    config.SYNTHETIC_COUNTS = {
        "direct": 10,
        "evasion": 5,
        "loitering": 3,
        "deceptive": 2,
    }

    trajectories = gen.generate_all()

    # Восстанавливаем
    config.SYNTHETIC_COUNTS = original_counts

    # Статистика
    for ttype in ["direct", "evasion", "loitering", "deceptive"]:
        subset = [t for t in trajectories if t["trajectory_type"] == ttype]
        if subset:
            lengths = [len(t["trajectory"]) for t in subset]
            dists = [t["distance_km"] for t in subset]
            print(f"\n{ttype}: {len(subset)} траекторий")
            print(f"  Длина: {np.min(lengths)}-{np.max(lengths)} точек")
            print(f"  Расстояние: {np.min(dists):.1f}-{np.max(dists):.1f} км")

    # Пример одной траектории
    if trajectories:
        sample = trajectories[0]
        print(f"\nПример траектории:")
        print(f"  Тип: {sample['trajectory_type']}")
        print(f"  Цель: {gen.target_db.targets[sample['target_id']]['name']}")
        print(f"  Точек: {len(sample['trajectory'])}")
        print(f"  Расстояние: {sample['distance_km']:.1f} км")
        print(f"  Скорость: {sample['speed_ms']:.1f} м/с ({sample['speed_ms']*3.6:.0f} км/ч)")
