"""
parser.py — Парсинг CSV-логов полётов, фильтрация GPS-аномалий,
преобразование в локальную метрическую систему ENU.

Входные данные: CSV-файлы телеметрии формата OpenTX
Выходные данные: numpy-массивы (N, 4) = [x_enu, y_enu, z, dt]
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import warnings

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================================
# ENU-преобразование (из WGS84 градусов в локальные метры)
# ============================================================================

def geodetic_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                    lat0: float = config.ENU_ORIGIN_LAT,
                    lon0: float = config.ENU_ORIGIN_LON,
                    alt0: float = config.ENU_ORIGIN_ALT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Преобразование координат из WGS84 (градусы) в ENU (метры).

    ENU (East-North-Up) — локальная метрическая система координат:
    - E (East): метры на восток от начала координат
    - N (North): метры на север от начала координат
    - U (Up): метры вверх от начала координат

    Используем упрощённую формулу (подходит для расстояний до 500 км):
    - 1 градус широты ≈ 111320 м (постоянно)
    - 1 градус долготы ≈ 111320 * cos(lat) м (зависит от широты)

    Args:
        lat, lon, alt: массивы координат в градусах/метрах
        lat0, lon0, alt0: координаты начала координат (опорная точка)

    Returns:
        e, n, u: массивы ENU-координат в метрах
    """
    lat_rad = np.radians(lat0)

    # Метров на градус
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)

    e = (lon - lon0) * meters_per_deg_lon  # East
    n = (lat - lat0) * meters_per_deg_lat  # North
    u = alt - alt0                          # Up

    return e, n, u


def enu_to_geodetic(e: np.ndarray, n: np.ndarray, u: np.ndarray,
                    lat0: float = config.ENU_ORIGIN_LAT,
                    lon0: float = config.ENU_ORIGIN_LON,
                    alt0: float = config.ENU_ORIGIN_ALT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Обратное преобразование: ENU (метры) → WGS84 (градусы).
    """
    lat_rad = np.radians(lat0)

    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)

    lon = lon0 + e / meters_per_deg_lon
    lat = lat0 + n / meters_per_deg_lat
    alt = alt0 + u

    return lat, lon, alt


# ============================================================================
# Парсинг одного CSV-файла
# ============================================================================

def parse_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Парсит CSV-файл телеметрии OpenTX.

    Обрабатывает два формата:
    1. С колонкой GPS (строка "lat lon") + Alt(m) + Time
    2. С готовыми колонками X, Y, Z, Time

    Returns:
        DataFrame с колонками [lat, lon, alt, time_sec] или None при ошибке
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        warnings.warn(f"Не удалось прочитать {filepath}: {e}")
        return None

    # Формат 1: уже обработанные данные (X, Y, Z, Time)
    if all(col in df.columns for col in ["X", "Y", "Z", "Time"]):
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f", errors="coerce")
        df = df.dropna(subset=["X", "Y", "Z", "Time"])
        if len(df) == 0:
            return None

        # X и Y здесь — широта и долгота (legacy формат)
        result = pd.DataFrame({
            "lat": pd.to_numeric(df["X"], errors="coerce"),
            "lon": pd.to_numeric(df["Y"], errors="coerce"),
            "alt": pd.to_numeric(df["Z"], errors="coerce"),
            "time_sec": _time_to_seconds(df["Time"]),
        })
        return result.dropna()

    # Формат 2: стандартный OpenTX с GPS
    if "GPS" not in df.columns:
        warnings.warn(f"Нет колонки GPS в {filepath}")
        return None

    # Парсинг GPS: "lat lon" (строка с пробелом)
    df["GPS"] = df["GPS"].astype(str).str.strip().str.replace(",", " ")
    gps_split = df["GPS"].str.split(r"\s+", expand=True)

    if gps_split.shape[1] < 2:
        warnings.warn(f"Некорректный формат GPS в {filepath}")
        return None

    # Определяем колонку высоты
    alt_col = None
    for col_name in ["Alt(m)", "Alt", "Altitude"]:
        if col_name in df.columns:
            alt_col = col_name
            break

    if alt_col is None:
        warnings.warn(f"Нет колонки высоты в {filepath}")
        return None

    # Парсинг времени
    if "Time" not in df.columns:
        warnings.warn(f"Нет колонки Time в {filepath}")
        return None

    time_parsed = pd.to_datetime(df["Time"], format="%H:%M:%S.%f", errors="coerce")

    result = pd.DataFrame({
        "lat": pd.to_numeric(gps_split[0], errors="coerce"),
        "lon": pd.to_numeric(gps_split[1], errors="coerce"),
        "alt": pd.to_numeric(df[alt_col], errors="coerce"),
        "time_sec": _time_to_seconds(time_parsed),
    })

    return result.dropna()


def _time_to_seconds(time_series: pd.Series) -> pd.Series:
    """Преобразует pd.Timestamp в секунды от начала записи."""
    # Если уже числовой тип
    if pd.api.types.is_numeric_dtype(time_series):
        return time_series - time_series.min()

    # Datetime → секунды
    epoch = time_series.min()
    return (time_series - epoch).dt.total_seconds()


# ============================================================================
# Фильтрация GPS-аномалий
# ============================================================================

def filter_gps_anomalies(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                         time_sec: np.ndarray,
                         max_step_m: float = config.GPS_MAX_STEP_METERS
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Удаляет GPS-аномалии: точки со скачком > max_step_m от предыдущей.

    Также фильтрует:
    - Нереальные высоты (< -100 м или > 10000 м)
    - Нулевые координаты (lat=0 или lon=0)
    - Дублирующиеся временные метки

    Args:
        lat, lon, alt, time_sec: массивы координат
        max_step_m: максимальный допустимый скачок (метры)

    Returns:
        Отфильтрованные массивы (lat, lon, alt, time_sec)
    """
    n = len(lat)
    if n < 2:
        return lat, lon, alt, time_sec

    # Маска валидных точек
    valid = np.ones(n, dtype=bool)

    # 1. Фильтр нулевых/аномальных координат
    valid &= (lat != 0) & (lon != 0)
    valid &= (np.abs(lat) <= 90) & (np.abs(lon) <= 180)

    # 2. Фильтр нереальных высот
    valid &= (alt > -100) & (alt < 10000)

    # 3. Фильтр скачков GPS
    # Вычисляем расстояние между соседними точками
    lat_rad = np.radians(lat)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(np.median(lat[valid]))) if valid.sum() > 0 else 111320.0 * 0.5

    for i in range(1, n):
        if not valid[i]:
            continue

        # Находим последнюю валидную точку
        prev = -1
        for j in range(i - 1, -1, -1):
            if valid[j]:
                prev = j
                break

        if prev == -1:
            continue

        dx = (lon[i] - lon[prev]) * meters_per_deg_lon
        dy = (lat[i] - lat[prev]) * meters_per_deg_lat
        dz = alt[i] - alt[prev]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Максимальный скачок пропорционален времени
        dt = max(time_sec[i] - time_sec[prev], 0.1)
        max_allowed = max_step_m * max(dt / 1.0, 1.0)  # Масштабируем по dt

        if dist > max_allowed:
            valid[i] = False

    return lat[valid], lon[valid], alt[valid], time_sec[valid]


# ============================================================================
# Ресемплирование
# ============================================================================

def resample_trajectory(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        time_sec: np.ndarray,
                        target_dt: float = config.RESAMPLE_DT
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ресемплирование траектории на фиксированный шаг по времени.

    Используем линейную интерполяцию для получения равномерных
    временных отсчётов с шагом target_dt секунд.

    Args:
        x, y, z: координаты (ENU метры)
        time_sec: временные метки (секунды от начала)
        target_dt: целевой шаг (секунды)

    Returns:
        Ресемплированные массивы (x, y, z, time_sec)
    """
    if len(time_sec) < 2:
        return x, y, z, time_sec

    t_start = time_sec[0]
    t_end = time_sec[-1]

    # Новые равномерные временные метки
    new_time = np.arange(t_start, t_end, target_dt)

    if len(new_time) < 2:
        return x, y, z, time_sec

    # Линейная интерполяция
    new_x = np.interp(new_time, time_sec, x)
    new_y = np.interp(new_time, time_sec, y)
    new_z = np.interp(new_time, time_sec, z)

    return new_x, new_y, new_z, new_time


# ============================================================================
# Полный пайплайн обработки одного файла
# ============================================================================

def process_single_file(filepath: str) -> Optional[np.ndarray]:
    """
    Полный пайплайн обработки одного CSV-файла.

    Шаги:
    1. Парсинг CSV → lat, lon, alt, time_sec
    2. Фильтрация GPS-аномалий
    3. Преобразование в ENU (метры)
    4. Ресемплирование на фиксированный dt

    Args:
        filepath: путь к CSV-файлу

    Returns:
        numpy array (N, 4) = [x_enu, y_enu, z_enu, dt] или None
        где dt — секунды от начала записи
    """
    # 1. Парсинг
    df = parse_csv(filepath)
    if df is None or len(df) < config.MIN_TRAJECTORY_POINTS:
        return None

    lat = df["lat"].values
    lon = df["lon"].values
    alt = df["alt"].values
    time_sec = df["time_sec"].values

    # 2. Фильтрация GPS-аномалий
    lat, lon, alt, time_sec = filter_gps_anomalies(lat, lon, alt, time_sec)

    if len(lat) < config.MIN_TRAJECTORY_POINTS:
        return None

    # 3. Преобразование в ENU
    e, n, u = geodetic_to_enu(lat, lon, alt)

    # 4. Ресемплирование
    e, n, u, time_sec = resample_trajectory(e, n, u, time_sec)

    if len(e) < config.MIN_TRAJECTORY_POINTS:
        return None

    # Время от начала записи
    dt = time_sec - time_sec[0]

    # Результат: (N, 4) = [x, y, z, dt]
    trajectory = np.stack([e, n, u, dt], axis=1).astype(np.float32)

    return trajectory


# ============================================================================
# Обработка всех файлов
# ============================================================================

def process_all_flights(source_dirs: Optional[List[str]] = None) -> List[Dict]:
    """
    Обрабатывает все CSV-файлы из указанных директорий.

    Args:
        source_dirs: список директорий с CSV-файлами.
                     По умолчанию — config.FLIGHT_LOGS_SOURCE

    Returns:
        Список словарей:
        [
            {
                "filename": "Dolphin_001.csv",
                "model": "Dolphin",
                "trajectory": np.array (N, 4),
                "n_points": N,
                "duration_sec": float,
                "max_distance_m": float,
            },
            ...
        ]
    """
    if source_dirs is None:
        source_dirs = [config.FLIGHT_LOGS_SOURCE]

    # Собираем все CSV
    csv_files = []
    for d in source_dirs:
        csv_files.extend(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))

    # Дедупликация по имени файла. Один и тот же лог может лежать в нескольких
    # источниках (копии папок с логами). Повторный разбор давал дубликаты
    # траекторий, а при случайном разбиении один и тот же полёт попадал
    # и в train, и в test - утечка данных.
    unique = {}
    for path in sorted(csv_files):
        name = os.path.basename(path)
        if name not in unique:
            unique[name] = path
    n_duplicates = len(csv_files) - len(unique)
    csv_files = list(unique.values())

    msg = f"[Parser] Найдено {len(csv_files)} уникальных CSV-файлов"
    if n_duplicates:
        msg += f" (пропущено дубликатов по имени: {n_duplicates})"
    print(msg)

    results = []
    skipped = 0

    for filepath in sorted(csv_files):
        trajectory = process_single_file(filepath)

        if trajectory is None:
            skipped += 1
            continue

        # Определяем модель БПЛА по имени файла
        filename = os.path.basename(filepath)
        model_name = _detect_model(filename)

        # Статистика
        duration = trajectory[-1, 3]  # dt последней точки
        positions = trajectory[:, :2]  # x, y
        start = positions[0]
        distances = np.sqrt(np.sum((positions - start)**2, axis=1))
        max_dist = np.max(distances)

        results.append({
            "filename": filename,
            "model": model_name,
            "trajectory": trajectory,
            "n_points": len(trajectory),
            "duration_sec": float(duration),
            "max_distance_m": float(max_dist),
        })

    print(f"[Parser] Обработано: {len(results)} траекторий, пропущено: {skipped}")
    _print_stats(results)

    return results


def _detect_model(filename: str) -> str:
    """Определяет модель БПЛА по имени файла."""
    filename_lower = filename.lower()

    models = {
        "chimera": "Chimera7",
        "dolphin": "Dolphin",
        "f450": "F450",
        "godiscover": "GoDiscover",
        "go_discover": "GoDiscover",
        "sim": "Sim",
        "z84": "Z84",
        "s1100": "S1100",
        "t1": "T1_VTOL",
        "vtol": "T1_VTOL",
        "mini_talon": "Mini_Talon",
        "minitalon": "Mini_Talon",
        "chernika": "Chernika",
    }

    for key, model in models.items():
        if key in filename_lower:
            return model

    return "Unknown"


def _print_stats(results: List[Dict]):
    """Выводит статистику по обработанным данным."""
    if not results:
        print("[Parser] Нет данных для статистики")
        return

    n_points = [r["n_points"] for r in results]
    durations = [r["duration_sec"] for r in results]
    distances = [r["max_distance_m"] for r in results]
    models = {}
    for r in results:
        models[r["model"]] = models.get(r["model"], 0) + 1

    print(f"\n[Parser] === Статистика ===")
    print(f"  Траекторий:    {len(results)}")
    print(f"  Точек:         медиана={np.median(n_points):.0f}, "
          f"мин={np.min(n_points)}, макс={np.max(n_points)}")
    print(f"  Длительность:  медиана={np.median(durations)/60:.1f} мин, "
          f"макс={np.max(durations)/60:.1f} мин")
    print(f"  Удаление:      медиана={np.median(distances):.0f} м, "
          f"макс={np.max(distances):.0f} м")
    print(f"  Модели БПЛА:")
    for model, count in sorted(models.items(), key=lambda x: -x[1]):
        print(f"    {model}: {count}")
    print()


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    import sys

    # Ищем CSV файлы
    source = config.FLIGHT_LOGS_SOURCE
    print(f"Источник данных: {source}")

    if not os.path.exists(source):
        print(f"Директория не найдена: {source}")
        sys.exit(1)

    results = process_all_flights()

    if results:
        # Сохраняем обработанные данные
        config.ensure_dirs()
        save_path = os.path.join(config.PROCESSED_DATA_DIR, "real_trajectories.npz")

        trajectories = [r["trajectory"] for r in results]
        models = [r["model"] for r in results]
        filenames = [r["filename"] for r in results]

        np.savez(save_path,
                 trajectories=np.array(trajectories, dtype=object),
                 models=np.array(models),
                 filenames=np.array(filenames))

        print(f"Сохранено в {save_path}")
