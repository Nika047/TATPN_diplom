import os
import glob
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset

import globals


def load_and_extract(file):
    """Читает файл и возвращает DataFrame с X,Y,Z,Time"""
    df = pd.read_csv(file)

    # Если уже есть готовые X/Y/Z/Time
    if all(col in df.columns for col in ["X", "Y", "Z", "Time"]):
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f", errors="coerce")
        df = df.dropna(subset=["X", "Y", "Z", "Time"])
        return df[["X", "Y", "Z", "Time"]]

    # Если есть GPS — парсим
    if "GPS" in df.columns and "Alt(m)" in df.columns:
        df["GPS"] = df["GPS"].astype(str).str.strip().str.replace(",", " ")
        gps_split = df["GPS"].str.split(r"\s+", expand=True)
        if gps_split.shape[1] < 2:
            return None

        df["X"] = pd.to_numeric(gps_split[0], errors="coerce")
        df["Y"] = pd.to_numeric(gps_split[1], errors="coerce")
        df["Z"] = pd.to_numeric(df["Alt(m)"], errors="coerce")
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f", errors="coerce")

        df = df.dropna(subset=["X", "Y", "Z", "Time"])
        return df[["X", "Y", "Z", "Time"]]

    return None

def log_file_stats(files, label):
    total_points = 0
    for f in files:
        try:
            total_points += len(pd.read_csv(f))
        except:
            pass
    print(f"[INFO] {label}: {len(files)} файлов, {total_points} точек\n")


def create_datasets_with_generated_data(
        source_dir="flight_logs_source",
        aug_dir="flight_logs_augmented",
        gen_data_dir="flight_logs_generated/data",
        work_dir="flight_logs_generated",
        seq_len=globals.seq_len,
        pred_horizon=globals.pred_horizon,
        dt=globals.dt,
        test_size=globals.test_size,
        val_size=globals.val_size,
        random_state=globals.random_state,
        num_generated_files=globals.num_generated_files
):
    # os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(gen_data_dir, exist_ok=True)

    def generate_new_trajectories(existing_files, num_to_generate):
        """Генерирует новые траектории на основе существующих с помощью генеративной модели"""
        generated_files = []

        # Загружаем все существующие траектории для обучения генератора
        all_trajectories = []
        trajectory_sources = []

        for file in existing_files:
            df = load_and_extract(file)
            if df is not None and not df.empty:
                traj = df[["X", "Y", "Z"]].values
                if len(traj) >= globals.min_total_points:  # Минимум 5 точек
                    all_trajectories.append(traj)
                    trajectory_sources.append(os.path.basename(file))

        if len(all_trajectories) < 2:
            print(f"[WARNING] Недостаточно траекторий для генерации. Найдено: {len(all_trajectories)}, требуется минимум 2\n")
            return generated_files

        print(f"[INFO] Загружено {len(all_trajectories)} траекторий для обучения генератора\n")

        for i in range(num_to_generate):
            try:
                base_traj_idx = np.random.randint(0, len(all_trajectories))
                base_traj = all_trajectories[base_traj_idx]
                base_source = trajectory_sources[base_traj_idx]

                if np.random.random() < 0.3:
                    new_traj = interpolate_with_noise(base_traj)
                    method_name = "interpolate"
                    source_info = f"from_{os.path.splitext(base_source)[0]}"

                elif np.random.random() < 0.6:
                    second_traj_idx = np.random.randint(0, len(all_trajectories))
                    while second_traj_idx == base_traj_idx:  # Чтобы не совпали
                        second_traj_idx = np.random.randint(0, len(all_trajectories))

                    second_traj = all_trajectories[second_traj_idx]
                    second_source = trajectory_sources[second_traj_idx]
                    new_traj = combine_trajectories(base_traj, second_traj)
                    method_name = "combine"
                    source_info = f"{os.path.splitext(base_source)[0]}_{os.path.splitext(second_source)[0]}"

                else:
                    new_traj = apply_affine_transformations(base_traj)
                    method_name = "affine"
                    source_info = f"from_{os.path.splitext(base_source)[0]}"

                time_points = pd.date_range(
                    start=pd.Timestamp.now(),
                    periods=len(new_traj),
                    freq=f"{dt}s"
                )

                gen_df = pd.DataFrame({
                    "X": new_traj[:, 0],
                    "Y": new_traj[:, 1],
                    "Z": new_traj[:, 2],
                    "Date": time_points.strftime("%Y-%m-%d"),
                    "Time": time_points.strftime("%H:%M:%S.%f")
                })

                gen_filename = f"gen_{method_name}_{source_info}_{i:04d}.csv"
                gen_path = os.path.join(gen_data_dir, gen_filename)
                gen_df.to_csv(gen_path, index=False)
                generated_files.append(gen_path)

                print(f"[INFO] Сгенерирован файл: {gen_filename}\n")

            except Exception as e:
                print(f"[ERROR] Ошибка генерации траектории {i}: {e}\n")
                continue

        return generated_files

    def interpolate_with_noise(trajectory, noise_factor=0.0001):
        """Интерполяция траектории с добавлением шума"""
        orig_len = len(trajectory)
        new_len = int(orig_len * np.random.uniform(1.5, 2.0))

        t_old = np.linspace(0, 1, orig_len)
        t_new = np.linspace(0, 1, new_len)

        new_traj = []
        for dim in range(3):
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(t_old, trajectory[:, dim])
            interpolated = cs(t_new)
            noise = np.random.normal(0, noise_factor, new_len)
            new_traj.append(interpolated + noise)

        return np.column_stack(new_traj)

    def combine_trajectories(traj1, traj2):
        """Комбинирует две траектории"""
        split_idx1 = np.random.randint(1, len(traj1) - 1)
        split_idx2 = np.random.randint(1, len(traj2) - 1)

        first_part = traj1[:split_idx1]
        second_part = traj2[split_idx2:]

        transition_len = min(5, len(first_part), len(second_part))
        if transition_len > 1:
            alpha = np.linspace(0, 1, transition_len).reshape(-1, 1)
            transition = first_part[-transition_len:] * (1 - alpha) + second_part[:transition_len] * alpha
            combined = np.vstack([first_part[:-transition_len], transition, second_part[transition_len:]])
        else:
            combined = np.vstack([first_part, second_part])

        return combined

    def apply_affine_transformations(trajectory):
        """Применяет аффинные преобразования к траектории"""
        scale = np.random.uniform(0.8, 1.2, 3)
        translation = np.random.uniform(-0.001, 0.001, 3)
        rotation_angle = np.random.uniform(-0.3, 0.3)

        transformed = trajectory.copy()
        transformed = transformed * scale
        transformed = transformed + translation

        cos_angle, sin_angle = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
                                    [sin_angle, cos_angle, 0],
                                    [0, 0, 1]])
        transformed = transformed @ rotation_matrix.T

        return transformed

    # ---------- 1. Исходные ----------
    source_files = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
    log_file_stats(source_files, "Исходные")

    # ---------- 3. Генерация с помощью генеративной модели ----------
    all_existing_files = source_files
    print(f"[INFO] Запуск генеративной модели на основе {len(all_existing_files)} файлов...\n")

    gen_files = generate_new_trajectories(all_existing_files, num_generated_files)
    log_file_stats(gen_files, "Сгенерированные генеративной моделью")

    # ---------- 4. Деление на train/val/test ----------
    all_files = source_files + gen_files
    log_file_stats(all_files, "Всего перед делением")

    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=random_state)
    train_files, val_files = train_test_split(train_files, test_size=val_size / (1 - test_size),
                                              random_state=random_state)

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_dir = os.path.join(work_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            shutil.copy(f, split_dir)
        log_file_stats(files, f"{split_name} выборка")

    # ---------- 5. Создание TensorDataset ----------
    train_ds = files_to_dataset(train_files, pred_horizon)
    val_ds = files_to_dataset(val_files, pred_horizon)
    test_ds = files_to_dataset(test_files, pred_horizon)

    print(f"[INFO] train_ds: {len(train_ds)} сэмплов\n")
    print(f"[INFO] val_ds: {len(val_ds)} сэмплов\n")
    print(f"[INFO] test_ds: {len(test_ds)} сэмплов\n")

    return train_ds, val_ds, test_ds


def files_to_dataset(files, pred_horizon=20, min_input_points=10):
    sequences, targets = [], []
    for f in files:
        df = load_and_extract(f)
        if df is None or df.empty:
            continue

        arr = df[["X", "Y", "Z", "Time"]].copy()
        arr["Time"] = (arr["Time"] - arr["Time"].iloc[0]).dt.total_seconds()
        arr = arr.to_numpy(dtype=np.float32)

        # Используем всю траекторию кроме последних pred_horizon точек как вход
        # и последние pred_horizon точек как цель
        if len(arr) >= pred_horizon + min_input_points:
            seq_x = arr[:-pred_horizon]  # Все точки кроме последних pred_horizon
            seq_y = arr[-pred_horizon:, :3]  # Последние pred_horizon точек (только координаты)

            sequences.append(torch.tensor(seq_x))
            targets.append(torch.tensor(seq_y))

    if not sequences:
        return TensorDataset(torch.empty(0), torch.empty(0))

    # Паддинг до максимальной длины
    max_len = max(seq.shape[0] for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            padding = torch.zeros(max_len - seq.shape[0], seq.shape[1], dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
            padded_sequences.append(padded_seq)
        else:
            padded_sequences.append(seq)

    return TensorDataset(torch.stack(padded_sequences), torch.stack(targets))

def load_datasets_from_folders(
    base_dir="flight_logs_generated",
    pred_horizon=20,
    min_input_points=10
):
    # Загружаем файлы из папок
    train_files = sorted(glob.glob(os.path.join(base_dir, "train", "*.csv")))
    val_files   = sorted(glob.glob(os.path.join(base_dir, "val", "*.csv")))
    test_files  = sorted(glob.glob(os.path.join(base_dir, "test", "*.csv")))

    print(f"[INFO] Найдено файлов: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}\n")

    # Создаем датасеты используя общую функцию files_to_tensor_dataset
    train_ds = files_to_dataset(train_files, pred_horizon, min_input_points)
    val_ds   = files_to_dataset(val_files, pred_horizon, min_input_points)
    test_ds  = files_to_dataset(test_files, pred_horizon, min_input_points)

    print(f"[INFO] train_ds: {len(train_ds)} сэмплов\n")
    print(f"[INFO] val_ds: {len(val_ds)} сэмплов\n")
    print(f"[INFO] test_ds: {len(test_ds)} сэмплов\n")

    return train_ds, val_ds, test_ds
