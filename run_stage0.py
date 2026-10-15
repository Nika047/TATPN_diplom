"""
run_stage0.py — Главный скрипт Этапа 0: данные и инфраструктура.

Выполняет полный пайплайн подготовки данных:
1. Парсинг реальных CSV-логов → ENU координаты
2. Вычисление кинематических признаков
3. Расширение реальных данных (Слой 2)
4. Генерация синтетических боевых траекторий (Слой 3)
5. Разбиение на train/val/test по исходным полётам и обучение нормализатора
6. Сохранение данных и визуализация

Запуск: python run_stage0.py
"""

import os
import sys
import time
import numpy as np

# Добавляем путь проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_processing.parser import process_all_flights
from data_processing.kinematics import compute_kinematics
from data_processing.target_database import TargetDatabase
from data_processing.generator import TrajectoryGenerator
from data_processing.augmentor import TrajectoryAugmentor
from data_processing.splits import (SPLIT_NAMES, random_labels, split_counts,
                                    fit_normalizer)
from data.windowing import WindowedDataset

try:
    from data_processing.visualize_data import generate_html_report
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("[!] matplotlib не установлен — визуализация будет пропущена")


def main():
    print("=" * 70)
    print("  TATPN — Этап 0: Подготовка данных")
    print("=" * 70)

    start_time = time.time()
    config.ensure_dirs()
    config.print_config()

    # ==================================================================
    # 1. Загрузка базы целей
    # ==================================================================
    print("\n" + "=" * 50)
    print("  1. Загрузка базы целей")
    print("=" * 50)

    target_db = TargetDatabase()
    target_db.print_targets()

    # ==================================================================
    # 2. Парсинг реальных данных (Слой 1)
    # ==================================================================
    print("\n" + "=" * 50)
    print("  2. Парсинг реальных данных (Слой 1)")
    print("=" * 50)

    # Источник реальных логов - только flight_logs_source/ внутри проекта.
    # Благодаря этому репозиторий самодостаточен: после git clone на любой
    # машине датасет пересоздаётся идентично (генератор и аугментор засеяны
    # RANDOM_SEED). Путь можно переопределить переменной окружения
    # FLIGHT_LOGS_SOURCE (см. config.py).
    #
    # Раньше дополнительно читались внешние папки рядом с проектом
    # ("Логи полетов для девушки новое"), но они содержали копии тех же самых
    # логов - каждый полёт парсился дважды, что давало дубликаты реальных
    # траекторий и утечку между train и test.
    source_dirs = []
    main_source = config.FLIGHT_LOGS_SOURCE
    if os.path.exists(main_source):
        source_dirs.append(main_source)

    if not source_dirs:
        print("[!] Не найдены директории с данными. Используем только синтетику.")
        real_results = []
    else:
        real_results = process_all_flights(source_dirs)

    # Извлекаем траектории
    real_trajectories_raw = [r["trajectory"] for r in real_results]
    print(f"\nСлой 1: {len(real_trajectories_raw)} реальных траекторий")

    # Реальные полёты делятся на train/val/test ДО аугментации: расширенные
    # траектории каждой выборки порождаются только из полётов этой выборки,
    # поэтому производные от одного полёта не попадают в разные выборки.
    real_labels = random_labels(len(real_results), seed=config.RANDOM_SEED)

    # Назначаем цели для реальных траекторий (по конечной точке)
    layer1_data = []
    for r, label in zip(real_results, real_labels):
        traj = r["trajectory"]
        end_x, end_y = traj[-1, 0], traj[-1, 1]
        target_id = target_db.find_nearest_target(end_x, end_y)
        layer1_data.append({
            "trajectory": traj,
            "target_id": target_id,
            "trajectory_type": "real",
            "model": r.get("model", "Unknown"),
            "split": int(label),
        })

    # ==================================================================
    # 3. Расширение реальных данных (Слой 2)
    # ==================================================================
    print("\n" + "=" * 50)
    print("  3. Расширение реальных данных (Слой 2)")
    print("=" * 50)

    layer2_data = []
    if real_trajectories_raw:
        aug_counts = split_counts(config.AUGMENTED_COUNT)
        for s, name in enumerate(SPLIT_NAMES):
            sources = [d["trajectory"] for d in layer1_data if d["split"] == s]
            print(f"\n--- выборка {name}: {len(sources)} реальных полётов, "
                  f"нужно {aug_counts[s]} расширенных ---")
            if not sources:
                print("[!] В выборке нет реальных полётов - пропуск")
                continue
            augmentor = TrajectoryAugmentor(
                target_db=target_db, seed=config.RANDOM_SEED + 10 * s)
            for item in augmentor.augment_all(sources, total_count=aug_counts[s]):
                item["split"] = s
                layer2_data.append(item)
    else:
        print("[!] Нет реальных данных для расширения")

    print(f"\nСлой 2: {len(layer2_data)} расширенных траекторий")

    # ==================================================================
    # 4. Генерация синтетических боевых траекторий (Слой 3)
    # ==================================================================
    print("\n" + "=" * 50)
    print("  4. Генерация синтетических траекторий (Слой 3)")
    print("=" * 50)

    generator = TrajectoryGenerator(target_db=target_db)
    layer3_data = generator.generate_all()

    # Синтетика независима от реальных логов - делим случайно.
    synth_labels = random_labels(len(layer3_data), seed=config.RANDOM_SEED + 1)
    for item, label in zip(layer3_data, synth_labels):
        item["split"] = int(label)

    print(f"Слой 3: {len(layer3_data)} синтетических траекторий")

    # ==================================================================
    # 5. Объединение всех данных
    # ==================================================================
    print("\n" + "=" * 50)
    print("  5. Объединение данных")
    print("=" * 50)

    all_trajectories = layer1_data + layer2_data + layer3_data
    print(f"\nИТОГО: {len(all_trajectories)} траекторий")
    print(f"  Слой 1 (реальные):      {len(layer1_data)}")
    print(f"  Слой 2 (расширенные):   {len(layer2_data)}")
    print(f"  Слой 3 (синтетические): {len(layer3_data)}")

    # Статистика по целям
    target_counts = {}
    for t in all_trajectories:
        tid = t["target_id"]
        target_counts[tid] = target_counts.get(tid, 0) + 1

    print(f"\nРаспределение по целям:")
    for tid in sorted(target_counts.keys()):
        tinfo = target_db.get_target_by_index(tid)
        print(f"  {tinfo['id']} {tinfo['name']:30s}: {target_counts[tid]} траекторий")

    # ==================================================================
    # 6. Разбиение и нормализатор
    # ==================================================================
    print("\n" + "=" * 50)
    print("  6. Разбиение и нормализатор")
    print("=" * 50)

    split_labels = np.array([t["split"] for t in all_trajectories], dtype=np.int8)
    split_sizes = {name: int((split_labels == s).sum())
                   for s, name in enumerate(SPLIT_NAMES)}
    print("\nВыборки (по типам траекторий):")
    kinds = sorted({t.get("trajectory_type", t.get("augmentation_type", "real"))
                    for t in all_trajectories})
    print(f"  {'тип':<12}" + "".join(f"{n:>8}" for n in SPLIT_NAMES))
    for kind in kinds:
        row = [sum(1 for t in all_trajectories
                   if t["split"] == s and
                   t.get("trajectory_type", t.get("augmentation_type", "real")) == kind)
               for s in range(len(SPLIT_NAMES))]
        print(f"  {kind:<12}" + "".join(f"{v:>8}" for v in row))
    print(f"  {'ВСЕГО':<12}" + "".join(f"{split_sizes[n]:>8}" for n in SPLIT_NAMES))

    # Нормализатор - только по обучающей выборке (без утечки статистик).
    train_trajs = [t["trajectory"] for t in all_trajectories if t["split"] == 0]
    normalizer = fit_normalizer(train_trajs)
    print(f"\nНормализатор обучен по {len(train_trajs)} обучающим траекториям")

    # Проверка: окна строятся и имеют ожидаемую форму
    check = WindowedDataset(
        [{"trajectory": t["trajectory"], "target_id": t["target_id"]}
         for t in all_trajectories if t["split"] == 0][:32],
        normalizer, mode="random")
    sample = check.make(0)
    print("\nПроверка окна:")
    for key in ("observed", "mask", "last_point", "future_points", "future_mask"):
        print(f"  {key}: shape={np.asarray(sample[key]).shape}")

    # ==================================================================
    # 7. Сохранение данных
    # ==================================================================
    print("\n" + "=" * 50)
    print("  7. Сохранение данных")
    print("=" * 50)

    # Сохраняем нормализатор
    norm_path = os.path.join(config.PROCESSED_DATA_DIR, "normalizer.npz")
    normalizer.save(norm_path)
    print(f"Normalizer → {norm_path}")

    # Сохраняем обработанные данные
    save_path = os.path.join(config.PROCESSED_DATA_DIR, "all_trajectories.npz")
    trajectories_arr = [t["trajectory"] for t in all_trajectories]
    target_ids = [t["target_id"] for t in all_trajectories]
    traj_types = [t.get("trajectory_type", t.get("augmentation_type", "real"))
                  for t in all_trajectories]

    np.savez(save_path,
             trajectories=np.array(trajectories_arr, dtype=object),
             target_ids=np.array(target_ids),
             trajectory_types=np.array(traj_types),
             splits=split_labels)
    print(f"Данные → {save_path}")

    # ==================================================================
    # 8. Визуализация
    # ==================================================================
    print("\n" + "=" * 50)
    print("  8. Визуализация")
    print("=" * 50)

    report_path = os.path.join(config.REPORTS_DIR, "stage0_data_report.html")
    if HAS_VISUALIZATION:
        generate_html_report(all_trajectories, target_db, save_path=report_path)
    else:
        print("[!] Визуализация пропущена (нет matplotlib)")
        print("    Установи: pip install matplotlib")
        report_path = "(не создан)"

    # ==================================================================
    # Итог
    # ==================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  Этап 0 завершён за {:.1f} сек".format(elapsed))
    print("  Данные: {} траекторий".format(len(all_trajectories)))
    print("  Train: {} / Val: {} / Test: {}".format(
          split_sizes["train"], split_sizes["val"], split_sizes["test"]))
    print("  Отчёт: {}".format(report_path))
    print("=" * 70)


if __name__ == "__main__":
    main()
