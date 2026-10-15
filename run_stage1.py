"""
run_stage1.py - Этап 1: пайплайн оценки + не-нейросетевые бейзлайны.

Обучаемых параметров НЕТ. Цель этапа - доказать, что тракт
данные -> нарезка окон -> метрики корректен и даёт разумные числа, и
зафиксировать опорные значения ADE/FDE, ниже которых обязаны опуститься
нейросети следующих этапов.

Что делает скрипт:
  1. Загружает датасет Этапа 0 (processed_data/all_trajectories.npz) и
     сохранённый нормализатор.
  2. Воспроизводит то же разбиение train/val/test, что и Этап 0 (seed=42).
  3. Формирует окна observed/future НОВЫМ корректным способом
     (хвост наблюдения + смежный future, см. data/windowing.py).
  4. Считает два бейзлайна на тесте: "стоять на месте" и линейная
     экстраполяция по последней скорости.
  5. Диагностика: масштаб координат и расстояние last_point -> future[0]
     (проверка, что баг рассинхрона окон устранён).
  6. Сохраняет history.json, summary.txt и полный лог запуска.

Запуск:
    python run_stage1.py
"""

import os
import sys
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_processing.kinematics import FeatureNormalizer
from data_processing.splits import load_splits
from data.windowing import WindowedDataset
from eval.metrics import trajectory_metrics
from eval.logging_utils import (Tee, sanity_scale, format_metrics_table,
                                save_json, save_summary)
from baselines import constant_position, linear_velocity


STAGE = "stage1_baselines"

# Порог для проверки корректности нарезки окон: среднее расстояние
# last_point -> future[0] обязано быть мало (порядка одного шага dt).
# Прежний баг давал десятки км; ставим щедрый порог 500 м.
WINDOW_CONTIGUITY_MAX_M = 500.0
# Порог "разумности" бейзлайна (не десятки км, как в сломанной версии).
BASELINE_SANE_MAX_M = 20000.0


def load_dataset():
    """Загружает траектории Этапа 0 и тестовую выборку (метки из датасета)."""
    norm_path = os.path.join(config.PROCESSED_DATA_DIR, "normalizer.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Не найден {norm_path}. Сначала выполните Этап 0: python run_stage0.py")

    splits = load_splits()      # метки выборок хранятся в датасете (см. splits.py)
    test_items = splits["test"]

    normalizer = FeatureNormalizer()
    normalizer.load(norm_path)

    return test_items, normalizer, splits["n_all"], len(test_items)


def evaluate_baselines(batch: dict, horizon: int) -> dict:
    """Считает метрики обоих бейзлайнов на подготовленном батче окон."""
    last_point = batch["last_point"]
    last_velocity = batch["last_velocity"]
    target = batch["future_points"]
    fmask = batch["future_mask"]

    pred_const = constant_position.predict(last_point, horizon)
    pred_lin = linear_velocity.predict(last_point, last_velocity, horizon)

    return {
        "constant_position": trajectory_metrics(pred_const, target, fmask),
        "linear_velocity":   trajectory_metrics(pred_lin, target, fmask),
    }, pred_lin


def main():
    config.ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(config.LOGS_DIR, "run_logs", f"{STAGE}_{ts}.log")
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print("=" * 70)
        print("  Этап 1 - Пайплайн оценки + бейзлайны (без обучения)")
        print("=" * 70)
        print(f"  Время старта:  {ts}")
        print(f"  Device:        {config.DEVICE}")
        print(f"  PRED_HORIZON:  {config.PRED_HORIZON}")
        print(f"  MAX_SEQ_LEN:   {config.MAX_SEQ_LEN}")
        print(f"  seed:          {config.RANDOM_SEED}")

        # --- Данные ---
        test_items, normalizer, n_all, n_test = load_dataset()
        print(f"\n[data] всего траекторий: {n_all}, тестовых: {n_test}")

        H = config.PRED_HORIZON

        # --- Основная оценка при 50% наблюдения ---
        print("\n" + "=" * 50)
        print("  Оценка при доле наблюдения 50% (test)")
        print("=" * 50)

        ds = WindowedDataset(test_items, normalizer=normalizer,
                             mode="fixed", fixed_ratio=0.5)
        batch = ds.build_all(observe_ratio=0.5)

        # Проверка детерминированности: повторная сборка даёт то же самое
        batch2 = ds.build_all(observe_ratio=0.5)
        deterministic = bool(
            np.array_equal(batch["future_points"], batch2["future_points"]) and
            np.array_equal(batch["last_point"], batch2["last_point"]))

        metrics, pred_lin = evaluate_baselines(batch, H)

        # --- Диагностика масштаба / контигуальности окон ---
        sanity = sanity_scale(
            pred_lin, batch["future_points"], batch["future_mask"],
            batch["last_point"], batch["future_points"], batch["future_mask"])

        print("\n[sanity] диапазоны координат (по валидным точкам):")
        print(f"  pred (linear) xyz: {sanity['pred_xyz']}")
        print(f"  target        xyz: {sanity['target_xyz']}")
        print(f"  last_point -> future[0], м: {sanity['last_to_future0_m']}")

        cols = ["ADE", "FDE", "RMSE", "MAE", f"ADE@10", f"ADE@20"]
        print("\n[metrics] бейзлайны на тесте (метры), доля наблюдения 50%:")
        print(format_metrics_table(metrics, cols))

        # --- Бонус: как деградируют бейзлайны с уменьшением наблюдения ---
        print("\n" + "=" * 50)
        print("  Бейзлайн linear_velocity vs доля наблюдения")
        print("=" * 50)
        metrics_by_ratio = {}
        for r in (0.2, 0.4, 0.6, 0.8):
            b = ds.build_all(observe_ratio=r)
            m_lin = trajectory_metrics(
                linear_velocity.predict(b["last_point"], b["last_velocity"], H),
                b["future_points"], b["future_mask"])
            metrics_by_ratio[f"{r:.1f}"] = m_lin
            print(f"  наблюдение {int(r*100):2d}%:  "
                  f"ADE={m_lin['ADE']:8.1f}м  FDE={m_lin['FDE']:8.1f}м")

        # --- Проверки этапа ---
        step_mean = sanity["last_to_future0_m"]["mean"]
        lin_ade = metrics["linear_velocity"]["ADE"]
        checks = {
            "window_contiguity": {
                "value_m": step_mean,
                "threshold_m": WINDOW_CONTIGUITY_MAX_M,
                "pass": bool(step_mean < WINDOW_CONTIGUITY_MAX_M),
            },
            "baseline_sane": {
                "value_m": lin_ade,
                "threshold_m": BASELINE_SANE_MAX_M,
                "pass": bool(lin_ade < BASELINE_SANE_MAX_M),
            },
            "deterministic": {"pass": deterministic},
        }
        all_pass = all(c["pass"] for c in checks.values())

        print("\n" + "=" * 50)
        print("  Проверки Этапа 1")
        print("=" * 50)
        print(f"  [{'OK' if checks['window_contiguity']['pass'] else 'FAIL'}] "
              f"контигуальность окон: last_point->future[0] = "
              f"{step_mean:.1f} м (< {WINDOW_CONTIGUITY_MAX_M:.0f})")
        print(f"  [{'OK' if checks['baseline_sane']['pass'] else 'FAIL'}] "
              f"бейзлайн разумен: linear ADE = {lin_ade:.1f} м "
              f"(< {BASELINE_SANE_MAX_M:.0f})")
        print(f"  [{'OK' if checks['deterministic']['pass'] else 'FAIL'}] "
              f"детерминированность (seed={config.RANDOM_SEED})")
        print(f"\n  ИТОГ: {'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_pass else 'ЕСТЬ ПРОВАЛЫ'}")

        # --- Сохранение ---
        history = {
            "stage": STAGE,
            "timestamp": ts,
            "device": str(config.DEVICE),
            "config": {
                "pred_horizon": H,
                "max_seq_len": config.MAX_SEQ_LEN,
                "observe_ratio_test": 0.5,
                "seed": config.RANDOM_SEED,
                "num_targets": config.NUM_TARGETS,
            },
            "dataset": {"n_all": n_all, "n_test": n_test},
            "metrics": metrics,
            "metrics_by_ratio_linear": metrics_by_ratio,
            "sanity": sanity,
            "checks": checks,
            "all_pass": all_pass,
        }
        hist_path = os.path.join(config.LOGS_DIR, STAGE, "history.json")
        save_json(hist_path, history)

        summary = _build_summary(history, metrics, cols)
        sum_path = os.path.join(config.LOGS_DIR, STAGE, "summary.txt")
        save_summary(sum_path, summary)

        print("\n" + "=" * 70)
        print("  Этап 1 завершён.")
        print(f"  Полный лог: {log_path}")
        print("=" * 70)

    finally:
        sys.stdout = tee.stdout
        tee.close()


def _build_summary(history: dict, metrics: dict, cols: list) -> str:
    lines = []
    lines.append("СВОДКА ЭТАПА 1 - бейзлайны прогноза траектории")
    lines.append("=" * 60)
    lines.append(f"Время:        {history['timestamp']}")
    lines.append(f"Device:       {history['device']}")
    lines.append(f"PRED_HORIZON: {history['config']['pred_horizon']}")
    lines.append(f"Тестовых:     {history['dataset']['n_test']} "
                 f"из {history['dataset']['n_all']}")
    lines.append("")
    lines.append("Метрики на тесте (метры), доля наблюдения 50%:")
    lines.append(format_metrics_table(metrics, cols))
    lines.append("")
    s = history["sanity"]["last_to_future0_m"]
    lines.append("Диагностика нарезки окон:")
    lines.append(f"  last_point -> future[0]: "
                 f"min={s['min']:.1f} mean={s['mean']:.1f} max={s['max']:.1f} м")
    lines.append("  (должно быть мало - подтверждает устранение бага 70 км)")
    lines.append("")
    lines.append("Линейная экстраполяция vs доля наблюдения:")
    for r, m in history["metrics_by_ratio_linear"].items():
        lines.append(f"  {int(float(r)*100):2d}%:  ADE={m['ADE']:8.1f}  FDE={m['FDE']:8.1f}")
    lines.append("")
    lines.append("Проверки:")
    for name, c in history["checks"].items():
        lines.append(f"  [{'OK' if c['pass'] else 'FAIL'}] {name}")
    lines.append("")
    lines.append(f"ИТОГ: {'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if history['all_pass'] else 'ЕСТЬ ПРОВАЛЫ'}")
    lines.append("")
    lines.append("Опорные значения ADE/FDE линейной экстраполяции - нижняя")
    lines.append("планка: нейросети Этапов 2+ обязаны быть не хуже.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
