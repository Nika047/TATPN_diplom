"""
logging_utils.py - утилиты диагностического логирования (единый формат).

Содержит:
  - Tee            - дублирование stdout в файл (полный лог консоли запуска);
  - sanity_scale   - проверка масштаба: min/mean/max предсказанных и истинных
                     координат + среднее расстояние last_point -> future[0].
                     Именно эта проверка ловит баг рассинхрона окон (70 км);
  - save_json      - сохранение метрик по эпохам (history.json);
  - format_metrics_table - человекочитаемая таблица метрик;
  - save_summary   - сохранение краткой сводки (summary.txt).
"""

import os
import sys
import json
import numpy as np


class Tee:
    """Дублирует всё, что пишется в stdout, ещё и в файл."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


def sanity_scale(pred: np.ndarray,
                 target: np.ndarray,
                 mask: np.ndarray,
                 last_point: np.ndarray,
                 future_points: np.ndarray,
                 future_mask: np.ndarray) -> dict:
    """
    Диагностика масштаба и корректности нарезки окон.

    Args:
        pred:          (B, H, 3) прогноз
        target:        (B, H, 3) истина
        mask:          (B, H)
        last_point:    (B, 3) последняя наблюдаемая точка
        future_points: (B, H, 3) истинные будущие точки
        future_mask:   (B, H)

    Returns:
        dict с диапазонами координат и статистикой расстояния
        last_point -> future[0] (ключевая проверка бага 70 км).
    """
    m = mask.astype(bool)
    pred_valid = pred[m]
    tgt_valid = target[m]

    # Расстояние от последней наблюдаемой точки до первой истинной будущей.
    # Должно быть порядка одного шага (метры-десятки метров), НЕ километры.
    step = np.linalg.norm(future_points[:, 0, :] - last_point, axis=-1)  # (B,)
    valid0 = future_mask[:, 0].astype(bool)
    step = step[valid0] if valid0.any() else step

    def _rng(a):
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        return {"min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}

    return {
        "pred_xyz":   _rng(pred_valid),
        "target_xyz": _rng(tgt_valid),
        "last_to_future0_m": _rng(step),
    }


def format_metrics_table(rows: dict, cols: list) -> str:
    """
    Форматирует таблицу метрик.

    Args:
        rows: {имя_модели: {метрика: значение}}
        cols: список метрик-колонок в нужном порядке.
    """
    name_w = max([len("модель")] + [len(k) for k in rows]) + 2
    header = "модель".ljust(name_w) + "".join(c.rjust(12) for c in cols)
    lines = [header, "-" * len(header)]
    for name, met in rows.items():
        line = name.ljust(name_w)
        for c in cols:
            v = met.get(c)
            line += ("-" if v is None else f"{v:.1f}").rjust(12)
        lines.append(line)
    return "\n".join(lines)


def save_json(path: str, obj: dict):
    """Сохраняет словарь метрик в JSON (history.json)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[log] JSON -> {path}")


def save_summary(path: str, text: str):
    """Сохраняет человекочитаемую сводку (summary.txt)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[log] summary -> {path}")
