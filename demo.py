"""
demo.py - демонстрация: одна траектория -> карта с прогнозом угрозы.

Источник траектории:
  - реальный CSV-лог полёта:   python demo.py --csv "путь/к/log.csv"
  - тестовый пример по индексу: python demo.py --test-index 3
По умолчанию берётся тестовый пример 0.

Результат - HTML-карта reports/stage6/demo.html (наблюдаемая и предсказанная
траектории, цели с вероятностями, ETA) + текстовая оценка угрозы в консоли.

Запуск:
    python demo.py --test-index 5
    python demo.py --csv "flight_logs_source/Dolphin_1.csv"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_processing.parser import process_single_file
from run_stage2 import load_split
from run_stage6 import load_tatpn, make_example_map, OUT_DIR


def main():
    ap = argparse.ArgumentParser(description="Демо TATPN: траектория -> карта угрозы")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--csv", type=str, help="путь к CSV-логу полёта")
    g.add_argument("--test-index", type=int, default=0, help="индекс тестовой траектории")
    args = ap.parse_args()

    device = config.DEVICE
    model, target_db, normalizer = load_tatpn(device)
    print(f"[model] TATPN загружена ({model.num_parameters():,} параметров)")

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"[!] Файл не найден: {args.csv}"); return
        traj = process_single_file(args.csv)
        if traj is None or len(traj) < config.MIN_TRAJECTORY_POINTS:
            print(f"[!] Не удалось получить валидную траекторию из {args.csv}"); return
        src = os.path.basename(args.csv)
    else:
        _, _, test_items, _, _ = load_split()
        idx = max(0, min(args.test_index, len(test_items) - 1))
        traj = test_items[idx]["trajectory"]
        src = f"тестовая траектория #{idx}"

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "demo.html")
    threat = make_example_map(model, target_db, normalizer, traj, out,
                              title=f"Демо: {src}", device=device)

    print("\n" + "=" * 60)
    print(f"  Источник: {src}")
    print(f"  ОЦЕНКА УГРОЗЫ: {threat}")
    print(f"  Карта: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
