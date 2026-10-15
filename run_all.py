"""
run_all.py - полный цикл: этапы 0-6 подряд одной командой.

Запускает run_stage0.py ... run_stage6.py по очереди, каждый как отдельный
процесс, и замеряет реальное время каждого этапа. Обучение идёт на всех
эпохах, которые назначены этапу (на GPU - до 100 с ранней остановкой).

Правила:
  - если этап упал (ненулевой код возврата) - цикл останавливается, так как
    следующие этапы зависят от результатов предыдущих;
  - если этап завершился, но его обязательные проверки не пройдены (all_pass
    = False в history.json) - цикл продолжается, но этап помечается в итоге;
    с ключом --stop-on-fail цикл остановится на таком этапе.

Итог пишется в logs/run_all/summary.txt и summary.json. Подробные журналы
каждого этапа лежат, как обычно, в logs/<этап>/ и logs/run_logs/.

Запуск:
    python run_all.py                  # этапы 0-6
    python run_all.py --from 3         # продолжить с этапа 3
    python run_all.py --from 2 --to 5  # только этапы 2-5
    python run_all.py --stop-on-fail

Для долгого обучения запускать в фоне (см. md/DAILY_ROUTINE.md).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")

# Номер этапа -> (скрипт, подпапка logs с history.json или None)
STAGES = {
    0: ("run_stage0.py", None),
    1: ("run_stage1.py", "stage1_baselines"),
    2: ("run_stage2.py", "stage2_cnn"),
    3: ("run_stage3.py", "stage3_cnn_mamba"),
    4: ("run_stage4.py", "stage4_target"),
    5: ("run_stage5.py", "stage5_tatpn"),
    6: ("run_stage6.py", None),
}


def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} ч {m:02d} мин {s:02d} с"
    if m:
        return f"{m} мин {s:02d} с"
    return f"{s} с"


def read_all_pass(log_dir):
    """True/False из history.json этапа, либо None, если файла нет."""
    if not log_dir:
        return None
    path = os.path.join(LOGS, log_dir, "history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("all_pass"))
    except Exception:
        return None


def run_stage(num: int, env: dict) -> int:
    script, _ = STAGES[num]
    cmd = [sys.executable, "-u", os.path.join(ROOT, script)]
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main():
    ap = argparse.ArgumentParser(description="Полный цикл этапов 0-6")
    ap.add_argument("--from", dest="first", type=int, default=0)
    ap.add_argument("--to", dest="last", type=int, default=6)
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="остановиться, если у этапа не пройдены проверки")
    args = ap.parse_args()

    nums = [n for n in sorted(STAGES) if args.first <= n <= args.last]
    if not nums:
        print("Нечего запускать: проверьте --from и --to (0..6).")
        return 2

    # UTF-8 для дочерних процессов: журналы с русским текстом при
    # перенаправлении вывода в файл на Windows иначе идут в кодировке консоли.
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    started = datetime.now()
    print("=" * 72)
    print(f"  Полный цикл: этапы {nums[0]}-{nums[-1]}   старт {started:%Y-%m-%d %H:%M:%S}")
    print("=" * 72, flush=True)

    results = []
    t_all = time.time()
    stopped_reason = None

    for n in nums:
        script, log_dir = STAGES[n]
        print("\n" + "#" * 72)
        print(f"#  ЭТАП {n}: {script}   {datetime.now():%H:%M:%S}")
        print("#" * 72, flush=True)

        t0 = time.time()
        code = run_stage(n, env)
        sec = time.time() - t0
        all_pass = read_all_pass(log_dir)

        if code != 0:
            status = f"ОШИБКА (код {code})"
        elif all_pass is False:
            status = "завершён, ПРОВЕРКИ НЕ ПРОЙДЕНЫ"
        elif all_pass is True:
            status = "завершён, проверки пройдены"
        else:
            status = "завершён"
        results.append({"stage": n, "script": script, "code": code,
                        "all_pass": all_pass, "seconds": sec, "status": status})
        print(f"\n>>> Этап {n}: {status}, {fmt_time(sec)}", flush=True)

        if code != 0:
            stopped_reason = f"этап {n} завершился с ошибкой (код {code})"
            break
        if args.stop_on_fail and all_pass is False:
            stopped_reason = f"у этапа {n} не пройдены проверки (--stop-on-fail)"
            break

    total = time.time() - t_all

    lines = ["ПОЛНЫЙ ЦИКЛ - итог", "=" * 60,
             f"Старт: {started:%Y-%m-%d %H:%M:%S}   Общее время: {fmt_time(total)}", ""]
    lines.append(f"{'этап':<6} {'время':<18} статус")
    for r in results:
        lines.append(f"{r['stage']:<6} {fmt_time(r['seconds']):<18} {r['status']}")
    if stopped_reason:
        lines += ["", f"ОСТАНОВЛЕНО: {stopped_reason}"]
    else:
        lines += ["", "Все запрошенные этапы выполнены."]
    text = "\n".join(lines)

    out_dir = os.path.join(LOGS, "run_all")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"started": started.isoformat(timespec="seconds"),
                   "total_seconds": total, "stopped": stopped_reason,
                   "stages": results}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(text)
    print(f"\nИтог сохранён: {os.path.join(out_dir, 'summary.txt')}")
    print("=" * 72)
    return 1 if stopped_reason and results and results[-1]["code"] != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
