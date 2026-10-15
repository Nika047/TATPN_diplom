"""
status.py - состояние полного цикла run_all.py одной командой.

Показывает: работает ли процесс, какой этап идёт, на какой он эпохе, какие
этапы уже завершены и за какое время, загрузку видеокарты, и - если цикл
закончился - итог. Ничего не запускает и не изменяет, только читает журнал
run_all.out и logs/run_all/summary.txt.

Запуск (в папке проекта):
    .venv\\Scripts\\python.exe status.py
Слежение с автообновлением каждые 30 с (Ctrl+C - выход, цикл не затрагивается):
    .venv\\Scripts\\python.exe status.py --watch 30
С рабочего компьютера, без интерактивного входа:
    ssh kirill@26.244.190.10 "cd /d C:\\Users\\kirill\\Desktop\\TATPN_diplom && .venv\\Scripts\\python.exe status.py"
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_READ = 20 * 1024 * 1024     # читаем не больше 20 МБ с конца журнала


def read_segments(path, limit=MAX_READ):
    """
    Читает журнал с конца и возвращает непустые фрагменты (по \\r и \\n).

    Кодировку определяем отдельно для каждого фрагмента: в одном файле могут
    оказаться и UTF-8 (этапы), и cp1251 (сам run_all.py старых запусков, если
    он писал в файл без явной кодировки).
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > limit:
            f.seek(size - limit)
        data = f.read()
    segments = []
    for part in re.split(rb"[\r\n]+", data):
        if not part.strip():
            continue
        try:
            segments.append(part.decode("utf-8"))
        except UnicodeDecodeError:
            segments.append(part.decode("cp1251", errors="replace"))
    return segments


def fmt_ago(sec):
    sec = int(max(sec, 0))
    if sec < 90:
        return f"{sec} с"
    if sec < 5400:
        return f"{sec // 60} мин"
    return f"{sec / 3600:.1f} ч"


def python_processes():
    """
    Число процессов python.exe, кроме самого status.py и его родителя
    (запуск из venv идёт через лаунчер: он тоже python.exe).
    """
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=15).stdout
        mine = {str(os.getpid()), str(os.getppid())}
        n = 0
        for row in csv.reader(out.splitlines()):
            if len(row) >= 2 and row[0].lower() == "python.exe" and row[1] not in mine:
                n += 1
        return n
    except Exception:
        return None


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        name, util, used, total = [x.strip() for x in out.splitlines()[0].split(",")]
        return f"{name}: загрузка {util}%, память {int(used) / 1024:.1f} из {int(total) / 1024:.1f} ГБ"
    except Exception:
        return "нет данных (nvidia-smi недоступна)"


# Признак "наблюдать больше нечего" (цикл завершён или остановился): режим
# --watch по нему выходит из цикла обновления.
_STATE = {"done": False}


def show(args) -> int:
    """Один снимок состояния. Возвращает 0, либо 1, если журнала нет."""
    now = time.time()
    procs = python_processes()

    print("=" * 66)
    print(f"  СОСТОЯНИЕ ЦИКЛА   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 66)

    if not os.path.exists(args.out):
        print(f"Журнал {os.path.basename(args.out)} не найден - цикл не запускался "
              f"из этой папки.")
        print(f"Процессов python: {procs if procs is not None else 'неизвестно'}")
        return 1

    mtime = os.path.getmtime(args.out)
    segments = read_segments(args.out)
    text = "\n".join(segments)

    # --- старт цикла ---
    starts = re.findall(r"старт (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", text)
    started = datetime.strptime(starts[-1], "%Y-%m-%d %H:%M:%S") if starts else None
    # Берём только записи текущего запуска (после последнего баннера).
    if starts:
        cut = text.rfind("Полный цикл: этапы")
        if cut >= 0:
            text = text[cut:]
            segments = text.split("\n")

    # --- этапы ---
    headers = re.findall(r"ЭТАП (\d+): (\S+)\s+(\d\d:\d\d:\d\d)", text)
    done = re.findall(r">>> Этап (\d+): (.+?), ((?:\d+ ч )?(?:\d+ мин )?\d+ с)\s*$",
                      text, flags=re.M)
    done_nums = {int(n) for n, _, _ in done}
    current = None
    if headers:
        n, script, at = headers[-1]
        if int(n) not in done_nums:
            current = (int(n), script, at)

    # --- эпохи: только строки текущего этапа (после его заголовка) ---
    stage_segments = segments
    if current:
        marker = f"ЭТАП {current[0]}:"
        for idx in range(len(segments) - 1, -1, -1):
            if marker in segments[idx]:
                stage_segments = segments[idx:]
                break
    epoch_lines = [s for s in stage_segments
                   if re.search(r"\[\s*\d+/\d+\]\s+[\d.]+s\b", s)]
    last_epoch = epoch_lines[-1] if epoch_lines else None

    finished = os.path.exists(args.summary) and \
        os.path.getmtime(args.summary) >= (mtime - 120)
    alive = bool(procs)

    # --- вывод ---
    if started:
        print(f"Старт цикла:        {started:%Y-%m-%d %H:%M:%S} "
              f"(прошло {fmt_ago(now - started.timestamp())})")
    print(f"Процессов python:   {procs if procs is not None else 'неизвестно'}")
    print(f"Последняя запись:   {fmt_ago(now - mtime)} назад")
    print(f"Видеокарта:         {gpu_line()}")
    print()

    if finished and not alive:
        _STATE["done"] = True
        print("ЦИКЛ ЗАВЕРШЁН. Итог:")
        print("-" * 66)
        with open(args.summary, "r", encoding="utf-8", errors="replace") as f:
            print(f.read().rstrip())
        return 0

    if not alive:
        _STATE["done"] = True
        print("ВНИМАНИЕ: процесс python не найден, а итога нет. Цикл остановился")
        print("(ошибка этапа, перезагрузка или остановка). Последние строки журнала:")
        print("-" * 66)
        for s in segments[-12:]:
            print("  " + s[:180])
        print()

    if done:
        print("Завершённые этапы:")
        for n, status, t in done:
            print(f"  этап {n}: {status}, {t}")
    if current:
        n, script, at = current
        print(f"Идёт этап {n} ({script}), начат в {at}")
    if last_epoch:
        m = re.search(r"\[\s*(\d+)/(\d+)\]\s+([\d.]+)s", last_epoch)
        if m and current and current[0] in (2, 3, 4, 5):
            print(f"Эпоха {m.group(1)} из {m.group(2)} (лимит; обучение может "
                  f"закончиться раньше), последняя заняла {m.group(3)} с")
        print("Последняя строка эпохи:")
        print("  " + last_epoch.strip()[:200])
    es = [s for s in stage_segments if "early stopping" in s]
    if es and current:
        print("Ранняя остановка:   " + es[-1].strip()[:150])

    if alive and (now - mtime) > 1800:
        print()
        print("ВНИМАНИЕ: журнал не обновлялся более 30 минут при живом процессе.")
        print("Возможно, обучение зависло - проверить загрузку видеокарты выше.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Состояние полного цикла run_all.py")
    ap.add_argument("--out", default=os.path.join(ROOT, "run_all.out"),
                    help="файл с выводом цикла (по умолчанию run_all.out)")
    ap.add_argument("--summary", default=os.path.join(ROOT, "logs", "run_all", "summary.txt"))
    ap.add_argument("--watch", type=int, default=0, metavar="СЕК",
                    help="обновлять состояние каждые СЕК секунд, пока цикл не "
                         "завершится (Ctrl+C - выход)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    if args.watch <= 0:
        return show(args)

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            code = show(args)
            if _STATE["done"] or code != 0:
                break
            print(f"\n(обновление каждые {args.watch} с, Ctrl+C - выход)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nВыход из режима слежения (цикл продолжает работать).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
