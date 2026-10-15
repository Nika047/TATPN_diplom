"""
benchmark_gpu.py - замер скорости обучения и оценка полного времени этапов 2-5.

Что делает:
  - строит модели этапов 2, 3, 4, 5 в тех же настройках, что и сами этапы
    (длина окна, размерности Mamba берутся из config и run_stageN);
  - на синтетических тензорах нужной формы измеряет время одного шага
    обучения (forward + backward + optimizer) и одного шага оценки (forward),
    а также пик видеопамяти - для нескольких размеров батча;
  - выбирает батч так же, как это делает автоподбор в этапах (наибольший,
    пик которого не превышает 85% видеопамяти);
  - пересчитывает в время эпохи и полного обучения.

Реальные данные не нужны, поэтому скрипт можно запускать до этапа 0.

Запуск:
    .venv\\Scripts\\python.exe benchmark_gpu.py
Параметры:
    --train N   число обучающих траекторий (по умолчанию из датасета, если он
                уже создан, иначе 2500)
    --val N     число валидационных (по умолчанию 540)
    --batches   размеры батча для проб, через запятую (по умолчанию 128,64,32)
    --steps N   число замеряемых шагов на пробу (по умолчанию 5)
    --models    какие этапы мерить, через запятую (по умолчанию 2,3,4,5)
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_processing.target_database import TargetDatabase
from models.cnn_baseline import ConvTrajectoryNet
from models.cnn_mamba import ConvMambaTrajectory
from models.cnn_mamba_target import ConvMambaTarget
from models.tatpn import TATPN
from training.losses import masked_huber_traj
from training.gpu_tuning import setup_cuda, DEFAULT_VRAM_FRACTION

import run_stage3
import run_stage4
import run_stage5

GB = 1024 ** 3
NAMES = {2: "Этап 2: CNN", 3: "Этап 3: CNN+Mamba",
         4: "Этап 4: +цель", 5: "Этап 5: TATPN"}
OVERFIT = {2: 200, 3: 300, 4: 300, 5: 300}
# Допущение: накладные расходы вне модели (сбор батча, метрики на numpy,
# прогресс-бар) - в измерениях не учтены, добавляется запас.
OVERHEAD = 1.10


def fmt_time(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} с"
    if sec < 3600 * 1.5:
        return f"{sec / 60:.0f} мин"
    return f"{sec / 3600:.1f} ч"


def stage_window(stage: int) -> int:
    """Длина окна, с которой обучается этап."""
    if stage == 2:
        return config.MAX_SEQ_LEN
    return {3: run_stage3.SEQ_LEN, 4: run_stage4.SEQ_LEN, 5: run_stage5.SEQ_LEN}[stage]


def build_model(stage, coords, types):
    if stage == 2:
        return ConvTrajectoryNet()
    if stage == 3:
        return ConvMambaTrajectory()
    if stage == 4:
        return ConvMambaTarget(target_coords=coords)
    return TATPN(target_coords=coords, target_types=types)


def make_inputs(B, L, device):
    H, K = config.PRED_HORIZON, config.NUM_TARGETS
    return dict(
        obs=torch.randn(B, L, config.INPUT_FEATURES, device=device),
        mask=torch.ones(B, L, device=device),
        lp=torch.randn(B, 3, device=device) * 1e3,
        lv=torch.randn(B, 3, device=device) * 30,
        fut=torch.randn(B, H, 3, device=device) * 100,
        fmask=torch.ones(B, H, device=device),
        tid=torch.randint(0, K, (B,), device=device),
        eta=torch.rand(B, device=device) * 3000,
    )


def compute_loss(stage, model, d, train):
    if stage in (2, 3):
        pred = model(d["obs"], d["mask"], d["lp"])
        return masked_huber_traj(pred, d["fut"], d["fmask"], scale=100.0)
    if stage == 4:
        out = model(d["obs"], d["mask"], d["lp"], d["lv"])
        return (masked_huber_traj(out["pred_future"], d["fut"], d["fmask"], scale=100.0)
                + F.cross_entropy(out["target_logits"], d["tid"]))
    out = model(d["obs"], d["mask"], d["lp"], d["lv"],
                future_points=d["fut"] if train else None,
                teacher_forcing_ratio=0.5 if train else 0.0)
    return (masked_huber_traj(out["pred_future"], d["fut"], d["fmask"], scale=100.0)
            + F.cross_entropy(out["target_logits"], d["tid"])
            + 0.5 * F.mse_loss(out["eta"] / 1000.0, d["eta"] / 1000.0))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(stage, B, device, coords, types, steps):
    """Возвращает dict: train_s, eval_s, peak_gb или {'oom': True}."""
    L = stage_window(stage)
    model = opt = d = loss = None
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(stage, coords, types).to(device)
        opt = AdamW(model.parameters(), lr=1e-4)
        d = make_inputs(B, L, device)

        model.train()
        # прогрев (выделение памяти, cudnn.benchmark)
        for _ in range(2):
            loss = compute_loss(stage, model, d, True)
            opt.zero_grad(); loss.backward(); opt.step()
        sync(device)
        t0 = time.time()
        for _ in range(steps):
            loss = compute_loss(stage, model, d, True)
            opt.zero_grad(); loss.backward(); opt.step()
        sync(device)
        train_s = (time.time() - t0) / steps
        peak = (torch.cuda.max_memory_allocated(device) / GB
                if device.type == "cuda" else 0.0)

        model.eval()
        with torch.no_grad():
            compute_loss(stage, model, d, False)       # прогрев
            sync(device)
            t0 = time.time()
            for _ in range(steps):
                compute_loss(stage, model, d, False)
            sync(device)
        eval_s = (time.time() - t0) / steps
        return {"train_s": train_s, "eval_s": eval_s, "peak_gb": peak}
    except torch.cuda.OutOfMemoryError:
        return {"oom": True}
    finally:
        del model, opt, d, loss
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def dataset_sizes(args):
    """Размеры train/val: из аргументов, из готового датасета или по умолчанию."""
    if args.train and args.val:
        return args.train, args.val, "заданы аргументами"
    npz = os.path.join(config.PROCESSED_DATA_DIR, "all_trajectories.npz")
    if os.path.exists(npz):
        try:
            n = len(np.load(npz, allow_pickle=True)["target_ids"])
            tr = int(n * config.TRAIN_RATIO)
            va = int(n * config.VAL_RATIO)
            return args.train or tr, args.val or va, f"из датасета ({n} траекторий)"
        except Exception:
            pass
    return args.train or 2500, args.val or 540, "оценка (датасет ещё не создан)"


def main():
    ap = argparse.ArgumentParser(description="Замер скорости обучения на GPU")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--batches", type=str, default="128,64,32")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--models", type=str, default="2,3,4,5")
    args = ap.parse_args()

    device = torch.device(config.DEVICE)
    batches = [int(x) for x in args.batches.split(",")]
    stages = [int(x) for x in args.models.split(",")]
    n_train, n_val, src = dataset_sizes(args)

    print("=" * 72)
    print("  Замер скорости обучения (синтетические данные нужной формы)")
    print("=" * 72)
    setup_cuda(device)
    if device.type != "cuda":
        print("[!] CUDA недоступна - замер идёт на процессоре, "
              "числа к видеокарте НЕ относятся.")
    print(f"  Mamba: {config.MAMBA_NUM_LAYERS} слоёв, d_model={config.MAMBA_D_MODEL}, "
          f"d_state={config.MAMBA_D_STATE}; горизонт H={config.PRED_HORIZON}")
    print(f"  Обучающих: {n_train}, валидационных: {n_val} - {src}")

    db = TargetDatabase()
    coords, types = db.get_enu_coords(), db.get_type_indices()
    total_mem = (torch.cuda.get_device_properties(device).total_memory
                 if device.type == "cuda" else None)

    summary = []   # (stage, B, epoch_s, overfit_s)
    for stage in stages:
        L = stage_window(stage)
        cand = [config.BATCH_SIZE] if stage == 2 else batches
        print("\n" + "-" * 72)
        print(f"  {NAMES[stage]}   (окно L={L})")
        print("-" * 72)
        print(f"  {'batch':>6} {'пик VRAM, ГБ':>13} {'шаг обуч., с':>13} "
              f"{'шаг оценки, с':>14} {'эпоха, с':>9}")

        chosen = None
        for B in sorted(cand, reverse=True):
            r = measure(stage, B, device, coords, types, args.steps)
            if r.get("oom"):
                print(f"  {B:>6} {'-':>13} {'-':>13} {'-':>14} {'-':>9}   нехватка памяти")
                continue
            epoch = ((n_train // B) * r["train_s"]
                     + math.ceil(n_val / B) * r["eval_s"]) * OVERHEAD
            fits = (total_mem is None) or (r["peak_gb"] * GB <= DEFAULT_VRAM_FRACTION * total_mem)
            mark = ""
            if fits and chosen is None:
                chosen = (B, epoch, r["train_s"])
                mark = "  <- будет выбран автоподбором"
            elif not fits:
                mark = f"  (выше {DEFAULT_VRAM_FRACTION:.0%} VRAM)"
            print(f"  {B:>6} {r['peak_gb']:>13.2f} {r['train_s']:>13.3f} "
                  f"{r['eval_s']:>14.3f} {epoch:>9.0f}{mark}")

        if chosen is None:
            print("  [!] ни один размер батча не прошёл")
            continue
        B, epoch, train_s = chosen
        summary.append((stage, B, epoch, OVERFIT[stage] * train_s * OVERHEAD))

    if not summary:
        return

    print("\n" + "=" * 72)
    print("  ИТОГО: оценка полного времени обучения")
    print("=" * 72)
    print(f"  Накладные расходы вне модели учтены коэффициентом {OVERHEAD:.2f} "
          f"(допущение, не измерено).")
    print(f"  {'этап':<22} {'batch':>6} {'эпоха':>9} {'10 эпох':>9} "
          f"{'50 эпох':>9} {'100 эпох':>10}")
    tot = {10: 0.0, 50: 0.0, 100: 0.0}
    for stage, B, epoch, ofit in summary:
        row = []
        for E in (10, 50, 100):
            t = ofit + E * epoch
            tot[E] += t
            row.append(fmt_time(t))
        print(f"  {NAMES[stage]:<22} {B:>6} {fmt_time(epoch):>9} "
              f"{row[0]:>9} {row[1]:>9} {row[2]:>10}")
    print(f"  {'ВСЕ ЭТАПЫ':<22} {'':>6} {'':>9} "
          f"{fmt_time(tot[10]):>9} {fmt_time(tot[50]):>9} {fmt_time(tot[100]):>10}")
    print()
    print("  Как читать: лимит эпох на GPU - 100, но обучение останавливается")
    print("  раньше, когда val ADE перестаёт улучшаться (patience 12). Реальное")
    print("  число эпох заранее неизвестно: на CPU этапы останавливались, сделав")
    print("  от трети до всех эпох. Ориентир - столбец '50 эпох'; '100 эпох' -")
    print("  верхняя граница. Этапы 0, 1 и 6 занимают считанные минуты.")


if __name__ == "__main__":
    main()
