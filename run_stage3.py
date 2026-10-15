"""
run_stage3.py - Этап 3: CNN + Mamba энкодер (только координаты).

Добавляем к свёрточному фронту временное моделирование (Mamba), сохраняя
формулировку Этапа 2 (дельты от последней точки, без физического prior'а).
Цель - обогнать линейную экстраполяцию (та же нижняя планка, что и на всех
этапах) и улучшить CNN-бейзлайн Этапа 2.

Особенности CPU: Mamba со scan-циклом по длине дорогая. По замерам время
эпохи определяется длиной наблюдаемого окна L (а не размером батча), поэтому
на CPU берём укороченное окно L=64 и батч 32 (эпоха ~2.8 мин). Якорь
сравнения - линейная экстраполяция - от L не зависит, сравнение честное.

Переиспользуются проверенные функции Этапа 2 (load_split, run_epoch,
overfit_one_batch, eval_baselines, _forward, _save_examples) - логика обучения
и метрик та же, меняется только модель и гиперпараметры окна/батча.

Запуск:
    python run_stage3.py
"""

import os
import sys
import copy
import json
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.windowing import WindowedDataset
from data.torch_dataset import TorchWindowed
from models.cnn_mamba import ConvMambaTrajectory
from eval.metrics import trajectory_metrics
from eval.logging_utils import (Tee, format_metrics_table, save_json, save_summary)
from run_stage2 import (load_split, run_epoch, overfit_one_batch, eval_baselines,
                        _forward, _save_examples, HAS_TQDM, LOSS_SCALE)
from training.losses import masked_huber_traj
from training.gpu_tuning import setup_cuda, tune_batch

STAGE = "stage3_cnn_mamba"
LEARN_IMPROVE_RATIO = 0.8

# Настройки зависят от устройства.
# На CPU Mamba (scan-цикл) дорогая, и важен не размер батча, а число шагов
# градиента за эпоху (= N/B). Поэтому берём МАЛЫЙ batch (больше обновлений),
# повышенный lr (кривая обучения гладкая - есть запас) и короткое окно L.
# На GPU per-step matmul параллелится - можно полное окно, большой batch,
# много эпох (для Colab).
if str(config.DEVICE) == "cuda":
    SEQ_LEN = 192
    BATCH = 64                 # стартовое значение; реальное подбирается по VRAM
    MAX_EPOCHS = 100
    EARLY_STOP_PATIENCE = 12
    LR = 1e-3
    OVERFIT_STEPS = 600
else:  # cpu
    SEQ_LEN = 48
    BATCH = 8
    MAX_EPOCHS = 12
    EARLY_STOP_PATIENCE = 6
    LR = 1e-3
    OVERFIT_STEPS = 600


def make_loaders3(train_items, val_items, test_items, normalizer):
    """Как make_loaders Этапа 2, но с укороченным окном L и батчем 32."""
    kw = dict(max_seq_len=SEQ_LEN)
    train_w = WindowedDataset(train_items, normalizer, mode="random", **kw)
    val_w = WindowedDataset(val_items, normalizer, mode="fixed", fixed_ratio=0.5, **kw)
    test_w = WindowedDataset(test_items, normalizer, mode="fixed", fixed_ratio=0.5, **kw)

    train_loader = DataLoader(TorchWindowed(train_w), batch_size=BATCH,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(TorchWindowed(val_w), batch_size=BATCH,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(TorchWindowed(test_w), batch_size=BATCH,
                             shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, test_w


def measure_inference_ms(model, test_loader, device, repeats=3):
    """Среднее время forward на одном батче (мс)."""
    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        _ = _forward(model, batch, device)   # прогрев
        t0 = time.time()
        for _ in range(repeats):
            _ = _forward(model, batch, device)
    return (time.time() - t0) / repeats * 1000.0


def load_stage2_cnn_ade():
    """Справочный ADE CNN Этапа 2 (если есть). Возвращает dict метрик или None."""
    path = os.path.join(config.LOGS_DIR, "stage2_cnn", "history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            h = json.load(f)
        return h["test_metrics"].get("CNN (этап 2)")
    except Exception:
        return None


def _probe_step(model, batch, device):
    """Один шаг для подбора батча под видеопамять: возвращает loss."""
    pred = _forward(model, batch, device)
    return masked_huber_traj(pred, batch["future_points"].to(device),
                             batch["future_mask"].to(device), scale=LOSS_SCALE)


def main():
    global BATCH
    config.ensure_dirs()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(config.LOGS_DIR, "run_logs", f"{STAGE}_{ts}.log")
    tee = Tee(log_path)
    sys.stdout = tee
    device = config.DEVICE

    try:
        setup_cuda(device)
        print("=" * 70)
        print("  Этап 3 - CNN + Mamba энкодер (только координаты)")
        print("=" * 70)
        print(f"  Время старта: {ts}")
        print(f"  Device:       {device}")
        print(f"  Окно L={SEQ_LEN}"
              f"{' (укорочено для CPU)' if str(device) == 'cpu' else ''}, "
              f"H={config.PRED_HORIZON}, Batch={BATCH}")
        print(f"  Mamba: {config.MAMBA_NUM_LAYERS} слоёв, d_model={config.MAMBA_D_MODEL}, "
              f"d_state={config.MAMBA_D_STATE}")
        print(f"  Epochs<={MAX_EPOCHS} (ранняя остановка, patience={EARLY_STOP_PATIENCE}), "
              f"lr={LR}")

        # --- Данные ---
        train_items, val_items, test_items, normalizer, n_all = load_split()
        print(f"\n[data] всего {n_all}: train={len(train_items)}, "
              f"val={len(val_items)}, test={len(test_items)}")
        train_loader, val_loader, test_loader, test_w = make_loaders3(
            train_items, val_items, test_items, normalizer)

        H = config.PRED_HORIZON
        build_model = lambda: ConvMambaTrajectory()

        # --- GPU: размер батча под видеопамять (на CPU ничего не меняет) ---
        BATCH, train_loader, val_loader, test_loader = tune_batch(
            build_model, _probe_step, train_loader, val_loader, test_loader,
            device, default_batch=BATCH)

        # --- overfit-one-batch ---
        overfit = overfit_one_batch(build_model, train_loader, device,
                                    steps=OVERFIT_STEPS)

        # --- Полное обучение ---
        model = build_model().to(device)
        print(f"\n[model] параметров: {model.num_parameters():,}")
        optimizer = AdamW(model.parameters(), lr=LR,
                          weight_decay=config.WEIGHT_DECAY, betas=config.BETAS)
        scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                      factor=config.SCHEDULER_FACTOR,
                                      patience=config.SCHEDULER_PATIENCE,
                                      min_lr=config.MIN_LEARNING_RATE)

        history = {"train_loss": [], "val_loss": [], "train_ADE": [],
                   "val_ADE": [], "val_FDE": [], "lr": [], "epoch_seconds": []}
        best_val_ade = float("inf")
        best_state = None
        best_epoch = 0
        epochs_no_improve = 0

        print("\n" + "=" * 50)
        print("  Обучение")
        print("=" * 50)
        print(f"  батчей: train={len(train_loader)}, val={len(val_loader)} "
              f"(batch={BATCH}); tqdm={'ON' if HAS_TQDM else 'OFF'}", flush=True)
        t0 = datetime.now()
        for epoch in range(1, MAX_EPOCHS + 1):
            te = datetime.now()
            print(f"\n>>> Эпоха {epoch}/{MAX_EPOCHS} — старт "
                  f"{te.strftime('%H:%M:%S')}", flush=True)
            train_loss, train_m = run_epoch(model, train_loader, device, optimizer,
                                            desc=f"эпоха {epoch} [train]")
            val_loss, val_m = run_epoch(model, val_loader, device, None,
                                        desc=f"эпоха {epoch} [val]")
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            secs = (datetime.now() - te).total_seconds()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_ADE"].append(train_m["ADE"])
            history["val_ADE"].append(val_m["ADE"])
            history["val_FDE"].append(val_m["FDE"])
            history["lr"].append(lr)
            history["epoch_seconds"].append(secs)

            print(f"[{epoch:2d}/{MAX_EPOCHS}] {secs:5.1f}s lr={lr:.1e} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"train_ADE={train_m['ADE']:.1f} val_ADE={val_m['ADE']:.1f} "
                  f"val_FDE={val_m['FDE']:.1f}", flush=True)

            if val_m["ADE"] < best_val_ade - 1e-6:
                best_val_ade = val_m["ADE"]
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
                print(f"   -> новый лучший val_ADE = {best_val_ade:.1f} м")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"\n[early stopping] нет улучшения {EARLY_STOP_PATIENCE} "
                          f"эпох, останавливаемся на эпохе {epoch}")
                    break

        total_s = (datetime.now() - t0).total_seconds()
        avg_epoch = float(np.mean(history["epoch_seconds"])) if history["epoch_seconds"] else 0.0
        print(f"\n  Обучение завершено за {timedelta(seconds=int(total_s))} "
              f"(лучшая эпоха {best_epoch}, val_ADE={best_val_ade:.1f} м; "
              f"средняя эпоха {avg_epoch:.0f} с)")

        # --- Тест на лучшей модели ---
        if best_state is not None:
            model.load_state_dict(best_state)
        _, mamba_test = run_epoch(model, test_loader, device, None, desc="тест")
        inf_ms = measure_inference_ms(model, test_loader, device)
        print(f"\n[timing] средняя эпоха: {avg_epoch:.0f} с, "
              f"inference: {inf_ms:.1f} мс/батч (B={BATCH})")

        baselines = eval_baselines(test_w, H)
        cnn2 = load_stage2_cnn_ade()

        rows = {
            "constant_position": baselines["constant_position"],
            "linear_velocity":   baselines["linear_velocity"],
        }
        if cnn2 is not None:
            rows["CNN (этап 2, L=192)"] = cnn2
        rows["CNN+Mamba (этап 3)"] = mamba_test

        cols = ["ADE", "FDE", "RMSE", "MAE", "ADE@10", "ADE@20"]
        print("\n" + "=" * 50)
        print("  Тест: модели vs бейзлайны (метры, доля наблюдения 50%)")
        print("=" * 50)
        print(format_metrics_table(rows, cols))

        # --- Проверки ---
        first_val_ade = history["val_ADE"][0]
        learned = best_val_ade < first_val_ade * LEARN_IMPROVE_RATIO
        lin_ade = baselines["linear_velocity"]["ADE"]
        beats_linear = mamba_test["ADE"] < lin_ade
        beats_cnn = (cnn2 is not None) and (mamba_test["ADE"] < cnn2["ADE"])

        checks = {
            "overfit_one_batch": {"pass": overfit["pass"]},
            "learned": {"first_val_ADE": first_val_ade, "best_val_ADE": best_val_ade,
                        "pass": bool(learned)},
        }
        goal = {
            "beats_linear": {"mamba_ADE": mamba_test["ADE"], "linear_ADE": lin_ade,
                             "pass": bool(beats_linear)},
            "beats_cnn_stage2": {"mamba_ADE": mamba_test["ADE"],
                                 "cnn_ADE": (cnn2["ADE"] if cnn2 else None),
                                 "pass": bool(beats_cnn)},
        }
        all_pass = all(c["pass"] for c in checks.values())

        print("\n" + "=" * 50)
        print("  Проверки Этапа 3")
        print("=" * 50)
        print(f"  [{'OK' if overfit['pass'] else 'FAIL'}] overfit-one-batch "
              f"(loss {overfit['loss_start']:.3f} -> {overfit['loss_end']:.3f})")
        print(f"  [{'OK' if learned else 'FAIL'}] модель обучилась: "
              f"val ADE {first_val_ade:.1f} -> {best_val_ade:.1f} м "
              f"(нужно < {first_val_ade * LEARN_IMPROVE_RATIO:.1f})")
        print(f"  ИТОГ обязательных: {'ПРОЙДЕНЫ' if all_pass else 'ЕСТЬ ПРОВАЛЫ'}")
        print(f"\n  Цели этапа:")
        print(f"    [{'достигнута' if beats_linear else 'нет'}] обгон линейной: "
              f"{mamba_test['ADE']:.1f} vs {lin_ade:.1f} м")
        if cnn2 is not None:
            print(f"    [{'достигнута' if beats_cnn else 'нет'}] лучше CNN Этапа 2: "
                  f"{mamba_test['ADE']:.1f} vs {cnn2['ADE']:.1f} м "
                  f"(CNN на окне L=192, Mamba на L={SEQ_LEN})")

        # --- Сохранение ---
        ckpt_dir = os.path.join(config.PROJECT_ROOT, "artifacts", "stage3")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "test_metrics": mamba_test, "best_val_ade": best_val_ade},
                   os.path.join(ckpt_dir, "best.pt"))
        print(f"\n[ckpt] -> {os.path.join(ckpt_dir, 'best.pt')}")

        history_out = {
            "stage": STAGE, "timestamp": ts, "device": str(device),
            "config": {"seq_len": SEQ_LEN, "pred_horizon": H, "batch_size": BATCH,
                       "max_epochs": MAX_EPOCHS, "epochs_run": len(history["val_ADE"]),
                       "best_epoch": best_epoch, "lr": LR,
                       "mamba_layers": config.MAMBA_NUM_LAYERS,
                       "mamba_d_model": config.MAMBA_D_MODEL, "seed": config.RANDOM_SEED},
            "timing": {"avg_epoch_seconds": avg_epoch, "inference_ms_per_batch": inf_ms},
            "history": history, "overfit": overfit,
            "test_metrics": rows, "checks": checks, "goal": goal,
            "all_pass": all_pass, "num_parameters": model.num_parameters(),
        }
        save_json(os.path.join(config.LOGS_DIR, STAGE, "history.json"), history_out)
        save_summary(os.path.join(config.LOGS_DIR, STAGE, "summary.txt"),
                     _summary(history_out, rows, cols))

        _save_examples(model, test_loader, device, H)
        # Переименуем картинку примеров под этап (у _save_examples имя stage2)
        src = os.path.join(config.REPORTS_DIR, "stage2_examples.png")
        dst = os.path.join(config.REPORTS_DIR, "stage3_examples.png")
        if os.path.exists(src):
            os.replace(src, dst)
            print(f"[examples] -> {dst}")

        print("\n" + "=" * 70)
        print("  Этап 3 завершён.")
        print(f"  Полный лог: {log_path}")
        print("=" * 70)

    finally:
        sys.stdout = tee.stdout
        tee.close()


def _summary(h, rows, cols):
    L = []
    L.append("СВОДКА ЭТАПА 3 - CNN + Mamba")
    L.append("=" * 60)
    L.append(f"Время: {h['timestamp']}   Device: {h['device']}")
    L.append(f"Параметров: {h['num_parameters']:,}   "
             f"Окно L={h['config']['seq_len']}  Batch={h['config']['batch_size']}")
    L.append(f"Эпох: {h['config']['epochs_run']} (лучшая {h['config']['best_epoch']}, "
             f"из <= {h['config']['max_epochs']})")
    t = h["timing"]
    L.append(f"Тайминг: эпоха {t['avg_epoch_seconds']:.0f} с, "
             f"inference {t['inference_ms_per_batch']:.1f} мс/батч")
    L.append("")
    o = h["overfit"]
    L.append(f"overfit-one-batch: loss {o['loss_start']:.3f} -> {o['loss_end']:.3f} "
             f"[{'OK' if o['pass'] else 'FAIL'}]")
    L.append("")
    L.append("Динамика по эпохам (val):")
    for i, (vl, va) in enumerate(zip(h["history"]["val_loss"],
                                     h["history"]["val_ADE"]), 1):
        L.append(f"  эпоха {i}: val_loss={vl:.4f}  val_ADE={va:.1f} м")
    L.append("")
    L.append("Тест (метры, доля наблюдения 50%):")
    L.append(format_metrics_table(rows, cols))
    L.append("")
    L.append("Обязательные проверки:")
    for name, c in h["checks"].items():
        L.append(f"  [{'OK' if c['pass'] else 'FAIL'}] {name}")
    L.append(f"ИТОГ обязательных: {'ПРОЙДЕНЫ' if h['all_pass'] else 'ЕСТЬ ПРОВАЛЫ'}")
    L.append("")
    gl = h["goal"]["beats_linear"]
    L.append(f"Цель (обгон линейной) [{'достигнута' if gl['pass'] else 'нет'}]: "
             f"Mamba ADE {gl['mamba_ADE']:.1f} vs линейная {gl['linear_ADE']:.1f} м")
    gc = h["goal"]["beats_cnn_stage2"]
    if gc["cnn_ADE"] is not None:
        L.append(f"Цель (лучше CNN Этапа 2) [{'достигнута' if gc['pass'] else 'нет'}]: "
                 f"Mamba {gc['mamba_ADE']:.1f} vs CNN {gc['cnn_ADE']:.1f} м")
    return "\n".join(L)


if __name__ == "__main__":
    main()
