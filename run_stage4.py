"""
run_stage4.py - Этап 4: + Target Head (классификация цели).

К модели Этапа 3 (CNN + Mamba) добавляется голова классификации цели.
Обучение становится многозадачным: L = alpha*L_traj + beta*L_target, где
L_traj - Huber по траектории (как раньше), L_target - CrossEntropy по 15 целям.

Помимо контекста траектории target-голова использует геометрические подсказки
(расстояние, углы до каждой цели), вычисляемые из последней наблюдаемой точки
и скорости (features/geometry.py).

Проверяем:
  - оба компонента loss убывают;
  - Acc@1 заметно выше случайного (1/15 = 0.067);
  - траектория не деградировала относительно Этапа 3;
  - accuracy vs доля наблюдения (20/40/60/80%).

CPU/GPU-настройки - как на Этапе 3 (Mamba дорогая на CPU).

Запуск:
    python run_stage4.py
"""

import os
import sys
import copy
import json
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_processing.target_database import TargetDatabase
from data.windowing import WindowedDataset
from data.torch_dataset import TorchWindowed
from models.cnn_mamba_target import ConvMambaTarget
from training.losses import masked_huber_traj
from eval.metrics import trajectory_metrics, target_classification_metrics
from eval.logging_utils import (Tee, format_metrics_table, save_json, save_summary)
from baselines import constant_position, linear_velocity
from run_stage2 import load_split, HAS_TQDM

if HAS_TQDM:
    from tqdm import tqdm

STAGE = "stage4_target"
LEARN_IMPROVE_RATIO = 0.8
LOSS_SCALE = 100.0
RANDOM_ACC = 1.0 / config.NUM_TARGETS          # случайный Acc@1 = 1/15
TARGET_ACC_MIN = 0.25                          # порог "цель выучена"
TRAJ_DEGRADE_TOL = 1.20                        # ADE не должен вырасти >20%

if str(config.DEVICE) == "cuda":
    SEQ_LEN, BATCH, MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, OVERFIT_STEPS = \
        192, 64, 60, 8, 1e-3, 300
else:
    SEQ_LEN, BATCH, MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, OVERFIT_STEPS = \
        48, 8, 12, 6, 1e-3, 400


# ============================================================================
# Данные / модель
# ============================================================================

def make_loaders4(train_items, val_items, test_items, normalizer):
    kw = dict(max_seq_len=SEQ_LEN)
    train_w = WindowedDataset(train_items, normalizer, mode="random", **kw)
    val_w = WindowedDataset(val_items, normalizer, mode="fixed", fixed_ratio=0.5, **kw)
    test_w = WindowedDataset(test_items, normalizer, mode="fixed", fixed_ratio=0.5, **kw)
    tl = DataLoader(TorchWindowed(train_w), batch_size=BATCH, shuffle=True,
                    drop_last=True, num_workers=0)
    vl = DataLoader(TorchWindowed(val_w), batch_size=BATCH, shuffle=False, num_workers=0)
    xl = DataLoader(TorchWindowed(test_w), batch_size=BATCH, shuffle=False, num_workers=0)
    return tl, vl, xl, test_w


def _forward4(model, batch, device):
    return model(batch["observed"].to(device), batch["mask"].to(device),
                 batch["last_point"].to(device), batch["last_velocity"].to(device))


def multitask_loss(outputs, batch, device):
    target = batch["future_points"].to(device)
    fmask = batch["future_mask"].to(device)
    tid = batch["target_id"].to(device)
    l_traj = masked_huber_traj(outputs["pred_future"], target, fmask, scale=LOSS_SCALE)
    l_tgt = F.cross_entropy(outputs["target_logits"], tid)
    loss = config.LOSS_ALPHA * l_traj + config.LOSS_BETA * l_tgt
    return loss, float(l_traj), float(l_tgt)


def run_epoch4(model, loader, device, optimizer=None, desc=""):
    """Проход по loader. Возвращает (loss, l_traj, l_tgt, traj_metrics, tgt_metrics)."""
    train = optimizer is not None
    model.train() if train else model.eval()

    tot, tot_traj, tot_tgt, n = 0.0, 0.0, 0.0, 0
    preds, targets, masks, logits_all, ids_all = [], [], [], [], []
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    t_start = time.time()

    iterator = tqdm(loader, desc=desc, total=n_batches, leave=False) \
        if (HAS_TQDM and desc) else loader
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(iterator):
            out = _forward4(model, batch, device)
            loss, lt, lg = multitask_loss(out, batch, device)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()

            b = batch["target_id"].shape[0]
            tot += float(loss) * b; tot_traj += lt * b; tot_tgt += lg * b; n += b
            preds.append(out["pred_future"].detach().cpu().numpy())
            targets.append(batch["future_points"].numpy())
            masks.append(batch["future_mask"].numpy())
            logits_all.append(out["target_logits"].detach().cpu().numpy())
            ids_all.append(batch["target_id"].numpy())

            if HAS_TQDM and desc:
                iterator.set_postfix(loss=f"{tot / max(n,1):.3f}")
            elif desc and (i % log_every == 0 or i == n_batches - 1):
                rate = (i + 1) / max(time.time() - t_start, 1e-6)
                print(f"    [{desc}] батч {i+1}/{n_batches} loss={tot/max(n,1):.3f} "
                      f"({rate:.1f} батч/с)", flush=True)

    tm = trajectory_metrics(np.concatenate(preds), np.concatenate(targets),
                            np.concatenate(masks))
    cm = target_classification_metrics(np.concatenate(logits_all),
                                       np.concatenate(ids_all))
    return tot/max(n,1), tot_traj/max(n,1), tot_tgt/max(n,1), tm, cm


def overfit4(build_model, train_loader, device, steps):
    print("\n" + "=" * 50)
    print(f"  Диагностика: overfit-one-batch ({steps} шагов)")
    print("=" * 50)
    model = build_model().to(device); model.train()
    opt = AdamW(model.parameters(), lr=1e-3)
    batch = next(iter(train_loader))
    l0 = None
    for s in range(steps):
        out = _forward4(model, batch, device)
        loss, lt, lg = multitask_loss(out, batch, device)
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 0:
            l0 = float(loss)
    l1 = float(loss)
    ok = l1 < 0.1 * l0 or l1 < 1e-2
    print(f"  loss: {l0:.4f} -> {l1:.4f} (traj={lt:.4f}, target={lg:.4f})")
    print(f"  [{'OK' if ok else 'FAIL'}] сеть способна подогнать один батч")
    return {"loss_start": l0, "loss_end": l1, "pass": bool(ok)}


def predict_logits(model, batch_np, device, bs=128):
    """Логиты целей для набора окон (numpy) - для accuracy vs доля наблюдения."""
    model.eval()
    N = batch_np["observed"].shape[0]
    out = []
    with torch.no_grad():
        for i in range(0, N, bs):
            sl = slice(i, i + bs)
            o = model(torch.tensor(batch_np["observed"][sl]).to(device),
                      torch.tensor(batch_np["mask"][sl]).to(device),
                      torch.tensor(batch_np["last_point"][sl]).to(device),
                      torch.tensor(batch_np["last_velocity"][sl]).to(device))
            out.append(o["target_logits"].cpu().numpy())
    return np.concatenate(out)


def accuracy_vs_observation(model, test_w, device, ratios=(0.2, 0.4, 0.6, 0.8)):
    res = {}
    for r in ratios:
        b = test_w.build_all(observe_ratio=r)
        logits = predict_logits(model, b, device)
        cm = target_classification_metrics(logits, b["target_id"])
        res[f"{r:.1f}"] = cm
    return res


def load_stage3_ade():
    path = os.path.join(config.LOGS_DIR, "stage3_cnn_mamba", "history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            h = json.load(f)
        return h["test_metrics"]["CNN+Mamba (этап 3)"]["ADE"]
    except Exception:
        return None


# ============================================================================
# main
# ============================================================================

def main():
    config.ensure_dirs()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(config.LOGS_DIR, "run_logs", f"{STAGE}_{ts}.log")
    tee = Tee(log_path); sys.stdout = tee
    device = config.DEVICE

    try:
        print("=" * 70)
        print("  Этап 4 - CNN + Mamba + Target Head (классификация цели)")
        print("=" * 70)
        print(f"  Время старта: {ts}   Device: {device}")
        print(f"  Окно L={SEQ_LEN}, H={config.PRED_HORIZON}, Batch={BATCH}, lr={LR}")
        print(f"  Epochs<={MAX_EPOCHS} (patience={EARLY_STOP_PATIENCE}); "
              f"loss: alpha={config.LOSS_ALPHA} (traj) + beta={config.LOSS_BETA} (target)")

        # --- База целей ---
        target_db = TargetDatabase()
        target_coords = target_db.get_enu_coords()          # (K, 2)

        # --- Данные ---
        train_items, val_items, test_items, normalizer, n_all = load_split()
        print(f"\n[data] всего {n_all}: train={len(train_items)}, "
              f"val={len(val_items)}, test={len(test_items)}")
        train_loader, val_loader, test_loader, test_w = make_loaders4(
            train_items, val_items, test_items, normalizer)

        H = config.PRED_HORIZON
        build_model = lambda: ConvMambaTarget(target_coords=target_coords)

        # --- overfit ---
        overfit = overfit4(build_model, train_loader, device, OVERFIT_STEPS)

        # --- Обучение ---
        model = build_model().to(device)
        print(f"\n[model] параметров: {model.num_parameters():,}")
        optimizer = AdamW(model.parameters(), lr=LR,
                          weight_decay=config.WEIGHT_DECAY, betas=config.BETAS)
        scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                      factor=config.SCHEDULER_FACTOR,
                                      patience=config.SCHEDULER_PATIENCE,
                                      min_lr=config.MIN_LEARNING_RATE)

        history = {"train_loss": [], "val_loss": [], "train_traj": [], "train_target": [],
                   "val_traj": [], "val_target": [], "val_ADE": [], "val_Acc1": [],
                   "val_Acc3": [], "lr": [], "epoch_seconds": []}
        best_val = float("inf"); best_state = None; best_epoch = 0; no_improve = 0

        print("\n" + "=" * 50 + "\n  Обучение\n" + "=" * 50)
        print(f"  батчей: train={len(train_loader)}, val={len(val_loader)} "
              f"(batch={BATCH}); tqdm={'ON' if HAS_TQDM else 'OFF'}", flush=True)
        t0 = datetime.now()
        for epoch in range(1, MAX_EPOCHS + 1):
            te = datetime.now()
            print(f"\n>>> Эпоха {epoch}/{MAX_EPOCHS} — старт {te.strftime('%H:%M:%S')}",
                  flush=True)
            tr_loss, tr_tj, tr_tg, tr_m, tr_c = run_epoch4(
                model, train_loader, device, optimizer, desc=f"эпоха {epoch} [train]")
            va_loss, va_tj, va_tg, va_m, va_c = run_epoch4(
                model, val_loader, device, None, desc=f"эпоха {epoch} [val]")
            scheduler.step(va_loss)
            lr = optimizer.param_groups[0]["lr"]
            secs = (datetime.now() - te).total_seconds()

            for k, v in [("train_loss", tr_loss), ("val_loss", va_loss),
                         ("train_traj", tr_tj), ("train_target", tr_tg),
                         ("val_traj", va_tj), ("val_target", va_tg),
                         ("val_ADE", va_m["ADE"]), ("val_Acc1", va_c["Acc@1"]),
                         ("val_Acc3", va_c["Acc@3"]), ("lr", lr), ("epoch_seconds", secs)]:
                history[k].append(v)

            print(f"[{epoch:2d}/{MAX_EPOCHS}] {secs:5.1f}s lr={lr:.1e} "
                  f"loss={va_loss:.3f} (traj={va_tj:.3f} tgt={va_tg:.3f}) | "
                  f"val_ADE={va_m['ADE']:.1f} Acc@1={va_c['Acc@1']:.3f} "
                  f"Acc@3={va_c['Acc@3']:.3f}", flush=True)

            if va_loss < best_val - 1e-6:
                best_val = va_loss; best_epoch = epoch; no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
                print(f"   -> новый лучший val_loss = {best_val:.4f}")
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"\n[early stopping] эпоха {epoch}")
                    break

        total_s = (datetime.now() - t0).total_seconds()
        avg_epoch = float(np.mean(history["epoch_seconds"]))
        print(f"\n  Обучение завершено за {timedelta(seconds=int(total_s))} "
              f"(лучшая эпоха {best_epoch}; средняя эпоха {avg_epoch:.0f} с)")

        # --- Тест ---
        if best_state is not None:
            model.load_state_dict(best_state)
        _, _, _, test_tm, test_cm = run_epoch4(model, test_loader, device, None, desc="тест")

        # Бейзлайны траектории (тот же якорь)
        b = test_w.build_all(observe_ratio=0.5)
        lin = trajectory_metrics(
            linear_velocity.predict(b["last_point"], b["last_velocity"], H),
            b["future_points"], b["future_mask"])
        const = trajectory_metrics(
            constant_position.predict(b["last_point"], H),
            b["future_points"], b["future_mask"])
        stage3_ade = load_stage3_ade()

        # --- Таблица траектории ---
        traj_rows = {"constant_position": const, "linear_velocity": lin}
        if stage3_ade is not None:
            traj_rows["CNN+Mamba (этап 3)"] = {"ADE": stage3_ade}
        traj_rows["Этап 4 (traj)"] = test_tm
        tcols = ["ADE", "FDE", "RMSE", "MAE", "ADE@10", "ADE@20"]
        print("\n" + "=" * 50)
        print("  Тест: траектория (метры, доля наблюдения 50%)")
        print("=" * 50)
        print(format_metrics_table(traj_rows, tcols))

        # --- Классификация цели ---
        print("\n" + "=" * 50)
        print("  Тест: классификация цели (доля наблюдения 50%)")
        print("=" * 50)
        print(f"  Acc@1 = {test_cm['Acc@1']:.3f}  (случайно {RANDOM_ACC:.3f})")
        print(f"  Acc@3 = {test_cm['Acc@3']:.3f}")
        print(f"  Brier = {test_cm['Brier']:.3f}")

        # --- Accuracy vs доля наблюдения ---
        acc_by_ratio = accuracy_vs_observation(model, test_w, device)
        print("\n  Acc@1 vs доля наблюдения:")
        for r, cm in acc_by_ratio.items():
            print(f"    {int(float(r)*100):2d}%:  Acc@1={cm['Acc@1']:.3f}  "
                  f"Acc@3={cm['Acc@3']:.3f}")

        # --- Проверки ---
        first_val_ade = history["val_ADE"][0]
        best_val_ade = min(history["val_ADE"])
        traj_learned = best_val_ade < first_val_ade * LEARN_IMPROVE_RATIO
        target_learned = test_cm["Acc@1"] > TARGET_ACC_MIN
        traj_ok = (stage3_ade is None) or (test_tm["ADE"] < stage3_ade * TRAJ_DEGRADE_TOL)

        checks = {
            "overfit_one_batch": {"pass": overfit["pass"]},
            "traj_learned": {"first_val_ADE": first_val_ade, "best_val_ADE": best_val_ade,
                             "pass": bool(traj_learned)},
            "target_learned": {"Acc@1": test_cm["Acc@1"], "threshold": TARGET_ACC_MIN,
                               "pass": bool(target_learned)},
            "traj_not_degraded": {"ADE": test_tm["ADE"], "stage3_ADE": stage3_ade,
                                  "pass": bool(traj_ok)},
        }
        all_pass = all(c["pass"] for c in checks.values())

        print("\n" + "=" * 50 + "\n  Проверки Этапа 4\n" + "=" * 50)
        print(f"  [{'OK' if overfit['pass'] else 'FAIL'}] overfit-one-batch "
              f"({overfit['loss_start']:.3f} -> {overfit['loss_end']:.3f})")
        print(f"  [{'OK' if traj_learned else 'FAIL'}] траектория обучилась: "
              f"val ADE {first_val_ade:.1f} -> {best_val_ade:.1f}")
        print(f"  [{'OK' if target_learned else 'FAIL'}] цель выучена: "
              f"Acc@1 {test_cm['Acc@1']:.3f} > {TARGET_ACC_MIN} (случайно {RANDOM_ACC:.3f})")
        deg = f"{stage3_ade:.1f}" if stage3_ade is not None else "н/д"
        print(f"  [{'OK' if traj_ok else 'FAIL'}] траектория не деградировала: "
              f"ADE {test_tm['ADE']:.1f} vs Этап 3 {deg}")
        print(f"\n  ИТОГ: {'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_pass else 'ЕСТЬ ПРОВАЛЫ'}")

        # --- Сохранение ---
        ckpt_dir = os.path.join(config.PROJECT_ROOT, "artifacts", "stage4")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "test_traj": test_tm, "test_target": test_cm},
                   os.path.join(ckpt_dir, "best.pt"))
        print(f"\n[ckpt] -> {os.path.join(ckpt_dir, 'best.pt')}")

        history_out = {
            "stage": STAGE, "timestamp": ts, "device": str(device),
            "config": {"seq_len": SEQ_LEN, "pred_horizon": H, "batch_size": BATCH,
                       "max_epochs": MAX_EPOCHS, "epochs_run": len(history["val_ADE"]),
                       "best_epoch": best_epoch, "lr": LR, "seed": config.RANDOM_SEED,
                       "loss_alpha": config.LOSS_ALPHA, "loss_beta": config.LOSS_BETA},
            "history": history, "overfit": overfit,
            "test_traj": test_tm, "test_target": test_cm,
            "acc_by_ratio": acc_by_ratio, "baselines": {"linear": lin, "constant": const},
            "stage3_ADE": stage3_ade, "checks": checks, "all_pass": all_pass,
            "num_parameters": model.num_parameters(),
        }
        save_json(os.path.join(config.LOGS_DIR, STAGE, "history.json"), history_out)
        save_summary(os.path.join(config.LOGS_DIR, STAGE, "summary.txt"),
                     _summary(history_out, traj_rows, tcols))
        _save_acc_plot(acc_by_ratio)

        print("\n" + "=" * 70)
        print("  Этап 4 завершён.")
        print(f"  Полный лог: {log_path}")
        print("=" * 70)

    finally:
        sys.stdout = tee.stdout
        tee.close()


def _summary(h, traj_rows, tcols):
    L = []
    L.append("СВОДКА ЭТАПА 4 - CNN + Mamba + Target Head")
    L.append("=" * 60)
    L.append(f"Время: {h['timestamp']}   Device: {h['device']}")
    L.append(f"Параметров: {h['num_parameters']:,}   Окно L={h['config']['seq_len']}  "
             f"Batch={h['config']['batch_size']}")
    L.append(f"Эпох: {h['config']['epochs_run']} (лучшая {h['config']['best_epoch']})")
    o = h["overfit"]
    L.append(f"overfit-one-batch: {o['loss_start']:.3f} -> {o['loss_end']:.3f} "
             f"[{'OK' if o['pass'] else 'FAIL'}]")
    L.append("")
    L.append("Динамика по эпохам (val):")
    for i in range(len(h["history"]["val_loss"])):
        L.append(f"  эпоха {i+1}: loss={h['history']['val_loss'][i]:.3f} "
                 f"(traj={h['history']['val_traj'][i]:.3f} "
                 f"tgt={h['history']['val_target'][i]:.3f})  "
                 f"ADE={h['history']['val_ADE'][i]:.1f}  "
                 f"Acc@1={h['history']['val_Acc1'][i]:.3f}")
    L.append("")
    L.append("Тест - траектория (метры, доля наблюдения 50%):")
    L.append(format_metrics_table(traj_rows, tcols))
    L.append("")
    tc = h["test_target"]
    L.append(f"Тест - цель: Acc@1={tc['Acc@1']:.3f} (случайно {1.0/config.NUM_TARGETS:.3f}), "
             f"Acc@3={tc['Acc@3']:.3f}, Brier={tc['Brier']:.3f}")
    L.append("")
    L.append("Acc@1 vs доля наблюдения:")
    for r, cm in h["acc_by_ratio"].items():
        L.append(f"  {int(float(r)*100):2d}%:  Acc@1={cm['Acc@1']:.3f}  Acc@3={cm['Acc@3']:.3f}")
    L.append("")
    L.append("Проверки:")
    for name, c in h["checks"].items():
        L.append(f"  [{'OK' if c['pass'] else 'FAIL'}] {name}")
    L.append(f"ИТОГ: {'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if h['all_pass'] else 'ЕСТЬ ПРОВАЛЫ'}")
    return "\n".join(L)


def _save_acc_plot(acc_by_ratio):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib недоступен ({e})")
        return
    rs = [float(r) * 100 for r in acc_by_ratio]
    a1 = [cm["Acc@1"] for cm in acc_by_ratio.values()]
    a3 = [cm["Acc@3"] for cm in acc_by_ratio.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, a1, "o-", label="Acc@1")
    ax.plot(rs, a3, "s--", label="Acc@3")
    ax.axhline(1.0 / config.NUM_TARGETS, color="gray", ls=":", label="случайно")
    ax.set_xlabel("доля наблюдения, %"); ax.set_ylabel("accuracy")
    ax.set_title("Этап 4: точность цели vs доля наблюдения")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(config.REPORTS_DIR, "stage4_accuracy_vs_observation.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[plot] -> {out}")


if __name__ == "__main__":
    main()
