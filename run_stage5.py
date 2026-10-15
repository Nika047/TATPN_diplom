"""
run_stage5.py - Этап 5: полная модель TATPN.

Cross-Attention target head (обучаемые эмбеддинги целей) + авторегрессионный
GRU-декодер + ETA-голова. Три задачи одновременно:
    L = alpha*L_traj + beta*L_target + gamma*L_eta

Teacher forcing на декодере убывает по эпохам (0.5 -> 0.0): в начале модель
опирается на истинные точки, к концу работает полностью автономно (как при
инференсе). На оценке teacher forcing всегда выключен.

Сравниваем с предыдущими этапами:
  - траектория: линейная (257.8), Этап 4 (269.9);
  - цель: Этап 4 Acc@1 (0.424) - Cross-Attention против ручной геометрии;
  - ETA: MAE в секундах против наивного предсказания среднего.

CPU/GPU-настройки - как на Этапах 3-4.

Запуск:
    python run_stage5.py
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
from models.tatpn import TATPN
from training.losses import masked_huber_traj
from training.gpu_tuning import setup_cuda, tune_batch
from training.diagnostics import overfit_check
from eval.metrics import (trajectory_metrics, target_classification_metrics,
                          eta_metrics)
from eval.logging_utils import (Tee, format_metrics_table, save_json, save_summary)
from baselines import constant_position, linear_velocity
from run_stage2 import load_split, HAS_TQDM

if HAS_TQDM:
    from tqdm import tqdm

STAGE = "stage5_tatpn"
LEARN_IMPROVE_RATIO = 0.8
LOSS_SCALE = 100.0
ETA_SCALE = config.ETA_SCALE
RANDOM_ACC = 1.0 / config.NUM_TARGETS
TARGET_ACC_MIN = 0.25
TF_START, TF_END = 0.5, 0.0

if str(config.DEVICE) == "cuda":
    SEQ_LEN, BATCH, MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, OVERFIT_STEPS = \
        192, 64, 100, 12, 1e-3, 600   # BATCH - стартовое, реальное подбирается по VRAM
else:
    SEQ_LEN, BATCH, MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, OVERFIT_STEPS = \
        48, 8, 12, 6, 1e-3, 600


def make_loaders5(train_items, val_items, test_items, normalizer):
    kw = dict(max_seq_len=SEQ_LEN)
    tw = WindowedDataset(train_items, normalizer, mode="random", **kw)
    vw = WindowedDataset(val_items, normalizer, mode="fixed", fixed_ratio=0.5, **kw)
    xw = WindowedDataset(test_items, normalizer, mode="fixed", fixed_ratio=0.5, **kw)
    tl = DataLoader(TorchWindowed(tw), batch_size=BATCH, shuffle=True,
                    drop_last=True, num_workers=0)
    vl = DataLoader(TorchWindowed(vw), batch_size=BATCH, shuffle=False, num_workers=0)
    xl = DataLoader(TorchWindowed(xw), batch_size=BATCH, shuffle=False, num_workers=0)
    return tl, vl, xl, xw


def _forward5(model, batch, device, train, tf_ratio):
    future = batch["future_points"].to(device) if train else None
    return model(batch["observed"].to(device), batch["mask"].to(device),
                 batch["last_point"].to(device), batch["last_velocity"].to(device),
                 future_points=future, teacher_forcing_ratio=(tf_ratio if train else 0.0))


def loss5(out, batch, device):
    fut = batch["future_points"].to(device); fmask = batch["future_mask"].to(device)
    tid = batch["target_id"].to(device); eta_t = batch["eta"].to(device)
    l_traj = masked_huber_traj(out["pred_future"], fut, fmask, scale=LOSS_SCALE)
    l_tgt = F.cross_entropy(out["target_logits"], tid)
    l_eta = F.mse_loss(out["eta"] / ETA_SCALE, eta_t / ETA_SCALE)
    loss = config.LOSS_ALPHA * l_traj + config.LOSS_BETA * l_tgt + config.LOSS_GAMMA * l_eta
    return loss, l_traj.item(), l_tgt.item(), l_eta.item()


def run_epoch5(model, loader, device, optimizer, tf_ratio, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = dict(loss=0.0, traj=0.0, tgt=0.0, eta=0.0); n = 0
    preds, targets, masks, logits, ids, eta_p, eta_t = [], [], [], [], [], [], []
    nb = len(loader); log_every = max(1, nb // 5); t_start = time.time()
    it = tqdm(loader, desc=desc, total=nb, leave=False) if (HAS_TQDM and desc) else loader
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(it):
            out = _forward5(model, batch, device, train, tf_ratio)
            loss, lt, lg, le = loss5(out, batch, device)
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()
            b = batch["target_id"].shape[0]; n += b
            tot["loss"] += loss.item()*b; tot["traj"] += lt*b; tot["tgt"] += lg*b; tot["eta"] += le*b
            preds.append(out["pred_future"].detach().cpu().numpy())
            targets.append(batch["future_points"].numpy()); masks.append(batch["future_mask"].numpy())
            logits.append(out["target_logits"].detach().cpu().numpy()); ids.append(batch["target_id"].numpy())
            eta_p.append(out["eta"].detach().cpu().numpy()); eta_t.append(batch["eta"].numpy())
            if HAS_TQDM and desc:
                it.set_postfix(loss=f"{tot['loss']/max(n,1):.3f}")
            elif desc and (i % log_every == 0 or i == nb-1):
                rate = (i+1)/max(time.time()-t_start, 1e-6)
                print(f"    [{desc}] батч {i+1}/{nb} loss={tot['loss']/max(n,1):.3f} "
                      f"({rate:.1f} батч/с)", flush=True)
    tm = trajectory_metrics(np.concatenate(preds), np.concatenate(targets), np.concatenate(masks))
    cm = target_classification_metrics(np.concatenate(logits), np.concatenate(ids))
    em = eta_metrics(np.concatenate(eta_p), np.concatenate(eta_t))
    avg = {k: v/max(n,1) for k, v in tot.items()}
    return avg, tm, cm, em


def overfit5(build_model, train_loader, device, steps):
    """Проверка "сеть способна подогнать малый набор" (teacher forcing 0.5)."""
    batch = next(iter(train_loader))

    def step_fn(model, b, dev):
        loss, _, _, _ = loss5(_forward5(model, b, dev, True, 0.5), b, dev)
        return loss

    return overfit_check(build_model, step_fn, batch, device, steps)


def predict_logits5(model, batch_np, device, bs=128):
    model.eval(); N = batch_np["observed"].shape[0]; out = []
    with torch.no_grad():
        for i in range(0, N, bs):
            sl = slice(i, i+bs)
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
        res[f"{r:.1f}"] = target_classification_metrics(
            predict_logits5(model, b, device), b["target_id"])
    return res


def load_prev(stage_dir, keys):
    path = os.path.join(config.LOGS_DIR, stage_dir, "history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            h = json.load(f)
        cur = h
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return None


def _probe_step(model, batch, device):
    """Один шаг для подбора батча под видеопамять: возвращает loss."""
    out = _forward5(model, batch, device, True, 0.5)
    loss, _, _, _ = loss5(out, batch, device)
    return loss


def main():
    global BATCH
    config.ensure_dirs()
    torch.manual_seed(config.RANDOM_SEED); np.random.seed(config.RANDOM_SEED)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(config.LOGS_DIR, "run_logs", f"{STAGE}_{ts}.log")
    tee = Tee(log_path); sys.stdout = tee
    device = config.DEVICE

    try:
        setup_cuda(device)
        print("=" * 70)
        print("  Этап 5 - Полная модель TATPN (Cross-Attention + декодер + ETA)")
        print("=" * 70)
        print(f"  Время старта: {ts}   Device: {device}")
        print(f"  Окно L={SEQ_LEN}, H={config.PRED_HORIZON}, Batch={BATCH}, lr={LR}")
        print(f"  Epochs<={MAX_EPOCHS} (patience={EARLY_STOP_PATIENCE}); "
              f"loss: a={config.LOSS_ALPHA} b={config.LOSS_BETA} g={config.LOSS_GAMMA}; "
              f"teacher forcing {TF_START}->{TF_END}")

        target_db = TargetDatabase()
        coords = target_db.get_enu_coords()          # (K, 2)
        types = target_db.get_type_indices()         # (K,)

        train_items, val_items, test_items, normalizer, n_all = load_split()
        print(f"\n[data] всего {n_all}: train={len(train_items)}, "
              f"val={len(val_items)}, test={len(test_items)}")
        train_loader, val_loader, test_loader, test_w = make_loaders5(
            train_items, val_items, test_items, normalizer)

        H = config.PRED_HORIZON
        build_model = lambda: TATPN(target_coords=coords, target_types=types)

        # --- GPU: размер батча под видеопамять (на CPU ничего не меняет) ---
        BATCH, train_loader, val_loader, test_loader = tune_batch(
            build_model, _probe_step, train_loader, val_loader, test_loader,
            device, default_batch=BATCH)

        overfit = overfit5(build_model, train_loader, device, OVERFIT_STEPS)

        model = build_model().to(device)
        print(f"\n[model] параметров: {model.num_parameters():,}")
        optimizer = AdamW(model.parameters(), lr=LR,
                          weight_decay=config.WEIGHT_DECAY, betas=config.BETAS)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.SCHEDULER_FACTOR,
                                      patience=config.SCHEDULER_PATIENCE,
                                      min_lr=config.MIN_LEARNING_RATE)

        history = {"train_loss": [], "val_loss": [], "val_traj": [], "val_tgt": [],
                   "val_eta": [], "val_ADE": [], "val_Acc1": [], "val_MAE_ETA": [],
                   "tf": [], "lr": [], "epoch_seconds": []}
        best_val = float("inf"); best_state = None; best_epoch = 0; no_improve = 0

        print("\n" + "=" * 50 + "\n  Обучение\n" + "=" * 50)
        print(f"  батчей: train={len(train_loader)}, val={len(val_loader)} "
              f"(batch={BATCH}); tqdm={'ON' if HAS_TQDM else 'OFF'}", flush=True)
        t0 = datetime.now()
        for epoch in range(1, MAX_EPOCHS + 1):
            prog = (epoch - 1) / max(MAX_EPOCHS - 1, 1)
            tf = TF_START + (TF_END - TF_START) * prog
            te = datetime.now()
            print(f"\n>>> Эпоха {epoch}/{MAX_EPOCHS} — старт {te.strftime('%H:%M:%S')}, "
                  f"tf={tf:.2f}", flush=True)
            tr, tr_m, tr_c, tr_e = run_epoch5(model, train_loader, device, optimizer, tf,
                                              desc=f"эпоха {epoch} [train]")
            va, va_m, va_c, va_e = run_epoch5(model, val_loader, device, None, 0.0,
                                              desc=f"эпоха {epoch} [val]")
            scheduler.step(va["loss"]); lr = optimizer.param_groups[0]["lr"]
            secs = (datetime.now() - te).total_seconds()

            for k, v in [("train_loss", tr["loss"]), ("val_loss", va["loss"]),
                         ("val_traj", va["traj"]), ("val_tgt", va["tgt"]), ("val_eta", va["eta"]),
                         ("val_ADE", va_m["ADE"]), ("val_Acc1", va_c["Acc@1"]),
                         ("val_MAE_ETA", va_e["MAE_ETA"]), ("tf", tf), ("lr", lr),
                         ("epoch_seconds", secs)]:
                history[k].append(v)

            print(f"[{epoch:2d}/{MAX_EPOCHS}] {secs:5.1f}s lr={lr:.1e} val_loss={va['loss']:.3f} "
                  f"| ADE={va_m['ADE']:.1f} Acc@1={va_c['Acc@1']:.3f} "
                  f"MAE_ETA={va_e['MAE_ETA']:.0f}с", flush=True)

            if va["loss"] < best_val - 1e-6:
                best_val = va["loss"]; best_epoch = epoch; no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
                print(f"   -> новый лучший val_loss = {best_val:.4f}")
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"\n[early stopping] эпоха {epoch}"); break

        total_s = (datetime.now() - t0).total_seconds()
        avg_epoch = float(np.mean(history["epoch_seconds"]))
        print(f"\n  Обучение завершено за {timedelta(seconds=int(total_s))} "
              f"(лучшая эпоха {best_epoch}; средняя эпоха {avg_epoch:.0f} с)")

        if best_state is not None:
            model.load_state_dict(best_state)
        _, test_tm, test_cm, test_em = run_epoch5(model, test_loader, device, None, 0.0, desc="тест")

        # Бейзлайны и предыдущие этапы
        b = test_w.build_all(observe_ratio=0.5)
        lin = trajectory_metrics(linear_velocity.predict(b["last_point"], b["last_velocity"], H),
                                 b["future_points"], b["future_mask"])
        const = trajectory_metrics(constant_position.predict(b["last_point"], H),
                                   b["future_points"], b["future_mask"])
        stage4_ade = load_prev("stage4_target", ["test_traj", "ADE"])
        stage4_acc = load_prev("stage4_target", ["test_target", "Acc@1"])
        # Наивный ETA-бейзлайн: предсказание среднего eta
        eta_true_all = b["eta"]
        naive_mae = float(np.abs(eta_true_all - eta_true_all.mean()).mean())

        traj_rows = {"constant_position": const, "linear_velocity": lin}
        if stage4_ade is not None:
            traj_rows["Этап 4"] = {"ADE": stage4_ade}
        traj_rows["TATPN (этап 5)"] = test_tm
        tcols = ["ADE", "FDE", "RMSE", "MAE", "ADE@10", "ADE@20"]
        print("\n" + "=" * 50)
        print("  Тест: траектория (метры, доля наблюдения 50%)")
        print("=" * 50)
        print(format_metrics_table(traj_rows, tcols))

        print("\n" + "=" * 50)
        print("  Тест: цель и ETA")
        print("=" * 50)
        s4 = f"{stage4_acc:.3f}" if stage4_acc is not None else "н/д"
        print(f"  Цель:  Acc@1={test_cm['Acc@1']:.3f} (случайно {RANDOM_ACC:.3f}, "
              f"Этап 4 {s4}), Acc@3={test_cm['Acc@3']:.3f}, Brier={test_cm['Brier']:.3f}")
        print(f"  ETA:   MAE={test_em['MAE_ETA']:.0f} с, RMSE={test_em['RMSE_ETA']:.0f} с "
              f"(наивный MAE={naive_mae:.0f} с)")

        acc_by_ratio = accuracy_vs_observation(model, test_w, device)
        print("\n  Acc@1 vs доля наблюдения:")
        for r, cm in acc_by_ratio.items():
            print(f"    {int(float(r)*100):2d}%:  Acc@1={cm['Acc@1']:.3f}  Acc@3={cm['Acc@3']:.3f}")

        # --- Проверки ---
        # Траектория с velocity-prior стартует near-optimal, поэтому проверяем
        # не относительное улучшение, а конкурентность с линейной экстраполяцией
        # (в пределах 10%) - осмысленный порог "модель не хуже классики".
        first_ade = history["val_ADE"][0]; best_ade = min(history["val_ADE"])
        traj_competitive = test_tm["ADE"] <= lin["ADE"] * 1.10
        target_learned = test_cm["Acc@1"] > TARGET_ACC_MIN
        eta_learned = test_em["MAE_ETA"] < naive_mae
        checks = {
            "overfit_one_batch": {"pass": overfit["pass"]},
            "traj_competitive": {"test_ADE": test_tm["ADE"], "linear_ADE": lin["ADE"],
                                 "pass": bool(traj_competitive)},
            "target_learned": {"Acc@1": test_cm["Acc@1"], "pass": bool(target_learned)},
            "eta_learned": {"MAE_ETA": test_em["MAE_ETA"], "naive_MAE": naive_mae,
                            "pass": bool(eta_learned)},
        }
        goal = {
            "target_vs_stage4": {"acc": test_cm["Acc@1"], "stage4": stage4_acc,
                                 "pass": bool(stage4_acc is not None and test_cm["Acc@1"] > stage4_acc)},
            "traj_vs_linear": {"ade": test_tm["ADE"], "linear": lin["ADE"],
                               "pass": bool(test_tm["ADE"] < lin["ADE"])},
        }
        all_pass = all(c["pass"] for c in checks.values())

        print("\n" + "=" * 50 + "\n  Проверки Этапа 5\n" + "=" * 50)
        print(f"  [{'OK' if overfit['pass'] else 'FAIL'}] overfit-one-batch "
              f"({overfit['loss_start']:.3f} -> {overfit['loss_end']:.3f})")
        print(f"  [{'OK' if traj_competitive else 'FAIL'}] траектория конкурентна с линейной: "
              f"ADE {test_tm['ADE']:.1f} <= {lin['ADE'] * 1.10:.1f} "
              f"(val ADE {first_ade:.1f} -> {best_ade:.1f})")
        print(f"  [{'OK' if target_learned else 'FAIL'}] цель выучена: "
              f"Acc@1 {test_cm['Acc@1']:.3f} > {TARGET_ACC_MIN}")
        print(f"  [{'OK' if eta_learned else 'FAIL'}] ETA лучше наивного: "
              f"{test_em['MAE_ETA']:.0f} < {naive_mae:.0f} с")
        print(f"  ИТОГ обязательных: {'ПРОЙДЕНЫ' if all_pass else 'ЕСТЬ ПРОВАЛЫ'}")
        print("\n  Цели этапа:")
        print(f"    [{'да' if goal['target_vs_stage4']['pass'] else 'нет'}] "
              f"Cross-Attention лучше геометрии Этапа 4: Acc@1 {test_cm['Acc@1']:.3f} vs {s4}")
        print(f"    [{'да' if goal['traj_vs_linear']['pass'] else 'нет'}] "
              f"обгон линейной: ADE {test_tm['ADE']:.1f} vs {lin['ADE']:.1f}")

        # --- Сохранение ---
        ckpt_dir = os.path.join(config.PROJECT_ROOT, "artifacts", "stage5")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "test_traj": test_tm,
                    "test_target": test_cm, "test_eta": test_em},
                   os.path.join(ckpt_dir, "best.pt"))
        print(f"\n[ckpt] -> {os.path.join(ckpt_dir, 'best.pt')}")

        history_out = {
            "stage": STAGE, "timestamp": ts, "device": str(device),
            "config": {"seq_len": SEQ_LEN, "pred_horizon": H, "batch_size": BATCH,
                       "max_epochs": MAX_EPOCHS, "epochs_run": len(history["val_ADE"]),
                       "best_epoch": best_epoch, "lr": LR, "seed": config.RANDOM_SEED,
                       "loss_alpha": config.LOSS_ALPHA, "loss_beta": config.LOSS_BETA,
                       "loss_gamma": config.LOSS_GAMMA},
            "history": history, "overfit": overfit, "test_traj": test_tm,
            "test_target": test_cm, "test_eta": test_em, "naive_eta_mae": naive_mae,
            "acc_by_ratio": acc_by_ratio, "baselines": {"linear": lin, "constant": const},
            "prev": {"stage4_ADE": stage4_ade, "stage4_Acc1": stage4_acc},
            "checks": checks, "goal": goal, "all_pass": all_pass,
            "num_parameters": model.num_parameters(),
        }
        save_json(os.path.join(config.LOGS_DIR, STAGE, "history.json"), history_out)
        save_summary(os.path.join(config.LOGS_DIR, STAGE, "summary.txt"),
                     _summary(history_out, traj_rows, tcols))
        _save_acc_plot(acc_by_ratio)

        print("\n" + "=" * 70)
        print("  Этап 5 завершён.")
        print(f"  Полный лог: {log_path}")
        print("=" * 70)

    finally:
        sys.stdout = tee.stdout
        tee.close()


def _summary(h, traj_rows, tcols):
    L = []
    L.append("СВОДКА ЭТАПА 5 - Полная модель TATPN")
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
        L.append(f"  эпоха {i+1}: loss={h['history']['val_loss'][i]:.3f}  "
                 f"ADE={h['history']['val_ADE'][i]:.1f}  "
                 f"Acc@1={h['history']['val_Acc1'][i]:.3f}  "
                 f"MAE_ETA={h['history']['val_MAE_ETA'][i]:.0f}с  "
                 f"(tf={h['history']['tf'][i]:.2f})")
    L.append("")
    L.append("Тест - траектория (метры, доля наблюдения 50%):")
    L.append(format_metrics_table(traj_rows, tcols))
    L.append("")
    tc = h["test_target"]; te = h["test_eta"]
    L.append(f"Тест - цель: Acc@1={tc['Acc@1']:.3f} (случайно {1.0/config.NUM_TARGETS:.3f}), "
             f"Acc@3={tc['Acc@3']:.3f}, Brier={tc['Brier']:.3f}")
    L.append(f"Тест - ETA: MAE={te['MAE_ETA']:.0f} с, RMSE={te['RMSE_ETA']:.0f} с "
             f"(наивный {h['naive_eta_mae']:.0f} с)")
    L.append("")
    L.append("Acc@1 vs доля наблюдения:")
    for r, cm in h["acc_by_ratio"].items():
        L.append(f"  {int(float(r)*100):2d}%:  Acc@1={cm['Acc@1']:.3f}  Acc@3={cm['Acc@3']:.3f}")
    L.append("")
    L.append("Обязательные проверки:")
    for name, c in h["checks"].items():
        L.append(f"  [{'OK' if c['pass'] else 'FAIL'}] {name}")
    L.append(f"ИТОГ: {'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if h['all_pass'] else 'ЕСТЬ ПРОВАЛЫ'}")
    L.append("")
    g = h["goal"]
    L.append(f"Цель (Cross-Attn > Этап 4): "
             f"[{'да' if g['target_vs_stage4']['pass'] else 'нет'}]")
    L.append(f"Цель (обгон линейной по ADE): "
             f"[{'да' if g['traj_vs_linear']['pass'] else 'нет'}]")
    return "\n".join(L)


def _save_acc_plot(acc_by_ratio):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib недоступен ({e})"); return
    rs = [float(r) * 100 for r in acc_by_ratio]
    a1 = [cm["Acc@1"] for cm in acc_by_ratio.values()]
    a3 = [cm["Acc@3"] for cm in acc_by_ratio.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, a1, "o-", label="Acc@1"); ax.plot(rs, a3, "s--", label="Acc@3")
    ax.axhline(1.0 / config.NUM_TARGETS, color="gray", ls=":", label="случайно")
    ax.set_xlabel("доля наблюдения, %"); ax.set_ylabel("accuracy")
    ax.set_title("Этап 5 (TATPN): точность цели vs доля наблюдения")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    out = os.path.join(config.REPORTS_DIR, "stage5_accuracy_vs_observation.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[plot] -> {out}")


if __name__ == "__main__":
    main()
