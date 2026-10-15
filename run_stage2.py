"""
run_stage2.py - Этап 2: CNN baseline (только координаты).

Первая обучаемая модель. Свёрточная сеть прогнозирует коррекции к линейной
экстраполяции (см. models/cnn_baseline.py). Задача - опуститься по ADE ниже
линейного бейзлайна Этапа 1.

Диагностика (по плану):
  1. overfit-one-batch: перед полным обучением сеть должна почти в ноль
     подогнать один батч (детектор оторванного графа / кривых таргетов / lr).
  2. train/val ADE логируются по эпохам.
  3. финальное сравнение CNN vs линейная экстраполяция vs константа на тесте.
  4. eval без всякого teacher forcing (CNN не авторегрессионный).

Запуск:
    python run_stage2.py
"""

import os
import sys
import copy
from datetime import datetime, timedelta

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_processing.kinematics import FeatureNormalizer
from data.windowing import WindowedDataset
from data.torch_dataset import TorchWindowed
from models.cnn_baseline import ConvTrajectoryNet
from training.losses import masked_huber_traj
from training.gpu_tuning import setup_cuda
from eval.metrics import trajectory_metrics
from eval.logging_utils import Tee, format_metrics_table, save_json, save_summary
from baselines import constant_position, linear_velocity

STAGE = "stage2_cnn"
OVERFIT_STEPS = 200
# Масштаб координат в loss (м). Меньше, чем COORD_SCALE=1000: при ошибках в
# сотни метров Huber работает в линейном (устойчивом) режиме и даёт здоровые
# градиенты. Метрики ADE/FDE по-прежнему в метрах.
LOSS_SCALE = 100.0
# CNN лёгкая (~15 с/эпоха на CPU), поэтому не ограничиваемся CPU-профилем
# NUM_EPOCHS=5: учим дольше с ранней остановкой по val ADE.
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
if str(config.DEVICE) == "cuda":
    # На GPU эпоха CNN занимает секунды - можно учить дольше.
    MAX_EPOCHS = 100
    EARLY_STOP_PATIENCE = 12
# Порог "модель реально обучилась": лучший val ADE ниже стартового на >=20%.
LEARN_IMPROVE_RATIO = 0.8


# ============================================================================
# Данные
# ============================================================================


def load_split():
    """Загружает траектории Этапа 0 и восстанавливает train/val/test (seed=42)."""
    npz_path = os.path.join(config.PROCESSED_DATA_DIR, "all_trajectories.npz")
    norm_path = os.path.join(config.PROCESSED_DATA_DIR, "normalizer.npz")
    for p in (npz_path, norm_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Не найден {p}. Сначала: python run_stage0.py")

    data = np.load(npz_path, allow_pickle=True)
    trajectories = data["trajectories"]
    target_ids = data["target_ids"]
    n_all = len(trajectories)

    np.random.seed(config.RANDOM_SEED)
    perm = np.random.permutation(n_all)
    n_train = int(n_all * config.TRAIN_RATIO)
    n_val = int(n_all * config.VAL_RATIO)
    idx = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }

    def items(ii):
        return [
            {"trajectory": trajectories[i], "target_id": int(target_ids[i])} for i in ii
        ]

    normalizer = FeatureNormalizer()
    normalizer.load(norm_path)
    return items(idx["train"]), items(idx["val"]), items(idx["test"]), normalizer, n_all


def make_loaders(train_items, val_items, test_items, normalizer):
    train_w = WindowedDataset(train_items, normalizer, mode="random")
    val_w = WindowedDataset(val_items, normalizer, mode="fixed", fixed_ratio=0.5)
    test_w = WindowedDataset(test_items, normalizer, mode="fixed", fixed_ratio=0.5)

    bs = config.BATCH_SIZE
    train_loader = DataLoader(
        TorchWindowed(train_w),
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TorchWindowed(val_w), batch_size=bs, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TorchWindowed(test_w), batch_size=bs, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader, test_w


# ============================================================================
# Прогон и метрики
# ============================================================================


def _forward(model, batch, device):
    observed = batch["observed"].to(device)
    mask = batch["mask"].to(device)
    last_point = batch["last_point"].to(device)
    return model(observed, mask, last_point)


def run_epoch(model, loader, device, optimizer=None, desc=""):
    """Один проход по loader. optimizer=None -> оценка. Возвращает (loss, metrics).

    Печатает живой прогресс (tqdm, либо периодический вывод), чтобы длинная
    эпоха на CPU не выглядела как зависание.
    """
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, total_n = 0.0, 0
    preds, targets, masks = [], [], []

    n_batches = len(loader)
    log_every = max(1, n_batches // 20)  # ~20 отметок прогресса на эпоху
    t_start = time.time()

    iterator = loader
    if HAS_TQDM and desc:
        iterator = tqdm(loader, desc=desc, total=n_batches, leave=False)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(iterator):
            target = batch["future_points"].to(device)
            fmask = batch["future_mask"].to(device)
            pred = _forward(model, batch, device)
            loss = masked_huber_traj(pred, target, fmask, scale=LOSS_SCALE)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.GRAD_CLIP_NORM
                )
                optimizer.step()

            b = target.shape[0]
            total_loss += float(loss) * b
            total_n += b
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            masks.append(fmask.detach().cpu().numpy())

            if HAS_TQDM and desc:
                iterator.set_postfix(loss=f"{total_loss / max(total_n, 1):.4f}")
            elif desc and (i % log_every == 0 or i == n_batches - 1):
                rate = (i + 1) / max(time.time() - t_start, 1e-6)
                print(
                    f"    [{desc}] батч {i + 1}/{n_batches} "
                    f"loss={total_loss / max(total_n, 1):.4f} "
                    f"({rate:.1f} батч/с)",
                    flush=True,
                )

    metrics = trajectory_metrics(
        np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)
    )
    return total_loss / max(total_n, 1), metrics


def overfit_one_batch(build_model, train_loader, device, steps=OVERFIT_STEPS):
    """Подгонка одного батча: loss должен упасть почти в ноль."""
    print("\n" + "=" * 50)
    print(f"  Диагностика: overfit-one-batch ({steps} шагов)")
    print("=" * 50)
    model = build_model().to(device)
    model.train()
    opt = AdamW(model.parameters(), lr=1e-3)
    batch = next(iter(train_loader))
    target = batch["future_points"].to(device)
    fmask = batch["future_mask"].to(device)

    loss_start = None
    for step in range(steps):
        pred = _forward(model, batch, device)
        loss = masked_huber_traj(pred, target, fmask, scale=LOSS_SCALE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            loss_start = float(loss)
    loss_end = float(loss)
    ade_end = trajectory_metrics(
        pred.detach().cpu().numpy(), target.cpu().numpy(), fmask.cpu().numpy()
    )["ADE"]

    ok = loss_end < 0.1 * loss_start or loss_end < 1e-2
    print(
        f"  loss: {loss_start:.4f} -> {loss_end:.4f} за {steps} шагов, "
        f"ADE на батче = {ade_end:.2f} м"
    )
    print(f"  [{'OK' if ok else 'FAIL'}] сеть способна подогнать один батч")
    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "ade_end": ade_end,
        "pass": bool(ok),
    }


def eval_baselines(test_w, horizon):
    """Линейная экстраполяция и константа на тех же тестовых окнах (доля 0.5)."""
    b = test_w.build_all(observe_ratio=0.5)
    lin = trajectory_metrics(
        linear_velocity.predict(b["last_point"], b["last_velocity"], horizon),
        b["future_points"],
        b["future_mask"],
    )
    const = trajectory_metrics(
        constant_position.predict(b["last_point"], horizon),
        b["future_points"],
        b["future_mask"],
    )
    return {"linear_velocity": lin, "constant_position": const}


# ============================================================================
# main
# ============================================================================


def main():
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
        print("  Этап 2 - CNN baseline (только координаты)")
        print("=" * 70)
        print(f"  Время старта: {ts}")
        print(f"  Device:       {device}")
        print(
            f"  PRED_HORIZON: {config.PRED_HORIZON}  MAX_SEQ_LEN: {config.MAX_SEQ_LEN}"
        )
        print(
            f"  Batch: {config.BATCH_SIZE}  Epochs<={MAX_EPOCHS} "
            f"(ранняя остановка, patience={EARLY_STOP_PATIENCE})  "
            f"lr: {config.LEARNING_RATE}"
        )

        # --- Данные ---
        train_items, val_items, test_items, normalizer, n_all = load_split()
        print(
            f"\n[data] всего {n_all}: train={len(train_items)}, "
            f"val={len(val_items)}, test={len(test_items)}"
        )
        train_loader, val_loader, test_loader, test_w = make_loaders(
            train_items, val_items, test_items, normalizer
        )

        H = config.PRED_HORIZON
        build_model = lambda: ConvTrajectoryNet()

        # --- overfit-one-batch ---
        overfit = overfit_one_batch(build_model, train_loader, device)

        # --- Полное обучение (свежая модель) ---
        model = build_model().to(device)
        print(f"\n[model] параметров: {model.num_parameters():,}")
        optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_ADE": [],
            "val_ADE": [],
            "val_FDE": [],
            "lr": [],
            "epoch_seconds": [],
        }
        best_val_ade = float("inf")
        best_state = None
        best_epoch = 0
        epochs_no_improve = 0

        print("\n" + "=" * 50)
        print("  Обучение")
        print("=" * 50)
        print(
            f"  батчей: train={len(train_loader)}, val={len(val_loader)} "
            f"(batch={config.BATCH_SIZE}); tqdm={'ON' if HAS_TQDM else 'OFF'}",
            flush=True,
        )
        t0 = datetime.now()
        for epoch in range(1, MAX_EPOCHS + 1):
            te = datetime.now()
            print(
                f"\n>>> Эпоха {epoch}/{MAX_EPOCHS} — старт "
                f"{te.strftime('%H:%M:%S')}",
                flush=True,
            )
            train_loss, train_m = run_epoch(
                model, train_loader, device, optimizer, desc=f"эпоха {epoch} [train]"
            )
            val_loss, val_m = run_epoch(
                model, val_loader, device, None, desc=f"эпоха {epoch} [val]"
            )
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

            print(
                f"[{epoch:2d}/{MAX_EPOCHS}] {secs:5.1f}s lr={lr:.1e} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_ADE={train_m['ADE']:.1f} val_ADE={val_m['ADE']:.1f} "
                f"val_FDE={val_m['FDE']:.1f}",
                flush=True,
            )

            if val_m["ADE"] < best_val_ade - 1e-6:
                best_val_ade = val_m["ADE"]
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
                print(f"   -> новый лучший val_ADE = {best_val_ade:.1f} м")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(
                        f"\n[early stopping] нет улучшения {EARLY_STOP_PATIENCE} "
                        f"эпох подряд, останавливаемся на эпохе {epoch}"
                    )
                    break

        total_s = (datetime.now() - t0).total_seconds()
        print(
            f"\n  Обучение завершено за {timedelta(seconds=int(total_s))} "
            f"(лучшая эпоха {best_epoch}, val_ADE={best_val_ade:.1f} м)"
        )

        # --- Тест на лучшей модели ---
        if best_state is not None:
            model.load_state_dict(best_state)
        _, cnn_test = run_epoch(model, test_loader, device, None, desc="тест")

        baselines = eval_baselines(test_w, H)

        rows = {
            "constant_position": baselines["constant_position"],
            "linear_velocity": baselines["linear_velocity"],
            "CNN (этап 2)": cnn_test,
        }
        cols = ["ADE", "FDE", "RMSE", "MAE", "ADE@10", "ADE@20"]
        print("\n" + "=" * 50)
        print("  Тест: CNN vs бейзлайны (метры, доля наблюдения 50%)")
        print("=" * 50)
        print(format_metrics_table(rows, cols))

        # --- Проверки этапа ---
        # Обязательные: сеть способна переобучить один батч И реально
        # обучилась (лучший val ADE заметно ниже стартового).
        # Цель этапа (информативно, не блокирует): обгон линейной экстраполяции.
        first_val_ade = history["val_ADE"][0]
        learned = best_val_ade < first_val_ade * LEARN_IMPROVE_RATIO
        lin_ade = baselines["linear_velocity"]["ADE"]
        beats = cnn_test["ADE"] < lin_ade

        checks = {
            "overfit_one_batch": {"pass": overfit["pass"]},
            "learned": {
                "first_val_ADE": first_val_ade,
                "best_val_ADE": best_val_ade,
                "pass": bool(learned),
            },
        }
        goal = {
            "beats_linear": {
                "cnn_ADE": cnn_test["ADE"],
                "linear_ADE": lin_ade,
                "pass": bool(beats),
            }
        }
        all_pass = all(c["pass"] for c in checks.values())

        print("\n" + "=" * 50)
        print("  Проверки Этапа 2")
        print("=" * 50)
        print(
            f"  [{'OK' if overfit['pass'] else 'FAIL'}] overfit-one-batch "
            f"(loss {overfit['loss_start']:.3f} -> {overfit['loss_end']:.3f})"
        )
        print(
            f"  [{'OK' if learned else 'FAIL'}] модель обучилась: "
            f"val ADE {first_val_ade:.1f} -> {best_val_ade:.1f} м "
            f"(нужно < {first_val_ade * LEARN_IMPROVE_RATIO:.1f})"
        )
        print(f"  ИТОГ обязательных: " f"{'ПРОЙДЕНЫ' if all_pass else 'ЕСТЬ ПРОВАЛЫ'}")
        print(
            f"\n  Цель этапа [{'достигнута' if beats else 'не достигнута'}]: "
            f"CNN vs линейная экстраполяция на тесте — "
            f"ADE {cnn_test['ADE']:.1f} vs {lin_ade:.1f} м"
        )
        if not beats:
            print(
                "    (простая CNN не обогнала классическую линейную "
                "экстраполяцию — честный результат, мотивирует Mamba на Этапе 3)"
            )

        # --- Сохранение модели, истории, сводки, примеров ---
        ckpt_dir = os.path.join(config.PROJECT_ROOT, "artifacts", "stage2")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "test_metrics": cnn_test,
                "best_val_ade": best_val_ade,
            },
            os.path.join(ckpt_dir, "best.pt"),
        )
        print(f"\n[ckpt] -> {os.path.join(ckpt_dir, 'best.pt')}")

        history_out = {
            "stage": STAGE,
            "timestamp": ts,
            "device": str(device),
            "config": {
                "pred_horizon": H,
                "max_seq_len": config.MAX_SEQ_LEN,
                "batch_size": config.BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "epochs_run": len(history["val_ADE"]),
                "best_epoch": best_epoch,
                "lr": config.LEARNING_RATE,
                "loss_scale": LOSS_SCALE,
                "seed": config.RANDOM_SEED,
            },
            "history": history,
            "overfit": overfit,
            "test_metrics": rows,
            "checks": checks,
            "goal": goal,
            "all_pass": all_pass,
            "num_parameters": model.num_parameters(),
        }
        save_json(os.path.join(config.LOGS_DIR, STAGE, "history.json"), history_out)
        save_summary(
            os.path.join(config.LOGS_DIR, STAGE, "summary.txt"),
            _summary(history_out, rows, cols),
        )

        _save_examples(model, test_loader, device, H)

        print("\n" + "=" * 70)
        print("  Этап 2 завершён.")
        print(f"  Полный лог: {log_path}")
        print("=" * 70)

    finally:
        sys.stdout = tee.stdout
        tee.close()


def _summary(h, rows, cols):
    L = []
    L.append("СВОДКА ЭТАПА 2 - CNN baseline")
    L.append("=" * 60)
    L.append(f"Время: {h['timestamp']}   Device: {h['device']}")
    L.append(
        f"Параметров: {h['num_parameters']:,}   "
        f"Эпох: {h['config']['epochs_run']} (лучшая {h['config']['best_epoch']}, "
        f"из <= {h['config']['max_epochs']})"
    )
    L.append("")
    o = h["overfit"]
    L.append(
        f"overfit-one-batch: loss {o['loss_start']:.3f} -> {o['loss_end']:.3f} "
        f"[{'OK' if o['pass'] else 'FAIL'}]"
    )
    L.append("")
    L.append("Динамика по эпохам (val):")
    for i, (vl, va) in enumerate(
        zip(h["history"]["val_loss"], h["history"]["val_ADE"]), 1
    ):
        L.append(f"  эпоха {i}: val_loss={vl:.4f}  val_ADE={va:.1f} м")
    L.append("")
    L.append("Тест (метры, доля наблюдения 50%):")
    L.append(format_metrics_table(rows, cols))
    L.append("")
    L.append("Обязательные проверки:")
    for name, c in h["checks"].items():
        L.append(f"  [{'OK' if c['pass'] else 'FAIL'}] {name}")
    L.append(
        f"ИТОГ обязательных: " f"{'ПРОЙДЕНЫ' if h['all_pass'] else 'ЕСТЬ ПРОВАЛЫ'}"
    )
    L.append("")
    g = h["goal"]["beats_linear"]
    L.append(
        f"Цель этапа [{'достигнута' if g['pass'] else 'не достигнута'}]: "
        f"CNN ADE {g['cnn_ADE']:.1f} vs линейная {g['linear_ADE']:.1f} м"
    )
    return "\n".join(L)


def _save_examples(model, test_loader, device, horizon, n_examples=10):
    """Сохраняет PNG с примерами: предсказанная vs истинная траектория (x-y)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[examples] matplotlib недоступен, пропуск ({e})")
        return

    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        pred = _forward(model, batch, device).cpu().numpy()
    target = batch["future_points"].numpy()
    fmask = batch["future_mask"].numpy()
    last = batch["last_point"].numpy()

    n = min(n_examples, pred.shape[0])
    # если в батче меньше n_examples - берём сколько есть
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        m = fmask[i].astype(bool)
        ax = axes[i]
        ax.plot(target[i, m, 0], target[i, m, 1], "g.-", label="истина", ms=3)
        ax.plot(pred[i, m, 0], pred[i, m, 1], "r.-", label="прогноз", ms=3)
        ax.plot(last[i, 0], last[i, 1], "ko", ms=5)
        ax.set_title(f"пример {i+1}", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Этап 2 (CNN): прогноз vs истина, проекция x-y", fontsize=11)
    fig.tight_layout()
    out = os.path.join(config.REPORTS_DIR, "stage2_examples.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[examples] -> {out}")


if __name__ == "__main__":
    main()
