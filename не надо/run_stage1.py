"""
run_stage1.py — Главный скрипт Этапа 1: обучение базовой модели TATPN.

Что делает:
  1. Загружает обработанные данные из processed_data/all_trajectories.npz
     (артефакт Этапа 0). Если их нет — выводит инструкцию и завершается.
  2. Создаёт train/val/test DataLoader-ы и FeatureNormalizer.
  3. Инициализирует модель TATPN, печатает архитектуру и число параметров.
  4. Запускает Trainer (AdamW, scheduler, early stopping, чекпоинты, TB-логи).
  5. По окончании обучения проводит оценку на test и сохраняет:
       - artifacts/TATPN/best.pt   — веса лучшей модели
       - artifacts/TATPN/last.pt   — веса последней эпохи
       - logs/<run_name>/history.json — история loss/метрик по эпохам
       - reports/stage1_train_report.json — итоговые метрики на test

Запуск:
    python run_stage1.py
"""

import os
import sys
import json
import time
import faulthandler
import traceback
from datetime import datetime
import numpy as np

# Включаем faulthandler — он выведет стек даже при segfault / OOM-kill
faulthandler.enable()

# Добавляем путь проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


# ============================================================================
# Дублирование stdout/stderr в файл с датой и временем
# ============================================================================
# Включается до старта main(): всё, что выводится в консоль, одновременно
# пишется в logs/run_logs/stage1_<YYYY-MM-DD>_<HH-MM-SS>.log

class _Tee:
    """Простой "тройник": write одновременно в несколько потоков."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        return False


def _setup_run_logging(stage: str = "stage1"):
    """Подключает Tee на stdout/stderr и возвращает путь к log-файлу."""
    import config as _cfg
    log_dir = os.path.join(_cfg.LOGS_DIR, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{stage}_{ts}.log")

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    # Шапка лога
    log_file.write(f"=== Лог запуска: {stage} ===\n")
    log_file.write(f"Начало: {datetime.now().isoformat(timespec='seconds')}\n")
    log_file.write(f"Файл:   {log_path}\n")
    log_file.write("=" * 60 + "\n\n")
    log_file.flush()

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path

import config
from data_processing.dataloader import create_dataloaders
from data_processing.kinematics import FeatureNormalizer
from models import TATPN
from train import Trainer
from metrics import compute_all_metrics, MetricAggregator
from losses import TATPNLoss
from train import extract_last_point, extract_last_velocity


# ============================================================================
# Загрузка данных Этапа 0
# ============================================================================

def load_stage0_data():
    """
    Загружает обработанные траектории и нормализатор из processed_data.
    Возвращает список словарей в формате, который принимает create_dataloaders.
    """
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "all_trajectories.npz")
    norm_path = os.path.join(config.PROCESSED_DATA_DIR, "normalizer.npz")

    if not os.path.exists(data_path):
        print(f"[!] Не найден файл {data_path}")
        print("    Сначала запустите Этап 0:")
        print("        python run_stage0.py")
        sys.exit(1)

    print(f"[Load] {data_path}")
    data = np.load(data_path, allow_pickle=True)

    trajectories = data["trajectories"]   # array of (Ni, 4) np.float32
    target_ids   = data["target_ids"]     # (N,) int
    traj_types   = data["trajectory_types"]  # (N,) str

    n = len(trajectories)
    print(f"   Загружено {n} траекторий")

    # Конвертируем в формат для create_dataloaders
    # Одновременно фильтруем траектории-выбросы. В исходном датасете часть
    # аугментированных траекторий получила некорректный масштаб координат
    # (до 10^8 м) — это раздувает std нормализатора в миллионы раз
    # и делает модель неспособной что-либо предсказать. Отсеиваем такие
    # траектории по геометрическому правилу: |x|, |y| <= 1 000 000 м (1000 км).
    MAX_COORD_M = 1_000_000.0
    all_trajectories = []
    n_dropped = 0
    for traj, tid, ttype in zip(trajectories, target_ids, traj_types):
        arr = np.asarray(traj, dtype=np.float32)
        x_max = float(np.max(np.abs(arr[:, 0])))
        y_max = float(np.max(np.abs(arr[:, 1])))
        z_max = float(np.max(np.abs(arr[:, 2])))
        if x_max > MAX_COORD_M or y_max > MAX_COORD_M or z_max > MAX_COORD_M:
            n_dropped += 1
            continue
        all_trajectories.append({
            "trajectory":      arr,
            "target_id":       int(tid),
            "trajectory_type": str(ttype),
        })

    if n_dropped > 0:
        print(f"   [!] Отфильтровано {n_dropped} траекторий-выбросов "
              f"(|координата| > {int(MAX_COORD_M)} м)")
    print(f"   После фильтрации: {len(all_trajectories)} траекторий")

    # Контрольная статистика по координатам (помогает убедиться,
    # что выбросы действительно убраны).
    xs = np.concatenate([t["trajectory"][:, 0] for t in all_trajectories])
    ys = np.concatenate([t["trajectory"][:, 1] for t in all_trajectories])
    zs = np.concatenate([t["trajectory"][:, 2] for t in all_trajectories])
    print(f"   Статистика координат после фильтрации:")
    print(f"     x: [{xs.min():.0f}, {xs.max():.0f}] м, std={xs.std():.0f} м")
    print(f"     y: [{ys.min():.0f}, {ys.max():.0f}] м, std={ys.std():.0f} м")
    print(f"     z: [{zs.min():.0f}, {zs.max():.0f}] м, std={zs.std():.0f} м")

    return all_trajectories


# ============================================================================
# Финальная оценка на test
# ============================================================================

def evaluate_on_test(model, test_loader, normalizer, device):
    """
    Прогоняет модель по test-DataLoader-у и возвращает усреднённые метрики и loss.
    """
    model.eval()
    criterion = TATPNLoss()

    norm_mean = torch.tensor(normalizer.mean, dtype=torch.float32)
    norm_std  = torch.tensor(normalizer.std,  dtype=torch.float32)

    agg = MetricAggregator()
    loss_agg = MetricAggregator()

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            observed = batch["observed"]
            mask     = batch["mask"]

            last_point = extract_last_point(observed, mask, norm_mean, norm_std)
            last_velocity = extract_last_velocity(observed, mask, norm_mean, norm_std)

            outputs = model(
                observed=observed,
                mask=mask,
                last_point=last_point,
                last_velocity=last_velocity,
                future_points=None,
                teacher_forcing_ratio=0.0,
            )

            loss, loss_dict = criterion(outputs, batch)
            metrics = compute_all_metrics(outputs, batch)
            B = observed.shape[0]
            loss_agg.update({k: float(v) for k, v in loss_dict.items()}, B)
            agg.update(metrics, B)

    return {**loss_agg.average(), **agg.average()}


# ============================================================================
# Main
# ============================================================================

def main():
    # ВАЖНО: подключаем логирование в файл ДО любых print, чтобы
    # весь вывод запуска (включая шапку и конфиг) попал в лог-файл.
    _log_path = _setup_run_logging(stage="stage1")

    print("=" * 70)
    print("  TATPN — Этап 1: Обучение базовой модели")
    print(f"  Лог-файл: {_log_path}")
    print("=" * 70)

    start_time = time.time()
    config.ensure_dirs()
    config.print_config()

    # Фиксируем seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # ==================================================================
    # 1. Загрузка данных Этапа 0
    # ==================================================================
    print("\n" + "=" * 50)
    print("  1. Загрузка данных Этапа 0")
    print("=" * 50)

    all_trajectories = load_stage0_data()

    # ==================================================================
    # 2. Создание DataLoader-ов
    # ==================================================================
    print("\n" + "=" * 50)
    print("  2. Создание DataLoader-ов")
    print("=" * 50)

    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        all_trajectories, batch_size=config.BATCH_SIZE
    )

    print(f"\n   Train: {len(train_loader.dataset)} сэмплов, {len(train_loader)} батчей")
    print(f"   Val:   {len(val_loader.dataset)} сэмплов, {len(val_loader)} батчей")
    print(f"   Test:  {len(test_loader.dataset)} сэмплов, {len(test_loader)} батчей")

    # ==================================================================
    # 3. Инициализация модели
    # ==================================================================
    print("\n" + "=" * 50)
    print("  3. Инициализация модели TATPN")
    print("=" * 50)

    model = TATPN(
        input_features=config.INPUT_FEATURES,
        d_model=config.MAMBA_D_MODEL,
        num_targets=config.NUM_TARGETS,
        pred_horizon=config.PRED_HORIZON,
        targets_json=config.TARGETS_JSON,
    )

    print(f"\n   Архитектура:")
    print(f"     MultiScaleCNN     : kernels={config.CNN_KERNEL_SIZES}, "
          f"channels={config.CNN_CHANNELS}")
    print(f"     MambaEncoder      : d_model={config.MAMBA_D_MODEL}, "
          f"layers={config.MAMBA_NUM_LAYERS}")
    print(f"     AdaptiveCtxPool   : K={config.CONTEXT_POOL_K}")
    print(f"     CrossAttnTarget   : heads={config.CROSS_ATTN_HEADS}, "
          f"d_attn={config.CROSS_ATTN_DIM}")
    print(f"     TrajectoryDecoder : hidden={config.DECODER_HIDDEN_DIM}, "
          f"layers={config.DECODER_NUM_LAYERS}")
    print(f"     ETAHead           : MLP {2*config.MAMBA_D_MODEL} → "
          f"{config.ETA_HEAD_HIDDEN} → 1")
    print(f"\n   Параметров: {model.num_parameters():,}")

    # ==================================================================
    # 4. Запуск обучения
    # ==================================================================
    print("\n" + "=" * 50)
    print("  4. Обучение")
    print("=" * 50)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINTS_DIR,
        log_dir=os.path.join(config.LOGS_DIR, "tatpn_stage1"),
        run_name="tatpn_stage1",
    )

    trainer.fit(
        num_epochs=config.NUM_EPOCHS,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
    )

    # ==================================================================
    # 5. Финальная оценка на test
    # ==================================================================
    print("\n" + "=" * 50)
    print("  5. Оценка на test")
    print("=" * 50)

    test_stats = evaluate_on_test(model, test_loader, normalizer, config.DEVICE)
    print("\n   Test метрики:")
    for key, val in test_stats.items():
        print(f"     {key:14s}: {val:.4f}")

    # Сохраняем отчёт
    report_path = os.path.join(config.REPORTS_DIR, "stage1_train_report.json")
    report = {
        "run_name":       "tatpn_stage1",
        "device":         str(config.DEVICE),
        "num_parameters": model.num_parameters(),
        "best_val_loss":  trainer.best_val_loss,
        "test_metrics":   test_stats,
        "config": {
            "batch_size":      config.BATCH_SIZE,
            "num_epochs":      config.NUM_EPOCHS,
            "learning_rate":   config.LEARNING_RATE,
            "weight_decay":    config.WEIGHT_DECAY,
            "d_model":         config.MAMBA_D_MODEL,
            "mamba_layers":    config.MAMBA_NUM_LAYERS,
            "pred_horizon":    config.PRED_HORIZON,
            "loss_alpha":      config.LOSS_ALPHA,
            "loss_beta":       config.LOSS_BETA,
            "loss_gamma":      config.LOSS_GAMMA,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n   Отчёт → {report_path}")

    # ==================================================================
    # Итог
    # ==================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Этап 1 завершён за {elapsed/60:.1f} мин")
    print(f"  Лучший val_loss: {trainer.best_val_loss:.4f}")
    print(f"  Test ADE:        {test_stats.get('ADE', 0):.2f} м")
    print(f"  Test FDE:        {test_stats.get('FDE', 0):.2f} м")
    print(f"  Test Acc@1:      {test_stats.get('Acc@1', 0):.3f}")
    print(f"  Test MAE_ETA:    {test_stats.get('MAE_ETA', 0):.2f} с")
    print(f"  Чекпоинт:        {os.path.join(config.CHECKPOINTS_DIR, 'best.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        # Перехватываем ВСЁ, чтобы при падении точно увидеть стек
        # (Windows иногда глотает stderr при аварийном завершении).
        print(f"\n[!!!] Аварийное завершение: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
