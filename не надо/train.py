"""
train.py — Тренировочный цикл TATPN (Этап 1).

Возможности:
  - AdamW + ReduceLROnPlateau + gradient clipping
  - Early stopping по val loss
  - Чекпоинтинг лучшей модели и последней
  - TensorBoard логи (если установлен tensorboard, иначе пропуск)
  - Прогресс-бар через tqdm
  - Работает универсально на CPU и GPU (config.DEVICE)
  - Корректное восстановление last_point в исходных метрах из нормализованных observed

Использование (из run_stage1.py):
    trainer = Trainer(model, train_loader, val_loader, normalizer)
    trainer.fit(num_epochs=config.NUM_EPOCHS)
"""

import os
import sys
import time
import json
import traceback
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

import config
from losses import TATPNLoss
from metrics import compute_all_metrics, MetricAggregator


# ============================================================================
# Вспомогательная функция: восстановление last_point в исходных координатах
# ============================================================================

def extract_last_point(observed_norm: torch.Tensor,
                       mask: torch.Tensor,
                       norm_mean: torch.Tensor,
                       norm_std: torch.Tensor) -> torch.Tensor:
    """
    Извлекает последнюю РЕАЛЬНУЮ точку каждого сэмпла в исходных координатах.

    Args:
        observed_norm: (B, L, F)  — нормализованные признаки
        mask:          (B, L)      — 1 для реальных, 0 для padding
        norm_mean:     (F,)        — средние нормализатора
        norm_std:      (F,)        — std нормализатора

    Returns:
        last_point: (B, 3) — последняя реальная точка (x, y, z) в метрах
    """
    B, L, F = observed_norm.shape
    device = observed_norm.device

    lengths = mask.sum(dim=1).long().clamp(min=1)            # (B,)
    last_idx = (lengths - 1).clamp(min=0)                    # (B,)

    # Собираем последнюю реальную точку
    idx_exp = last_idx.view(B, 1, 1).expand(-1, 1, F)        # (B, 1, F)
    last = observed_norm.gather(1, idx_exp).squeeze(1)       # (B, F)

    # Денормализуем (нас интересуют только первые 3 канала — x, y, z)
    last_xyz_norm = last[:, :3]                              # (B, 3)
    mean_xyz = norm_mean[:3].to(device).unsqueeze(0)         # (1, 3)
    std_xyz  = norm_std[:3].to(device).unsqueeze(0)          # (1, 3)
    last_xyz = last_xyz_norm * std_xyz + mean_xyz            # (B, 3)
    return last_xyz


def extract_last_velocity(observed_norm: torch.Tensor,
                          mask: torch.Tensor,
                          norm_mean: torch.Tensor,
                          norm_std: torch.Tensor) -> torch.Tensor:
    """
    Извлекает компоненты скорости (vx, vy, vz) последней реальной точки
    в исходных единицах м/с. По устройству признаков (см. kinematics.py)
    скорости лежат в каналах 4, 5, 6.

    Эти значения используются декодером как baseline линейной экстраполяции:
    next_point = prev + last_velocity * dt + correction. Такой prior резко
    повышает обучаемость декодера, поскольку он стартует с физически
    разумного "продолжай движение" вместо "стой на месте".
    """
    B, L, F = observed_norm.shape
    device = observed_norm.device

    lengths = mask.sum(dim=1).long().clamp(min=1)
    last_idx = (lengths - 1).clamp(min=0)

    idx_exp = last_idx.view(B, 1, 1).expand(-1, 1, F)
    last = observed_norm.gather(1, idx_exp).squeeze(1)       # (B, F)

    last_v_norm = last[:, 4:7]                               # (B, 3)
    mean_v = norm_mean[4:7].to(device).unsqueeze(0)
    std_v  = norm_std[4:7].to(device).unsqueeze(0)
    last_v = last_v_norm * std_v + mean_v                    # (B, 3)
    return last_v


# ============================================================================
# Класс Trainer
# ============================================================================

class Trainer:
    """
    Универсальный тренер для TATPN.

    Args:
        model:        nn.Module — экземпляр TATPN
        train_loader: DataLoader
        val_loader:   DataLoader
        normalizer:   FeatureNormalizer — нужен для денормализации last_point
        device:       torch.device — куда переносить модель и батчи
        checkpoint_dir: куда сохранять модели
        log_dir:      куда писать TensorBoard логи
        run_name:     имя запуска (для логов)
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 normalizer,
                 device=None,
                 checkpoint_dir: str = None,
                 log_dir: str = None,
                 run_name: str = "tatpn_stage1"):
        self.device = device or config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_name = run_name

        # Нормализатор → тензоры для GPU
        self.norm_mean = torch.tensor(normalizer.mean, dtype=torch.float32)
        self.norm_std  = torch.tensor(normalizer.std,  dtype=torch.float32)

        # Loss
        self.criterion = TATPNLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
        )

        # LR scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
        )

        # Пути
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINTS_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        log_dir = log_dir or os.path.join(config.LOGS_DIR, run_name)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir) if HAS_TB else None

        # История метрик (для последующего HTML отчёта Этапа 1)
        self.history = {
            "train_loss":  [], "val_loss":  [],
            "train_metrics": [], "val_metrics": [],
            "lr": [],
            "epoch_seconds": [],  # время каждой эпохи в секундах
        }

        # Early stopping
        self.best_val_loss = float("inf")
        self.best_state = None
        self.epochs_no_improve = 0

        # Scheduled sampling: teacher forcing ratio падает по эпохам.
        # На эпохе 1 — 0.5 (даём модели опору в начале обучения),
        # к финальной эпохе — 0.0 (модель должна работать полностью
        # автономно, как при инференсе). Это устраняет exposure bias —
        # ситуацию, когда модель отлично работает с подсказками в train
        # (val_loss / train_loss = 5x), но разваливается без них.
        self._current_epoch = 1
        self._total_epochs = config.NUM_EPOCHS
        self._tf_start = 0.5
        self._tf_end = 0.0

    # ------------------------------------------------------------------
    # Один шаг обучения / валидации
    # ------------------------------------------------------------------

    def _move_batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    def _step(self, batch: dict, train: bool) -> tuple:
        """
        Один шаг обучения или оценки. Возвращает (loss_dict, metrics_dict, B).
        """
        batch = self._move_batch_to_device(batch)

        observed   = batch["observed"]
        mask       = batch["mask"]
        future_pts = batch["future_points"]

        # last_point в исходных координатах
        last_point = extract_last_point(observed, mask, self.norm_mean, self.norm_std)
        # last_velocity (м/с) для velocity-prior в декодере
        last_velocity = extract_last_velocity(observed, mask, self.norm_mean, self.norm_std)

        # Forward.
        # На train teacher_forcing падает с self._tf_start до self._tf_end
        # по эпохам (scheduled sampling). На val/test всегда 0.0.
        if train:
            if self._total_epochs > 1:
                progress = (self._current_epoch - 1) / (self._total_epochs - 1)
            else:
                progress = 0.0
            teacher_forcing = self._tf_start + (self._tf_end - self._tf_start) * progress
        else:
            teacher_forcing = 0.0
        outputs = self.model(
            observed=observed,
            mask=mask,
            last_point=last_point,
            last_velocity=last_velocity,
            future_points=future_pts if train else None,
            teacher_forcing_ratio=teacher_forcing,
        )

        # Loss
        loss, loss_dict = self.criterion(outputs, batch)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GRAD_CLIP_NORM
            )
            self.optimizer.step()

        # Метрики (без градиентов)
        metrics = compute_all_metrics(outputs, batch)

        B = observed.shape[0]
        # Конвертируем тензоры loss_dict в float
        loss_dict = {k: float(v) for k, v in loss_dict.items()}
        return loss_dict, metrics, B

    # ------------------------------------------------------------------
    # Одна эпоха
    # ------------------------------------------------------------------

    def _run_epoch(self, loader, train: bool, desc: str) -> dict:
        if train:
            self.model.train()
        else:
            self.model.eval()

        agg = MetricAggregator()
        loss_agg = MetricAggregator()

        iterator = loader
        if HAS_TQDM:
            iterator = tqdm(loader, desc=desc, leave=False)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(iterator):
                try:
                    loss_dict, metrics, B = self._step(batch, train)
                except Exception as e:
                    # Если упало внутри батча — пытаемся вывести полную трассировку,
                    # чтобы пользователь увидел истинную причину (а не молчаливый exit).
                    print(f"\n[!!!] Ошибка на батче {batch_idx} ({desc}): {type(e).__name__}: {e}",
                          flush=True)
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    raise
                agg.update(metrics, B)
                loss_agg.update(loss_dict, B)
                if HAS_TQDM:
                    iterator.set_postfix(
                        loss=f"{loss_dict['loss']:.3f}",
                        ade=f"{metrics.get('ADE', 0.0):.1f}",
                    )
                # Периодически даём пользователю понять, что обучение идёт.
                if not HAS_TQDM and (batch_idx % 10 == 0 or batch_idx == len(loader) - 1):
                    print(f"    [{desc}] batch {batch_idx+1}/{len(loader)} "
                          f"loss={loss_dict['loss']:.3f}", flush=True)

        return {**loss_agg.average(), **agg.average()}

    # ------------------------------------------------------------------
    # Полный цикл обучения
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int = config.NUM_EPOCHS,
            early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE):
        """
        Запуск обучения.
        """
        run_started_at = datetime.now()
        print("\n" + "=" * 70)
        print("  TATPN — Тренировка (Этап 1)")
        print("=" * 70)
        print(f"  Device:           {self.device}")
        print(f"  Train batches:    {len(self.train_loader)}")
        print(f"  Val batches:      {len(self.val_loader)}")
        print(f"  Parameters:       {self.model.num_parameters():,}")
        print(f"  TensorBoard:      {'ON  ('+self.log_dir+')' if HAS_TB else 'OFF'}")
        print(f"  tqdm:             {'ON' if HAS_TQDM else 'OFF'}")
        print(f"  Старт обучения:   {run_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        start_time = time.time()

        # Синхронизируем total_epochs с фактическим количеством эпох
        self._total_epochs = num_epochs

        for epoch in range(1, num_epochs + 1):
            self._current_epoch = epoch
            # Текущий TF-ratio для логирования
            if num_epochs > 1:
                _prog = (epoch - 1) / (num_epochs - 1)
            else:
                _prog = 0.0
            cur_tf = self._tf_start + (self._tf_end - self._tf_start) * _prog

            epoch_started_at = datetime.now()
            print(f"\n>>> Эпоха {epoch}/{num_epochs} — старт "
                  f"{epoch_started_at.strftime('%H:%M:%S')}, "
                  f"teacher_forcing={cur_tf:.2f}", flush=True)
            t_ep = time.time()

            try:
                # === Train ===
                train_stats = self._run_epoch(
                    self.train_loader, train=True,
                    desc=f"Epoch {epoch}/{num_epochs} [train]"
                )

                # === Val ===
                val_stats = self._run_epoch(
                    self.val_loader, train=False,
                    desc=f"Epoch {epoch}/{num_epochs} [val]"
                )
            except Exception as e:
                print(f"\n[!!!] Эпоха {epoch} прервана: {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                # Сохраним то, что успели собрать в истории
                self._save_history()
                raise

            # === Scheduler ===
            self.scheduler.step(val_stats["loss"])
            cur_lr = self.optimizer.param_groups[0]["lr"]

            # === Время эпохи (ВАЖНО: считаем до использования elapsed ниже) ===
            elapsed = time.time() - t_ep
            epoch_finished_at = datetime.now()
            cumulative = time.time() - start_time

            # === Сохраняем историю ===
            self.history["train_loss"].append(train_stats["loss"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["train_metrics"].append(train_stats)
            self.history["val_metrics"].append(val_stats)
            self.history["lr"].append(cur_lr)
            self.history["epoch_seconds"].append(elapsed)

            # === Логи ===
            print(f"    эпоха завершена в {epoch_finished_at.strftime('%H:%M:%S')}, "
                  f"длительность {timedelta(seconds=int(elapsed))}, "
                  f"всего прошло {timedelta(seconds=int(cumulative))}", flush=True)
            self._print_epoch(epoch, num_epochs, train_stats, val_stats, cur_lr, elapsed)
            self._tb_log(epoch, train_stats, val_stats, cur_lr)

            # === Early stopping и чекпоинты ===
            improved = val_stats["loss"] < self.best_val_loss - 1e-6
            if improved:
                self.best_val_loss = val_stats["loss"]
                self.epochs_no_improve = 0
                self.best_state = deepcopy(self.model.state_dict())
                self._save_checkpoint("best.pt", epoch, val_stats)
                print(f"   ✓ Новый лучший val_loss = {val_stats['loss']:.4f}")
            else:
                self.epochs_no_improve += 1

            # Всегда сохраняем "last"
            self._save_checkpoint("last.pt", epoch, val_stats)

            # Сохраняем историю в JSON каждый шаг
            self._save_history()

            if self.epochs_no_improve >= early_stopping_patience:
                print(f"\n[!] Early stopping после {epoch} эпох "
                      f"(нет улучшения {early_stopping_patience} эпох)")
                break

        total = time.time() - start_time
        run_finished_at = datetime.now()
        print("\n" + "=" * 70)
        print(f"  Старт обучения:   {run_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Конец обучения:   {run_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Общая длительность: {timedelta(seconds=int(total))} "
              f"(= {total/60:.1f} мин = {total/3600:.2f} ч)")
        print(f"  Лучший val_loss:    {self.best_val_loss:.4f}")
        print(f"  Чекпоинт:           {os.path.join(self.checkpoint_dir, 'best.pt')}")
        print("=" * 70)

        # Сохраняем длительность в историю (для отчётов)
        self.history["run_started_at"] = run_started_at.isoformat(timespec="seconds")
        self.history["run_finished_at"] = run_finished_at.isoformat(timespec="seconds")
        self.history["total_seconds"] = total
        self._save_history()

        # Восстанавливаем веса лучшей модели
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        if self.writer is not None:
            self.writer.close()

    # ------------------------------------------------------------------
    # Логирование и сохранение
    # ------------------------------------------------------------------

    def _print_epoch(self, epoch, num_epochs, tr, va, lr, elapsed):
        print(
            f"[{epoch:3d}/{num_epochs}] "
            f"{elapsed:5.1f}s  lr={lr:.2e}  "
            f"train_loss={tr['loss']:.3f}  val_loss={va['loss']:.3f}  "
            f"| ADE={va.get('ADE',0):.1f}m  FDE={va.get('FDE',0):.1f}m  "
            f"Acc@1={va.get('Acc@1',0):.3f}  MAE_ETA={va.get('MAE_ETA',0):.2f}s"
        )

    def _tb_log(self, epoch, train_stats, val_stats, lr):
        if self.writer is None:
            return
        for k, v in train_stats.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_stats.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        self.writer.add_scalar("lr", lr, epoch)
        self.writer.flush()

    def _save_checkpoint(self, name: str, epoch: int, val_stats: dict):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_stats": val_stats,
            "best_val_loss": self.best_val_loss,
            "norm_mean": self.norm_mean.cpu().numpy(),
            "norm_std":  self.norm_std.cpu().numpy(),
        }, path)

    def _save_history(self):
        path = os.path.join(self.log_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
