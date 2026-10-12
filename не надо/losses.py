"""
losses.py — Multi-task loss для TATPN.

L_total = α · L_traj + β · L_target + γ · L_eta,

где:
  L_traj   — Huber loss по координатам прогноза с учётом future_mask
             (паддинг будущих точек не штрафуется);
  L_target — CrossEntropy по логиту цели;
  L_eta    — MSE по ETA.

Веса α, β, γ берутся из config.LOSS_ALPHA / LOSS_BETA / LOSS_GAMMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def masked_huber_loss(pred: torch.Tensor,
                       target: torch.Tensor,
                       mask: torch.Tensor,
                       delta: float = config.HUBER_DELTA,
                       coord_scale: float = config.COORD_SCALE) -> torch.Tensor:
    """
    Huber loss с маской (паддинг не штрафуется).

    Координаты приводятся к безразмерному виду делением на coord_scale.
    Это критично: без нормализации Huber loss на ненормированных метрах
    даёт значения ~10^5, что полностью забивает остальные слагаемые
    multi-task loss и приводит к взрыву градиентов.

    Args:
        pred:        (B, H, 3) — предсказанные точки, м
        target:      (B, H, 3) — истинные точки, м
        mask:        (B, H)    — 1 для реальных, 0 для padding
        delta:       параметр Huber (в безразмерных единицах)
        coord_scale: масштаб нормализации координат, м

    Returns:
        scalar loss (безразмерный)
    """
    # Нормируем разность в безразмерные единицы
    pred_n   = pred / coord_scale
    target_n = target / coord_scale
    loss = F.smooth_l1_loss(pred_n, target_n, reduction="none", beta=delta)
    loss = loss.mean(dim=-1)  # (B, H) — среднее по координатам x, y, z

    # Применяем маску
    mask = mask.float()
    masked = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


class TATPNLoss(nn.Module):
    """
    Сводный multi-task loss TATPN.

    Args:
        alpha: вес траекторного loss
        beta:  вес классификационного loss
        gamma: вес ETA loss
        huber_delta: параметр Huber
    """

    def __init__(self,
                 alpha: float = config.LOSS_ALPHA,
                 beta: float = config.LOSS_BETA,
                 gamma: float = config.LOSS_GAMMA,
                 huber_delta: float = config.HUBER_DELTA):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.huber_delta = huber_delta

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self,
                outputs: dict,
                batch: dict) -> tuple:
        """
        Args:
            outputs: dict из TATPN.forward (pred_future, target_logits, eta).
            batch:   dict из DataLoader-а (future_points, future_mask,
                                            target_id, eta).

        Returns:
            total_loss: scalar tensor
            loss_dict: dict с компонентами для логирования
        """
        # === L_traj: Huber по координатам ===
        pred_future   = outputs["pred_future"]      # (B, H, 3)
        target_future = batch["future_points"]      # (B, H, 3)
        future_mask   = batch["future_mask"]        # (B, H)

        loss_traj = masked_huber_loss(
            pred_future, target_future, future_mask, delta=self.huber_delta
        )

        # === L_target: CrossEntropy ===
        target_logits = outputs["target_logits"]   # (B, T)
        target_id     = batch["target_id"].long()  # (B,)
        loss_target = self.ce(target_logits, target_id)

        # === L_eta: MSE ===
        # ETA в секундах может достигать нескольких тысяч; без нормализации
        # MSE даёт значения ~10^7, что несовместимо с остальными слагаемыми.
        # Нормируем ETA на config.ETA_SCALE - тогда MSE получается ~ 1.
        eta_pred = outputs["eta"]                  # (B,)
        eta_true = batch["eta"].float()            # (B,)
        eta_scale = float(config.ETA_SCALE)
        loss_eta = self.mse(eta_pred / eta_scale, eta_true / eta_scale)

        # === Сумма ===
        total = self.alpha * loss_traj + self.beta * loss_target + self.gamma * loss_eta

        return total, {
            "loss":        total.detach(),
            "loss_traj":   loss_traj.detach(),
            "loss_target": loss_target.detach(),
            "loss_eta":    loss_eta.detach(),
        }
