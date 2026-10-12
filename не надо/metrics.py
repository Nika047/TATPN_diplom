"""
metrics.py — Метрики качества для оценки TATPN.

Метрики:
  - Траектория:
      ADE  (Average Displacement Error) — средняя L2-ошибка по всем шагам
      FDE  (Final  Displacement Error)  — L2-ошибка в последней точке
      RMSE / MAE — стандартные регрессионные
      Horizon-ADE — отдельные ADE на горизонтах 10/20/50 шагов
  - Классификация цели:
      Accuracy@1, Accuracy@3
      Brier score (среднеквадратичная ошибка вероятностей)
  - ETA:
      MAE_ETA (секунды)

Все функции корректно учитывают future_mask
(паддинг будущих точек не входит в усреднение).
"""

import math
import torch
import torch.nn.functional as F


# ============================================================================
# Траекторные метрики
# ============================================================================

@torch.no_grad()
def trajectory_metrics(pred: torch.Tensor,
                       target: torch.Tensor,
                       mask: torch.Tensor = None,
                       horizons=(10, 20, 50)) -> dict:
    """
    Метрики траектории с учётом маски.

    Args:
        pred:    (B, H, 3) — прогноз
        target:  (B, H, 3) — истина
        mask:    (B, H)    — 1 для валидных шагов, 0 для padding
        horizons: горизонты для оценки ADE на коротких подпрогнозах

    Returns:
        dict с метриками в метрах.
    """
    diff = pred - target                         # (B, H, 3)
    l2 = torch.linalg.norm(diff, dim=-1)         # (B, H)
    B, H = l2.shape

    if mask is None:
        mask = torch.ones_like(l2)
    mask = mask.float()

    # === ADE ===
    denom = mask.sum().clamp(min=1.0)
    ade = (l2 * mask).sum() / denom

    # === FDE: ошибка в последней РЕАЛЬНОЙ точке каждого сэмпла ===
    # last_idx[b] = последняя валидная позиция в маске
    lengths = mask.sum(dim=1).long().clamp(min=1)            # (B,)
    last_idx = (lengths - 1).clamp(min=0)                    # (B,)
    fde = l2.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

    # === RMSE / MAE ===
    sq = (diff ** 2).sum(dim=-1)                              # (B, H)
    rmse = torch.sqrt(((sq * mask).sum() / denom).clamp(min=1e-12))
    mae = (diff.abs().mean(dim=-1) * mask).sum() / denom

    out = {
        "ADE":  ade.item(),
        "FDE":  fde.item(),
        "RMSE": rmse.item(),
        "MAE":  mae.item(),
    }

    # === Horizon-ADE: ADE на первых K шагах ===
    for k in horizons:
        if k > H:
            continue
        # маска первых k шагов
        head_mask = mask.clone()
        head_mask[:, k:] = 0
        denom_k = head_mask.sum().clamp(min=1.0)
        ade_k = (l2 * head_mask).sum() / denom_k
        out[f"ADE@{k}"] = ade_k.item()

    return out


# ============================================================================
# Метрики классификации цели
# ============================================================================

@torch.no_grad()
def target_classification_metrics(logits: torch.Tensor,
                                  target_id: torch.Tensor,
                                  topk=(1, 3)) -> dict:
    """
    Args:
        logits:    (B, T) — логиты по целям
        target_id: (B,)   — истинные индексы цели

    Returns:
        dict с accuracy@k и Brier score
    """
    target_id = target_id.long()
    B, T = logits.shape
    out = {}

    # === Top-K accuracy ===
    maxk = max(topk)
    _, pred_topk = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
    correct = pred_topk.eq(target_id.unsqueeze(1))                      # (B, maxk)
    for k in topk:
        acc_k = correct[:, :k].any(dim=1).float().mean().item()
        out[f"Acc@{k}"] = acc_k

    # === Brier score ===
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(target_id, num_classes=T).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()
    out["Brier"] = brier

    return out


# ============================================================================
# Метрика ETA
# ============================================================================

@torch.no_grad()
def eta_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Args:
        pred:   (B,) — предсказанный ETA (сек)
        target: (B,) — истинный ETA (сек)

    Returns:
        dict с MAE_ETA и RMSE_ETA
    """
    diff = pred.float() - target.float()
    return {
        "MAE_ETA":  diff.abs().mean().item(),
        "RMSE_ETA": math.sqrt((diff ** 2).mean().item()),
    }


# ============================================================================
# Совмещённая функция для удобства
# ============================================================================

@torch.no_grad()
def compute_all_metrics(outputs: dict, batch: dict) -> dict:
    """
    Собирает все метрики в один словарь.

    Args:
        outputs: dict из TATPN.forward
        batch:   dict из DataLoader

    Returns:
        Словарь со всеми метриками.
    """
    traj = trajectory_metrics(
        outputs["pred_future"], batch["future_points"], batch.get("future_mask")
    )
    tgt = target_classification_metrics(
        outputs["target_logits"], batch["target_id"]
    )
    eta = eta_metrics(outputs["eta"], batch["eta"])
    return {**traj, **tgt, **eta}


# ============================================================================
# Вспомогательная: агрегатор для эпохи
# ============================================================================

class MetricAggregator:
    """
    Накапливает метрики по батчам и возвращает средние по эпохе.
    Усреднение взвешенное по размеру батча.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}
        self.counts = {}

    def update(self, metrics: dict, batch_size: int):
        for key, val in metrics.items():
            if val is None:
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(val) * batch_size
            self.counts[key] = self.counts.get(key, 0) + batch_size

    def average(self) -> dict:
        out = {}
        for key, s in self.sums.items():
            c = self.counts.get(key, 1)
            out[key] = s / max(c, 1)
        return out
