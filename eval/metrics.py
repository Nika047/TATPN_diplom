"""
metrics.py - траекторные метрики (numpy), единые для всех этапов.

Метрики (в метрах), все учитывают future_mask (паддинг не штрафуется):
  ADE  - средняя L2-ошибка по всем валидным шагам
  FDE  - L2-ошибка в последней валидной точке каждого образца
  RMSE - корень из среднего квадрата ошибки
  MAE  - средняя абсолютная ошибка по координатам
  ADE@k - ADE на первых k шагах прогноза (k из horizons)

На поздних этапах сюда добавятся метрики цели (Acc@k, Brier) и ETA.
"""

import numpy as np


def trajectory_metrics(pred: np.ndarray,
                       target: np.ndarray,
                       mask: np.ndarray = None,
                       horizons=(10, 20, 50)) -> dict:
    """
    Args:
        pred:   (B, H, 3) - прогноз
        target: (B, H, 3) - истина
        mask:   (B, H)    - 1 для валидных шагов, 0 для паддинга
        horizons: горизонты для ADE@k

    Returns:
        dict метрик (float, в метрах).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = pred - target                                  # (B, H, 3)
    l2 = np.sqrt((diff ** 2).sum(axis=-1))                # (B, H)
    B, H = l2.shape

    if mask is None:
        mask = np.ones_like(l2)
    mask = mask.astype(np.float64)

    denom = max(mask.sum(), 1.0)

    ade = float((l2 * mask).sum() / denom)

    # FDE: ошибка в последней валидной точке каждого образца
    lengths = mask.sum(axis=1).astype(int)                # (B,)
    last_idx = np.clip(lengths - 1, 0, H - 1)             # (B,)
    fde = float(l2[np.arange(B), last_idx].mean())

    sq = (diff ** 2).sum(axis=-1)                         # (B, H)
    rmse = float(np.sqrt((sq * mask).sum() / denom))
    mae = float((np.abs(diff).mean(axis=-1) * mask).sum() / denom)

    out = {"ADE": ade, "FDE": fde, "RMSE": rmse, "MAE": mae}

    for k in horizons:
        if k > H:
            continue
        head = mask.copy()
        head[:, k:] = 0.0
        denom_k = max(head.sum(), 1.0)
        out[f"ADE@{k}"] = float((l2 * head).sum() / denom_k)

    return out


def target_classification_metrics(logits: np.ndarray,
                                  target_id: np.ndarray,
                                  topk=(1, 3)) -> dict:
    """
    Метрики классификации цели.

    Args:
        logits:    (B, K) - логиты по целям
        target_id: (B,)   - истинные индексы цели
        topk:      кортеж k для Acc@k

    Returns:
        dict: Acc@1, Acc@3, Brier.
    """
    logits = np.asarray(logits, dtype=np.float64)
    target_id = np.asarray(target_id).astype(int)
    B, K = logits.shape

    # softmax (со сдвигом для устойчивости)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)

    out = {}
    order = np.argsort(-logits, axis=1)               # индексы по убыванию
    for k in topk:
        topk_idx = order[:, :k]
        correct = (topk_idx == target_id[:, None]).any(axis=1)
        out[f"Acc@{k}"] = float(correct.mean())

    onehot = np.zeros_like(p)
    onehot[np.arange(B), target_id] = 1.0
    out["Brier"] = float(((p - onehot) ** 2).sum(axis=1).mean())
    return out


def eta_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Метрики оценки времени до цели (ETA).

    Args:
        pred:   (B,) - предсказанный ETA (сек)
        target: (B,) - истинный ETA (сек)

    Returns:
        dict: MAE_ETA, RMSE_ETA (секунды).
    """
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "MAE_ETA":  float(np.abs(diff).mean()),
        "RMSE_ETA": float(np.sqrt((diff ** 2).mean())),
    }
