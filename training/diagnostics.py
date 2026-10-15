"""
diagnostics.py - диагностика обучения, общая для этапов 2-5.

overfit_check - проверка "сеть способна подогнать малый набор". Это детектор
оторванного графа вычислений, несогласованных таргетов и неверного масштаба
loss: если модель не может запомнить несколько образцов, полное обучение
бессмысленно.

Условия проверки выбраны так, чтобы она была достижимой для исправной сети:
  - малый набор (OVERFIT_SAMPLES образцов), а не весь батч: запомнить 96
    траекторий за сотни шагов нереально даже для исправной модели;
  - dropout выключен: при случайном обнулении точно подогнать нельзя;
  - итоговый loss - среднее по последним шагам (на этапе 5 teacher forcing
    случайный, и один шаг шумит);
  - порог - падение loss в 5 раз (OVERFIT_REDUCE_TO), а не в 10.
Прежний вариант (весь батч, dropout включён, порог в 10 раз) давал ложный FAIL
на исправных моделях.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW

OVERFIT_SAMPLES = 8        # на скольких образцах пробуем переобучиться
OVERFIT_REDUCE_TO = 0.2    # итоговый loss должен быть не выше 20% от начального
OVERFIT_TAIL = 10          # итоговый loss - среднее по последним шагам


def disable_dropout(model: nn.Module) -> None:
    """Обнуляет вероятность dropout во всех слоях (включая межслойный в GRU)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        elif isinstance(m, nn.RNNBase):
            m.dropout = 0.0


def take_samples(batch: dict, n: int) -> dict:
    """Первые n образцов батча (все тензоры режутся по нулевой оси)."""
    return {k: (v[:n] if torch.is_tensor(v) else v) for k, v in batch.items()}


def overfit_check(build_model,
                  step_fn,
                  batch: dict,
                  device,
                  steps: int,
                  n_samples: int = OVERFIT_SAMPLES,
                  lr: float = 1e-3,
                  reduce_to: float = OVERFIT_REDUCE_TO) -> dict:
    """
    Args:
        build_model: функция без аргументов, создаёт новую модель.
        step_fn:     (model, batch, device) -> скалярный loss (с градиентом).
        batch:       батч из DataLoader (словарь тензоров).
        steps:       число шагов оптимизации.
        n_samples:   сколько образцов брать из батча.
        reduce_to:   доля начального loss, ниже которой проверка пройдена.

    Returns:
        dict: loss_start, loss_end, pass, n_samples, steps, reduce_to.
    """
    print("\n" + "=" * 50)
    print(f"  Диагностика: overfit ({steps} шагов, {n_samples} образцов, без dropout)")
    print("=" * 50)

    model = build_model().to(device)
    model.train()
    disable_dropout(model)
    opt = AdamW(model.parameters(), lr=lr)
    small = take_samples(batch, n_samples)
    n_real = next(v.shape[0] for v in small.values() if torch.is_tensor(v))

    losses = []
    for _ in range(steps):
        loss = step_fn(model, small, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    start = losses[0]
    tail = losses[-OVERFIT_TAIL:]
    end = sum(tail) / len(tail)
    ok = end < reduce_to * start or end < 1e-2

    print(f"  loss: {start:.4f} -> {end:.4f} "
          f"(нужно < {reduce_to * start:.4f}, образцов {n_real})")
    print(f"  [{'OK' if ok else 'FAIL'}] сеть способна подогнать малый набор")
    return {"loss_start": start, "loss_end": end, "pass": bool(ok),
            "n_samples": n_real, "steps": steps, "reduce_to": reduce_to}
