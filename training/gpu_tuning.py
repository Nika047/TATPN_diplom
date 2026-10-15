"""
gpu_tuning.py - настройка обучения под конкретную видеокарту.

Что делает:
  - setup_cuda(): включает режимы PyTorch, ускоряющие обучение на NVIDIA GPU
    (TF32 для матричных умножений, cudnn.benchmark) и печатает параметры
    карты: название, объём видеопамяти, compute capability;
  - probe_batch_size(): подбирает наибольший размер батча, который помещается
    в видеопамять. Пробует кандидатов от большего к меньшему, на каждом
    выполняя два настоящих шага обучения (forward + backward + optimizer),
    и берёт первый, где не случилось нехватки памяти и пик потребления не
    превысил заданную долю видеопамяти (запас на валидацию и фрагментацию);
  - tune_batch(): обёртка для скриптов этапов: подбирает батч и пересоздаёт
    загрузчики данных.

Почему батч подбирается автоматически, а не задаётся числом: потребление
памяти Mamba (selective scan) растёт как B * L * d_inner * d_state, поэтому
разумное значение зависит от объёма видеопамяти, длины окна и числа слоёв.
Подбор по факту надёжнее оценки на глаз и не зависит от модели видеокарты.

На CPU все функции ничего не делают (возвращают исходные значения).
"""

import gc
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, default_collate

# Кандидаты размера батча от большего к меньшему. Верхняя граница намеренно
# ограничена: при очень большом батче на ~2600 обучающих траекториях остаётся
# слишком мало шагов оптимизации за эпоху (N / B).
DEFAULT_CANDIDATES = (128, 96, 64, 48, 32, 24, 16, 8)

# Доля видеопамяти, которую можно занять на шаге обучения. Остальное -
# запас на валидацию, накопители метрик и фрагментацию памяти.
DEFAULT_VRAM_FRACTION = 0.85


def is_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


def setup_cuda(device) -> None:
    """Включает ускоряющие режимы и печатает параметры видеокарты."""
    if not is_cuda(device):
        return

    # TF32: матричные умножения на Ampere и новее заметно быстрее при
    # незначительной потере точности (для обучения допустимо).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Свёртки CNN имеют фиксированные формы - выбор самого быстрого алгоритма
    # окупается.
    torch.backends.cudnn.benchmark = True

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"[gpu] {props.name}, {total_gb:.1f} ГБ VRAM, "
          f"compute capability {props.major}.{props.minor}, "
          f"torch {torch.__version__}, CUDA {torch.version.cuda}")


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probe_batch_size(build_model,
                     dataset,
                     step_fn,
                     device,
                     candidates=DEFAULT_CANDIDATES,
                     vram_fraction: float = DEFAULT_VRAM_FRACTION):
    """
    Подбирает наибольший размер батча, помещающийся в видеопамять.

    Args:
        build_model: функция без аргументов, создаёт новую модель.
        dataset:     обучающий датасет (элементы - словари тензоров).
        step_fn:     функция (model, batch, device) -> скалярный loss.
        device:      torch.device.
        candidates:  размеры батча для проверки.
        vram_fraction: допустимая доля видеопамяти для пика потребления.

    Returns:
        (chosen_batch, rows): выбранный размер и таблица проб
        [(batch, peak_gb, sec_per_step, status), ...].
    """
    if not is_cuda(device):
        return None, []

    total = torch.cuda.get_device_properties(device).total_memory
    limit = vram_fraction * total
    rows = []
    chosen = None

    for b in sorted(candidates, reverse=True):
        if b > len(dataset):
            continue
        model = opt = batch = loss = None
        status = "ok"
        peak = None
        sec = None
        try:
            _cleanup()
            torch.cuda.reset_peak_memory_stats(device)
            model = build_model().to(device)
            model.train()
            opt = AdamW(model.parameters(), lr=1e-4)
            batch = default_collate([dataset[i] for i in range(b)])

            # Два шага: первый "прогревает" выделение памяти и cudnn,
            # по второму меряем скорость.
            t0 = None
            for k in range(2):
                if k == 1:
                    torch.cuda.synchronize(device)
                    t0 = time.time()
                loss = step_fn(model, batch, device)
                opt.zero_grad()
                loss.backward()
                opt.step()
            torch.cuda.synchronize(device)
            sec = time.time() - t0
            peak = torch.cuda.max_memory_allocated(device)

            if peak > limit:
                status = f"не проходит запас ({vram_fraction:.0%} VRAM)"
            else:
                chosen = b
        except torch.cuda.OutOfMemoryError:
            status = "нехватка памяти"
        finally:
            del model, opt, batch, loss
            _cleanup()

        rows.append((b, None if peak is None else peak / (1024 ** 3), sec, status))
        if chosen is not None:
            break

    return chosen, rows


def rebuild_loader(loader: DataLoader, batch_size: int) -> DataLoader:
    """Пересоздаёт DataLoader с новым размером батча, сохраняя остальное."""
    shuffle = isinstance(loader.sampler, RandomSampler)
    return DataLoader(loader.dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=loader.drop_last,
                      num_workers=0)


def tune_batch(build_model, step_fn, train_loader, val_loader, test_loader,
               device, default_batch: int,
               candidates=DEFAULT_CANDIDATES,
               vram_fraction: float = DEFAULT_VRAM_FRACTION):
    """
    Подбирает батч под видеопамять и пересоздаёт загрузчики.

    Returns:
        (batch, train_loader, val_loader, test_loader). На CPU или если ни один
        кандидат не прошёл - исходные значения без изменений.
    """
    if not is_cuda(device):
        return default_batch, train_loader, val_loader, test_loader

    print("\n" + "=" * 50)
    print("  GPU: подбор размера батча по видеопамяти")
    print("=" * 50)

    chosen, rows = probe_batch_size(build_model, train_loader.dataset, step_fn,
                                    device, candidates, vram_fraction)

    print(f"  {'batch':>6} {'пик VRAM, ГБ':>14} {'сек/шаг':>9}  результат")
    for b, peak_gb, sec, status in rows:
        peak_s = "-" if peak_gb is None else f"{peak_gb:.2f}"
        sec_s = "-" if sec is None else f"{sec:.2f}"
        print(f"  {b:>6} {peak_s:>14} {sec_s:>9}  {status}")

    if chosen is None:
        print(f"  [!] ни один кандидат не прошёл, остаётся batch={default_batch}")
        return default_batch, train_loader, val_loader, test_loader

    print(f"  -> выбран batch={chosen}")
    if chosen == train_loader.batch_size:
        return chosen, train_loader, val_loader, test_loader
    return (chosen,
            rebuild_loader(train_loader, chosen),
            rebuild_loader(val_loader, chosen),
            rebuild_loader(test_loader, chosen))
