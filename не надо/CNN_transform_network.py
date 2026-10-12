# Общая структура:
#     1. Свёрточный блок (извлекает локальные временные паттерны)
#     2. Трансформерный блок (анализирует глобальные зависимости)
#     3. Полносвязный блок (финальный прогноз)
#
# 1. Свёрточный блок (1D-CNN)
#     Вход:
#         (B, T, 4) где 4 = (x,y,z,Δt)
#     Слои:
#         Последовательность 1D-свёрток + MaxPooling:
#         Conv1d(4→32, kernel=5) → BatchNorm → ReLU → MaxPool1d(2)
#         Conv1d(32→64, kernel=3) → BatchNorm → ReLU → MaxPool1d(2)
#         Conv1d(64→128, kernel=3) → BatchNorm → ReLU → MaxPool1d(2)
#     Выход:
#         (B, T', 128) где T' = T/(2³) (т.к. 3 слоя пулинга с stride=2)
#
# 2. Трансформерный блок
#     Линейная проекция:
#         Linear(128→256) (приводит к d_model)
#     Позиционное кодирование:
#         Синусоидальное (как в оригинальном Transformer)
#     Transformer-Encoder:
#         4 слоя (num_encoder_layers)
#         8 голов внимания (num_heads)
#         FFN-размерность: 512 (dim_feedforward)
#         Нормализация перед слоями (norm_first=True)
#         Dropout=0.1
#     Pooling:
#         Усреднение по временной оси (mean(dim=1))
#
# 3. Полносвязный блок
#     Слои:
#         Linear(256→256) → ReLU → Dropout
#         Linear(256→128) → ReLU → Dropout
#         Linear(128→60) (20 точек × 3 координаты)
#     Выход:
#         (B, 20, 3) (прогноз на pred_horizon шагов)
#
# Ключевые особенности:
#     Гибридная архитектура: CNN (локальные паттерны) + Transformer (глобальные зависимости)
#     Обработка траекторий: Специально адаптирована для последовательностей (x,y,z,Δt)
#     Прогнозирование: Предсказывает сразу весь горизонт (20 точек) одним вызовом
#
# Архитектура сочетает преимущества свёрточных сетей для работы с локальными временными зависимостями и трансформеров
# для анализа долгосрочных взаимосвязей в траекториях.


import torch
import torch.nn as nn
import time
import os
import sys
import datetime
import numpy as np

import metrics
import globals


import sys
import os
import datetime

class Logger:
    def __init__(self, base_dir="logs/CNN_transform"):
        # Создаём папку, если её нет
        os.makedirs(base_dir, exist_ok=True)

        # Формируем имя файла по дате и времени
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(base_dir, f"{timestamp}.txt")

        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        if message.strip():  # чтобы не логировать пустые строки
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            self.terminal.write(timestamp + message)
            self.log.write(timestamp + message)
            self.log.flush()

    def flush(self):
        pass  # нужен для совместимости с sys.stdout

# Используем логгер
sys.stdout = Logger()

class ConvTransformerTrajectory(nn.Module):
    def __init__(
        self,
        conv_channels       = (32, 64, 128),    # кол‑во фильтров в 1‑D свёртках
        kernel_sizes        = (5, 3, 3),        # размеры ядер свертки (время)
        pool_kernel         =  2,               # MaxPool1d kernel size, шаг пулинга (уменьшает длину последовательности в 2 раза на каждом слое)

        d_model             = 256,              # скрытое размер‑во трансформера
        num_encoder_layers  = 4,                # глубина Transformer‑Encoder
        num_heads           = 8,                # кол‑во голов внимания
        dim_feedforward     = 512,              # FFN (скрытый слой) внутри Encoder

        dropout             = 0.1,
        fc_hidden           = (256, 128),       # полносвязные слои после Encoder
        pred_horizon        = 20,               # сколько точек вперёд предсказываем
        max_pos_len         = 2000,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.d_model = d_model
        self.max_pos_len = max_pos_len

        # ---- Блок свёрток -----------------------------------------------
        convs, in_ch = [], 4  # 4 входных признака: x,y,z,Δt
        for out_ch, k in zip(conv_channels, kernel_sizes):
            convs.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(pool_kernel)
            ))
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)

        # ---- Проекция для трансформера ----------------------------------
        # Преобразует выход свёрток в пространство нужной размерности (d_model=256)

        self.proj = nn.Linear(in_ch, d_model)

        # Позиционное кодирование (синусоидальное)
        pe = self.build_sinusoid(max_len=max_pos_len, d_model=d_model)
        self.register_buffer("pos_embedding", pe, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # ---- Полносвязный головной блок --------------------------------
        # Превращает выход Transformer в последовательность из pred_horizon точек по 3 координаты (x,y,z)

        fc_layers, in_f = [], d_model

        for h in fc_hidden:
            fc_layers += [nn.Linear(in_f, h), nn.ReLU(), nn.Dropout(dropout)]
            in_f = h

        fc_layers.append(nn.Linear(in_f, pred_horizon * 3))   # 3 координаты
        self.fc = nn.Sequential(*fc_layers)

    # ---------------------------------------------------------------------
    # Добавляет информацию о позиции точки в последовательности (Transformer сам по себе не чувствителен к порядку)

    @staticmethod
    def build_sinusoid(max_len: int, d_model: int):
        # стандартная синусоидальная PE, аккуратный dtype
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)         # (L, 1)
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)           # (1, D)
        div_term = torch.pow(10000.0, (2 * torch.floor(i / 2)) / d_model)     # (1, D)
        angles = pos / div_term                                               # (L, D)
        pe = torch.zeros((max_len, d_model), dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe                                                             # (max_len, d_model)

    # ---------------------------------------------------------------------
    def forward(self, x):
        """
        x: (B, T, 4) = (размер батча, длина последовательности, признаки)
        """
        B, T, _ = x.shape

        # 1‑D Conv блок – работаем вдоль временной оси
        x = x.transpose(1, 2)      # (B, C_in=4, T)
        x = self.conv(x)           # (B, C_out, T')
        x = x.transpose(1, 2)      # (B, T', C_out)

        # линейная проекция
        x = self.proj(x)           # (B, T', d_model)

        # позиционное кодирование, подготовка для Transformer
        Tprime = x.size(1)
        if Tprime > self.pos_embedding.size(0):
            raise ValueError(f"pos_embedding length {self.pos_embedding.size(0)} < sequence length {Tprime}")
        pe = self.pos_embedding[:Tprime, :].unsqueeze(0).to(x.dtype).to(x.device)   # (1, T', d_model)
        x = x + pe

        # Transformer encoder
        x = self.encoder(x)        # (B, T', d_model)

        # Global mean pooling по времени
        # Усреднение по времени -> (B, 256)
        x = x.mean(dim=1)          # (B, d_model)

        # FC‑блок reshape в секвенцию прогноза
        # head -> (B, H*3) -> (B, H, 3)
        x = self.fc(x)                        # (B, pred_T*3)
        x = x.view(B, self.pred_horizon, 3)   # Разделение координат
        return x

def train_model(train_loader, val_loader):
    print("Тренировка началась")

    # model, loss, optim, scheduler
    model = ConvTransformerTrajectory(pred_horizon=globals.pred_horizon).to(globals.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=globals.lr, weight_decay=globals.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_key = "ADE"      # ключевая метрика для выбора лучшей модели
    best_val = float("inf")
    best_state = None

    train_losses = []
    epoch_times = []

    for epoch in range(1, globals.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        batch_losses = []

        for batch_idx, (inp, tgt) in enumerate(train_loader, start=1):  # Распаковываем кортеж (inp, tgt)
            inp = inp.to(globals.device)  # (B, seq_len, 4)
            tgt = tgt.to(globals.device)  # (B, pred_horizon, 3)

            pred = model(inp)
            loss = criterion(pred, tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if globals.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), globals.grad_clip)
            optimizer.step()

            batch_losses.append(loss.item())

            # === Вывод по батчу ===
            print(f"[Epoch {epoch}/{globals.num_epochs}] "
                  f"Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.6f}")

        # Вычисление статистик
        avg_train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        train_losses.append(avg_train_loss)

        # Валидация
        val_loss, val_mets = evaluate(model, val_loader, criterion, globals.device)
        scheduler.step(val_mets["ADE"])  # адаптивно по ADE

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEpoch:    {epoch}/{globals.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}   | ADE: {val_mets['ADE']:.4f} | FDE: {val_mets['FDE']:.4f} | RMSE: {val_mets['RMSE']:.4f} | MAE: {val_mets['MAE']:.4f}")
        print(f"Time:       {epoch_time:.2f} s      | Speed: {len(train_loader.dataset)/epoch_time:.1f} samples/s")

        # Сохранить по ADE
        if val_mets[best_val_key] < best_val:
            best_val = val_mets[best_val_key]
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_mets,
            }
            torch.save(best_state, os.path.join(globals.save_dir, "best_model.pt"))
            print(f"\nНовая модель сохранена (по критерию {best_val_key})")

        print(f"__________________________________________________________________________________________________________")

    # Финал обучения
    print("\n=== Обучение завершено ===")
    print(f"Лучший val loss (по ADE): {best_val:.4f}")
    print(f"Время обучения: {sum(epoch_times)/60:.1f} минут")


    torch.save({
        "epoch": globals.num_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_losses": train_losses,
        "epoch_times": epoch_times
    }, os.path.join(globals.save_dir, "last_checkpoint.pt"))

    print("\n=== Модель сохранена в last_checkpoint.pt ===")

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    agg_pred, agg_tgt = [], []

    for inp, tgt in loader:
        inp = inp.to(device)
        tgt = tgt.to(device)
        pred = model(inp)
        loss = criterion(pred, tgt)
        losses.append(loss.item())
        agg_pred.append(pred.detach().cpu())
        agg_tgt.append(tgt.detach().cpu())

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    pred_all = torch.cat(agg_pred, dim=0) if agg_pred else torch.empty(0, 1, 3)
    tgt_all  = torch.cat(agg_tgt,  dim=0) if agg_tgt  else torch.empty(0, 1, 3)
    mets = metrics.compute_metrics(pred_all, tgt_all) if pred_all.numel() else {"ADE": float("nan"), "FDE": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}

    return avg_loss, mets

def make_zone_grid(coords=(15.0, 50.0, 85.0), device='cpu', dtype=torch.float32):
    grid = torch.tensor(
        [(cx, cy, cz) for cz in coords for cy in coords for cx in coords],
        dtype=dtype, device=device
    )  # (K=27, 3)
    return grid

def zone_probs(pred_xyz, centers, tau=10.0):
    """
    pred_xyz: (B, H, 3), centers: (K, 3)
    Возвращает probs: (B, H, K)
    """
    B, H, _ = pred_xyz.shape
    K = centers.shape[0]

    dist = torch.cdist(pred_xyz.reshape(-1, 3), centers, p=2).reshape(B, H, K)
    scores = torch.exp(- (dist / tau) ** 2)  # RBF
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
    return probs  # (B, H, K)

def targets_to_zone_labels(tgt_xyz, centers, r_zone=8.0):
    """
    tgt_xyz: (B, H, 3) → labels: (B, H)
    """
    dist = torch.cdist(tgt_xyz.reshape(-1, 3), centers, p=2).reshape(tgt_xyz.shape[0], tgt_xyz.shape[1], centers.shape[0])
    closest = dist.argmin(dim=-1)  # ближайшая зона
    return closest  # (B, H)

def test_model(test_loader):
    print("Тестирование началось")

    model = ConvTransformerTrajectory(pred_horizon=globals.pred_horizon).to(globals.device)

    try:
        state_dict = torch.load(os.path.join(globals.save_dir, "best_model.pt"), map_location=globals.device)
        model.load_state_dict(state_dict['model'])
        print("Загружена best_model.pt")

    except RuntimeError as e:
        print("Ошибка при загрузке best_model.pt:", e)
        checkpoint = torch.load(os.path.join(globals.save_dir, "last_checkpoint.pt"), map_location=globals.device)
        model.load_state_dict(checkpoint['model'])
        print("Загружен last_checkpoint.pt")

    model.eval()

    centers = make_zone_grid(coords=globals.ZONE_COORDS, device=globals.device)

    all_predictions, all_probs, all_labels, test_losses = [], [], [], []

    step_correct = 0
    step_total = 0
    final_step_correct = 0

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(test_loader):
            inp = inp.to(globals.device)  # (B, T, 4)
            tgt = tgt.to(globals.device)  # (B, H, 3) фактическое будущее

            pred_xyz = model(inp)  # (B, H, 3)

            # Вероятности по зонам + 'вне зоны'
            probs = zone_probs(pred_xyz, centers, tau=globals.TAU)  # (B, H, K)

            # Фактические метки зон
            labels = targets_to_zone_labels(tgt, centers, r_zone=globals.R_ZONE)  # (B, H), значения в [0..K] (K='вне зоны')

            # Классификационные метрики
            pred_classes = probs.argmax(dim=-1)  # (B, H)
            step_correct += (pred_classes == labels).sum().item()
            step_total += labels.numel()

            final_step_correct += (pred_classes[:, -1] == labels[:, -1]).sum().item()

            # Координатные метрики (ADE/FDE)
            ade, fde, _, _ = metrics.compute_metrics(pred_xyz, tgt)
            test_losses.append(ade)

            all_predictions.append(pred_xyz.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    # Объединение
    all_predictions = torch.cat(all_predictions, dim=0)  # (N, H, 3)
    all_probs = torch.cat(all_probs, dim=0)  # (N, H, K)
    all_labels = torch.cat(all_labels, dim=0)  # (N, H)

    # Финальные метрики
    step_acc = step_correct / step_total
    final_step_acc = final_step_correct / all_labels.shape[0]
    K = centers.shape[0]
    brier = metrics.brier_score(all_probs.reshape(-1, K), all_labels.reshape(-1), num_classes=K)

    # ADE/FDE заново и точнее по всему массиву
    with torch.no_grad():
        diff = all_predictions - torch.cat([t for _, t in test_loader], dim=0)  # (осторожно: если shuffle=False)
        l2 = torch.norm(diff, dim=-1)
        ade = l2.mean().item()
        fde = l2[:, -1].mean().item()

    print("\n=== Результаты на тесте ===")
    print(f"Step Top-1 Acc (включая 'вне зоны'):  {step_acc:.4f}")
    print(f"Final-step Top-1 Acc:                 {final_step_acc:.4f}")
    print(f"Brier score (ниже — лучше):           {brier:.4f}")
    print(f"ADE (↓):                              {ade:.4f}")
    print(f"FDE (↓):                              {fde:.4f}")

    # Сохранение
    torch.save({
        'predictions': all_predictions,  # (N, H, 3)
        'probs': all_probs,  # (N, H, K)
        'labels': all_labels,  # (N, H)
        'zone_centers': centers.cpu(),
        'r_zone': globals.R_ZONE,
        'tau': globals.TAU,
        'tau_none': globals.TAU_NONE
    }, 'zone_predictions.pt')

    print("\n=== Предсказания и метки сохранены в zone_predictions.pt ===")

    return all_predictions, all_probs, pred_xyz
