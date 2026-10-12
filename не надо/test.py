import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    # Пути
    DATA_DIR = "flight_logs_generated/data"  # Папка с CSV файлами
    MODEL_SAVE_PATH = "trajectory_predictor.pth"

    # Параметры данных
    SEQ_LEN = 60  # Длина исторического контекста (60 точек)
    PRED_LEN = 30  # Длина прогноза (30 точек)
    FEATURES = ['X', 'Y', 'Z']  # Используемые признаки
    BATCH_SIZE = 32
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0  # Исправлено для Windows

    # Параметры модели
    CNN_OUT_CHANNELS = 64
    EMBED_SIZE = 128
    NUM_HEADS = 4  # Уменьшено для стабильности
    NUM_LAYERS = 3  # Уменьшено для стабильности
    DROPOUT = 0.1
    FF_DIM = 256

    # Обучение
    EPOCHS = 50  # Уменьшено для тестирования
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # для Early Stopping

    # Устройство
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()


# ==================== ПОДГОТОВКА ДАННЫХ ====================
class TrajectoryDataset(Dataset):
    """Датасет для траекторий с переменной длиной"""

    def __init__(self, trajectories, scaler=None, fit_scaler=True):
        """
        trajectories: список numpy массивов формы (L, 3), где L - длина траектории
        """
        self.trajectories = trajectories

        if fit_scaler:
            # Объединяем все траектории для обучения скейлера
            all_data = np.vstack(trajectories)
            self.scaler = StandardScaler()
            self.scaler.fit(all_data)
        else:
            self.scaler = scaler

        # Нормализуем все траектории
        self.normalized_trajectories = []
        for traj in trajectories:
            normalized = self.scaler.transform(traj)
            self.normalized_trajectories.append(normalized)

        # Создаем пары (история, цель)
        self.samples = []
        for traj in self.normalized_trajectories:
            if len(traj) >= config.SEQ_LEN + config.PRED_LEN:
                for i in range(0, len(traj) - config.SEQ_LEN - config.PRED_LEN + 1, 5):  # Шаг 5 для уменьшения данных
                    history = traj[i:i + config.SEQ_LEN]
                    target = traj[i + config.SEQ_LEN:i + config.SEQ_LEN + config.PRED_LEN]
                    self.samples.append((history, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        return (
            torch.FloatTensor(history),  # [SEQ_LEN, 3]
            torch.FloatTensor(target)  # [PRED_LEN, 3]
        )


def load_trajectories_from_csv(data_dir):
    """Загрузка всех CSV файлов из папки"""
    trajectories = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)

                # Проверяем наличие необходимых колонок
                required_cols = ['X', 'Y', 'Z']
                if all(col in df.columns for col in required_cols):
                    # Создаем массив траектории
                    traj = df[required_cols].values.astype(np.float32)

                    # Проверяем на NaN и бесконечные значения
                    if (not np.any(np.isnan(traj))) and (not np.any(np.isinf(traj))):
                        # Проверяем минимальную длину
                        if len(traj) >= config.SEQ_LEN + config.PRED_LEN:
                            trajectories.append(traj)
                            print(f"✓ Загружена траектория из {filename}: {len(traj)} точек")
                        else:
                            print(
                                f"✗ Пропущен {filename}: слишком короткая ({len(traj)} < {config.SEQ_LEN + config.PRED_LEN})")
                    else:
                        print(f"✗ Пропущен {filename}: содержит NaN/Inf значения")
                else:
                    print(f"✗ Пропущен {filename}: отсутствуют необходимые колонки")

            except Exception as e:
                print(f"✗ Ошибка при загрузке {filename}: {e}")

    print(f"\nВсего загружено траекторий: {len(trajectories)}")

    if trajectories:
        # Статистика по длинам траекторий
        lengths = [len(t) for t in trajectories]
        print(f"Минимальная длина: {min(lengths)}")
        print(f"Максимальная длина: {max(lengths)}")
        print(f"Средняя длина: {np.mean(lengths):.1f}")

    return trajectories


# ==================== МОДЕЛЬ CNN-TRANSFORMER ====================
class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return x


class CNNFeatureExtractor(nn.Module):
    """CNN для извлечения локальных признаков из траектории"""

    def __init__(self, input_channels=3, output_channels=64):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=5,
            padding=2,
            stride=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2,
            stride=1
        )

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_channels)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
        x = x.transpose(1, 2)

        # Первый сверточный блок
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        # Второй сверточный блок
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        # Третий сверточный блок
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        # Возвращаем к исходной размерности
        x = x.transpose(1, 2)  # [batch_size, seq_len_compressed, output_channels]

        return x


class TrajectoryTransformer(nn.Module):
    """Гибридная модель CNN-Transformer для прогнозирования траекторий"""

    def __init__(self):
        super().__init__()

        # CNN для извлечения признаков
        self.cnn = CNNFeatureExtractor(
            input_channels=len(config.FEATURES),
            output_channels=config.CNN_OUT_CHANNELS
        )

        # Линейный слой для проекции в размер эмбеддинга
        self.input_projection = nn.Linear(config.CNN_OUT_CHANNELS, config.EMBED_SIZE)

        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(config.EMBED_SIZE)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_SIZE,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.FF_DIM,
            dropout=config.DROPOUT,
            batch_first=True,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.NUM_LAYERS
        )

        # Декодер для прогнозирования
        self.decoder = nn.Sequential(
            nn.Linear(config.EMBED_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.PRED_LEN * len(config.FEATURES))
        )

        # Слои нормализации
        self.layer_norm1 = nn.LayerNorm(config.EMBED_SIZE)
        self.layer_norm2 = nn.LayerNorm(config.EMBED_SIZE)

        # Dropout
        self.dropout = nn.Dropout(config.DROPOUT)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов для лучшей сходимости"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [batch_size, seq_len, features]
        batch_size = x.size(0)

        # 1. Извлечение признаков CNN
        cnn_features = self.cnn(x)  # [batch_size, seq_len_compressed, cnn_channels]

        # 2. Проекция в пространство эмбеддингов
        projected = self.input_projection(cnn_features)  # [batch_size, seq_len_compressed, embed_size]

        # 3. Позиционное кодирование
        projected = self.pos_encoder(projected)
        projected = self.dropout(projected)

        # 4. Transformer Encoder
        transformer_out = self.transformer_encoder(projected)
        transformer_out = self.layer_norm1(transformer_out)

        # 5. Используем последний скрытый вектор для прогнозирования
        last_hidden = transformer_out[:, -1, :]  # [batch_size, embed_size]
        last_hidden = self.layer_norm2(last_hidden)

        # 6. Декодирование прогноза
        predictions = self.decoder(last_hidden)  # [batch_size, pred_len * features]

        # 7. Reshape в правильную форму
        predictions = predictions.view(
            batch_size,
            config.PRED_LEN,
            len(config.FEATURES)
        )  # [batch_size, pred_len, features]

        return predictions


# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================
class EarlyStopping:
    """Ранняя остановка для предотвращения переобучения"""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0

    for batch_idx, (history, target) in enumerate(dataloader):
        history = history.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(history)

        # Вычисление loss
        loss = criterion(predictions, target)

        # Backward pass
        loss.backward()

        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def validate(model, dataloader, criterion, device):
    """Валидация модели"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for history, target in dataloader:
            history = history.to(device)
            target = target.to(device)

            predictions = model(history)
            loss = criterion(predictions, target)
            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def train_model(model, train_loader, val_loader, epochs=50):
    """Полный цикл обучения"""
    device = config.DEVICE
    model = model.to(device)

    # Loss функция и оптимизатор
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler для уменьшения learning rate (без verbose параметра)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Ранняя остановка
    early_stopping = EarlyStopping(patience=config.PATIENCE)

    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_model_state = None

    print(f"Начинаем обучение на устройстве: {device}")
    print(f"Размер батча: {config.BATCH_SIZE}")
    print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
    print(f"Количество параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        print(f"\n{'=' * 50}")
        print(f"Эпоха {epoch + 1}/{epochs}")

        # Обучение
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Валидация
        val_loss = validate(model, val_loader, criterion, device)

        # Обновление истории
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Scheduler step
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"✓ Новый лучший результат! Сохранение модели...")

        # Проверка ранней остановки
        if early_stopping(val_loss):
            print(f"\nРанняя остановка на эпохе {epoch + 1}")
            break

    # Загрузка лучших весов
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, best_val_loss


# ==================== ВИЗУАЛИЗАЦИЯ ====================
def plot_training_history(history, best_val_loss):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # График loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].axhline(y=best_val_loss, color='r', linestyle='--', alpha=0.5, label=f'Best Val: {best_val_loss:.6f}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График в логарифмической шкале
    axes[1].semilogy(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].semilogy(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Training History (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # График learning rate
    axes[2].plot(history['lr'], label='Learning Rate', linewidth=2, color='purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, dataset, scaler, num_examples=3):
    """Визуализация прогнозов модели"""
    device = config.DEVICE
    model.eval()

    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4 * num_examples))

    if num_examples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)

        for idx, sample_idx in enumerate(indices):
            history, true_target = dataset[sample_idx]

            # Добавляем batch dimension
            history_input = history.unsqueeze(0).to(device)

            # Получаем прогноз
            prediction = model(history_input)
            prediction = prediction.cpu().squeeze(0).numpy()

            # Обратная нормализация
            history_original = scaler.inverse_transform(history.numpy())
            true_target_original = scaler.inverse_transform(true_target.numpy())
            prediction_original = scaler.inverse_transform(prediction)

            # Временные метки
            history_time = np.arange(len(history_original))
            target_time = np.arange(len(history_original),
                                    len(history_original) + len(true_target_original))

            # Вычисление ошибки
            mse = np.mean((true_target_original - prediction_original) ** 2)
            rmse = np.sqrt(mse)

            # Графики для каждой координаты
            for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
                ax = axes[idx, coord_idx]

                # История
                ax.plot(history_time, history_original[:, coord_idx],
                        'b-', linewidth=2, label='История', alpha=0.7)

                # Истинный будущий путь
                ax.plot(target_time, true_target_original[:, coord_idx],
                        'g-', linewidth=2, label='Истинный путь', alpha=0.7)

                # Прогноз
                ax.plot(target_time, prediction_original[:, coord_idx],
                        'r--', linewidth=2, label='Прогноз', alpha=0.9)

                # Заполнение между истинным и прогнозом
                ax.fill_between(target_time,
                                true_target_original[:, coord_idx],
                                prediction_original[:, coord_idx],
                                alpha=0.2, color='red')

                ax.set_xlabel('Время (шаги)')
                ax.set_ylabel(coord_name)
                ax.set_title(f'{coord_name} координата - Пример {idx + 1}')
                ax.legend()
                ax.grid(True, alpha=0.3)

            print(f"Пример {idx + 1}: RMSE = {rmse:.4f}")

    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== ОСНОВНОЙ ПРОЦЕСС ====================
def main():
    print("=" * 60)
    print("Система прогнозирования траекторий на основе CNN-Transformer")
    print("=" * 60)

    # Проверка PyTorch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Используемое устройство: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    if not os.path.exists(config.DATA_DIR):
        print(f"Ошибка: папка '{config.DATA_DIR}' не существует!")
        return

    trajectories = load_trajectories_from_csv(config.DATA_DIR)

    if len(trajectories) == 0:
        print("Ошибка: не найдено траекторий в указанной папке!")
        print("\nСоздайте папку 'trajectory_data' и добавьте CSV файлы с колонками:")
        print("X, Y, Z (обязательно), Date, Time (опционально)")
        print("\nПример содержимого CSV:")
        print("X,Y,Z,Date,Time")
        print("48.288974,36.758579,0.000282,2025-10-22,17:23:50.073798")
        print("48.288967,36.758583,-1.154545,2025-10-22,17:23:50.173798")
        return

    # 2. Разделение на train/val
    print("\n2. Разделение данных...")
    # Разделяем траектории, а не сэмплы
    train_size = int(0.8 * len(trajectories))
    val_size = len(trajectories) - train_size

    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:]

    print(f"Обучающие траектории: {len(train_trajectories)}")
    print(f"Валидационные траектории: {len(val_trajectories)}")

    # 3. Создание датасетов
    print("\n3. Создание датасетов...")
    train_dataset = TrajectoryDataset(train_trajectories, fit_scaler=True)
    val_dataset = TrajectoryDataset(
        val_trajectories,
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    print(f"Обучающих сэмплов: {len(train_dataset)}")
    print(f"Валидационных сэмплов: {len(val_dataset)}")

    # 4. Создание DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.BATCH_SIZE, len(train_dataset)),
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.BATCH_SIZE, len(val_dataset)),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 5. Создание модели
    print("\n4. Создание модели...")
    model = TrajectoryTransformer()

    # 6. Обучение модели
    print("\n5. Обучение модели...")
    history, best_val_loss = train_model(model, train_loader, val_loader, epochs=config.EPOCHS)

    # 7. Сохранение модели
    print("\n6. Сохранение модели...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': train_dataset.scaler,
        'config': config.__dict__,
        'history': history,
        'best_val_loss': best_val_loss
    }, config.MODEL_SAVE_PATH)
    print(f"✓ Модель сохранена в: {config.MODEL_SAVE_PATH}")

    # 8. Визуализация
    print("\n7. Визуализация результатов...")
    plot_training_history(history, best_val_loss)

    if len(val_dataset) > 0:
        visualize_predictions(model, val_dataset, train_dataset.scaler, num_examples=min(3, len(val_dataset)))

    # 9. Оценка на тестовых данных
    print("\n8. Финальная оценка...")
    final_val_loss = validate(model, val_loader, nn.MSELoss(), config.DEVICE)

    # Вычисление ошибки в оригинальных единицах измерения
    if hasattr(train_dataset.scaler, 'scale_'):
        feature_stds = train_dataset.scaler.scale_  # стандартные отклонения для каждого признака
        avg_std = np.mean(feature_stds)
        rmse_original = np.sqrt(final_val_loss) * avg_std
    else:
        # Если StandardScaler не имеет scale_ атрибута (старые версии sklearn)
        rmse_original = np.sqrt(final_val_loss)

    print(f"\nФинальная оценка:")
    print(f"Validation Loss (MSE нормализованное): {final_val_loss:.6f}")
    print(f"RMSE (нормализованное): {np.sqrt(final_val_loss):.6f}")
    print(f"RMSE (оригинальные единицы): {rmse_original:.4f}")

    print("\n" + "=" * 60)
    print("Обучение завершено успешно!")
    print("=" * 60)


# ==================== ФУНКЦИЯ ДЛЯ ИСПОЛЬЗОВАНИЯ ОБУЧЕННОЙ МОДЕЛИ ====================
class TrajectoryPredictor:
    """Класс для использования обученной модели"""

    def __init__(self, model_path=config.MODEL_SAVE_PATH):
        """Загрузка обученной модели"""
        print(f"Загрузка модели из {model_path}...")

        if not os.path.exists(model_path):
            print(f"Ошибка: файл модели '{model_path}' не найден!")
            print("Сначала обучите модель с помощью main()")
            return

        # Загружаем на CPU для совместимости
        checkpoint = torch.load(model_path, map_location='cpu')

        # Восстанавливаем конфигурацию
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)

        self.config = Config(checkpoint['config'])

        # Создаем модель с той же архитектурой
        self.model = TrajectoryTransformer()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Загружаем scaler
        self.scaler = checkpoint['scaler']

        # История обучения
        self.history = checkpoint.get('history', {})
        self.best_val_loss = checkpoint.get('best_val_loss', 0)

        print(f"✓ Модель успешно загружена")
        print(f"  SEQ_LEN: {self.config.SEQ_LEN}")
        print(f"  PRED_LEN: {self.config.PRED_LEN}")
        print(f"  Лучшая Val Loss: {self.best_val_loss:.6f}")

    def predict(self, trajectory, device=None):
        """
        Прогнозирование будущей траектории

        Args:
            trajectory: numpy array формы (N, 3) или список с X, Y, Z координатами
            device: устройство для вычислений (по умолчанию CPU)

        Returns:
            prediction: numpy array формы (PRED_LEN, 3) - прогноз
            history_original: исходная история (последние SEQ_LEN точек)
        """
        if device is None:
            device = torch.device('cpu')

        # Преобразование входных данных
        if isinstance(trajectory, list):
            trajectory = np.array(trajectory)

        # Проверка размерности
        if trajectory.ndim == 1:
            if len(trajectory) == 3:  # Одна точка [x, y, z]
                trajectory = trajectory.reshape(1, 3)
            else:
                raise ValueError(f"Неправильная размерность входных данных: {trajectory.shape}")

        # Проверка на NaN
        if np.any(np.isnan(trajectory)):
            raise ValueError("Входные данные содержат NaN значения")

        # Берем последние SEQ_LEN точек
        if len(trajectory) >= self.config.SEQ_LEN:
            history = trajectory[-self.config.SEQ_LEN:]
        else:
            # Если данных меньше, чем SEQ_LEN, дополняем последней точкой
            print(f"Предупреждение: траектория слишком короткая ({len(trajectory)} < {self.config.SEQ_LEN})")
            last_point = trajectory[-1:]
            repeats = self.config.SEQ_LEN - len(trajectory)
            padding = np.repeat(last_point, repeats, axis=0)
            history = np.vstack([trajectory, padding])

        # Нормализация
        history_normalized = self.scaler.transform(history)

        # Преобразование в тензор
        history_tensor = torch.FloatTensor(history_normalized).unsqueeze(0).to(device)

        # Прогнозирование
        with torch.no_grad():
            self.model.to(device)
            prediction_normalized = self.model(history_tensor)
            prediction_normalized = prediction_normalized.squeeze(0).cpu().numpy()

        # Обратная нормализация
        prediction = self.scaler.inverse_transform(prediction_normalized)
        history_original = self.scaler.inverse_transform(history_normalized)

        return prediction, history_original

    def predict_from_csv(self, csv_path, device=None):
        """Прогнозирование непосредственно из CSV файла"""
        try:
            df = pd.read_csv(csv_path)

            # Проверяем наличие необходимых колонок
            if not all(col in df.columns for col in ['X', 'Y', 'Z']):
                # Пробуем найти колонки с другими именами
                possible_cols = {
                    'x': 'X', 'y': 'Y', 'z': 'Z',
                    'lat': 'X', 'lon': 'Y', 'alt': 'Z',
                    'latitude': 'X', 'longitude': 'Y', 'altitude': 'Z'
                }

                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower in possible_cols:
                        df[possible_cols[col_lower]] = df[col]

                if not all(col in df.columns for col in ['X', 'Y', 'Z']):
                    raise ValueError("CSV файл должен содержать координаты X, Y, Z или их аналоги")

            # Извлекаем траекторию
            trajectory = df[['X', 'Y', 'Z']].values.astype(np.float32)

            # Проверяем данные
            if len(trajectory) == 0:
                raise ValueError("CSV файл не содержит данных")

            # Прогнозирование
            prediction, history = self.predict(trajectory, device)

            # Создаем DataFrame с результатами
            pred_df = pd.DataFrame(prediction, columns=['Pred_X', 'Pred_Y', 'Pred_Z'])

            # Добавляем временные метки (если они есть в исходных данных)
            if 'Time' in df.columns:
                try:
                    last_time = pd.to_datetime(df['Time'].iloc[-1])
                    # Предполагаем частоту 100мс (как в примере)
                    time_delta = pd.to_timedelta('100ms')
                    pred_times = [last_time + time_delta * (i + 1) for i in range(len(prediction))]
                    pred_df['Pred_Time'] = pred_times
                except:
                    pass

            return pred_df, history, trajectory

        except Exception as e:
            print(f"Ошибка при обработке файла {csv_path}: {e}")
            return None, None, None

    def plot_prediction(self, csv_path, save_path=None):
        """Визуализация прогноза для CSV файла"""
        pred_df, history, original_traj = self.predict_from_csv(csv_path)

        if pred_df is None:
            return

        # Создаем график
        fig = plt.figure(figsize=(15, 10))

        # 1. 3D траектория
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        # Оригинальная траектория (вся)
        ax1.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2],
                 'gray', alpha=0.3, label='Вся траектория')

        # История (последние SEQ_LEN точек)
        ax1.plot(history[:, 0], history[:, 1], history[:, 2],
                 'b-', linewidth=3, label='История', alpha=0.7)

        # Прогноз
        ax1.plot(pred_df['Pred_X'], pred_df['Pred_Y'], pred_df['Pred_Z'],
                 'r--', linewidth=3, label='Прогноз', alpha=0.9)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Траектория')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. XY проекция
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(history[:, 0], history[:, 1], 'b-', linewidth=2, label='История')
        ax2.plot(pred_df['Pred_X'], pred_df['Pred_Y'], 'r--', linewidth=2, label='Прогноз')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY проекция')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        # 3. XZ проекция
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(history[:, 0], history[:, 2], 'b-', linewidth=2, label='История')
        ax3.plot(pred_df['Pred_X'], pred_df['Pred_Z'], 'r--', linewidth=2, label='Прогноз')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ проекция (Высота)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Временные ряды
        ax4 = fig.add_subplot(2, 2, 4)
        time_points = range(len(history) + len(pred_df))

        # X координата
        ax4.plot(time_points[:len(history)], history[:, 0], 'b-', label='X (история)')
        ax4.plot(time_points[len(history):], pred_df['Pred_X'], 'b--', label='X (прогноз)')

        # Y координата
        ax4.plot(time_points[:len(history)], history[:, 1], 'g-', label='Y (история)')
        ax4.plot(time_points[len(history):], pred_df['Pred_Y'], 'g--', label='Y (прогноз)')

        # Z координата
        ax4.plot(time_points[:len(history)], history[:, 2], 'r-', label='Z (история)')
        ax4.plot(time_points[len(history):], pred_df['Pred_Z'], 'r--', label='Z (прогноз)')

        ax4.set_xlabel('Время (шаги)')
        ax4.set_ylabel('Значения координат')
        ax4.set_title('Временные ряды координат')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        # Выводим статистику
        print("\nСтатистика прогноза:")
        print(f"История: {len(history)} точек")
        print(f"Прогноз: {len(pred_df)} точек")
        print(f"\nПервые 5 точек прогноза:")
        print(pred_df.head())

        return pred_df


# ==================== ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ ====================
def generate_sample_data(num_files=5, points_per_file=200):
    """Генерация тестовых CSV файлов для демонстрации"""
    import os

    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    for file_idx in range(num_files):
        filename = os.path.join(config.DATA_DIR, f"trajectory_{file_idx + 1:03d}.csv")

        # Генерация реалистичной траектории полета
        time_steps = np.linspace(0, 20, points_per_file)

        # X координата - линейное движение с маневрами
        X = 48.0 + 0.2 * time_steps + 0.5 * np.sin(time_steps * 0.5)

        # Y координата - также линейное с маневрами
        Y = 36.0 + 0.15 * time_steps + 0.3 * np.cos(time_steps * 0.7)

        # Z координата - изменение высоты (набор -> полет -> снижение)
        climb_duration = int(points_per_file * 0.3)
        cruise_duration = int(points_per_file * 0.4)
        descend_duration = points_per_file - climb_duration - cruise_duration

        Z_climb = 0 + 0.5 * time_steps[:climb_duration]
        Z_cruise = 0.5 * time_steps[climb_duration] * np.ones(cruise_duration)
        Z_descend = Z_cruise[-1] - 0.3 * time_steps[:descend_duration]
        Z = np.concatenate([Z_climb, Z_cruise, Z_descend])

        # Обрезаем или дополняем до нужной длины
        if len(Z) > points_per_file:
            Z = Z[:points_per_file]
        elif len(Z) < points_per_file:
            Z = np.pad(Z, (0, points_per_file - len(Z)), mode='edge')

        # Добавляем случайный шум
        noise_xy = np.random.normal(0, 0.001, (points_per_file, 2))
        noise_z = np.random.normal(0, 0.01, points_per_file)

        X += noise_xy[:, 0]
        Y += noise_xy[:, 1]
        Z += noise_z

        # Создаем временные метки
        start_time = pd.Timestamp('2025-01-15 17:23:50')
        time_deltas = pd.to_timedelta(np.arange(points_per_file) * 100, unit='ms')
        times = start_time + time_deltas

        # Создаем DataFrame
        df = pd.DataFrame({
            'X': X,
            'Y': Y,
            'Z': Z,
            'Date': times.strftime('%Y-%m-%d'),
            'Time': times.strftime('%H:%M:%S.%f')
        })

        # Сохраняем в CSV
        df.to_csv(filename, index=False)
        print(f"✓ Создан тестовый файл: {filename}")

    print(f"\nСоздано {num_files} тестовых файлов в папке {config.DATA_DIR}")
    print("Теперь вы можете запустить обучение модели.")


# ==================== ТОЧКА ВХОДА ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Система прогнозирования траекторий')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'generate', 'test'],
                        help='Режим работы: train, predict, generate, test')
    parser.add_argument('--csv', type=str, help='Путь к CSV файлу для прогноза')
    parser.add_argument('--model', type=str, default=config.MODEL_SAVE_PATH,
                        help='Путь к файлу модели')

    args = parser.parse_args()

    if args.mode == 'train':
        # Проверяем, есть ли данные
        if not os.path.exists(config.DATA_DIR) or len(os.listdir(config.DATA_DIR)) == 0:
            print("Данные не найдены. Генерация демонстрационных данных...")
            generate_sample_data(num_files=5, points_per_file=200)

        # Запускаем обучение
        main()

    elif args.mode == 'predict':
        if not args.csv:
            print("Ошибка: для режима predict укажите CSV файл через --csv")
            exit(1)

        if not os.path.exists(args.csv):
            print(f"Ошибка: файл '{args.csv}' не найден")
            exit(1)

        # Создаем предсказатель
        predictor = TrajectoryPredictor(args.model)

        # Делаем прогноз
        pred_df = predictor.plot_prediction(args.csv, save_path='prediction_result.png')

        if pred_df is not None:
            # Сохраняем прогноз в CSV
            pred_csv_path = args.csv.replace('.csv', '_prediction.csv')
            pred_df.to_csv(pred_csv_path, index=False)
            print(f"\n✓ Прогноз сохранен в: {pred_csv_path}")

    elif args.mode == 'generate':
        generate_sample_data(num_files=10, points_per_file=300)

    elif args.mode == 'test':
        # Тестовый режим для проверки
        print("Тестовый режим...")
        print(f"PyTorch версия: {torch.__version__}")
        print(f"Устройство: {config.DEVICE}")

        # Проверяем наличие данных
        if os.path.exists(config.DATA_DIR):
            print(f"Папка данных существует: {config.DATA_DIR}")
            files = [f for f in os.listdir(config.DATA_DIR) if f.endswith('.csv')]
            print(f"Найдено CSV файлов: {len(files)}")
        else:
            print(f"Папка данных не существует: {config.DATA_DIR}")