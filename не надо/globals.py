import torch
import random
import numpy as np
import os


n_samples_train = 10_000                    # количество траекторий для тренировки
n_samples_val = int(0.2 * n_samples_train)  # количество траекторий для валидации
n_samples_test = 1_000                      # количество траекторий для тестов

seq_len = 100                   # длина входного окна
pred_horizon = 20               # сколько точек нужно предсказать
min_input_points = 40           # минимальный путь (без точек, которые нужно предсказать)
min_total_points = min_input_points + pred_horizon
dt = 0.1                        # фиксированный временной шаг

batch_size = 32
num_epochs = 20

val_size  = 0.15
test_size = 0.15

num_generated_files = 1000

lr = 1e-3
weight_decay = 1e-4
grad_clip = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "artifacts"
os.makedirs(save_dir, exist_ok=True)


ZONE_COORDS = (15.0, 50.0, 85.0)  # сетка
R_ZONE = 8.0  # радиус «попадания» в зону (единиц)
TAU = 10.0  # температура для зон
TAU_NONE = 10.0  # температура для «вне зоны»

# Температуры для софтмакса: τ управляет «резкостью» распределения, τ_none — чувствительностью «вне зоны»


random_state = 42

def set_seed(seed: int = random_state):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(random_state)
