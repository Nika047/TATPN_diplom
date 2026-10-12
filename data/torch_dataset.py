"""
torch_dataset.py - тонкая обёртка WindowedDataset для PyTorch.

WindowedDataset (numpy, без torch) формирует окна; здесь оборачиваем его в
torch.utils.data.Dataset, чтобы использовать стандартный DataLoader с
батчами и перемешиванием. Для train mode="random" (доля наблюдения 20-80%
меняется на каждом обращении), для val/test mode="fixed" 0.5.
"""

import numpy as np

import torch
from torch.utils.data import Dataset

from data.windowing import WindowedDataset

# Поля, возвращаемые как тензоры в батче
_ARRAY_KEYS = ["observed", "mask", "last_point", "last_velocity",
               "future_points", "future_mask"]


class TorchWindowed(Dataset):
    """Оборачивает WindowedDataset -> словарь torch-тензоров на образец."""

    def __init__(self, windowed: WindowedDataset):
        self.w = windowed

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        s = self.w.make(idx)   # доля наблюдения зависит от mode WindowedDataset
        out = {k: torch.from_numpy(np.asarray(s[k], dtype=np.float32))
               for k in _ARRAY_KEYS}
        out["target_id"] = torch.tensor(s["target_id"], dtype=torch.long)
        out["eta"] = torch.tensor(s["eta"], dtype=torch.float32)
        out["observe_ratio"] = torch.tensor(s["observe_ratio"], dtype=torch.float32)
        return out
