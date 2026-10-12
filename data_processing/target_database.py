"""
target_database.py — База целей (объектов инфраструктуры).

Загружает цели из targets.json, преобразует координаты в ENU,
предоставляет интерфейс для работы с целями в модели.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import torch
except ImportError:
    torch = None

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.parser import geodetic_to_enu


class TargetDatabase:
    """
    База целей для системы прогнозирования.

    Хранит:
    - Координаты целей в ENU (метры)
    - Типы целей (для TypeEmbedding)
    - Имена и метаданные

    Используется:
    - При генерации синтетических траекторий (выбор цели)
    - В модели TATPN (Cross-Attention Target Head)
    - При визуализации результатов
    """

    def __init__(self, json_path: str = config.TARGETS_JSON):
        """
        Загружает базу целей из JSON.

        Args:
            json_path: путь к файлу targets.json
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.target_types = data["target_types"]
        self.type_to_idx = {t: i for i, t in enumerate(self.target_types)}

        self.targets = []
        for t in data["targets"]:
            # Преобразуем координаты в ENU
            e, n, u = geodetic_to_enu(
                np.array([t["lat"]]),
                np.array([t["lon"]]),
                np.array([0.0])
            )

            self.targets.append({
                "id": t["id"],
                "name": t["name"],
                "type": t["type"],
                "type_idx": self.type_to_idx[t["type"]],
                "lat": t["lat"],
                "lon": t["lon"],
                "enu_e": float(e[0]),
                "enu_n": float(n[0]),
                "source": t.get("source", ""),
            })

        self.num_targets = len(self.targets)
        self.num_types = len(self.target_types)

        print(f"[TargetDB] Загружено {self.num_targets} целей, {self.num_types} типов")

    def get_enu_coords(self) -> np.ndarray:
        """
        Возвращает ENU-координаты всех целей.

        Returns:
            (K, 2) массив [east, north] в метрах
        """
        coords = np.array([
            [t["enu_e"], t["enu_n"]]
            for t in self.targets
        ], dtype=np.float32)
        return coords

    def get_type_indices(self) -> np.ndarray:
        """
        Возвращает индексы типов для всех целей.

        Returns:
            (K,) массив int
        """
        return np.array([t["type_idx"] for t in self.targets], dtype=np.int64)

    def get_target_embeddings_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает входные данные для TargetEmbedding модели.

        Returns:
            coords: (K, 2) tensor — координаты в ENU
            type_ids: (K,) tensor — индексы типов
        """
        coords = torch.tensor(self.get_enu_coords(), dtype=torch.float32)
        type_ids = torch.tensor(self.get_type_indices(), dtype=torch.long)
        return coords, type_ids

    def find_nearest_target(self, x: float, y: float) -> int:
        """
        Находит ближайшую цель к заданной точке (ENU).

        Args:
            x, y: координаты в ENU (метры)

        Returns:
            Индекс ближайшей цели (0..K-1)
        """
        coords = self.get_enu_coords()
        distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
        return int(np.argmin(distances))

    def get_target_by_index(self, idx: int) -> Dict:
        """Возвращает информацию о цели по индексу."""
        return self.targets[idx]

    def get_distances_from_point(self, x: float, y: float) -> np.ndarray:
        """
        Расстояния от точки до всех целей.

        Returns:
            (K,) массив расстояний в метрах
        """
        coords = self.get_enu_coords()
        return np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)

    def print_targets(self):
        """Выводит таблицу целей."""
        print(f"\n{'ID':>4} | {'Название':^35} | {'Тип':^15} | {'E (км)':>8} | {'N (км)':>8}")
        print("-" * 85)
        for t in self.targets:
            print(f"{t['id']:>4} | {t['name']:^35} | {t['type']:^15} | "
                  f"{t['enu_e']/1000:>8.1f} | {t['enu_n']/1000:>8.1f}")


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    db = TargetDatabase()
    db.print_targets()

    print(f"\nENU координаты shape: {db.get_enu_coords().shape}")
    print(f"Type indices: {db.get_type_indices()}")

    # Тест: ближайшая цель к началу координат
    nearest = db.find_nearest_target(0, 0)
    print(f"\nБлижайшая цель к центру зоны полётов: "
          f"{db.targets[nearest]['name']} ({db.targets[nearest]['id']})")

    # Расстояния от центра зоны полётов
    dists = db.get_distances_from_point(0, 0)
    print(f"\nРасстояния от центра зоны полётов:")
    for i, t in enumerate(db.targets):
        print(f"  {t['id']} {t['name']:30s}: {dists[i]/1000:.1f} км")
