"""
run_stage6.py - Этап 6: визуализация и демонстрация.

Загружает обученную модель TATPN (Этап 5), прогоняет несколько тестовых
траекторий и создаёт:
  - интерактивные карты угрозы (Leaflet + OpenStreetMap): наблюдаемая и
    предсказанная траектории, цели с вероятностями, ETA;
  - сводный HTML-отчёт по всем этапам (таблица метрик + графики).

Всё сохраняется в reports/stage6/. Карты открываются в браузере.

Запуск:
    python run_stage6.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_processing.target_database import TargetDatabase
from data_processing.kinematics import FeatureNormalizer, compute_kinematics
from data_processing.parser import enu_to_geodetic
from data.windowing import compute_split_index
from models.tatpn import TATPN
from run_stage2 import load_split
from visualization.map_html import build_threat_map
from visualization import report as report_mod

# Длина наблюдаемого окна модели (как при обучении Этапа 5 на CPU)
SEQ_LEN = 48
CKPT = os.path.join(config.PROJECT_ROOT, "artifacts", "stage5", "best.pt")
OUT_DIR = os.path.join(config.REPORTS_DIR, "stage6")
NUM_EXAMPLES = 6


def load_tatpn(device):
    """Загружает TATPN с чекпоинта Этапа 5 + нормализатор + базу целей."""
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"Не найден чекпоинт {CKPT}. Сначала: python run_stage5.py")
    target_db = TargetDatabase()
    coords = target_db.get_enu_coords()
    types = target_db.get_type_indices()

    normalizer = FeatureNormalizer()
    normalizer.load(os.path.join(config.PROCESSED_DATA_DIR, "normalizer.npz"))

    model = TATPN(target_coords=coords, target_types=types).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, target_db, normalizer


def prepare_example(traj, normalizer, H):
    """Готовит вход модели и сырые пути (ENU) для одной траектории (наблюдение 50%)."""
    traj = np.asarray(traj, dtype=np.float32)
    n = len(traj)
    split = compute_split_index(n, 0.5)
    feats = compute_kinematics(traj)
    start = max(0, split - SEQ_LEN)
    obs = normalizer.transform(feats[start:split]).astype(np.float32)
    L = obs.shape[0]

    observed = np.zeros((SEQ_LEN, config.INPUT_FEATURES), np.float32); observed[:L] = obs
    mask = np.zeros(SEQ_LEN, np.float32); mask[:L] = 1.0
    last_point = traj[split - 1, :3].astype(np.float32)
    last_velocity = feats[split - 1, 4:7].astype(np.float32)

    obs_path = traj[:split, :3]
    true_future = traj[split:min(split + H, n), :3]
    return observed, mask, last_point, last_velocity, obs_path, true_future


def _enu_to_ll(path_enu):
    """(M,3) ENU -> список [[lat,lon], ...]."""
    if len(path_enu) == 0:
        return []
    lat, lon, _ = enu_to_geodetic(path_enu[:, 0], path_enu[:, 1], path_enu[:, 2])
    return [[float(a), float(b)] for a, b in zip(lat, lon)]


def _downsample(path, max_points=150):
    if len(path) <= max_points:
        return path
    idx = np.linspace(0, len(path) - 1, max_points).astype(int)
    return path[idx]


def make_example_map(model, target_db, normalizer, traj, out_path, title, device):
    """Прогоняет модель по одной траектории и строит карту. Возвращает threat_text."""
    H = config.PRED_HORIZON
    observed, mask, last_point, last_velocity, obs_path, true_future = \
        prepare_example(traj, normalizer, H)

    with torch.no_grad():
        out = model(torch.tensor(observed[None]).to(device),
                    torch.tensor(mask[None]).to(device),
                    torch.tensor(last_point[None]).to(device),
                    torch.tensor(last_velocity[None]).to(device))
    logits = out["target_logits"][0].cpu().numpy()
    z = logits - logits.max(); probs = np.exp(z); probs /= probs.sum()
    eta_sec = float(out["eta"][0].cpu().numpy())
    pred_future = out["pred_future"][0].cpu().numpy()

    # Пути в широту/долготу
    observed_ll = _enu_to_ll(_downsample(obs_path))
    predicted_ll = _enu_to_ll(pred_future)
    truth_ll = _enu_to_ll(true_future) if len(true_future) else None

    # Цели с вероятностями
    targets = []
    for i, t in enumerate(target_db.targets):
        targets.append({"name": t["name"], "type": t["type"],
                        "lat": t["lat"], "lon": t["lon"], "prob": float(probs[i])})

    top = int(np.argmax(probs))
    top_name = target_db.targets[top]["name"]
    threat_text = (f"{top_name} — {probs[top]*100:.0f}%, ETA {eta_sec/60:.0f} мин")

    build_threat_map(out_path, observed_ll, predicted_ll, truth_ll, targets,
                     threat_text, title=title)
    return threat_text


def main():
    config.ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = config.DEVICE

    print("=" * 70)
    print("  Этап 6 - Визуализация и демонстрация")
    print("=" * 70)

    model, target_db, normalizer = load_tatpn(device)
    print(f"[model] TATPN загружена ({model.num_parameters():,} параметров)")

    _, _, test_items, _, _ = load_split()
    # Выбираем разнообразные примеры: равномерно по тесту, длиной >= 100 точек
    candidates = [it for it in test_items if len(it["trajectory"]) >= 100]
    if not candidates:
        candidates = test_items
    idxs = np.linspace(0, len(candidates) - 1, NUM_EXAMPLES).astype(int)

    map_infos = []
    print(f"\n[maps] генерация {len(idxs)} карт-примеров...")
    for k, ci in enumerate(idxs, 1):
        traj = candidates[ci]["trajectory"]
        fname = f"example_{k}.html"
        out = os.path.join(OUT_DIR, fname)
        title = f"Пример {k}: прогноз положения воздушного объекта"
        threat = make_example_map(model, target_db, normalizer, traj, out, title, device)
        map_infos.append({"file": out, "title": f"Пример {k}", "threat": threat})
        print(f"  [{k}] {fname}: {threat}")

    # Сводный отчёт
    rows, stage5 = report_mod.collect_stage_summary(config.LOGS_DIR)
    charts = report_mod.make_charts(rows, stage5)
    report_path = os.path.join(OUT_DIR, "report.html")
    report_mod.generate_report(report_path, rows, charts, map_infos)

    print(f"\n[report] сводный отчёт -> {report_path}")
    print("\n" + "=" * 70)
    print("  Этап 6 завершён. Откройте в браузере:")
    print(f"    {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
