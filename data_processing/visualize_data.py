"""
visualize_data.py — Визуализация сгенерированных данных всех 3 слоёв.

Создаёт HTML-отчёт с графиками:
- Карта всех целей (ENU)
- Примеры траекторий каждого типа
- Распределения длин, скоростей, расстояний
- Статистика по датасету
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import base64
from typing import List, Dict
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.target_database import TargetDatabase


def generate_html_report(all_trajectories: List[Dict],
                         target_db: TargetDatabase,
                         save_path: str = None) -> str:
    """
    Генерирует HTML-отчёт с визуализацией данных.

    Args:
        all_trajectories: все траектории (все 3 слоя)
        target_db: база целей
        save_path: путь для сохранения HTML (если None — не сохраняется)

    Returns:
        HTML-строка
    """
    figures = []

    # 1. Карта целей и примеры траекторий
    fig = _plot_targets_and_trajectories(all_trajectories, target_db)
    figures.append(("Карта целей и примеры траекторий", fig))

    # 2. Примеры каждого типа
    for ttype in ["direct", "evasion", "loitering", "deceptive", "scale", "stitch", "warp"]:
        subset = [t for t in all_trajectories
                  if t.get("trajectory_type") == ttype or t.get("augmentation_type") == ttype]
        if subset:
            fig = _plot_trajectory_examples(subset[:6], ttype, target_db)
            figures.append((f"Примеры: {ttype}", fig))

    # 3. Распределения
    fig = _plot_distributions(all_trajectories)
    figures.append(("Распределения", fig))

    # Собираем HTML
    html = _build_html(figures, all_trajectories, target_db)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[Visualize] Отчёт сохранён: {save_path}")

    return html


def _plot_targets_and_trajectories(trajectories: List[Dict],
                                   target_db: TargetDatabase) -> plt.Figure:
    """Карта целей + несколько примеров траекторий."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Цели
    coords = target_db.get_enu_coords() / 1000  # в км
    types = target_db.get_type_indices()

    type_colors = {
        0: 'red',      # airport
        1: 'orange',   # nuclear_plant
        2: 'purple',   # refinery
        3: 'blue',     # power_plant
        4: 'green',    # military_base
        5: 'cyan',     # port
        6: 'brown',    # oil_terminal
        7: 'magenta',  # city_center
        8: 'gray',     # city
        9: 'olive',    # transport_hub
    }

    for i, t in enumerate(target_db.targets):
        color = type_colors.get(types[i], 'black')
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=100, marker='*',
                   zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(t["id"], (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha='center', va='bottom', fontweight='bold')

    # Центр зоны полётов
    ax.scatter(0, 0, c='red', s=200, marker='x', zorder=6, linewidths=2)
    ax.annotate('Зона полётов', (0, 0), fontsize=8, ha='center', va='top', color='red')

    # Примеры траекторий (до 30 штук)
    colors_map = {
        "direct": "#2196F3",
        "evasion": "#FF9800",
        "loitering": "#4CAF50",
        "deceptive": "#F44336",
        "scale": "#9C27B0",
        "stitch": "#00BCD4",
        "warp": "#795548",
    }

    sample_indices = np.random.choice(len(trajectories),
                                      size=min(30, len(trajectories)),
                                      replace=False)
    for idx in sample_indices:
        t = trajectories[idx]
        traj = t["trajectory"]
        ttype = t.get("trajectory_type", t.get("augmentation_type", "real"))
        color = colors_map.get(ttype, "#666666")
        ax.plot(traj[:, 0] / 1000, traj[:, 1] / 1000, color=color,
                alpha=0.3, linewidth=0.5)

    ax.set_xlabel("East (км)")
    ax.set_ylabel("North (км)")
    ax.set_title("Карта целей и примеры траекторий (ENU)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=c, label=l, linewidth=2)
        for l, c in colors_map.items()
    ]
    legend_elements.append(Line2D([0], [0], marker='*', color='black',
                                   label='Цели', markersize=10, linestyle='None'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def _plot_trajectory_examples(trajectories: List[Dict], ttype: str,
                              target_db: TargetDatabase) -> plt.Figure:
    """Примеры траекторий конкретного типа."""
    n = min(len(trajectories), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    coords = target_db.get_enu_coords() / 1000

    for i in range(6):
        ax = axes[i]
        if i < n:
            t = trajectories[i]
            traj = t["trajectory"]
            ax.plot(traj[:, 0] / 1000, traj[:, 1] / 1000, 'b-', linewidth=0.8)
            ax.plot(traj[0, 0] / 1000, traj[0, 1] / 1000, 'go', markersize=8, label='Старт')
            ax.plot(traj[-1, 0] / 1000, traj[-1, 1] / 1000, 'r^', markersize=8, label='Конец')

            # Цель
            if "target_id" in t:
                tid = t["target_id"]
                ax.plot(coords[tid, 0], coords[tid, 1], 'r*', markersize=15, label='Цель')

            ax.set_title(f"{ttype} #{i+1} ({len(traj)} pts)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            if i == 0:
                ax.legend(fontsize=7)
        else:
            ax.axis('off')

    plt.suptitle(f"Примеры траекторий: {ttype}", fontsize=14)
    plt.tight_layout()
    return fig


def _plot_distributions(trajectories: List[Dict]) -> plt.Figure:
    """Распределения длин, расстояний, скоростей."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    lengths = [len(t["trajectory"]) for t in trajectories]
    distances = [t.get("distance_km", 0) for t in trajectories if "distance_km" in t]
    speeds = [t.get("speed_ms", 0) for t in trajectories if "speed_ms" in t]

    # Типы траекторий
    type_counts = {}
    for t in trajectories:
        ttype = t.get("trajectory_type", t.get("augmentation_type", "real"))
        type_counts[ttype] = type_counts.get(ttype, 0) + 1

    # 1. Длины траекторий
    axes[0, 0].hist(lengths, bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].set_xlabel("Длина (точек)")
    axes[0, 0].set_ylabel("Количество")
    axes[0, 0].set_title("Распределение длин траекторий")

    # 2. Расстояния
    if distances:
        axes[0, 1].hist(distances, bins=50, color='coral', edgecolor='white')
        axes[0, 1].set_xlabel("Расстояние (км)")
        axes[0, 1].set_ylabel("Количество")
        axes[0, 1].set_title("Расстояние до цели")

    # 3. Скорости
    if speeds:
        speeds_kmh = [s * 3.6 for s in speeds]
        axes[1, 0].hist(speeds_kmh, bins=50, color='seagreen', edgecolor='white')
        axes[1, 0].set_xlabel("Скорость (км/ч)")
        axes[1, 0].set_ylabel("Количество")
        axes[1, 0].set_title("Распределение скоростей")

    # 4. Типы
    if type_counts:
        bars = axes[1, 1].bar(type_counts.keys(), type_counts.values(), color='mediumpurple')
        axes[1, 1].set_ylabel("Количество")
        axes[1, 1].set_title("Типы траекторий")
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def _fig_to_base64(fig: plt.Figure) -> str:
    """Конвертирует matplotlib figure в base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _build_html(figures: list, trajectories: List[Dict],
                target_db: TargetDatabase) -> str:
    """Собирает HTML-отчёт."""

    # Статистика
    n_total = len(trajectories)
    lengths = [len(t["trajectory"]) for t in trajectories]

    type_counts = {}
    for t in trajectories:
        ttype = t.get("trajectory_type", t.get("augmentation_type", "real"))
        type_counts[ttype] = type_counts.get(ttype, 0) + 1

    stats_html = f"""
    <h2>Статистика датасета</h2>
    <table border="1" cellpadding="5" cellspacing="0">
    <tr><td>Всего траекторий</td><td><b>{n_total}</b></td></tr>
    <tr><td>Медиана длины</td><td>{np.median(lengths):.0f} точек</td></tr>
    <tr><td>Мин/Макс длина</td><td>{np.min(lengths)}/{np.max(lengths)} точек</td></tr>
    <tr><td>Количество целей</td><td>{target_db.num_targets}</td></tr>
    <tr><td>Типы целей</td><td>{target_db.num_types}</td></tr>
    </table>
    <h3>Типы траекторий</h3>
    <table border="1" cellpadding="5" cellspacing="0">
    <tr><th>Тип</th><th>Количество</th><th>Доля</th></tr>
    """
    for ttype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        stats_html += f"<tr><td>{ttype}</td><td>{count}</td><td>{count/n_total*100:.1f}%</td></tr>"
    stats_html += "</table>"

    # Графики
    figures_html = ""
    for title, fig in figures:
        b64 = _fig_to_base64(fig)
        figures_html += f"""
        <h2>{title}</h2>
        <img src="data:image/png;base64,{b64}" style="max-width:100%;">
        """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>TATPN Data Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
h1 {{ color: #333; }}
h2 {{ color: #555; margin-top: 30px; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
td, th {{ padding: 5px 10px; text-align: left; }}
img {{ border: 1px solid #ddd; margin: 10px 0; }}
</style>
</head><body>
<h1>TATPN — Отчёт по данным (Этап 0)</h1>
{stats_html}
{figures_html}
</body></html>"""

    return html


# ============================================================================
# Тестирование
# ============================================================================

if __name__ == "__main__":
    print("Тест визуализации...")

    target_db = TargetDatabase()

    # Генерируем тестовые данные
    rng = np.random.RandomState(42)
    fake_data = []
    for i in range(20):
        n = rng.randint(100, 1000)
        x = np.cumsum(rng.randn(n) * 1000)
        y = np.cumsum(rng.randn(n) * 1000)
        z = rng.uniform(50, 500, n)
        dt = np.arange(n, dtype=np.float32)
        fake_data.append({
            "trajectory": np.stack([x, y, z, dt], axis=1).astype(np.float32),
            "target_id": rng.randint(0, config.NUM_TARGETS),
            "trajectory_type": rng.choice(["direct", "evasion", "loitering", "deceptive"]),
            "distance_km": rng.uniform(10, 300),
            "speed_ms": rng.uniform(20, 60),
        })

    config.ensure_dirs()
    save_path = os.path.join(config.REPORTS_DIR, "data_report.html")
    generate_html_report(fake_data, target_db, save_path=save_path)
