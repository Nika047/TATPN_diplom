import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from datetime import datetime
import base64
from io import BytesIO

import CNN_transform_network as Network
import metrics
import globals


# Вычисляем "кривизну" траектории
def calculate_curvature(traj):
    diffs = np.diff(traj[:, :3], axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    return np.mean(norms)


def plot_trajectories_with_projections(loader, n_trajectories=3, pred=None, probs=None, pred_xyz=None, save_dir="trajectory_reports"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Начало генерации отчета с проекциями\n")

    inp_batch, tgt_batch = next(iter(loader))
    grid_coords_3d = Network.make_zone_grid()

    dateTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = get_html_content() + """
    <body>
        <div class="container">
            <h1>📊 Отчет по траекториям</h1>
            <div class="timestamp">Создано: """ + dateTime + """</div>
    """

    for i in range(min(n_trajectories, len(inp_batch))):
        fig = plt.figure(figsize=(20, 5))

        inp = inp_batch[i].numpy()  # (seq_len, 4)
        tgt = tgt_batch[i].numpy()  # (pred_horizon, 3)

        current_pred = None
        if pred is not None:
            current_pred = pred[i].numpy() if isinstance(pred, torch.Tensor) else pred[i]

        val_mets = None
        if current_pred is not None and pred_xyz is not None:
            val_mets = metrics.compute_metrics(pred_xyz, tgt)

        final_point = tgt[-1]

        z_level = max(0, min(2, int(round(final_point[2] / 33.3))))
        y_level = max(0, min(2, int(round(final_point[1] / 33.3))))
        x_level = max(0, min(2, int(round(final_point[0] / 33.3))))

        # 3D отображение
        ax = fig.add_subplot(1, 4, 1, projection='3d')
        draw_3d_trajectory(ax, grid_coords_3d, inp, tgt, current_pred)
        ax.set_title(f'Траектория {i + 1} (Кривизна: {calculate_curvature(inp):.2f})')

        # XY Проекция
        ax = fig.add_subplot(1, 4, 2)
        draw_2d_projection(ax, grid_coords_3d, ('X', 'Y'), (0, 1), f'XY Проекция (Z-level {z_level})', inp, tgt,
                           current_pred)

        # XZ Проекция
        ax = fig.add_subplot(1, 4, 3)
        draw_2d_projection(ax, grid_coords_3d, ('X', 'Z'), (0, 2), f'XZ Проекция (Y-level {y_level})', inp, tgt,
                           current_pred)

        # YZ Проекция
        ax = fig.add_subplot(1, 4, 4)
        draw_2d_projection(ax, grid_coords_3d, ('Y', 'Z'), (1, 2), f'YZ Проекция (X-level {x_level})', inp, tgt,
                           current_pred)

        # Конвертируем график в base64 для вставки в HTML
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Формируем информационный текст
        info_text = f"Траектория {i + 1}\n"
        info_text += f"Конечная точка (X,Y,Z): {final_point}\n"
        info_text += f"Уровни сетки (X,Y,Z): {x_level}, {y_level}, {z_level}\n"

        if current_pred is not None:
            info_text += f"Предсказанная конечная точка: {current_pred[-1]}\n"

            # Вычисляем ошибку по каждой координате
            coord_errors = np.abs(current_pred[-1] - final_point)
            info_text += f"Ошибки по координатам (X,Y,Z): {coord_errors}\n"

            if val_mets is not None:
                info_text += f"Средняя ошибка (ADE): {val_mets['ADE']:.4f}\n"
                info_text += f"Финальная ошибка (FDE): {val_mets['FDE']:.4f}\n"
                info_text += f"RMSE: {val_mets['RMSE']:.4f}\n"
                info_text += f"MAE: {val_mets['MAE']:.4f}\n"

        # Добавляем траекторию в HTML
        html_content += f"""
            <div class="trajectory">
                <h2>Траектория {i + 1}</h2>

                <div class="plot-container">
                    <div class="plot">
                        <img src="data:image/png;base64,{image_base64}" alt="График траектории {i + 1}">
                    </div>
                </div>

                <div class="info">
                    {info_text}
                </div>
        """

        # Добавляем таблицу с метриками, если есть предсказания
        if current_pred is not None and val_mets is not None:
            html_content += f"""
                <table class="metrics-table">
                    <tr>
                        <th>Метрика</th>
                        <th>Значение</th>
                        <th>Описание</th>
                    </tr>
                    <tr>
                        <td>ADE</td>
                        <td>{val_mets['ADE']:.4f}</td>
                        <td>Средняя дистанционная ошибка</td>
                    </tr>
                    <tr>
                        <td>FDE</td>
                        <td>{val_mets['FDE']:.4f}</td>
                        <td>Финальная дистанционная ошибка</td>
                    </tr>
                    <tr>
                        <td>RMSE</td>
                        <td>{val_mets['RMSE']:.4f}</td>
                        <td>Среднеквадратичная ошибка</td>
                    </tr>
                    <tr>
                        <td>MAE</td>
                        <td>{val_mets['MAE']:.4f}</td>
                        <td>Средняя абсолютная ошибка</td>
                    </tr>
                </table>
            """

        html_content += "</div>"

    html_content += """
        </div>
    </body>
    </html>
    """

    out_path = os.path.join(save_dir, f"trajectories_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML отчет сохранен: {out_path}\n")
    print(f"📊 Количество траекторий в отчете: {min(n_trajectories, len(inp_batch))}\n")

def get_html_content():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Отчет по траекториям</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f8f9fa;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .trajectory { 
                margin-bottom: 40px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #fff;
            }
            .plot-container { 
                display: flex; 
                flex-wrap: wrap; 
                gap: 20px;
                margin-bottom: 20px;
            }
            .plot { 
                flex: 1; 
                min-width: 300px; 
                text-align: center;
            }
            .plot img { 
                max-width: 100%; 
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .info { 
                background-color: #f8f9fa; 
                padding: 15px; 
                border-radius: 5px;
                border-left: 4px solid #007bff;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
            }
            h1 { 
                color: #2c3e50; 
                text-align: center;
                margin-bottom: 30px;
            }
            h2 { 
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .timestamp { 
                color: #7f8c8d; 
                text-align: center;
                margin-bottom: 30px;
            }
            .metrics-table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }
            .metrics-table th, .metrics-table td {
                padding: 8px 12px;
                text-align: left;
                border: 1px solid #ddd;
            }
            .metrics-table th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .metrics-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
        </style>
    </head>
    """

def draw_2d_projection(ax, grid_coords_3d, axis_labels, indexes, title, inp, tgt, pred=None, draw_zones=True):

    # Отрисовка траекторий
    ax.plot(inp[:, indexes[0]], inp[:, indexes[1]], 'b-', label='Input', alpha=0.7)
    ax.plot(tgt[:, indexes[0]], tgt[:, indexes[1]], 'g--', label='True', alpha=0.7)

    # ax.scatter(inp[0, indexes[0]], inp[0, indexes[1]], c='blue', s=30, label='Start')
    # ax.scatter(tgt[-1, indexes[0]], tgt[-1, indexes[1]], c='green', s=30, label='End')

    ax.scatter(inp[0, indexes[0]], inp[0, indexes[1]], c='blue', s=30)
    ax.scatter(tgt[-1, indexes[0]], tgt[-1, indexes[1]], c='green', s=30)

    if pred is not None:
        ax.plot(pred[:, indexes[0]], pred[:, indexes[1]], 'r--', label='Predicted', alpha=0.7)
        ax.scatter(pred[-1, indexes[0]], pred[-1, indexes[1]], c='red', s=30)
        # ax.scatter(pred[-1, indexes[0]], pred[-1, indexes[1]], c='red', s=30, label='End')

    if draw_zones:
        final_point = tgt[-1]

        z_level = max(0, min(2, int(round(final_point[2] / 33.3))))
        y_level = max(0, min(2, int(round(final_point[1] / 33.3))))
        x_level = max(0, min(2, int(round(final_point[0] / 33.3))))

        for zone_id, (cx, cy, cz) in enumerate(grid_coords_3d):
            show_zone = False
            center_x = 0
            center_y = 0

            # Выбираем координаты для текущей проекции
            # XY проекция - проверяем Z уровень
            if indexes == (0, 1):
                show_zone = int(cz // 33.3) == z_level
                center_x, center_y = cx, cy

            # XZ проекция - проверяем Y уровень
            elif indexes == (0, 2):
                show_zone = int(cy // 33.3) == y_level
                center_x, center_y = cx, cz

            # YZ проекция - проверяем X уровень
            elif indexes == (1, 2):
                show_zone = int(cx // 33.3) == x_level
                center_x, center_y = cy, cz

            if show_zone:
                # Рисуем прямоугольник зоны
                rect = Rectangle((center_x - globals.R_ZONE, center_y - globals.R_ZONE), globals.R_ZONE * 2, globals.R_ZONE * 2, fill=False, linestyle=':', alpha=0.3, color='gray')
                ax.add_patch(rect)

                # Подпись номера зоны
                ax.text(center_x, center_y, f'{zone_id}', color='red', ha='center', va='center', fontsize=8)

    # Подписи осей
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    # Настройки графиков
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_aspect('equal')

    ax.set_title(title)

def draw_3d_trajectory(ax, grid_coords_3d, inp, tgt, pred=None, draw_zones=True):
    """
    Отрисовывает 3D траекторию и зоны

    Параметры:
        ax: ось matplotlib 3D
        grid_coords_3d: координаты центров зон
        inp: входная траектория (N, 3)
        tgt: истинные траектории (B, H, 3)
        pred: предсказанная траектория (N, 3)
        draw_zones: флаг отрисовки зон (True/False)
    """

    # Отрисовка траекторий
    ax.plot(inp[:, 0], inp[:, 1], inp[:, 2], 'b-', label='Input', linewidth=1.5)
    ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], 'g--', label='True', linewidth=1.5)

    # ax.scatter(inp[0, 0], inp[0, 1], inp[0, 2], c='blue', s=30, label='Start')
    # ax.scatter(tgt[-1, 0], tgt[-1, 1], tgt[-1, 2], c='green', s=30, label='End')

    ax.scatter(inp[0, 0], inp[0, 1], inp[0, 2], c='blue', s=30)
    ax.scatter(tgt[-1, 0], tgt[-1, 1], tgt[-1, 2], c='green', s=30)

    if pred is not None:
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r--', label='Predicted', linewidth=1.5)
        ax.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], c='red', s=30)
        # ax.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], c='red', s=30, label='End')

    if draw_zones:
        # Отрисовка кубов 3x3x3
        for z, (cx, cy, cz) in enumerate(grid_coords_3d):
            r = globals.R_ZONE

            v = [
                [cx - r, cy - r, cz - r],
                [cx + r, cy - r, cz - r],
                [cx + r, cy + r, cz - r],
                [cx - r, cy + r, cz - r],
                [cx - r, cy - r, cz + r],
                [cx + r, cy - r, cz + r],
                [cx + r, cy + r, cz + r],
                [cx - r, cy + r, cz + r],
            ]
            faces = [
                [v[0], v[1], v[2], v[3]],
                [v[4], v[5], v[6], v[7]],
                [v[0], v[1], v[5], v[4]],
                [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]],
                [v[0], v[3], v[7], v[4]],
            ]

            cube = Poly3DCollection(faces, alpha=0.05, linewidths=1, edgecolor='k')
            cube.set_facecolor('lightgray')
            ax.add_collection3d(cube)
            ax.text(cx, cy, cz, f'{z}', color='red', ha='center', va='center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_zlim(-5, 105)

    ax.legend()

def plot_train_samples(loader, n_trajectories=15, traj_per_row=5):
    n_rows = int(np.ceil(n_trajectories / traj_per_row))

    # Создаем фигуру с несколькими строками
    fig, axes = plt.subplots(n_rows, traj_per_row,
                            figsize=(6 * traj_per_row, 5 * n_rows),
                            subplot_kw={'projection': '3d'})

    # Берем первые `n_trajectories` примеров из loader
    inp_batch, tgt_batch = next(iter(loader))

    for i in range(n_trajectories):
        # Определяем позицию в сетке
        row = i // traj_per_row
        col = i % traj_per_row

        # Для случая с одной строкой axes будет 1D массивом
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        # Берем i-й пример из батча
        inp = inp_batch[i].numpy()  # (seq_len, 4)
        tgt = tgt_batch[i].numpy()  # (pred_horizon, 3)

        # Входная траектория (синяя)
        ax.plot(inp[:, 0], inp[:, 1], inp[:, 2], 'b-', label="Input", linewidth=1.5)

        # Целевая траектория (красная пунктирная)
        ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], 'r--', label="Target", linewidth=1.5)

        # Отмечаем начало и конец
        ax.scatter(inp[0, 0], inp[0, 1], inp[0, 2], c='blue', s=30, label='Start')
        ax.scatter(tgt[-1, 0], tgt[-1, 1], tgt[-1, 2], c='red', s=30, label='End')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Траектория {i+1} (Средняя кривизна: {calculate_curvature(inp):.2f})')
        ax.legend(loc='upper right')

    # Скрываем пустые subplots, если количество не кратно 5
    for i in range(n_trajectories, n_rows * traj_per_row):
        row = i // traj_per_row
        col = i % traj_per_row
        if n_rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
