import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime

def plot_all_trajectories_from_folders(base_dir="flight_logs_generated", save_dir="trajectory_reports_all"):
    os.makedirs(save_dir, exist_ok=True)

    print("Начало генерации отчета по всем траекториям\n")

    # HTML-шапка
    html_content = get_html_content() + """
    <body>
        <div class="container">
            <h1>📊 Отчет по всем траекториям</h1>
            <div class="timestamp">Создано: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
    """

    # Функция для безопасного парсинга времени
    def parse_time_safe(time_series):
        """Безопасный парсинг времени с разными форматами"""
        formats = [
            "%H:%M:%S.%f",  # 15:17:57.250
            "%H:%M:%S",  # 15:17:57
            "%Y-%m-%d %H:%M:%S.%f",  # 2025-05-09 15:17:57.250
            "%Y-%m-%d %H:%M:%S",  # 2025-05-09 15:17:57
        ]

        for fmt in formats:
            try:
                parsed = pd.to_datetime(time_series, format=fmt, errors="coerce")
                if not parsed.isna().all():  # Если хоть некоторые значения распарсились
                    return parsed
            except:
                continue

        # Если ни один формат не подошел, используем общий парсинг
        return pd.to_datetime(time_series, errors="coerce")


    # Проходим по наборам
    for split in ["train", "val", "test"]:
        folder = os.path.join(base_dir, split)
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        html_content += f"<h2>Набор: {split} ({len(files)} файлов)</h2>"

        for idx, file in enumerate(files, 1):
            # df = pd.read_csv(file, parse_dates=["Time"])
            # if not all(col in df.columns for col in ["X", "Y", "Z"]):
            #     continue
            #
            # inp = df[["X", "Y", "Z", "Time"]].to_numpy(dtype=np.float32)
            # tgt = df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)

            df = pd.read_csv(file)  # Только читаем CSV, без parse_dates

            if not all(col in df.columns for col in ["X", "Y", "Z", "Time"]):
                continue

            # Парсим только колонку Time с правильным форматом
            time_parsed = pd.to_datetime(df["Time"], format="%H:%M:%S.%f", errors="coerce")

            # Преобразуем в секунды от начала траектории
            time_seconds = (time_parsed - time_parsed.iloc[0]).dt.total_seconds()

            # Создаем массивы для входа и цели
            inp = np.column_stack([
                df["X"].to_numpy(dtype=np.float32),
                df["Y"].to_numpy(dtype=np.float32),
                df["Z"].to_numpy(dtype=np.float32),
                time_seconds.astype(np.float32)
            ])

            tgt = np.column_stack([
                df["X"].to_numpy(dtype=np.float32),
                df["Y"].to_numpy(dtype=np.float32),
                df["Z"].to_numpy(dtype=np.float32)
            ])

            final_point = tgt[-1]
            z_level = max(0, min(2, int(round(final_point[2] / 33.3))))
            y_level = max(0, min(2, int(round(final_point[1] / 33.3))))
            x_level = max(0, min(2, int(round(final_point[0] / 33.3))))

            fig = plt.figure(figsize=(20, 5))

            # 3D
            ax = fig.add_subplot(1, 4, 1, projection='3d')
            ax.plot(inp[:, 0], inp[:, 1], inp[:, 2], label="Траектория")
            ax.set_title(f"3D Траектория {idx}")

            # XY
            ax = fig.add_subplot(1, 4, 2)
            ax.plot(inp[:, 0], inp[:, 1])
            ax.set_title(f"XY (Z-level {z_level})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            # XZ
            ax = fig.add_subplot(1, 4, 3)
            ax.plot(inp[:, 0], inp[:, 2])
            ax.set_title(f"XZ (Y-level {y_level})")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")

            # YZ
            ax = fig.add_subplot(1, 4, 4)
            ax.plot(inp[:, 1], inp[:, 2])
            ax.set_title(f"YZ (X-level {x_level})")
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)

            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            html_content += f"""
            <div class="trajectory">
                <h3>{split} — Траектория {idx} ({os.path.basename(file)})</h3>
                <div class="plot-container">
                    <div class="plot">
                        <img src="data:image/png;base64,{image_base64}" alt="График траектории {idx}">
                    </div>
                </div>
                <div class="info">
                    Конечная точка (X,Y,Z): {final_point}
                </div>
            </div>
            """

    html_content += """
        </div>
    </body>
    </html>
    """

    out_path = os.path.join(save_dir, f"all_trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Отчет сохранен: {out_path}\n")


# HTML-шаблон
def get_html_content():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Отчет по траекториям</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
            .container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }
            .trajectory { margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #fff; }
            .plot-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
            .plot { flex: 1; min-width: 300px; text-align: center; }
            .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
            .info { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; font-family: 'Courier New', monospace; white-space: pre-wrap; }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
            h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h3 { color: #2c3e50; }
        </style>
    </head>
    """
