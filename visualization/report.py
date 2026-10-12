"""
report.py - сводный HTML-отчёт по всем этапам проекта (Этап 6).

Собирает метрики из logs/*/history.json, строит графики (matplotlib -> base64,
самодостаточно) и генерирует единую HTML-страницу:
  - сводная таблица метрик по этапам (ADE, Acc@1, ETA);
  - график ADE по этапам;
  - график точности цели vs доля наблюдения (Этап 5);
  - ссылки на интерактивные карты-примеры.
"""

import os
import io
import json
import base64


def _load(logs_dir, stage):
    path = os.path.join(logs_dir, stage, "history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_stage_summary(logs_dir):
    """Собирает строки сводной таблицы (этап -> ADE / Acc@1 / ETA)."""
    rows = []

    b = _load(logs_dir, "stage1_baselines")
    if b:
        lin = b["metrics"]["linear_velocity"]
        rows.append({"stage": "Линейная экстраполяция (бейзлайн)",
                     "ADE": lin["ADE"], "Acc1": None, "ETA": None})
    s2 = _load(logs_dir, "stage2_cnn")
    if s2:
        rows.append({"stage": "Этап 2: CNN",
                     "ADE": s2["test_metrics"]["CNN (этап 2)"]["ADE"],
                     "Acc1": None, "ETA": None})
    s3 = _load(logs_dir, "stage3_cnn_mamba")
    if s3:
        rows.append({"stage": "Этап 3: CNN+Mamba",
                     "ADE": s3["test_metrics"]["CNN+Mamba (этап 3)"]["ADE"],
                     "Acc1": None, "ETA": None})
    s4 = _load(logs_dir, "stage4_target")
    if s4:
        rows.append({"stage": "Этап 4: +Target Head",
                     "ADE": s4["test_traj"]["ADE"],
                     "Acc1": s4["test_target"]["Acc@1"], "ETA": None})
    s5 = _load(logs_dir, "stage5_tatpn")
    if s5:
        rows.append({"stage": "Этап 5: TATPN (полная)",
                     "ADE": s5["test_traj"]["ADE"],
                     "Acc1": s5["test_target"]["Acc@1"],
                     "ETA": s5["test_eta"]["MAE_ETA"]})
    return rows, (s5 or {})


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def make_charts(rows, stage5):
    """Строит графики -> base64 PNG. Возвращает dict {name: b64}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    charts = {}

    # ADE по этапам
    labels = [r["stage"].split(":")[0].replace("Линейная экстраполяция (бейзлайн)", "Линейная")
              for r in rows]
    ades = [r["ADE"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(range(len(ades)), ades, color="#3b6fb0")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ADE, м"); ax.set_title("Ошибка прогноза траектории (ADE) по этапам")
    for b_, v in zip(bars, ades):
        ax.text(b_.get_x()+b_.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    charts["ade"] = _fig_to_b64(fig); plt.close(fig)

    # Acc@1 vs доля наблюдения (Этап 5)
    abr = stage5.get("acc_by_ratio")
    if abr:
        rs = [float(r) * 100 for r in abr]
        a1 = [cm["Acc@1"] for cm in abr.values()]
        a3 = [cm["Acc@3"] for cm in abr.values()]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(rs, a1, "o-", label="Acc@1", color="#c0392b")
        ax.plot(rs, a3, "s--", label="Acc@3", color="#e67e22")
        nt = 1.0 / 15
        ax.axhline(nt, color="gray", ls=":", label="случайно")
        ax.set_xlabel("доля наблюдения, %"); ax.set_ylabel("accuracy")
        ax.set_title("Точность определения цели vs доля наблюдения (TATPN)")
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        charts["acc"] = _fig_to_b64(fig); plt.close(fig)

    return charts


def generate_report(out_path, rows, charts, map_infos):
    """
    Args:
        out_path:  путь HTML.
        rows:      строки сводной таблицы.
        charts:    dict {name: base64 png}.
        map_infos: список dict {file, title, threat} - ссылки на карты-примеры.
    """
    def cell(v, kind):
        if v is None:
            return "-"
        if kind == "ADE":
            return f"{v:.1f}"
        if kind == "Acc1":
            return f"{v*100:.1f}%"
        if kind == "ETA":
            return f"{v:.0f} с"
        return str(v)

    trs = ""
    for r in rows:
        trs += (f"<tr><td>{r['stage']}</td><td>{cell(r['ADE'],'ADE')}</td>"
                f"<td>{cell(r['Acc1'],'Acc1')}</td><td>{cell(r['ETA'],'ETA')}</td></tr>")

    imgs = ""
    for key, cap in [("ade", "ADE по этапам"), ("acc", "Точность цели vs наблюдение")]:
        if key in charts:
            imgs += (f"<figure><img src='data:image/png;base64,{charts[key]}' "
                     f"style='max-width:100%'><figcaption>{cap}</figcaption></figure>")

    maps = ""
    for m in map_infos:
        rel = os.path.basename(m["file"])
        maps += (f"<li><a href='{rel}' target='_blank'>{m['title']}</a> — {m['threat']}</li>")

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт: прогнозирование положения воздушного объекта</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222}}
 h1{{font-size:22px}} h2{{font-size:18px;margin-top:28px}}
 table{{border-collapse:collapse;width:100%;margin:10px 0}}
 th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
 th{{background:#f2f5f9}}
 figure{{margin:16px 0}} figcaption{{font-size:13px;color:#666}}
 li{{margin:4px 0}}
</style></head><body>
<h1>Система прогнозирования положения воздушного объекта</h1>
<p>Сводный отчёт по этапам: прогноз траектории, классификация цели, оценка ETA.
Модель - Multi-Scale CNN + Mamba + Cross-Attention + авторегрессионный декодер (TATPN).</p>

<h2>Сводная таблица метрик</h2>
<table>
<tr><th>Этап / модель</th><th>ADE ↓</th><th>Acc@1 ↑</th><th>ETA MAE ↓</th></tr>
{trs}
</table>
<p style="font-size:13px;color:#666">ADE - средняя ошибка прогноза траектории (м);
Acc@1 - доля верного определения цели из 15 (случайно 6.7%); ETA MAE - средняя
ошибка времени до цели (с).</p>

<h2>Графики</h2>
{imgs}

<h2>Примеры прогноза на карте</h2>
<p>Интерактивные карты (наблюдаемая и предсказанная траектории, цели с
вероятностями, ETA). Открываются в браузере, карта - OpenStreetMap.</p>
<ul>{maps}</ul>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
