"""
map_html.py - интерактивная карта угрозы (Leaflet + OpenStreetMap).

Генерирует самодостаточный HTML-файл: подключает Leaflet с CDN и тайлы
OpenStreetMap (подгружаются в браузере при открытии), рисует:
  - наблюдаемую траекторию (синяя линия);
  - предсказанную траекторию (красная линия);
  - истинное будущее (зелёная пунктирная линия, если есть);
  - все цели как маркеры, размер/цвет по вероятности;
  - панель с оценкой угрозы: наиболее вероятная цель, вероятность, ETA.

Внешних python-зависимостей не требует (folium/plotly не нужны).
"""

import json
import os

# Шаблон HTML. Данные подставляются как JSON в один скрипт Leaflet.
_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,Arial,sans-serif}
  #map{width:100%;height:100vh}
  .panel{position:absolute;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.94);
    padding:12px 14px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.3);max-width:320px;font-size:13px}
  .panel h3{margin:0 0 6px;font-size:15px}
  .panel .top{font-weight:700;color:#c0392b}
  .panel table{border-collapse:collapse;margin-top:6px;width:100%}
  .panel td{padding:1px 4px}
  .legend{position:absolute;bottom:14px;left:10px;z-index:1000;background:rgba(255,255,255,.94);
    padding:8px 10px;border-radius:8px;font-size:12px;box-shadow:0 1px 6px rgba(0,0,0,.3)}
  .legend span{display:inline-block;width:12px;height:3px;margin-right:5px;vertical-align:middle}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h3>Оценка угрозы</h3>
  <div class="top">__THREAT_TEXT__</div>
  <table id="tgt-table"></table>
</div>
<div class="legend">
  <div><span style="background:#1f6fe0"></span>наблюдаемая</div>
  <div><span style="background:#c0392b"></span>прогноз</div>
  <div><span style="background:#2e9e4f;border-top:2px dashed #2e9e4f"></span>истина</div>
</div>
<script>
const D = __DATA__;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, attribution:'© OpenStreetMap'}).addTo(map);

function line(coords, color, dash){
  if(!coords || !coords.length) return null;
  return L.polyline(coords, {color:color, weight:3, opacity:.9,
    dashArray:dash?'6 6':null}).addTo(map);
}
const obs = line(D.observed, '#1f6fe0', false);
const pred = line(D.predicted, '#c0392b', false);
const truth = line(D.truth, '#2e9e4f', true);

// стартовая и конечная наблюдаемые точки
if(D.observed && D.observed.length){
  L.circleMarker(D.observed[0], {radius:5,color:'#1f6fe0',fillOpacity:1}).addTo(map)
    .bindPopup('старт наблюдения');
  L.circleMarker(D.observed[D.observed.length-1], {radius:6,color:'#111',fillColor:'#fff',
    fillOpacity:1}).addTo(map).bindPopup('текущая позиция');
}

// цели: размер и цвет по вероятности
let rows = '';
D.targets.forEach((t,i)=>{
  const p = t.prob;
  const col = p>0.5?'#c0392b':(p>0.2?'#e67e22':(p>0.08?'#f1c40f':'#95a5a6'));
  const r = 6 + 22*p;
  L.circleMarker([t.lat,t.lon], {radius:r, color:col, weight:1,
    fillColor:col, fillOpacity:.55}).addTo(map)
    .bindPopup('<b>'+t.name+'</b><br>тип: '+t.type+'<br>P='+(p*100).toFixed(1)+'%');
});
// таблица топ-5 целей
D.targets.slice().sort((a,b)=>b.prob-a.prob).slice(0,5).forEach(t=>{
  rows += '<tr><td>'+t.name+'</td><td style="text-align:right">'+(t.prob*100).toFixed(1)+'%</td></tr>';
});
document.getElementById('tgt-table').innerHTML =
  '<tr><td><b>цель</b></td><td style="text-align:right"><b>P</b></td></tr>'+rows;

// подгон карты под все объекты
const all = [];
[D.observed, D.predicted, D.truth].forEach(a=>{if(a) all.push(...a);});
D.targets.forEach(t=>all.push([t.lat,t.lon]));
if(all.length) map.fitBounds(L.latLngBounds(all).pad(0.1));
else map.setView([59.6,30.0], 8);
</script>
</body>
</html>
"""


def build_threat_map(out_path: str,
                     observed_ll,
                     predicted_ll,
                     truth_ll,
                     targets,
                     threat_text: str,
                     title: str = "Прогноз положения воздушного объекта"):
    """
    Args:
        out_path:     путь для сохранения HTML.
        observed_ll:  список [[lat,lon], ...] наблюдаемой траектории.
        predicted_ll: список [[lat,lon], ...] прогноза.
        truth_ll:     список [[lat,lon], ...] истинного будущего (может быть None).
        targets:      список dict {name, type, lat, lon, prob}.
        threat_text:  строка оценки угрозы (наиболее вероятная цель, ETA).
        title:        заголовок страницы.
    """
    data = {
        "observed": observed_ll,
        "predicted": predicted_ll,
        "truth": truth_ll,
        "targets": targets,
    }
    html = (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__THREAT_TEXT__", threat_text)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
