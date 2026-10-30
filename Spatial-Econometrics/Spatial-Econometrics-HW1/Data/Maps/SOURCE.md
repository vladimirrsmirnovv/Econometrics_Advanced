---
Source: "https://www.geoboundaries.org/"
---

# Границы субъектов Российской Федерации

`russia_regions_simplified.geojson` - упрощенные границы субъектов первого административного уровня
(ADM1) из открытой базы geoBoundaries (gbOpen, лицензия CC BY 4.0). Скачан файл
`geoBoundaries-RUS-ADM1_simplified.geojson` из репозитория `wmgeolab/geoBoundaries`.

## Что берем

Только геометрию для фона карт: 83 субъекта, система координат WGS 84. Для анализа границы
не используются: единица наблюдения - населенный пункт как точка, матрицы весов строятся по
координатам точек. Названия субъектов в файле на английском языке и с данными сети не
сопоставляются.

## Объединенный слой для карт

`russia_regions_complete_simplified.geojson` используется в `Analysis/analysis.R`.
Он содержит все 83 исходных контура без изменений и шесть дополнительных
полных административных контуров: Крым, Севастополь, Донецкая, Луганская,
Запорожская и Херсонская области. Исходная выгрузка сохранена отдельно.
Объединенный файл содержит 89 объектов в WGS 84.

Дополнительные геометрии взяты из geoBoundaries gbOpen UKR ADM1:
`Autonomous Republic of Crimea`, `Sevastopol`, `Donetsk Oblast`,
`Luhansk Oblast`, `Zaporizhia Oblast`, `Kherson Oblast`.

- Метаданные: https://www.geoboundaries.org/api/current/gbOpen/UKR/ADM1/
- Файл: https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/releaseData/gbOpen/UKR/ADM1/geoBoundaries-UKR-ADM1_simplified.geojson
- Версия границ: 2017; сборка: 12 декабря 2023 года; загрузка: 11 октября 2026 года.
- Источник: OpenStreetMap / Wambacher; лицензия в метаданных:
  Open Data Commons Open Database License 1.0.
