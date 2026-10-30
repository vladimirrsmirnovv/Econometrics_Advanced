# GeoNames Settlement Coordinates

Пайплайн собирает координаты населённых пунктов для всех `new_id` из
`Source/Qualifying/train.csv` на основе выгрузки GeoNames по России.

Что делает скрипт:

- скачивает `RU.zip` из GeoNames во временную директорию `Notes/Vladimir/tmp`;
- распаковывает `RU.txt` и `readme.txt`;
- сопоставляет каждому `new_id` регион и населённый пункт из train с записью GeoNames;
- сохраняет промежуточную таблицу `new_id, Регион, Населенный пункт, lon, lat, url`;
- отдельно сохраняет диагностику по матчингу населённых пунктов и регионов.

Запуск:

```bash
uv run python Parsing/geonames_settlement_coordinates/build_settlement_coordinates.py
```

Основные артефакты:

- `Parsing/geonames_settlement_coordinates/result/settlement_coordinates.csv`
- `Parsing/geonames_settlement_coordinates/result/settlement_coordinates_diagnostics.csv`
- `Parsing/geonames_settlement_coordinates/result/region_resolution.csv`
