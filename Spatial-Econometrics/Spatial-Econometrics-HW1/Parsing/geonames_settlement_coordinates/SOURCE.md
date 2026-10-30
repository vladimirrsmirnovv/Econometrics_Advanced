# Source

Источник координат населённых пунктов:

- GeoNames Russia dump: [RU.zip](https://download.geonames.org/export/dump/RU.zip)
- Документация формата: `readme.txt` внутри архива GeoNames
- Лицензия: Creative Commons Attribution 4.0 License

Что используется в задаче:

- файл `RU.txt` из архива;
- записи класса `P` для населённых пунктов;
- записи класса `A` и кода `ADM1` для сопоставления регионов.

Ключевые поля GeoNames:

- `name`, `asciiname`, `alternatenames` — варианты названия объекта;
- `latitude`, `longitude` — координаты центра объекта;
- `feature_class`, `feature_code` — тип географического объекта;
- `admin1_code` — код административной единицы первого уровня;
- `population` — численность населения, используется как tie-breaker при дублях.

Локальное размещение:

- архив и распакованные файлы кладутся в `Parsing/geonames_settlement_coordinates/tmp/geonames_ru/`;
- итоговые CSV сохраняются в `Parsing/geonames_settlement_coordinates/result/`.
