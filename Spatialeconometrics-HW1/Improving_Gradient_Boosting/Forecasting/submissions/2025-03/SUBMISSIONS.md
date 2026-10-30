# Готовые submission на март 2025

Все основные CSV имеют ровно 18 657 строк, столбцы `new_id,rto`, исходный порядок магазинов, конечные неотрицательные прогнозы. Обучение использует январь 2023 — февраль 2025. Мартовские реальные значения отсутствуют.

| Модель | Исходный CSV | CSV с выбранным улучшением |
|---|---|---|
| CatBoost уровней | [Исходная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/baseline/test_catboost_log_level.csv>) | [Улучшенная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/spatial/test_catboost_log_level.csv>) |
| LightGBM темпов | [Исходная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/baseline/test_lightgbm_log_diff.csv>) | [Улучшенная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/spatial/test_lightgbm_log_diff.csv>) |
| LightGBM уровней | [Исходная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/baseline/test_lightgbm_log_level.csv>) | [Улучшенная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/spatial/test_lightgbm_log_level.csv>) |
| XGBoost темпов | [Исходная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/baseline/test_xgboost_log_diff.csv>) | [Улучшенная](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/spatial/test_xgboost_log_diff.csv>) |

## Ансамбли

- [Исходный final_meta_logmean](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/ensemble/baseline_final_meta_logmean.csv>): четыре исходные модели и исходная формула ноутбука.
- [После изменений с той же формулой](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/ensemble/spatial_final_meta_logmean.csv>): четыре выбранные версии; формула смешивания сохранена.
- [Лучшее правило по основной валидации](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/ensemble/spatial_gated_1_08_all.csv>): те же улучшенные компоненты, правило `gated_1_08_all`.

[Архив основных submission](<C:/Users/smirn/OneDrive/Рабочий стол/Spatialeconometrics-HW1/Forecasting/submissions/2025-03/march_submissions.zip>) содержит 11 CSV и манифест. В `ensemble/reference_rules` сохранены остальные исходные правила смеси для сравнения.

## Что изменено

CatBoost: четыре исторических признака соседей и фиксированная коррекция общего смещения по февральскому OOF той же пространственной версии. Прогноз до коррекции находится в `diagnostics`. LightGBM темпов: восемь географических собственных векторов. LightGBM уровней: веса обучения по размеру рынка. XGBoost: четыре признака истории гибридных соседей. Цели и исходные гиперпараметры сохранены.

Полные модели обучены без ранней остановки: 6 500 / 7 000 / 8 000 / 3 750 деревьев в порядке таблицы. XGBoost — CUDA 3.1.3. Исходный мартовский XGBoost сопоставлен с ранее воспроизведённым прогнозом.

Метрики марта по этим данным вычислить нельзя. Отбор версий и правила смеси выполнен ранее на валидации, а не по неизвестному мартовскому результату. Проверки, хэши, параметры, версии и пути native-моделей сохранены в JSON рядом с CSV и в `submission_manifest.json`.
