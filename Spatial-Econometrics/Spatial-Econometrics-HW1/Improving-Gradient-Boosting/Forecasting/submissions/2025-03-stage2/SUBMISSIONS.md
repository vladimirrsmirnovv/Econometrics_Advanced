# Готовые мартовские прогнозы второго этапа

> **Актуальный статус после посылок:** это сохранённые файлы прежнего эксперимента. Их рекомендация не считается установленным мартовским улучшением; новые индивидуальные модели ухудшились на феврале. Объяснение одно-деревного XGBoost — `../../research-stage2/POST_SUBMISSION_REVIEW.md`. Дальнейшая работа использует только методы курса: `../../course-only/README.md`. Все CSV оставлены без изменений; исходный ZIP хранит исторический манифест.

**Рекомендован `test.csv`.** Он точно копирует `ensemble/test_frozen_new_ensemble.csv`; результат заранее заданной проверки: `passed`.

Все файлы: март 2025, 18 657 строк, столбцы `new_id,rto`, размер до 1 МБ. Посылки в соревнование не отправлялись.

Прежний отдельный spatial XGBoost CSV оказался пустым. Его контрольная копия восстановлена в этой новой папке из исходного ZIP; SHA-256 совпадает с прежним манифестом. Старая папка не изменялась.

| Модель | Исходная, полное обучение | Прежняя модификация, полное обучение | Новая выбранная, rolling refit | Новая конфигурация / деревья |
| --- | --- | --- | --- | --- |
| catboost_log_level | `reference-baseline/test_catboost_log_level.csv` | `reference-previous/test_catboost_log_level.csv` | `individual/test_catboost_log_level.csv` | hybrid8_history+seasonal_spatial / 856 |
| lightgbm_log_diff | `reference-baseline/test_lightgbm_log_diff.csv` | `reference-previous/test_lightgbm_log_diff.csv` | `individual/test_lightgbm_log_diff.csv` | innovations_spatial / 51 |
| lightgbm_log_level | `reference-baseline/test_lightgbm_log_level.csv` | `reference-previous/test_lightgbm_log_level.csv` | `individual/test_lightgbm_log_level.csv` | seasonal_spatial / 134 |
| xgboost_log_diff | `reference-baseline/test_xgboost_log_diff.csv` | `reference-previous/test_xgboost_log_diff.csv` | `individual/test_xgboost_log_diff.csv` | seasonal_spatial / 1 |

`ensemble/test_frozen_new_ensemble.csv` — ансамбль из выбранных новых версий, правило `gated_1_08_no_weak`. Он остаётся экспериментальным, если контрольная проверка не пройдена. Правило gated_1_08_no_weak использует CatBoost, LightGBM уровней и XGBoost; LightGBM приростов исключается.

`ensemble/test_mixed_original3_hybrid_xgb_final_meta_logmean.csv` — обязательное сравнение: три исходные полные модели плюс прежний полный hybrid8 XGBoost. Другие файлы с тем же префиксом содержат пять остальных фиксированных правил.

`ensemble/test_original_full_ensemble.csv` и `ensemble/test_previous_full_gated_1_08_all.csv` — точные копии прежних ансамблей для контроля.

В `march_submissions_stage2.zip` включены все перечисленные CSV и манифест. Native-модели и параметры новых вариантов находятся в `models` и `individual/*.json`, а не в архиве посылок.

Новые числа деревьев выбираются на феврале 2025, после чего модели обучаются на январе 2023 — феврале 2025. Разница с исходными полными CSV включает как признаки, так и протокол выбора длины обучения; их отдельные эффекты показаны в отчёте `Forecasting/research-stage2/REPORT.md`.
