# Март 2025: только учебный WX

Рекомендован подтверждённый состав.

Для загрузки используйте новый пакет `../2025-03-course-upload-ready/march_course_upload_ready.zip`: после распаковки каждый вариант находится в своей папке под именем `test.csv`. Выбирайте этот CSV напрямую из распакованной папки. Суффиксы `(1)` и пробелы в имени исключены. ZIP служит для доставки файлов; в соревнование загружается CSV. Значения всех прогнозов сохранены побайтово. Причина серверных CE пока не подтверждена отчётом проверяющей системы.

`test.csv` — рекомендованный файл. `baseline` — неизменённые исходные прогнозы; `experimental` — WX каждой модели, включая не подтвердившиеся варианты; `confirmed` — WX только при выигрыше в разработке и обоих контрольных месяцах, иначе исходная модель. Во всех файлах 18657 строк `new_id,rto`; каждый меньше 1 МБ. Прогнозы относятся к марту 2025. В соревнование автоматически не отправлены. Качество марта неизвестно.

| file | rows | bytes |
| --- | --- | --- |
| baseline/test_catboost_log_level.csv | 18657 | 461970 |
| baseline/test_lightgbm_log_diff.csv | 18657 | 462157 |
| baseline/test_lightgbm_log_level.csv | 18657 | 462134 |
| baseline/test_xgboost_log_diff.csv | 18657 | 443547 |
| confirmed/test_catboost_log_level.csv | 18657 | 461970 |
| confirmed/test_lightgbm_log_diff.csv | 18657 | 443398 |
| confirmed/test_lightgbm_log_level.csv | 18657 | 462134 |
| confirmed/test_xgboost_log_diff.csv | 18657 | 443547 |
| ensemble/test_baseline_final_meta_logmean.csv | 18657 | 461960 |
| ensemble/test_confirmed_final_meta_logmean.csv | 18657 | 462159 |
| ensemble/test_experimental_final_meta_logmean.csv | 18657 | 462070 |
| experimental/test_catboost_log_level.csv | 18657 | 462103 |
| experimental/test_lightgbm_log_diff.csv | 18657 | 443398 |
| experimental/test_lightgbm_log_level.csv | 18657 | 443485 |
| experimental/test_xgboost_log_diff.csv | 18657 | 443389 |
| test.csv | 18657 | 462159 |
