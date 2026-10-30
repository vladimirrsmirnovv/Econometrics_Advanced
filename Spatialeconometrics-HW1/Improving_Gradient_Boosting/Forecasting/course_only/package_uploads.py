"""Package existing forecasts under the competition's required filename test.csv.

Copies bytes without changing predictions, IDs, order, encoding or line endings.
Does not train models or submit anything to the competition.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import re
import zipfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'submissions' / '2025-03-course'
DESTINATION = HERE.parent / 'submissions' / '2025-03-course-upload-ready'
MODELS = [
    ('01_catboost', 'catboost_log_level', 'CatBoost, уровни', 'wx_knn12'),
    ('02_lightgbm_diff', 'lightgbm_log_diff', 'LightGBM, приросты', 'wx_knn12'),
    ('03_lightgbm_level', 'lightgbm_log_level', 'LightGBM, уровни', 'wx_knn6'),
    ('04_xgboost', 'xgboost_log_diff', 'XGBoost, приросты', 'wx_knn12'),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(data, expected_ids=None):
    assert 0 < len(data) < 1_000_000, 'CSV must be smaller than 1 MB'
    assert not data.startswith(b'\xef\xbb\xbf'), 'Unexpected BOM'
    rows = list(csv.reader(io.StringIO(data.decode('utf-8'))))
    assert rows[0] == ['new_id', 'rto'], 'Unexpected header'
    assert len(rows) == 18658, 'Expected 18657 data rows'
    assert all(len(row) == 2 for row in rows[1:]), 'Expected two columns'
    ids = [row[0] for row in rows[1:]]
    assert all(re.fullmatch(r'[0-9]+', value) for value in ids)
    assert len(set(ids)) == 18657
    if expected_ids is not None:
        assert sorted(map(int, ids)) == expected_ids, 'Store IDs differ from the reference'
    predictions = [float(row[1]) for row in rows[1:]]
    assert all(math.isfinite(value) and value > 0 for value in predictions)
    return dict(rows=18657, bytes=len(data), sha256=digest(data))


def build():
    manifest = json.loads((SOURCE / 'submission_manifest.json').read_text(encoding='utf-8'))
    expected = {row['file']: row['sha256'] for row in manifest['files']}
    decision = json.loads((SOURCE / 'CURRENT_STATUS.json').read_text(encoding='utf-8'))
    reference = (SOURCE / 'test.csv').read_bytes()
    assert validate(reference)['sha256'] == expected['test.csv']
    reference_ids = sorted(int(row[0]) for row in
                           list(csv.reader(io.StringIO(reference.decode('utf-8'))))[1:])
    entries = []

    def copy(source_name, destination_name, label, role):
        data = (SOURCE / source_name).read_bytes()
        checked = validate(data, reference_ids)
        assert checked['sha256'] == expected[source_name], 'Source changed since validation'
        destination = DESTINATION / destination_name
        assert destination.name == 'test.csv'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        assert destination.read_bytes() == data, 'Copy is not byte identical'
        entries.append(dict(file=destination_name, source=source_name, label=label,
                            role=role, byte_identical_to_source=True, **checked))

    for folder, model, label, wx in MODELS:
        copy(f'baseline/test_{model}.csv', f'{folder}/01_before/test.csv', label, 'baseline')
        copy(f'experimental/test_{model}.csv', f'{folder}/02_after_{wx}/test.csv', label, wx)
    for folder, composition, label in [
        ('01_before', 'baseline', 'Исходный ансамбль'),
        ('02_all_wx', 'experimental', 'Ансамбль всех четырёх WX-кандидатов'),
        ('03_confirmed', 'confirmed', 'Подтверждённый состав: WX только у LightGBM приростов'),
    ]:
        copy(f'ensemble/test_{composition}_final_meta_logmean.csv',
             f'05_ensembles/{folder}/test.csv', label, composition)

    lines = [
        '# Март 2025: файлы с именем test.csv', '',
        'Распакуйте архив с сохранением папок. Для посылки выберите CSV из нужной папки: '
        'его имя должно оставаться ровно `test.csv`, без `(1)`, пробелов и дополнительного расширения. '
        'В соревнование загружается этот CSV, а не общий ZIP. '
        'Папки разделяют модели и варианты; одинаковые имена файлов не конфликтуют.', '',
        'В каждой из первых четырёх папок: `01_before` — исходная модель; '
        '`02_after_...` — та же модель с учебными WX-признаками. '
        '«После» обозначает эксперимент, а не гарантированное улучшение. '
        'В подтверждённом ансамбле заменён только LightGBM приростов; '
        'остальные три модели оставлены исходными.', '',
        'Все 11 CSV побайтово совпадают с ранее подготовленными прогнозами. '
        'Изменена только упаковка. Деревья, значения прогнозов и метрики не менялись. '
        'В каждом CSV 18657 строк, столбцы `new_id,rto`, целочисленные уникальные ID, '
        'конечные положительные прогнозы; размер меньше 1 МБ.', '',
        'Одиночные файлы в пользовательских «Загрузках» имеют суффиксы `(1)`–`(4)`; '
        'смеси, которые пользователь сообщил как принятые, таких суффиксов не имеют. '
        'Содержимое этих скачанных CSV совпало с исходными файлами по SHA-256. '
        'Это указывает на возможную проблему имени при загрузке. '
        'Точная причина CE ещё не установлена: нужен текст отчёта проверяющей системы. '
        'Принятие нового пакета сервером пока не проверено.', '',
        'Предоставленный `Downloads/test.csv` содержит один и тот же прогноз '
        '`33503980.259272862` для всех магазинов. Это отдельный файл, '
        'не один из 11 прогнозов этого пакета; его значения здесь не использованы.', '',
        '| Вариант | Файл | Размер, байт |', '| --- | --- | --- |',
    ]
    lines.extend(f"| {entry['label']} / {entry['role']} | `{entry['file']}` | {entry['bytes']} |"
                 for entry in entries)
    (DESTINATION / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    output_manifest = dict(status='passed', forecast_month='2025-03',
                           packaging_only=True, required_upload_filename='test.csv',
                           exact_prediction_bytes_preserved=True,
                           server_acceptance_verified=False,
                           files=entries, decision=decision)
    (DESTINATION / 'manifest.json').write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    archive = DESTINATION / 'march_course_upload_ready.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for entry in entries:
            package.write(DESTINATION / entry['file'], entry['file'])
        for name in ['README.md', 'manifest.json']:
            package.write(DESTINATION / name, name)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        for entry in entries:
            assert digest(package.read(entry['file'])) == entry['sha256']
    print(json.dumps(dict(status='passed', csv_files=len(entries),
                          all_named_test_csv=True, byte_identical_copies=True,
                          archive=str(archive), archive_bytes=archive.stat().st_size),
                     ensure_ascii=False))
    return output_manifest


if __name__ == '__main__':
    build()
