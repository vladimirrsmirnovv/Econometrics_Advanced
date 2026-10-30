"""Prepare the last five requested blends of existing March forecasts.

Use equal arithmetic/logarithmic averages of predictions from the preceding
ten-file batch. Do not train models, tune weights, or submit to the contest.
"""


def _directory_path(value):
    """Resolve directory names from preserved metadata after the folder rename."""
    from pathlib import Path as _Path
    path = _Path(value)
    if path.is_absolute():
        return path
    directory_count = len(path.parts) - bool(path.suffix)
    return _Path(*(part.replace("_", "-") if index < directory_count else part
                   for index, part in enumerate(path.parts)))

from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

from evaluation_context import annotate_evaluation
from package_uploads import validate

HERE = Path(__file__).resolve().parent
FORECASTING = HERE.parent
SOURCE = FORECASTING / 'submissions/2025-03-course-final-ten'
OUT = FORECASTING / 'submissions/2025-03-course-final-five'
COMPONENTS = {
    'A': (4, '04-three-wx-original-mix/test.csv', 90.83),
    'B': (5, '05-three-wx-gated-all/test.csv', 90.83),
    'C': (1, '01-winner-gated-all/test.csv', 90.82),
    'D': (8, '08-xgb-ll-wx-gated-no-ld/test.csv', 90.82),
}
CANDIDATES = [
    ('01_leaders_equal', ('A', 'B'), 'equal'),
    ('02_leaders_log_average', ('A', 'B'), 'log_average'),
    ('03_leaders_plus_01_equal', ('A', 'B', 'C'), 'equal'),
    ('04_leaders_plus_08_equal', ('A', 'B', 'D'), 'equal'),
    ('05_leaders_plus_08_log_average', ('A', 'B', 'D'), 'log_average'),
]
KNOWN_PACKAGES = [
    ('2025-03-course', 'submission_manifest.json'),
    ('2025-03-course-ensemble-alternatives', 'manifest.json'),
    ('2025-03-course-ensemble-followups', 'manifest.json'),
    ('2025-03-course-final-ten', 'manifest.json'),
]
LABELS = {'equal': 'Обычное среднее', 'log_average': 'Логарифмическое среднее'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    frozen_before = sha(HERE / 'FINAL_DECISION.json')
    original_rules_before = sha(FORECASTING / 'run_validation.py')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    expected = {e['file']: e for e in source_manifest['files']}
    results_path = SOURCE / 'contest_results.json'
    assert sha(results_path) == source_manifest['contest_results_sha256']
    registry = json.loads(results_path.read_text(encoding='utf-8'))
    scores = {r['order']: r for r in registry['submissions']}
    components, sources, ids = {}, {}, None
    for key, (order, relative, score) in COMPONENTS.items():
        path = SOURCE / _directory_path(relative)
        assert sha(path) == expected[relative]['sha256'] == scores[order]['sha256']
        assert scores[order]['contest_score'] == score
        frame = pd.read_csv(path, float_precision='round_trip')
        if ids is None:
            ids = frame.new_id.to_numpy()
        np.testing.assert_array_equal(ids, frame.new_id.to_numpy())
        validate(path.read_bytes(), sorted(map(int, ids)))
        components[key] = frame.rto.to_numpy()
        sources[key] = dict(previous_batch_order=order, file=relative,
                            sha256=sha(path), observed_private_score=score,
                            submission_id=scores[order]['submission_id'],
                            composition=expected[relative]['composition_label'],
                            rule=expected[relative]['rule_label'])

    # Check every earlier forecast by ID, including all earlier blend rules.
    known = []
    sorted_ids = np.sort(ids)
    for folder, manifest_name in KNOWN_PACKAGES:
        directory = FORECASTING / 'submissions' / folder.replace("_", "-")
        metadata = json.loads((directory / manifest_name).read_text(encoding='utf-8'))
        for entry in metadata['files']:
            path = directory / _directory_path(entry['file'])
            assert sha(path) == entry['sha256']
            frame = pd.read_csv(path, float_precision='round_trip').sort_values('new_id')
            np.testing.assert_array_equal(sorted_ids, frame.new_id.to_numpy())
            known.append((str(path.relative_to(FORECASTING)), frame.rto.to_numpy()))

    outputs = []
    sorting = np.argsort(ids)
    for folder, keys, rule in CANDIDATES:
        matrix = np.column_stack([components[key] for key in keys])
        if rule == 'equal':
            prediction = matrix.mean(axis=1)
        else:
            # Same log1p/expm1 operation as the original notebook's log average.
            prediction = np.expm1(np.log1p(np.maximum(matrix, 0)).mean(axis=1))
        assert np.all(np.isfinite(prediction)) and np.all(prediction > 0)
        for prior_path, prior in known:
            assert not np.allclose(prediction[sorting], prior, rtol=1e-12, atol=1e-6), prior_path
        for _, prior_keys, prior_rule, prior in outputs:
            assert not np.allclose(prediction, prior, rtol=1e-12, atol=1e-6), (prior_keys, prior_rule)
        outputs.append((folder, keys, rule, prediction))

    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for number, (folder, keys, rule, prediction) in enumerate(outputs, 1):
        path = OUT / folder.replace("_", "-") / 'test.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'new_id': ids, 'rto': prediction}).to_csv(
            path, index=False, encoding='utf-8', lineterminator='\n')
        checked = validate(path.read_bytes(), sorted(map(int, ids)))
        round_trip = pd.read_csv(path, float_precision='round_trip')
        np.testing.assert_array_equal(ids, round_trip.new_id.to_numpy())
        np.testing.assert_array_equal(prediction, round_trip.rto.to_numpy())
        files.append(dict(order=number, file=path.relative_to(OUT).as_posix(),
                          components=list(keys), previous_batch_orders=[COMPONENTS[k][0] for k in keys],
                          averaging_rule=rule, rule_label=LABELS[rule],
                          weights=[1 / len(keys)] * len(keys), private_score=None, **checked))

    assert len(files) == 5
    assert len(list(OUT.glob('*/test.csv'))) == 5
    manifest = annotate_evaluation(dict(
        status='passed', forecast_month='2025-03', requested_final_batch_size=5,
        required_upload_filename='test.csv',
        source_package='2025-03-course-final-ten', source_components=sources,
        source_manifest_sha256=sha(SOURCE / 'manifest.json'),
        source_contest_results_sha256=sha(results_path),
        source_score_mapping_order_explicitly_confirmed=registry['mapping_order_explicitly_confirmed'],
        averages_existing_ensemble_predictions=True,
        original_averaging_operations_reused=True, no_models_retrained=True,
        no_new_spatial_methods=True, no_weights_fitted=True, no_new_thresholds=True,
        uniform_component_weights=True, exploratory_after_contest_feedback=True,
        original_rule_source_sha256=original_rules_before,
        original_decision_sha256=frozen_before,
        private_scores_available=False, server_acceptance_verified=False,
        automatic_submissions=False, files=files,
        duplicate_check=dict(prior_files_compared=len(known), new_files=5,
                             relative_tolerance=1e-12, absolute_tolerance=1e-6,
                             no_duplicates=True)))
    write_json(OUT / 'manifest.json', manifest)

    lines = [
        '# Последние пять сабмитов на март 2025', '',
        'Распакуйте архив. Отправляйте файл ровно `test.csv` из папок 01–05 '
        'по порядку. В проверяющую систему загружается CSV, а не общий ZIP. '
        'Сохраняйте имя без суффиксов `(1)` и без дополнительного расширения.', '',
        'Это новые равновесные смеси готовых прогнозов из предыдущей десятки. '
        'Номера в колонке «Компоненты» относятся именно к предыдущей десятке, '
        'а не к этому пакету.', '',
        'A = прежний №04, исходная meta-смесь с WX у XGBoost и обоих LightGBM, 90,83.  '
        '\nB = прежний №05, тот же состав с правилом среднего/медианы 1,08, 90,83.  '
        '\nC = прежний №01, WX у XGBoost и LGB приростов, исходный LGB уровней, '
        'правило 1,08, 90,82.  '
        '\nD = прежний №08, WX у XGBoost и LGB уровней, без LGB приростов, '
        'правило 1,08, 90,82.  '
        '\nCatBoost во всех исходных компонентах остаётся базовым.', '',
        '| Новый № | Компоненты из прежней десятки | Формула | Файл |',
        '| --- | --- | --- | --- |',
    ]
    for entry in files:
        orders = ' + '.join(f'№{n:02d}' for n in entry['previous_batch_orders'])
        lines.append(f"| {entry['order']:02d} | {orders} | {entry['rule_label']} | `{entry['file']}` |")
    lines.extend(['',
        'У двух компонентов равные доли 1/2, у трёх — 1/3. Обычное среднее: '
        'mean(p). Логарифмическое: exp(mean(log(1+p)))−1. Усредняются значения '
        'прогнозов каждого магазина. Баллы компонентов в формулу не входят.', '',
        'Итоговые приватные баллы этих пяти новых смесей неизвестны. '
        'Ранее сообщённые 90,83/90,82 относятся к компонентам. '
        'Пять файлов подготовлены по запросу пользователя как заключительная партия. '
        'Повторного обучения и новых пространственных признаков нет.', '',
        'Проверены SHA источников, соответствие исходным идентификаторам, '
        '18657 строк new_id,rto, положительные конечные прогнозы, '
        'размер каждого CSV меньше 1 МБ, UTF-8 без BOM, точное восстановление '
        'чисел после записи, отсутствие повторов с предыдущими 36 CSV '
        'и внутри пятёрки, CRC архива. Автоматических посылок нет.', '',
    ])
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')

    archive = OUT / 'final_five_ensembles.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for entry in files:
            z.write(OUT / _directory_path(entry['file']), entry['file'])
        for name in ['README.md', 'manifest.json']:
            z.write(OUT / _directory_path(name), name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in files:
            assert z.read(entry['file']) == (OUT / _directory_path(entry['file'])).read_bytes()
    assert sha(HERE / 'FINAL_DECISION.json') == frozen_before
    assert sha(FORECASTING / 'run_validation.py') == original_rules_before
    for key, source in sources.items():
        assert sha(SOURCE / _directory_path(source['file'])) == source['sha256']
    write_json(OUT / 'package_checks.json', dict(
        status='passed', csv_count=5, unique_forecasts=5,
        all_headers_ids_values_and_sizes_valid=True, float_round_trip_exact=True,
        prior_files_compared=len(known), no_forecast_duplicates=True,
        source_forecasts_unchanged=True, original_blend_code_unchanged=True,
        frozen_decision_unchanged=True, zip_crc_and_sha_passed=True,
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size))
    print(json.dumps(dict(status='passed', csv_files=5, archive=str(archive),
                          archive_bytes=archive.stat().st_size, no_duplicates=True,
                          private_scores_available=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
