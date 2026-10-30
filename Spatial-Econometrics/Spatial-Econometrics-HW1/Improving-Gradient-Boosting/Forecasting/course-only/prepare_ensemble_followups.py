"""Prepare seven finite, original-rule ensemble trials from existing course WX fits.

All variants are exploratory after final contest feedback. No model/weight training,
new thresholds, custom spatial estimators or competition uploads are performed.
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
import json
import zipfile

import numpy as np
import pandas as pd

from manage import MODELS, DEVELOPMENT, CONFIRM, load, mape, sha, write_json
from run_validation import original_blends
from package_uploads import validate
from evaluation_context import annotate_evaluation

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'submissions/2025-03-course'
PREVIOUS = HERE.parent / 'submissions/2025-03-course-ensemble-alternatives'
OUT = HERE.parent / 'submissions/2025-03-course-ensemble-followups'
REFERENCE_FILE = PREVIOUS / '01-only-xgb-wx-original-mix/test.csv'
CANDIDATES = [
    ('01_xgb_cat_wx_meta', 'xgb_cat', 'final_meta_logmean',
     'WX у XGBoost и CatBoost; оба LightGBM исходные', 'первая очередь'),
    ('02_xgb_ld_wx_meta', 'xgb_ld', 'final_meta_logmean',
     'WX у XGBoost и LightGBM приростов; остальные исходные', 'первая очередь'),
    ('03_xgb_wx_gated_no_ld', 'xgb', 'gated_1_08_no_weak',
     'XGBoost WX + исходные CatBoost и LightGBM уровней; смесь трёх с порогом 1,08',
     'первая очередь'),
    ('04_xgb_wx_gated_all', 'xgb', 'gated_1_08_all',
     'Только XGBoost WX; смесь четырёх с порогом 1,08', 'первая очередь'),
    ('05_xgb_wx_equal', 'xgb', 'equal',
     'Только XGBoost WX; обычное среднее четырёх', 'резерв'),
    ('06_xgb_wx_log_average', 'xgb', 'log_average',
     'Только XGBoost WX; логарифмическое среднее четырёх', 'резерв'),
    ('07_xgb_ll_wx_meta', 'xgb_ll', 'final_meta_logmean',
     'WX у XGBoost и LightGBM уровней; остальные исходные', 'резерв'),
]


def compositions(base, wx):
    xgb = dict(base, xgboost_log_diff=wx['xgboost_log_diff'])
    return dict(baseline=base, xgb=xgb,
                xgb_cat=dict(xgb, catboost_log_level=wx['catboost_log_level']),
                xgb_ld=dict(xgb, lightgbm_log_diff=wx['lightgbm_log_diff']),
                xgb_ll=dict(xgb, lightgbm_log_level=wx['lightgbm_log_level']))


def collect_validation(choices):
    records, sources = [], []
    for fold in DEVELOPMENT + CONFIRM:
        base, wx, ids, truth = {}, {}, None, None
        for model in MODELS:
            b, bi = load(fold, model, 'baseline')
            w, wi = load(fold, model, choices[model])
            if ids is None:
                ids, truth = b.new_id.to_numpy(), b.pto.to_numpy()
            for frame in [b, w]:
                np.testing.assert_array_equal(ids, frame.new_id)
                np.testing.assert_array_equal(truth, frame.pto)
            base[model], wx[model] = b.prediction.to_numpy(), w.prediction.to_numpy()
            sources.extend(dict(fold=fold, model=model, configuration=info['configuration'],
                                trees=info['trees'], prediction_sha256=info['prediction_sha256'])
                           for info in [bi, wi])
        for composition, components in compositions(base, wx).items():
            for rule, prediction in original_blends(components).items():
                records.append(dict(fold=fold, composition=composition, rule=rule,
                                    mape=mape(truth, prediction)))
    previous = json.loads((PREVIOUS / 'validation_metrics.json').read_text(encoding='utf-8'))
    lookup = {(r['fold'], r['composition'], r['rule']): r['mape']
              for r in previous['metrics']}
    for record in records:
        if record['composition'] not in ['baseline', 'xgb']:
            continue
        composition = 'only_xgb_wx' if record['composition'] == 'xgb' else 'baseline'
        assert abs(record['mape'] - lookup[(record['fold'], composition, record['rule'])]) < 1e-10
    return records, sources


def read_known_submissions():
    known = []
    for root, name in [(SOURCE, 'submission_manifest.json'), (PREVIOUS, 'manifest.json')]:
        manifest = json.loads((root / _directory_path(name)).read_text(encoding='utf-8'))
        for entry in manifest['files']:
            path = root / _directory_path(entry['file'])
            assert sha(path) == entry['sha256']
            frame = pd.read_csv(path, float_precision='round_trip').sort_values('new_id')
            known.append(dict(file=str(path), new_id=frame.new_id.to_numpy(),
                              rto=frame.rto.to_numpy(), sha256=entry['sha256']))
    return known


def main():
    choices = {model: json.loads((HERE / 'selection' / (model + '.json')).read_text(
        encoding='utf-8'))['selected_wx'] for model in MODELS}
    records, validation_sources = collect_validation(choices)
    known = read_known_submissions()
    expected = {r['file']: r['sha256'] for r in json.loads(
        (SOURCE / 'submission_manifest.json').read_text(encoding='utf-8'))['files']}
    base, wx, ids, march_sources = {}, {}, None, []
    for model in MODELS:
        for variant, destination in [('baseline', base), ('experimental', wx)]:
            relative = f'{variant.replace("_", "-")}/test_{model}.csv'
            path = SOURCE / _directory_path(relative)
            assert sha(path) == expected[relative]
            frame = pd.read_csv(path, float_precision='round_trip').sort_values('new_id')
            if ids is None:
                ids = frame.new_id.to_numpy()
            np.testing.assert_array_equal(ids, frame.new_id)
            destination[model] = frame.rto.to_numpy()
            march_sources.append(dict(file=relative, sha256=expected[relative]))
    march = {composition: original_blends(components)
             for composition, components in compositions(base, wx).items()}
    reference = pd.read_csv(REFERENCE_FILE, float_precision='round_trip').sort_values('new_id')
    np.testing.assert_array_equal(ids, reference.new_id)
    np.testing.assert_array_equal(march['xgb']['final_meta_logmean'], reference.rto)
    final_decision_sha256 = sha(HERE / 'FINAL_DECISION.json')
    OUT.mkdir(parents=True, exist_ok=True)

    def summary(composition, rule):
        part = [r for r in records if r['composition'] == composition and r['rule'] == rule]
        assert len(part) == 5
        result = {r['fold']: r['mape'] for r in part}
        result['development_mean'] = float(np.mean([result[f] for f in DEVELOPMENT]))
        result['confirmation_mean'] = float(np.mean([result[f] for f in CONFIRM]))
        return result

    original = summary('baseline', 'final_meta_logmean')
    benchmark = summary('xgb', 'final_meta_logmean')
    files, candidate_predictions = [], []
    for order, (folder, composition, rule, label, priority) in enumerate(CANDIDATES, 1):
        prediction = march[composition][rule]
        # Ignore round-trip/line-ending differences; prohibit practically identical forecasts.
        for existing in known:
            np.testing.assert_array_equal(ids, existing['new_id'])
            assert not np.allclose(prediction, existing['rto'], rtol=1e-12, atol=1e-6), (
                folder, 'duplicates an already prepared submission', existing['file'])
        for prior in candidate_predictions:
            assert not np.allclose(prediction, prior, rtol=1e-12, atol=1e-6), folder
        candidate_predictions.append(prediction)
        path = OUT / folder.replace("_", "-") / 'test.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(new_id=ids, rto=prediction)).to_csv(
            path, index=False, lineterminator='\n')
        checked = validate(path.read_bytes(), sorted(map(int, ids)))
        readback = pd.read_csv(path, float_precision='round_trip')
        np.testing.assert_array_equal(ids, readback.new_id)
        np.testing.assert_array_equal(prediction, readback.rto)
        metrics = summary(composition, rule)
        wx_models = ['xgboost_log_diff'] + {
            'xgb': [], 'xgb_cat': ['catboost_log_level'],
            'xgb_ld': ['lightgbm_log_diff'], 'xgb_ll': ['lightgbm_log_level'],
        }[composition]
        excluded_models = ['lightgbm_log_diff'] if rule == 'gated_1_08_no_weak' else []
        files.append(dict(order=order, file=path.relative_to(OUT).as_posix(), label=label,
                          priority=priority, composition=composition, original_rule=rule,
                          wx_models=wx_models, excluded_models=excluded_models,
                          original_models=[m for m in MODELS if m not in wx_models + excluded_models],
                          validation=metrics,
                          wins_both_months_vs_original=all(metrics[f] < original[f] for f in CONFIRM),
                          wins_both_months_vs_public_best_reference=all(
                              metrics[f] < benchmark[f] for f in CONFIRM), **checked))
    manifest = dict(status='passed', exploratory_after_public_feedback=True,
                    all_validation_months_previously_seen=True, no_models_retrained=True,
                    no_weights_fitted=True, only_original_notebook_blend_rules=True,
                    fixed_original_gate_threshold=1.08, spatial_method='Course WX only',
                    source_rule_file='Forecasting/run_validation.py:48',
                    source_rule_sha256=sha(HERE.parent / 'run_validation.py'),
                    original_decision_unchanged=True, original_decision_sha256=final_decision_sha256,
                    current_public_reference=dict(file=str(REFERENCE_FILE),
                                                  sha256=sha(REFERENCE_FILE),
                                                  public_score=90.80,
                                                  mapping_assumes_previous_upload_order=True,
                                                  validation=benchmark),
                    original_ensemble_validation=original, march_sources=march_sources,
                    validation_sources=validation_sources, files=files,
                    duplicate_check=dict(known_files=len(known),
                                         absolute_tolerance=1e-6, relative_tolerance=1e-12,
                                         no_duplicates=True),
                    automatic_submissions=False, public_scores_available=False)
    annotate_evaluation(manifest)
    write_json(OUT / 'manifest.json', manifest)
    write_json(OUT / 'validation_metrics.json', dict(status='retrospective_comparison',
                                                    metrics=records))
    text = [
        '# Семь следующих мартовских смесей', '',
        'Уточнение контекста: соревнование завершено, указанные баллы относятся '
        'к основному финальному мартовскому сету для рейтинга. Точный охват '
        'магазинов не подтверждён; локальная валидация служит диагностикой.', '',
        'Варианты расположены по рекомендуемому порядку проверки. Сначала папки 01–04. '
        'Папки 05–07 — резерв: локально они проигрывают текущему итоговому ориентиру '
        'в обоих контрольных месяцах. Текущий ориентир — исходная смесь с WX только '
        'у XGBoost, 90,80 итоговый балл при сохранённом порядке предыдущих отправок.', '',
        'Распакуйте архив с сохранением папок и отправляйте CSV `test.csv` из выбранной '
        'папки. Общий ZIP не отправляется. Ни один из семи прогнозов не дублирует '
        'ранее подготовленные варианты, включая отличие только в переносах строк '
        'или числовой точности записи.', '',
        '| № | Состав и правило | Приоритет |', '| --- | --- | --- |',
    ]
    text.extend(f"| {entry['order']:02d} | {entry['label']} | {entry['priority']} |"
                for entry in files)
    text.extend(['',
        'В 01,02,07 используется та же исходная meta-формула, что у варианта с 90,80; '
        'меняется только версия ещё одной модели. В 03,04,05,06 состав фиксирован: '
        'исходные CatBoost и LightGBM, XGBoost с WX. В 03 дополнительно исключён '
        'LightGBM приростов, как в исходной ветви no_weak.', '',
        'Правило с порогом 1,08 взято из исходного решения: если максимум / минимум '
        'прогнозов больше 1,08, берётся медиана; иначе обычное среднее. '
        'В 03 правило применяется к трём моделям, в 04 — к четырём. '
        'В 05 используется обычное среднее, в 06 — exp(mean(log(1+prediction)))−1. '
        'Порог и веса не подбирались. Пространственные модели — только ранее '
        'обученные учебные WX, полные исходные числа деревьев сохранены.', '',
        '| № | Январь, MAPE % | Февраль, MAPE % | Средняя | Лучше ориентира в обоих? |',
        '| --- | ---: | ---: | ---: | --- |',
        f"| Исходный ансамбль | {original['2025-01']:.6f} | {original['2025-02']:.6f} | {original['confirmation_mean']:.6f} | — |",
        f"| Ориентир 90,80 итоговый балл | {benchmark['2025-01']:.6f} | {benchmark['2025-02']:.6f} | {benchmark['confirmation_mean']:.6f} | — |",
    ])
    for entry in files:
        row = entry['validation']
        wins = 'да' if entry['wins_both_months_vs_public_best_reference'] else 'нет'
        text.append(f"| {entry['order']:02d} | {row['2025-01']:.6f} | {row['2025-02']:.6f} | {row['confirmation_mean']:.6f} | {wins} |")
    text.extend(['',
        '01 и 02 улучшают оба месяца относительно исходного ансамбля и ориентира '
        'с 90,80. 03 улучшает среднюю ошибку, но ухудшает январь. 04 улучшает январь, '
        'но ухудшает февраль. 05–07 проигрывают ориентиру в обоих месяцах и '
        'оставлены как необязательные проверки переноса на март.', '',
        'Все месяцы уже использовались, составы проверяются после обратной '
        'связи. Это ретроспективные эксперименты, не новый слепой тест; мартовские '
        'баллы новых вариантов неизвестны. Первоначальные критерии, FINAL_DECISION, '
        'модели и эконометрическая часть не меняются. Нельзя выводить преимущество '
        'ансамбля только из баллов его отдельных участников.', '',
        'В каждом файле 18657 строк, столбцы new_id,rto, уникальные исходные ID, '
        'положительные конечные прогнозы, размер <1 МБ. Проверены SHA источников, '
        'точное восстановление чисел после записи, отсутствие повторов и CRC архива. '
        'В соревнование автоматически не отправлялись.', '',
    ])
    (OUT / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    archive = OUT / 'seven_ensemble_followups.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for entry in files:
            z.write(OUT / _directory_path(entry['file']), entry['file'])
        for name in ['README.md', 'manifest.json', 'validation_metrics.json']:
            z.write(OUT / _directory_path(name), name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in files:
            assert z.read(entry['file']) == (OUT / _directory_path(entry['file'])).read_bytes()
    assert sha(HERE / 'FINAL_DECISION.json') == final_decision_sha256
    print(json.dumps(dict(status='passed', files=len(files), no_duplicates=True,
                          archive=str(archive), archive_bytes=archive.stat().st_size,
                          first_two_both_months_better=all(
                              entry['wins_both_months_vs_public_best_reference']
                              for entry in files[:2])), ensure_ascii=False))


if __name__ == '__main__':
    main()
