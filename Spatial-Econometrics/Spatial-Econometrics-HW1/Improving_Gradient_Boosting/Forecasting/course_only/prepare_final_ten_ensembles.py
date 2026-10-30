"""Ten final exploratory mixes using original formulas and existing course WX fits.

No new weights, thresholds, model fits, spatial estimators or contest submissions.
This batch follows post-competition ranking-test feedback; validation is retrospective.
"""
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
EARLIER = HERE.parent / 'submissions/2025-03-course-ensemble-alternatives'
FOLLOWUPS = HERE.parent / 'submissions/2025-03-course-ensemble-followups'
OUT = HERE.parent / 'submissions/2025-03-course-final-ten'
WX_MODELS = [model for model in MODELS if model != 'catboost_log_level']
CANDIDATES = [
    ('01_winner_gated_all', 'xgb_ld', 'gated_1_08_all'),
    ('02_winner_log_average', 'xgb_ld', 'log_average'),
    ('03_winner_equal', 'xgb_ld', 'equal'),
    ('04_three_wx_original_mix', 'xgb_ld_ll', 'final_meta_logmean'),
    ('05_three_wx_gated_all', 'xgb_ld_ll', 'gated_1_08_all'),
    ('06_three_wx_log_average', 'xgb_ld_ll', 'log_average'),
    ('07_three_wx_equal', 'xgb_ld_ll', 'equal'),
    ('08_xgb_ll_wx_gated_no_ld', 'xgb_ll', 'gated_1_08_no_weak'),
    ('09_xgb_ll_wx_gated_all', 'xgb_ll', 'gated_1_08_all'),
    ('10_xgb_ll_wx_log_average', 'xgb_ll', 'log_average'),
]
COMPOSITION_LABELS = {
    'xgb_ld': 'WX: XGBoost и LightGBM приростов',
    'xgb_ll': 'WX: XGBoost и LightGBM уровней',
    'xgb_ld_ll': 'WX: XGBoost и оба LightGBM',
}
RULE_LABELS = {
    'gated_1_08_all': 'Порог 1,08, четыре модели',
    'gated_1_08_no_weak': 'Порог 1,08, три модели без LightGBM приростов',
    'log_average': 'Логарифмическое среднее',
    'equal': 'Обычное среднее',
    'final_meta_logmean': 'Исходная meta-смесь',
}


def compositions(base, wx):
    xgb = dict(base, xgboost_log_diff=wx['xgboost_log_diff'])
    return dict(baseline=base,
                xgb_ld=dict(xgb, lightgbm_log_diff=wx['lightgbm_log_diff']),
                xgb_ll=dict(xgb, lightgbm_log_level=wx['lightgbm_log_level']),
                xgb_ld_ll=dict(xgb, lightgbm_log_diff=wx['lightgbm_log_diff'],
                               lightgbm_log_level=wx['lightgbm_log_level']))


def read_checked_march_sources():
    expected = {entry['file']: entry['sha256'] for entry in json.loads(
        (SOURCE / 'submission_manifest.json').read_text(encoding='utf-8'))['files']}
    base, wx, ids, sources = {}, {}, None, []
    for model in MODELS:
        variants = [('baseline', base)]
        if model in WX_MODELS:
            variants.append(('experimental', wx))
        for variant, destination in variants:
            relative = f'{variant}/test_{model}.csv'
            path = SOURCE / relative
            assert sha(path) == expected[relative]
            frame = pd.read_csv(path, float_precision='round_trip').sort_values('new_id')
            if ids is None:
                ids = frame.new_id.to_numpy()
            np.testing.assert_array_equal(ids, frame.new_id)
            destination[model] = frame.rto.to_numpy()
            sources.append(dict(file=relative, sha256=expected[relative]))
    return base, wx, ids, sources


def collect_validation(choices):
    records, sources = [], []
    for fold in DEVELOPMENT + CONFIRM:
        base, wx, ids, truth = {}, {}, None, None
        for model in MODELS:
            variants = [('baseline', base)]
            if model in WX_MODELS:
                variants.append((choices[model], wx))
            for configuration, destination in variants:
                frame, info = load(fold, model, configuration)
                if ids is None:
                    ids, truth = frame.new_id.to_numpy(), frame.pto.to_numpy()
                np.testing.assert_array_equal(ids, frame.new_id)
                np.testing.assert_array_equal(truth, frame.pto)
                destination[model] = frame.prediction.to_numpy()
                sources.append(dict(fold=fold, model=model, configuration=configuration,
                                    trees=info['trees'], prediction_sha256=info['prediction_sha256']))
        for composition, components in compositions(base, wx).items():
            for rule, prediction in original_blends(components).items():
                records.append(dict(fold=fold, composition=composition, rule=rule,
                                    mape=mape(truth, prediction)))
    previous = json.loads((FOLLOWUPS / 'validation_metrics.json').read_text(encoding='utf-8'))
    lookup = {(r['fold'], r['composition'], r['rule']): r['mape'] for r in previous['metrics']}
    for record in records:
        if record['composition'] == 'xgb_ld_ll':
            continue
        assert abs(record['mape'] - lookup[(record['fold'], record['composition'], record['rule'])]) < 1e-10
    return records, sources


def known_submissions():
    known = []
    for root, manifest_name in [(SOURCE, 'submission_manifest.json'),
                                (EARLIER, 'manifest.json'), (FOLLOWUPS, 'manifest.json')]:
        manifest = json.loads((root / manifest_name).read_text(encoding='utf-8'))
        for entry in manifest['files']:
            path = root / entry['file']
            assert sha(path) == entry['sha256']
            frame = pd.read_csv(path, float_precision='round_trip').sort_values('new_id')
            known.append(dict(file=str(path), sha256=entry['sha256'],
                              new_id=frame.new_id.to_numpy(), rto=frame.rto.to_numpy()))
    return known


def main():
    choices = {model: json.loads((HERE / 'selection' / (model + '.json')).read_text(
        encoding='utf-8'))['selected_wx'] for model in WX_MODELS}
    public = json.loads((FOLLOWUPS / 'public_results.json').read_text(encoding='utf-8'))
    assert public['status'] == 'user_reported_order_confirmed'
    best = max(public['submissions'], key=lambda row: row['public_score'])
    assert best['order'] == 2 and best['public_score'] == 90.82
    best_path = FOLLOWUPS / best['file']
    assert sha(best_path) == best['sha256']
    previous_manifest = json.loads((FOLLOWUPS / 'manifest.json').read_text(encoding='utf-8'))
    rule_sha = sha(HERE.parent / 'run_validation.py')
    assert rule_sha == previous_manifest['source_rule_sha256']
    decision_sha = sha(HERE / 'FINAL_DECISION.json')
    records, validation_sources = collect_validation(choices)
    known = known_submissions()
    base, wx, ids, march_sources = read_checked_march_sources()
    march = {composition: original_blends(components)
             for composition, components in compositions(base, wx).items()}
    reference = pd.read_csv(best_path, float_precision='round_trip').sort_values('new_id')
    np.testing.assert_array_equal(ids, reference.new_id)
    np.testing.assert_array_equal(march['xgb_ld']['final_meta_logmean'], reference.rto)

    def summary(composition, rule):
        part = [r for r in records if r['composition'] == composition and r['rule'] == rule]
        assert len(part) == 5
        result = {r['fold']: r['mape'] for r in part}
        result['development_mean'] = float(np.mean([result[f] for f in DEVELOPMENT]))
        result['confirmation_mean'] = float(np.mean([result[f] for f in CONFIRM]))
        return result

    benchmark = summary('xgb_ld', 'final_meta_logmean')
    original = summary('baseline', 'final_meta_logmean')
    # Excluding LGBdiff would produce an already tested forecast (90.81).
    redundant = []
    for composition in ['xgb_ld', 'xgb_ld_ll']:
        prediction = march[composition]['gated_1_08_no_weak']
        matches = [existing['file'] for existing in known
                   if np.allclose(prediction, existing['rto'], rtol=1e-12, atol=1e-6)]
        if matches:
            redundant.append(dict(composition=composition, rule='gated_1_08_no_weak',
                                  matches=matches, excluded_from_batch=True))
    OUT.mkdir(parents=True, exist_ok=True)
    files, new_predictions = [], []
    for order, (folder, composition, rule) in enumerate(CANDIDATES, 1):
        prediction = march[composition][rule]
        for existing in known:
            np.testing.assert_array_equal(ids, existing['new_id'])
            assert not np.allclose(prediction, existing['rto'], rtol=1e-12, atol=1e-6), (
                folder, 'duplicate', existing['file'])
        for prior in new_predictions:
            assert not np.allclose(prediction, prior, rtol=1e-12, atol=1e-6), folder
        new_predictions.append(prediction)
        path = OUT / folder / 'test.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(new_id=ids, rto=prediction)).to_csv(path, index=False,
                                                           lineterminator='\n')
        checked = validate(path.read_bytes(), sorted(map(int, ids)))
        readback = pd.read_csv(path, float_precision='round_trip')
        np.testing.assert_array_equal(ids, readback.new_id)
        np.testing.assert_array_equal(prediction, readback.rto)
        row = summary(composition, rule)
        files.append(dict(order=order, file=path.relative_to(OUT).as_posix(),
                          composition=composition, composition_label=COMPOSITION_LABELS[composition],
                          rule=rule, rule_label=RULE_LABELS[rule], validation=row,
                          wins_both_months_vs_current_public_reference=all(
                              row[f] < benchmark[f] for f in CONFIRM),
                          excluded_models=['lightgbm_log_diff'] if rule == 'gated_1_08_no_weak' else [],
                          **checked))
    assert len(files) == 10
    manifest = dict(status='passed', exploratory_after_public_feedback=True,
                    all_validation_months_previously_seen=True, no_models_retrained=True,
                    no_new_spatial_methods=True, spatial_method='Existing course WX only',
                    no_weights_fitted=True, no_new_thresholds=True, original_gate_threshold=1.08,
                    only_original_notebook_blend_rules=True,
                    blend_source_file='Forecasting/run_validation.py:48', blend_source_sha256=rule_sha,
                    original_decision_unchanged=True, original_decision_sha256=decision_sha,
                    current_public_reference=dict(file=str(best_path), sha256=best['sha256'],
                                                  submission_id=best['submission_id'], public_score=90.82,
                                                  validation=benchmark),
                    source_public_results_sha256=sha(FOLLOWUPS / 'public_results.json'),
                    original_ensemble_validation=original, validation_sources=validation_sources,
                    march_sources=march_sources, files=files, excluded_duplicate_candidates=redundant,
                    duplicate_check=dict(known_files=len(known), absolute_tolerance=1e-6,
                                         relative_tolerance=1e-12, no_duplicates=True),
                    automatic_submissions=False, public_scores_available=False)
    annotate_evaluation(manifest)
    write_json(OUT / 'manifest.json', manifest)
    write_json(OUT / 'validation_metrics.json', dict(status='retrospective_comparison', metrics=records))
    text = [
        '# Последние десять мартовских проверок ансамбля', '',
        'Соревнование завершено. Указанные баллы относятся к основному финальному '
        'мартовскому сету для рейтинга; точный охват магазинов не подтверждён. '
        'Итоговый конкурсный балл — основной критерий выбора, локальные месяцы '
        'служат диагностикой. Новые варианты выбираются после обратной связи этого теста.', '',
        'Текущий итоговый максимум — 90,82: исходные CatBoost и LightGBM уровней, '
        'WX у XGBoost и LightGBM приростов, исходная meta-формула. Последние семь '
        'проверок по порядку дали 90,76 / 90,82 / 90,81 / 90,79 / 90,78 / 90,78 / 90,81. '
        'Локально сильный CatWX+XGBWX получил только 90,76. '
        'Во всех десяти новых вариантах CatBoost остаётся исходным.', '',
        'Распакуйте архив и отправляйте только `test.csv` из нужной папки. '
        'Папки 01–10 задают порядок отправки. Общий ZIP предназначен для доставки, '
        'а не для отправки в соревнование. Во всех составах остальные модели '
        'исходные, кроме явно перечисленных WX; в 08 LightGBM приростов исключён.', '',
        '| № | Пространственные версии | Формула |', '| --- | --- | --- |',
    ]
    text.extend(f"| {entry['order']:02d} | {entry['composition_label']} | {entry['rule_label']} |"
                for entry in files)
    text.extend(['',
        '01–03 меняют только правило усреднения состава с 90,82. '
        '04 добавляет WX у LightGBM уровней при прежней формуле — соединяет '
        'пространственные версии из составов с 90,82 и 90,81. '
        '05–07 проверяют остальные исходные способы смешивания этого тройного '
        'WX-состава. 08 проверяет три модели без LightGBM приростов, с WX у '
        'XGBoost и LightGBM уровней. 09–10 меняют правило для состава с 90,81 '
        '(XGBoost WX + LightGBM уровней WX).', '',
        'Все правила взяты из исходного original_blends. Порог 1,08 означает: '
        'если максимальный прогноз / минимальный больше 1,08, взять медиану, '
        'иначе обычное среднее. В 08 это применяется к трём прогнозам, '
        'в остальных gated-вариантах к четырём. Логарифмическое среднее равно '
        'exp(mean(log(1+prediction)))−1. Исходная meta-смесь — логарифмическое '
        'среднее четырёх исходных кандидатов. Веса и порог не подбираются; '
        'новые пространственные методы и повторное обучение не применяются.', '',
        '| № | Январь, MAPE % | Февраль, MAPE % | Средняя |',
        '| --- | ---: | ---: | ---: |',
        f"| Исходный ансамбль | {original['2025-01']:.6f} | {original['2025-02']:.6f} | {original['confirmation_mean']:.6f} |",
        f"| Ориентир 90,82 итоговый балл | {benchmark['2025-01']:.6f} | {benchmark['2025-02']:.6f} | {benchmark['confirmation_mean']:.6f} |",
    ])
    for entry in files:
        row = entry['validation']
        text.append(f"| {entry['order']:02d} | {row['2025-01']:.6f} | {row['2025-02']:.6f} | {row['confirmation_mean']:.6f} |")
    text.extend(['',
        'Ни один из десяти новых вариантов не улучшил ориентир с 90,82 в обоих '
        'локальных контрольных месяцах. Это заключительные диагностические '
        'проверки переноса на март после обратной связи итоговой проверяющей системы, '
        'а не подтверждённые улучшения и не новый слепой тест. '
        'Итоговые баллы новых вариантов неизвестны; истинные РТО '
        'мартовского рейтингового теста недоступны. Первоначальное FINAL_DECISION и исходное '
        'эконометрическое исследование не изменяются.', '',
        'Повторы исключены по значениям прогнозов с учётом точности записи, '
        'а не только по имени или переносу строк. В частности, gated без LGBdiff '
        'для состава с 90,82 уже совпадает с ранее отправленным вариантом '
        'с 90,81. Gated без LGBdiff для тройного WX-состава совпадает с новым 08 '
        'и не занимает второй слот.', '',
        'В каждом CSV 18657 строк new_id,rto, уникальные исходные ID, конечные '
        'положительные прогнозы и размер <1 МБ. Проверены SHA источников, '
        'точный float round-trip, отсутствие повторов среди 26 прежних файлов '
        'и новых десяти, CRC архива. Автоматических посылок нет.', '',
    ])
    (OUT / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    archive = OUT / 'final_ten_ensembles.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for entry in files:
            z.write(OUT / entry['file'], entry['file'])
        for name in ['README.md', 'manifest.json', 'validation_metrics.json']:
            z.write(OUT / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in files:
            assert z.read(entry['file']) == (OUT / entry['file']).read_bytes()
    assert sha(HERE / 'FINAL_DECISION.json') == decision_sha
    print(json.dumps(dict(status='passed',files=len(files),no_duplicates=True,
                          archive=str(archive),archive_bytes=archive.stat().st_size,
                          local_both_months_winners=sum(
                              entry['wins_both_months_vs_current_public_reference']
                              for entry in files)), ensure_ascii=False))


if __name__ == '__main__':
    main()
