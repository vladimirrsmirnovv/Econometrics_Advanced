"""Compare original notebook blend rules and package three exploratory March mixes.

Uses checked existing predictions only. No training, weight fitting or submissions.
The XGBoost-only composition is motivated by final March contest results, so these are
exploratory trials after leaderboard feedback, not a new blind validation.
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
OUT = HERE.parent / 'submissions/2025-03-course-ensemble-alternatives'
CANDIDATES = [
    ('01_only_xgb_wx_original_mix', 'only_xgb_wx', 'final_meta_logmean',
     'WX только у XGBoost; исходная формула ансамбля'),
    ('02_all_wx_equal', 'all_wx', 'equal', 'Все четыре WX-модели; обычное среднее'),
    ('03_all_wx_log_average', 'all_wx', 'log_average',
     'Все четыре WX-модели; логарифмическое среднее'),
]


def compositions(base, wx):
    return {
        'baseline': base,
        'all_wx': wx,
        'only_xgb_wx': dict(base, xgboost_log_diff=wx['xgboost_log_diff']),
        'only_lgbdiff_wx': dict(base, lightgbm_log_diff=wx['lightgbm_log_diff']),
    }


def markdown_table(rows):
    return '\n'.join([
        '| Вариант | Январь, MAPE % | Февраль, MAPE % | Средняя |',
        '| --- | ---: | ---: | ---: |',
        *[f"| {r['label']} | {r['2025-01']:.6f} | {r['2025-02']:.6f} | {r['mean']:.6f} |"
          for r in rows],
    ])


def main():
    choices = {model: json.loads((HERE / 'selection' / (model + '.json')).read_text(
        encoding='utf-8'))['selected_wx'] for model in MODELS}
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
                                prediction_sha256=info['prediction_sha256'])
                           for info in [bi, wi])
        for composition, predictions in compositions(base, wx).items():
            for rule, prediction in original_blends(predictions).items():
                records.append(dict(fold=fold, composition=composition, rule=rule,
                                    mape=mape(truth, prediction)))

    # The two previously assembled compositions must reproduce their saved MAPE.
    saved = pd.read_csv(HERE / 'ensemble_comparison.csv')
    for record in records:
        if record['composition'] not in ['baseline', 'all_wx', 'only_lgbdiff_wx']:
            continue
        name = {'baseline': 'baseline', 'all_wx': 'experimental',
                'only_lgbdiff_wx': 'confirmed'}[record['composition']]
        reference = saved[(saved.fold == record['fold']) & (saved.composition == name)
                          & (saved.rule == record['rule'])]
        assert len(reference) == 1
        assert abs(record['mape'] - reference.mape.iloc[0]) < 1e-10

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'validation_metrics.json', dict(
        status='retrospective_comparison', all_months_previously_seen=True,
        metrics=records, sources=sources))
    base, wx, ids, march_sources = {}, {}, None, []
    original_manifest = json.loads((SOURCE / 'submission_manifest.json').read_text(
        encoding='utf-8'))
    expected = {row['file']: row['sha256'] for row in original_manifest['files']}
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
    predictions = {composition: original_blends(values)
                   for composition, values in compositions(base, wx).items()}
    files = []
    summaries = []

    def summary(composition, rule, label):
        selected = [r for r in records if r['composition'] == composition
                    and r['rule'] == rule and r['fold'] in CONFIRM]
        assert len(selected) == 2
        return dict(label=label, **{r['fold']: r['mape'] for r in selected},
                    mean=float(np.mean([r['mape'] for r in selected])))

    summaries.append(summary('baseline', 'final_meta_logmean', 'Исходный ансамбль (90,78 итоговый балл)'))
    summaries.append(summary('all_wx', 'final_meta_logmean', 'Все WX, исходная смесь (90,79 итоговый балл)'))
    summaries.append(summary('only_lgbdiff_wx', 'final_meta_logmean', 'Локально подтверждённый состав (90,78 итоговый балл)'))
    for folder, composition, rule, label in CANDIDATES:
        path = OUT / folder.replace("_", "-") / 'test.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction = predictions[composition][rule]
        pd.DataFrame(dict(new_id=ids, rto=prediction)).to_csv(path, index=False,
                                                           lineterminator='\n')
        checked = validate(path.read_bytes(), sorted(map(int, ids)))
        readback = pd.read_csv(path, float_precision='round_trip')
        np.testing.assert_array_equal(ids, readback.new_id)
        np.testing.assert_array_equal(prediction, readback.rto)
        row = summary(composition, rule, label)
        summaries.append(row)
        files.append(dict(file=path.relative_to(OUT).as_posix(), label=label,
                          composition=composition, original_notebook_rule=rule,
                          validation=row, **checked))
    manifest = dict(status='passed', exploratory_after_leaderboard_feedback=True,
                    all_validation_months_previously_seen=True, no_models_retrained=True,
                    no_new_weights_or_thresholds=True, server_acceptance_verified=False,
                    original_blends_source_sha256=sha(HERE.parent / 'run_validation.py'),
                    baseline_frozen_decision_changed=False, march_sources=march_sources,
                    files=files)
    annotate_evaluation(manifest)
    write_json(OUT / 'manifest.json', manifest)
    text = [
        '# Три дополнительных мартовских ансамбля', '',
        'Уточнение контекста: соревнование завершено, указанные баллы относятся '
        'к основному финальному мартовскому сету для рейтинга. Точный охват '
        'магазинов не подтверждён; локальная валидация служит диагностикой.', '',
        'Это исследовательские варианты после получения итоговых результатов контеста. '
        'Модели не переобучаются; используются исходные формулы из ноутбука. '
        'Новые веса и пороги не подбираются. Первоначальный FINAL_DECISION не меняется.', '',
        'Распакуйте архив с сохранением папок. Отправляйте CSV `test.csv` из выбранной '
        'папки, не общий ZIP. Порядок проб соответствует номерам папок:', '',
    ]
    text.extend(f"{index}. `{entry['file']}` — {entry['label']}."
                for index, entry in enumerate(files, 1))
    text.extend(['',
        'Вариант 1 меняет состав: исходные CatBoost и два LightGBM плюс XGBoost с WX kNN12. '
        'Он мотивирован улучшением итогового балла одиночного XGBoost на 0,13 балла; '
        'локально улучшил январь, но ухудшил февраль.', '',
        'Варианты 2 и 3 сохраняют все четыре WX-модели, а меняют только усреднение '
        'относительно принятой смеси с баллом 90,79. Обычное среднее — сумма четырёх '
        'прогнозов / 4; логарифмическое — exp(mean(log(1+prediction)))−1, '
        'как original_blends в исходном адаптере.', '',
        markdown_table(summaries), '',
        'Ни один из трёх новых вариантов не улучшил исходный ансамбль в обоих '
        'контрольных месяцах. Поэтому они предназначены для ограниченной мартовской '
        'диагностики и не объявляются подтверждённым улучшением. '
        'Итоговый балл новых вариантов при подготовке неизвестен. Месяцы уже изучались; '
        'эта проверка не является слепой.', '',
        'Все CSV: 18657 строк, new_id,rto, ID совпадают с исходными, '
        'положительные конечные прогнозы, размер меньше 1 МБ. '
        'Точное восстановление чисел после записи и CRC архива проверены. '
        'Автоматических посылок в соревнование нет.', '',
    ])
    (OUT / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    archive = OUT / 'three_ensemble_trials.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for entry in files:
            z.write(OUT / _directory_path(entry['file']), entry['file'])
        for name in ['README.md', 'manifest.json', 'validation_metrics.json']:
            z.write(OUT / _directory_path(name), name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in files:
            assert z.read(entry['file']) == (OUT / _directory_path(entry['file'])).read_bytes()
    print(json.dumps(dict(status='passed',new_csvs=len(files),archive=str(archive),
                          comparison=summaries), ensure_ascii=False))


if __name__ == '__main__':
    main()
