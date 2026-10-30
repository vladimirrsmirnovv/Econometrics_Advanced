"""Create matched frozen and required original-three-plus-hybrid ensembles."""
from pathlib import Path
import hashlib,io,json,shutil,sys,zipfile
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS
from selection import blend
from preserved_submissions import read_previous,check_previous
OUT=HERE.parent/'submissions/2025-03-stage2';OLD=HERE.parent/'submissions/2025-03'

def main():
    frozen=json.loads((HERE/'frozen_final_selection.json').read_text());decision=json.loads((HERE/'final_decision.json').read_text())
    previous_integrity=check_previous()
    ids=pd.read_csv(OLD/'baseline/test_catboost_log_level.csv').new_id.to_numpy()
    def read(path):
        source=io.BytesIO(read_previous(path.relative_to(OLD).as_posix())) if path.is_relative_to(OLD) else path
        frame=pd.read_csv(source);assert list(frame)==['new_id','rto'] and len(frame)==18657 and frame.new_id.is_unique
        frame=frame.set_index('new_id').reindex(ids);assert np.isfinite(frame.rto).all() and (frame.rto>0).all();return frame.rto.to_numpy()
    selected={m:read(OUT/'individual'/('test_'+m+'.csv')) for m in MODELS}
    original={m:read(OLD/'baseline'/('test_'+m+'.csv')) for m in MODELS}
    changes=[]
    for m in MODELS:
        ratio=selected[m]/original[m]-1
        changes.append(dict(model=m,median_change_percent=float(np.median(ratio)*100),
            mean_absolute_prediction_change_percent=float(np.mean(abs(ratio))*100),
            p05_change_percent=float(np.quantile(ratio,.05)*100),p95_change_percent=float(np.quantile(ratio,.95)*100),
            aggregate_rto_change_percent=float((selected[m].sum()/original[m].sum()-1)*100),
            actual_march_quality_known=False))
    pd.DataFrame(changes).to_csv(HERE/'march_forecast_changes.csv',index=False)
    references=[]
    for variant in ['baseline','spatial']:
        folder=OUT/('reference_baseline' if variant=='baseline' else 'reference_previous');folder.mkdir(exist_ok=True)
        for m in MODELS:
            source=OLD/variant/('test_'+m+'.csv');target=folder/source.name;target.write_bytes(read_previous(source.relative_to(OLD).as_posix()))
            references.append(dict(file=target.relative_to(OUT).as_posix(),source=source.relative_to(OLD).as_posix(),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    mixed=dict(original);mixed[MODELS[-1]]=read(OLD/'spatial/test_xgboost_log_diff.csv')
    ensemble=OUT/'ensemble';ensemble.mkdir(exist_ok=True);records=[]
    def write(name,pred):
        path=ensemble/name;pd.DataFrame(dict(new_id=ids,rto=pred)).to_csv(path,index=False)
        assert path.stat().st_size<=1000000;records.append(dict(file=path.relative_to(OUT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
        return path
    composition=frozen['ensemble']['composition'];components={}
    for m in MODELS:
        if composition=='new':components[m]=selected[m]
        elif composition=='cautious_new':components[m]=(selected[m]+read(OUT/'components/baseline'/('test_'+m+'.csv')))*.5
        else:
            config='previous' if composition=='previous' or (composition=='mandatory_mixed' and m==MODELS[-1]) else 'baseline'
            components[m]=read(OUT/'components'/config/('test_'+m+'.csv'))
    candidate=write('test_frozen_new_ensemble.csv',blend(components)[frozen['ensemble']['rule']])
    for rule,pred in blend(mixed).items():write('test_mixed_original3_hybrid_xgb_'+rule+'.csv',pred)
    original_path=write('test_original_full_ensemble.csv',blend(original)['final_meta_logmean'])
    previous_path=write('test_previous_full_gated_1_08_all.csv',blend({m:read(OLD/'spatial'/('test_'+m+'.csv')) for m in MODELS})['gated_1_08_all'])
    # References retain the exact bytes of the previously generated submissions.
    for target,source in [(original_path,OLD/'ensemble/baseline_final_meta_logmean.csv'),
                          (previous_path,OLD/'ensemble/spatial_gated_1_08_all.csv')]:
        np.testing.assert_allclose(read(target),read(source),rtol=1e-12,atol=1e-7)
        target.write_bytes(read_previous(source.relative_to(OLD).as_posix()))
        record=next(r for r in records if r['file']==target.relative_to(OUT).as_posix())
        record.update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size)
    recommendation=candidate if decision['status']=='passed' else original_path
    shutil.copy2(recommendation,OUT/'test.csv')
    manifest=dict(recommended_file='test.csv',source= recommendation.relative_to(OUT).as_posix(),decision=decision['status'],
        forecast_month='2025-03',rows=18657,maximum_csv_bytes=1000000,competition_submissions_sent=False,
        frozen_selection_sha256=hashlib.sha256((HERE/'frozen_final_selection.json').read_bytes()).hexdigest(),ensembles=records,references=references,
        previous_reference_recovery=previous_integrity['damaged_loose_files'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
    lines=['# Готовые мартовские прогнозы второго этапа','',
        '**Рекомендован `test.csv`.** Он точно копирует `'+manifest['source']+'`; результат заранее заданной проверки: `'+decision['status']+'`.',
        '', 'Все файлы: март 2025, 18 657 строк, столбцы `new_id,rto`, размер до 1 МБ. Посылки в соревнование не отправлялись.',
        '', 'Прежний отдельный spatial XGBoost CSV оказался пустым. Его контрольная копия восстановлена в этой новой папке из исходного ZIP; SHA-256 совпадает с прежним манифестом. Старая папка не изменялась.',
        '', '| Модель | Исходная, полное обучение | Прежняя модификация, полное обучение | Новая выбранная, rolling refit | Новая конфигурация / деревья |',
        '| --- | --- | --- | --- | --- |']
    for m in MODELS:
        name='test_'+m+'.csv';info=json.loads((OUT/'individual'/name).with_suffix('.json').read_text())
        lines.append('| '+m+' | `reference_baseline/'+name+'` | `reference_previous/'+name+'` | `individual/'+name+'` | '+info['configuration']+' / '+str(info['trees'])+' |')
    lines+=['',
        '`ensemble/test_frozen_new_ensemble.csv` — ансамбль из выбранных новых версий, правило `'+frozen['ensemble']['rule']+'`. Он остаётся экспериментальным, если контрольная проверка не пройдена. Правило gated_1_08_no_weak использует CatBoost, LightGBM уровней и XGBoost; LightGBM приростов исключается.',
        '', '`ensemble/test_mixed_original3_hybrid_xgb_final_meta_logmean.csv` — обязательное сравнение: три исходные полные модели плюс прежний полный hybrid8 XGBoost. Другие файлы с тем же префиксом содержат пять остальных фиксированных правил.',
        '', '`ensemble/test_original_full_ensemble.csv` и `ensemble/test_previous_full_gated_1_08_all.csv` — точные копии прежних ансамблей для контроля.',
        '', 'В `march_submissions_stage2.zip` включены все перечисленные CSV и манифест. Native-модели и параметры новых вариантов находятся в `models` и `individual/*.json`, а не в архиве посылок.',
        '', 'Новые числа деревьев выбираются на феврале 2025, после чего модели обучаются на январе 2023 — феврале 2025. Разница с исходными полными CSV включает как признаки, так и протокол выбора длины обучения; их отдельные эффекты показаны в отчёте `Forecasting/research_stage2/REPORT.md`.']
    if (HERE/'POST_SUBMISSION_REVIEW.md').exists():
        lines[1:1]=['', '> **Актуальный статус:** это сохранённые файлы исторического эксперимента, а не установленное мартовское улучшение. Объяснение и уточнённая граница методов: `../../research_stage2/POST_SUBMISSION_REVIEW.md`, `../../course_only/README.md`.']
    (OUT/'SUBMISSIONS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with zipfile.ZipFile(OUT/'march_submissions_stage2.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*.csv'):
            if p.name.startswith('february_inner_'):continue
            read(p);assert p.stat().st_size<=1000000;z.write(p,p.relative_to(OUT).as_posix())
        z.write(OUT/'manifest.json','manifest.json')

if __name__=='__main__':main()
