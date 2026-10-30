"""Final CSV schema/ID validation, original ensembles and selected blend."""
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,ROOT,NOTEBOOK_SHA256
from run_validation import original_blends
OUT=HERE.parent/'submissions/2025-03'

def main():
    cache=HERE.parent/'cache/final_2025-03'
    test=pd.read_parquet(cache/'test_features.parquet',columns=['new_id','year_month_index','pto'])
    ids=test.new_id.to_numpy();assert len(ids)==18657 and test.new_id.is_unique and test.pto.isna().all()
    choices=json.loads((HERE.parent/'research/best_model_configs.json').read_text())
    recipe=choices['recommended_ensemble_rule'];records=[];vectors={};main_files=[]
    def check(path,metadata,main=False):
        frame=pd.read_csv(path,float_precision='round_trip')
        assert list(frame)==['new_id','rto'] and len(frame)==18657 and frame.new_id.is_unique
        np.testing.assert_array_equal(frame.new_id.to_numpy(),ids)
        assert np.isfinite(frame.rto).all() and (frame.rto>=0).all()
        row=dict(file=path.relative_to(OUT).as_posix(),rows=len(frame),columns=list(frame),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),min_rto=float(frame.rto.min()),
            median_rto=float(frame.rto.median()),mean_rto=float(frame.rto.mean()),max_rto=float(frame.rto.max()),
            total_rto=float(frame.rto.sum()),**metadata)
        records.append(row)
        if main:main_files.append(path)
        return frame
    for variant in ['baseline','spatial']:
        vectors[variant]={}
        for model in MODELS:
            path=OUT/variant.replace("_", "-")/('test_'+model+'.csv');info=json.loads(path.with_suffix('.json').read_text())
            assert info['forecast_month']=='2025-03' and info['trees']==choices['final_tree_counts'][model]
            assert not info['early_stopping'] and info['notebook_sha256']==NOTEBOOK_SHA256
            frame=check(path,dict(model=model,variant=variant,configuration=info['configuration'],
                trees=info['trees'],backend=info['backend']),main=True)
            vectors[variant][model]=frame.rto.to_numpy()
    ensembles=OUT/'ensemble';ensembles.mkdir(exist_ok=True)
    additional=ensembles/'reference-rules';additional.mkdir(exist_ok=True)
    for variant in ['baseline','spatial']:
        rules=original_blends(vectors[variant])
        for rule,pred in rules.items():
            is_main=rule=='final_meta_logmean' or (variant=='spatial' and rule==recipe)
            path=(ensembles if is_main else additional)/(variant+'_'+rule+'.csv')
            pd.DataFrame({'new_id':ids,'rto':pred}).to_csv(path,index=False)
            check(path,dict(model='ensemble',variant=variant,rule=rule,components='original' if variant=='baseline' else choices['components']),main=is_main)
    diagnostic=OUT/'diagnostics/catboost_history_before_global_correction.csv'
    if diagnostic.exists():check(diagnostic,dict(model='catboost_log_level',variant='spatial_history_only',diagnostic=True))
    changes=[]
    for model in MODELS:
        a=vectors['baseline'][model];b=vectors['spatial'][model]
        changes.append(dict(model=model,mean_change_pct=float(np.mean((b/a-1)*100)),
            absolute_change_median_pct=float(np.median(np.abs(b/a-1)*100)),
            absolute_change_p95_pct=float(np.quantile(np.abs(b/a-1)*100,.95))))
    pd.DataFrame(changes).to_csv(OUT/'forecast_changes.csv',index=False)
    manifest=dict(status='passed',forecast_month='2025-03',shops=18657,
        training_months=['2023-01','2025-02'],notebook_sha256=NOTEBOOK_SHA256,
        full_fit_counts=choices['final_tree_counts'],no_early_stopping_in_full_fits=True,
        original_model_features_preserved=True,selected_configurations=choices['components'],
        recommended_rule_selected_on_validation=recipe,actual_march_targets_available=False,
        primary_csv_files=[p.relative_to(OUT).as_posix() for p in main_files],csv_files_checked=len(records),files=records,
        limitation='No March ground truth; validation MAPE does not certify competition score')
    (OUT/'submission_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=OUT/'march_submissions.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in main_files:z.write(path,path.relative_to(OUT).as_posix())
        z.write(OUT/'submission_manifest.json','submission_manifest.json')
    rows=['| Модель | Исходный CSV | CSV с выбранным улучшением |','|---|---|---|']
    def link(path,label):return f'[{label}](<{path.resolve().as_posix()}>)'
    names=['CatBoost уровней','LightGBM темпов','LightGBM уровней','XGBoost темпов']
    for model,name in zip(MODELS,names):
        filename='test_'+model+'.csv'
        rows.append(f'| {name} | '+link(OUT/'baseline'/filename,'Исходная')+' | '+link(OUT/'spatial'/filename,'Улучшенная')+' |')
    note=f'''# Готовые submission на март 2025

Все основные CSV имеют ровно 18 657 строк, столбцы `new-id,rto`, исходный порядок магазинов, конечные неотрицательные прогнозы. Обучение использует январь 2023 — февраль 2025. Мартовские реальные значения отсутствуют.

{chr(10).join(rows)}

## Ансамбли

- {link(ensembles/'baseline-final-meta-logmean.csv','Исходный final-meta-logmean')}: четыре исходные модели и исходная формула ноутбука.
- {link(ensembles/'spatial-final-meta-logmean.csv','После изменений с той же формулой')}: четыре выбранные версии; формула смешивания сохранена.
- {link(ensembles/('spatial-'+recipe+'.csv'),'Лучшее правило по основной валидации')}: те же улучшенные компоненты, правило `{recipe}`.

{link(archive,'Архив основных submission')} содержит {len(main_files)} CSV и манифест. В `ensemble/reference-rules` сохранены остальные исходные правила смеси для сравнения.

## Что изменено

CatBoost: четыре исторических признака соседей и фиксированная коррекция общего смещения по февральскому OOF той же пространственной версии. Прогноз до коррекции находится в `diagnostics`. LightGBM темпов: восемь географических собственных векторов. LightGBM уровней: веса обучения по размеру рынка. XGBoost: четыре признака истории гибридных соседей. Цели и исходные гиперпараметры сохранены.

Полные модели обучены без ранней остановки: 6 500 / 7 000 / 8 000 / 3 750 деревьев в порядке таблицы. XGBoost — CUDA 3.1.3. Исходный мартовский XGBoost сопоставлен с ранее воспроизведённым прогнозом.

Метрики марта по этим данным вычислить нельзя. Отбор версий и правила смеси выполнен ранее на валидации, а не по неизвестному мартовскому результату. Проверки, хэши, параметры, версии и пути native-моделей сохранены в JSON рядом с CSV и в `submission_manifest.json`.
'''
    (OUT/'SUBMISSIONS.md').write_text(note,encoding='utf-8')
    print('SUBMISSIONS_VERIFIED',len(main_files),'primary CSV;',len(records),'total CSV; archive',archive)

if __name__=='__main__':main()
