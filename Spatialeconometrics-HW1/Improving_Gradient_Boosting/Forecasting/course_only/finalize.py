"""Compare course WX at original full counts and produce clearly labelled March CSVs."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import io,json,sys,zipfile
import numpy as np
import pandas as pd
from scipy import sparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'research_stage2'))
from manage import MODELS,COUNTS,DEVELOPMENT,CONFIRM,load,mape,sha,write_json
from run_validation import original_blends
from preserved_submissions import read_previous
NAMES=dict(zip(MODELS,['CatBoost, уровни','LightGBM, приросты','LightGBM, уровни','XGBoost, приросты']))
def table(frame):
    f=frame.copy()
    for col in f:
        if pd.api.types.is_float_dtype(f[col]):f[col]=f[col].map(lambda v:f'{v:.6f}')
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in f.itertuples(index=False,name=None))
def native_count(path,model):
    if model=='catboost_log_level':
        from catboost import CatBoostRegressor
        est=CatBoostRegressor();est.load_model(str(path));return est.tree_count_
    if model.startswith('lightgbm'):
        import lightgbm as lgb
        return lgb.Booster(model_str=path.read_text(encoding='utf-8')).current_iteration()
    import xgboost as xgb
    est=xgb.Booster();est.load_model(bytearray(path.read_bytes()));return est.num_boosted_rounds()
def submit(path,ids,pred):
    path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(dict(new_id=ids,rto=pred)).to_csv(path,index=False)
def moran(frame,w,stores):
    work=frame.assign(node=frame.new_id.map(stores.node),error=(frame.pto-frame.prediction)/frame.pto)
    x=work.groupby('node').error.mean().reindex(range(w.shape[0])).to_numpy();assert np.isfinite(x).all()
    z=x-x.mean();factor=len(z)/w.sum();denom=z@z
    observed=float(factor*(z@(w@z))/denom);rng=np.random.default_rng(20261010);exceed=0
    for _ in range(999):
        draw=rng.permutation(z);exceed+=float(factor*(draw@(w@draw))/denom)>=observed
    return dict(moran_I=observed,permutation_p_greater=(exceed+1)/1000,null_expected_I=-1/(len(z)-1),nodes=len(z),permutations=999)
def main():
    selected={};individual=[];metrics=[];native=[];parity=[]
    for model in MODELS:
        tag=dict(zip(MODELS,['cat','ld','ll','xgb']))[model]
        for phase in ['development','confirmation']:
            check=json.loads((HERE/'imports'/(phase+'_'+tag+'.json')).read_text(encoding='utf-8'));assert check['status']=='passed'
            if phase=='development':
                location=check.get('execution_location',check['checks'].get('execution_location','Kaggle'))
                parity.append(dict(model=model,execution_location=location,**check['checks']['baseline_parity']))
        choice=json.loads((HERE/'selection'/(model+'.json')).read_text(encoding='utf-8'))
        wx=choice['selected_wx'];wins=[];row=dict(model=model,WX=wx,trees=COUNTS[model],development_gain=choice['development_gain'])
        for fold in DEVELOPMENT+CONFIRM:
            b,bi=load(fold,model,'baseline');n,ni=load(fold,model,wx)
            np.testing.assert_array_equal(b.new_id,n.new_id);np.testing.assert_array_equal(b.pto,n.pto)
            metrics.append(dict(model=model,fold=fold,configuration='baseline',mape=bi['mape'],trees=COUNTS[model]))
            metrics.append(dict(model=model,fold=fold,configuration=wx,mape=ni['mape'],trees=COUNTS[model]))
            if fold in CONFIRM:
                row[fold+'_before']=bi['mape'];row[fold+'_WX']=ni['mape'];row[fold+'_gain']=bi['mape']-ni['mape'];wins.append(ni['mape']<bi['mape'])
        row['eligible']=bool(choice['development_eligible'] and all(wins));selected[model]=wx if row['eligible'] else 'baseline'
        row['confirmed_version']=selected[model];individual.append(row)
    indiv=pd.DataFrame(individual);indiv.to_csv(HERE/'confirmation_model_comparison.csv',index=False)
    pd.DataFrame(metrics).to_csv(HERE/'all_selected_monthly_metrics.csv',index=False)
    development_rows=[]
    for model in MODELS:
        for config in ['baseline','wx_knn6','wx_knn12']:
            row=dict(model=model,configuration=config)
            for fold in DEVELOPMENT:row[fold]=load(fold,model,config)[1]['mape']
            row['mean_mape']=float(np.mean([row[f] for f in DEVELOPMENT]));development_rows.append(row)
    dev=pd.DataFrame(development_rows);dev.to_csv(HERE/'development_model_comparison.csv',index=False)
    ensemble=[];regional=[];moran_rows=[];stores=pd.read_parquet(HERE/'cache/stores.parquet').set_index('new_id')
    for fold in DEVELOPMENT+CONFIRM:
        sets={s:{} for s in ['baseline','experimental','confirmed']};truth=None;fold_ids=None
        for model in MODELS:
            wx=json.loads((HERE/'selection'/(model+'.json')).read_text(encoding='utf-8'))['selected_wx']
            b,_=load(fold,model,'baseline');n,_=load(fold,model,wx)
            if truth is None:truth=b.pto.to_numpy();fold_ids=b.new_id.to_numpy()
            else:
                np.testing.assert_array_equal(fold_ids,b.new_id.to_numpy())
                np.testing.assert_array_equal(truth,b.pto.to_numpy())
            np.testing.assert_array_equal(fold_ids,n.new_id.to_numpy())
            np.testing.assert_array_equal(truth,n.pto.to_numpy())
            sets['baseline'][model]=b.prediction.to_numpy();sets['experimental'][model]=n.prediction.to_numpy()
            sets['confirmed'][model]=(n if selected[model]!='baseline' else b).prediction.to_numpy()
            for config,frame in [('baseline',b),(wx,n)]:
                for region,part in frame.groupby('region',dropna=False):
                    regional.append(dict(fold=fold,model=model,configuration=config,region=region,shops=len(part),mape=mape(part.pto,part.prediction)))
                if fold in CONFIRM:
                    for k in [6,12]:
                        w=sparse.load_npz(HERE/'cache'/('knn'+str(k)+'.npz'))
                        moran_rows.append(dict(fold=fold,model=model,configuration=config,W='knn'+str(k),**moran(frame,w,stores)))
        for composition,components in sets.items():
            for rule,pred in original_blends(components).items():
                ensemble.append(dict(fold=fold,composition=composition,rule=rule,mape=mape(truth,pred)))
    ensemble=pd.DataFrame(ensemble);ensemble.to_csv(HERE/'ensemble_comparison.csv',index=False)
    pd.DataFrame(regional).to_csv(HERE/'regional_comparison.csv',index=False)
    pd.DataFrame(moran_rows).to_csv(HERE/'moran_diagnostics.csv',index=False)
    final=ensemble.query("rule=='final_meta_logmean' and fold in @CONFIRM").pivot(index='fold',columns='composition',values='mape')
    ensemble_pass=bool((final.confirmed<final.baseline).all())
    recommended='confirmed' if ensemble_pass else 'baseline'
    decision=dict(status='complete',method='Course SLX explanatory lags WX only',original_training_counts=COUNTS,
        individual_composition=selected,original_ensemble_rule='final_meta_logmean',ensemble_confirmed=ensemble_pass,
        recommended_composition=recommended,confirmation_metrics=final.reset_index().to_dict('records'),
        no_leaderboard_tuning=True,actual_march_targets_available=False,retrospective_previously_seen_months=True)
    write_json(HERE/'FINAL_DECISION.json',decision)
    out=HERE.parent/'submissions/2025-03-course';sets={s:{} for s in ['baseline','experimental','confirmed']};ids=None
    for model in MODELS:
        source=read_previous('baseline/test_'+model+'.csv');b=pd.read_csv(io.BytesIO(source)).sort_values('new_id')
        p=out/'experimental'/('test_'+model+'.csv');n=pd.read_csv(p).sort_values('new_id');info=json.loads(p.with_suffix('.json').read_text())
        assert sha(p)==info['submission_sha256'] and sha(out/info['model_file'])==info['model_sha256']
        assert info['trees']==COUNTS[model] and not info['early_stopping']
        np.testing.assert_array_equal(b.new_id,n.new_id)
        if ids is None:ids=b.new_id.to_numpy()
        else:np.testing.assert_array_equal(ids,b.new_id)
        for variant,frame in [('baseline',b),('experimental',n),('confirmed',n if selected[model]!='baseline' else b)]:sets[variant][model]=frame.rto.to_numpy()
        reference=out/'baseline'/('test_'+model+'.csv');reference.parent.mkdir(parents=True,exist_ok=True);reference.write_bytes(source)
        destination=out/'confirmed'/('test_'+model+'.csv');destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(p.read_bytes() if selected[model]!='baseline' else source)
        native.append(dict(model=model,scope='March experimental',file=info['model_file'],sha256=info['model_sha256'],trees=native_count(out/info['model_file'],model)))
        assert native[-1]['trees']==COUNTS[model]
    for composition,components in sets.items():
        submit(out/'ensemble'/('test_'+composition+'_final_meta_logmean.csv'),ids,original_blends(components)['final_meta_logmean'])
    oldmeta=pd.read_csv(io.BytesIO(read_previous('ensemble/baseline_final_meta_logmean.csv'))).sort_values('new_id')
    np.testing.assert_allclose(oldmeta.rto,original_blends(sets['baseline'])['final_meta_logmean'],rtol=1e-12,atol=1e-8)
    baseline_meta=out/'ensemble/test_baseline_final_meta_logmean.csv';baseline_meta.write_bytes(read_previous('ensemble/baseline_final_meta_logmean.csv'))
    (out/'test.csv').write_bytes((out/'ensemble'/('test_'+recommended+'_final_meta_logmean.csv')).read_bytes())
    write_json(out/'CURRENT_STATUS.json',decision)
    csv_rows=[];first_csv_by_hash={}
    for p in sorted(out.rglob('*.csv')):
        f=pd.read_csv(p);assert list(f.columns)==['new_id','rto'] and len(f)==18657 and f.new_id.is_unique
        np.testing.assert_array_equal(np.sort(f.new_id),ids);assert np.isfinite(f.rto).all() and (f.rto>0).all() and p.stat().st_size<1000000
        digest=sha(p);relative=p.relative_to(out).as_posix()
        csv_rows.append(dict(file=relative,sha256=digest,rows=len(f),bytes=p.stat().st_size,
            same_content_as=first_csv_by_hash.get(digest)))
        first_csv_by_hash.setdefault(digest,relative)
    for p in sorted((HERE/'results').glob('*/*/*.json')):
        info=json.loads(p.read_text(encoding='utf-8'));model=info['model'];path=HERE/info['model_file']
        assert sha(path)==info['model_sha256'];count=native_count(path,model);assert count==COUNTS[model]
        native.append(dict(model=model,scope=info['fold']+'/'+info['configuration'],file=info['model_file'],sha256=info['model_sha256'],trees=count))
    write_json(HERE/'native_count_checks.json',dict(status='passed',models=native))
    write_json(out/'submission_manifest.json',dict(status='passed',forecast_month='2025-03',files=csv_rows,
        byte_distinct_csv_contents=len(first_csv_by_hash),decision=decision))
    submission_text='# Март 2025: только учебный WX\n\n'+('Рекомендован подтверждённый состав.' if ensemble_pass else 'Рекомендован исходный полный ансамбль: новый состав не прошёл проверку обоих контрольных месяцев.')+'\n\n`test.csv` — рекомендованный файл. `baseline` — неизменённые исходные прогнозы; `experimental` — WX каждой модели, включая не подтвердившиеся варианты; `confirmed` — WX только при выигрыше в разработке и обоих контрольных месяцах, иначе исходная модель. Во всех файлах 18657 строк `new_id,rto`; каждый меньше 1 МБ. Прогнозы относятся к марту 2025. В соревнование автоматически не отправлены. Качество марта неизвестно.\n\n'+table(pd.DataFrame(csv_rows)[['file','rows','bytes']])+'\n'
    submission_text+='\nДля загрузки используйте `../2025-03-course-upload-ready/march_course_upload_ready.zip`: после распаковки каждый вариант называется ровно `test.csv` и находится в своей папке. В соревнование загружается CSV из выбранной папки, не общий ZIP. Прогнозы копируются побайтово без изменения.\n'
    (out/'SUBMISSIONS.md').write_text(submission_text,encoding='utf-8')
    with zipfile.ZipFile(out/'march_course_submissions.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for row in csv_rows:z.write(out/row['file'],row['file'])
        for name in ['SUBMISSIONS.md','submission_manifest.json','CURRENT_STATUS.json']:z.write(out/name,name)
    with zipfile.ZipFile(out/'march_course_submissions.zip') as z:assert z.testzip() is None
    title='Учебный WX снизил MAPE ансамбля в обоих контрольных месяцах' if ensemble_pass else 'Новый состав не подтвердил выигрыш; рекомендован исходный ансамбль'
    report='# Учебные пространственные методы в прогнозе X5\n\n'+title+'. Мартовские истинные значения неизвестны; баллы leaderboard не использованы для настройки.\n\n'
    parity_view=pd.DataFrame(parity)
    parity_view['max_relative_difference']=parity_view.max_relative_difference.map(lambda v:f'{v:.3e}')
    parity_view['model']=parity_view.model.map(NAMES)
    report+='Проверена одна прямая интеграция курса: лаги объясняющих переменных WX из SLX. Источники: `Source/Seminars/Seminar4/sim_cross_section_models.R`, строки 44–49,68 (Xβ+WXθ); `municipalities_models.R`, строки 15,45–47,109,153–154 (проекция, симметричные kNN6/12, нормировка, SLX). k12 приведена в учебном SARAR как вторая W; здесь это фиксированный вариант той же конструкции, а не пример SLX из семинара. Бустинг с WX не объявляется оценённой линейной SLX или причинной моделью.\n\n'
    report+='W построена на 3360 населённых пунктах в учебной проекции Albers Siberia: симметризация до построчной нормировки, нулевая диагональ. Исходные числовые регрессоры агрегированы средним внутри населённого пункта и месяца, затем умножены на W и возвращены магазинам. Добавлено 39/22/33/14 WX-признаков соответственно CatBoost/LGB-приростов/LGB-уровней/XGB. Неизвестный текущий РТО не используется, NA не заменяется собственным прогнозом. Первоначальные регрессоры и цели сохранены. Исходные числовые календарные регрессоры, не попавшие в явный список исключений, тоже умножаются на W: для постоянного по территории X результат WX=X, новой пространственной информации это не даёт. Поэтому сравнение измеряет эффект всего добавленного WX-блока, а не отдельный эффект каждого его столбца. Полные списки сохранены в метаданных fit.\n\n'
    report+='Число деревьев во всех сравнениях и марте 6500/7000/8000/3750, исходные objective, learning rate, регуляризация и seed. Early stopping не применяется. CatBoost 1.2.10 и LightGBM 4.7.0 работают на CPU, XGBoost 3.1.3 — на CUDA. Каждый исходный полный ноябрьский прогноз повторён и проверен против сохранённого; WX-подготовка проверена по полным fingerprint. Результаты исходных полных августовских/декабрьских fit переиспользованы с проверкой SHA и native-моделей. Место выполнения и паритет каждого повторного baseline:\n\n'+table(parity_view)+'\n\n'
    report+='## Разработка: август, ноябрь, декабрь 2024\n\n'+table(dev)+'\n\nЛучший WX по средней MAPE зафиксирован до его январских/февральских вычислений. Если baseline выиграл разработку, WX всё равно рассчитан для диагностики, но не допускается в итоговый состав.\n\n'
    readable=[]
    for row in individual:
        readable.append({'Модель':NAMES[row['model']],'Деревья':row['trees'],'W':row['WX'].removeprefix('wx_'),
            'MAPE до, средняя':np.mean([row[f+'_before'] for f in CONFIRM]),
            'MAPE WX, средняя':np.mean([row[f+'_WX'] for f in CONFIRM]),
            'Выигрыш января, п.п.':row['2025-01_gain'],'Выигрыш февраля, п.п.':row['2025-02_gain'],
            'В итоговом составе':'WX' if row['eligible'] else 'исходная'})
    report+='## Каждая модель: до и после\n\nMAPE в процентах; средняя — по январю и февралю, положительный выигрыш означает снижение ошибки. Все месяцы уже использовались в исследовании: это повторная ретроспективная проверка, не новая слепая выборка.\n\n'+table(pd.DataFrame(readable))+'\n\nПодробные метрики каждого месяца и критерии допуска сохранены в `confirmation_model_comparison.csv`.\n\n'
    report+=f'![Сравнение моделей до и после учебного WX](<{(HERE/"before_after.png").as_posix()}>)\n\n'
    report+='## Ансамбль с исходной формулой\n\nВо всех строках ниже одинаковая исходная формула `final_meta_logmean`; веса и порог 1.08 не перенастроены. experimental использует лучший WX для всех четырёх моделей; confirmed оставляет исходную версию там, где выигрыш не подтверждён. Для рекомендации confirmed обязан улучшить исходный ансамбль в обоих контрольных месяцах.\n\n'+table(ensemble.query("rule=='final_meta_logmean'"))+'\n\n'
    report+=f'Средняя MAPE исходного ансамбля за январь и февраль: {final.baseline.mean():.6f}%; состава с допущенными WX-моделями: {final.confirmed.mean():.6f}%. Рекомендованный мартовский файл использует состав `{recommended}`.\n\n'
    report+='## Диагностика и границы вывода\n\n`moran_diagnostics.csv` содержит Moran I средних относительных ошибок населённых пунктов и 999 перестановок, как `moran.mc` в семинаре 2, строки 243–244. Альтернатива — положительная автокорреляция, seed 20261010; это диагностика, не критерий отбора. `regional_comparison.csv` сохраняет срезы по регионам. MAPE оценивается равным весом магазинов, Moran — населённых пунктов; это разные задачи.\n\n'
    report+='SAR/SEM/SDM/SDEM/SARAR, пространственная панель и настоящая GWR в этом эксперименте не оценивались; их нельзя объявлять неработающими или подменять сглаживанием прогноза/локальным ridge. Не добавлялись собственные корреляционные графы, усадки, инновационные пакеты или eigenfilter. Эконометрическая часть исходного исследования не изменена.\n\n'
    report+=f'Проверены SHA и фактическое число деревьев {len(native)-4} валидационных и 4 мартовских native-моделей. MAPE валидационных прогнозов пересчитана из CSV; мартовские WX-модели перепроверены повторной загрузкой. Прогнозы и архив находятся в `Forecasting/submissions/2025-03-course`; `test.csv` соответствует итоговому решению, не обещанию мартовского выигрыша. Полные метрики, графы, код, выбор до подтверждения и приватные notebook сохранены в этом каталоге.\n'
    report+='\nДля загрузки подготовлен пакет `Forecasting/submissions/2025-03-course-upload-ready/march_course_upload_ready.zip`: 11 вариантов в отдельных папках с именем `test.csv`, без изменения прогнозов. После распаковки загружается CSV, не общий ZIP.\n'
    (HERE/'REPORT.md').write_text(report,encoding='utf-8')
    source_snapshot=json.loads((HERE/'data_and_course_sources.json').read_text(encoding='utf-8'))
    for source in source_snapshot['files']:
        assert sha(resource_path(source['file']))==source['sha256'],source['file']
    write_json(HERE/'source_preservation_checks.json',dict(status='passed',files=source_snapshot['files']))
    assert sha(HERE.parent/'baseline/original_solution.ipynb')=='15a74a6b30c47fb9821b7d8ad5574868b7a793b744c7ce1997d1919aaa9caeb5'
    assert sha(resource_path('Data/Shops/train_2.csv'))=='f1422045d80c673b28d3abcef91feb78308499eb511d4f5090a81adbebd44338'
    write_json(HERE/'artifact_checks.json',dict(status='passed',validation_fits=len(native)-4,march_native_models=4,
        csv_files=len(csv_rows),original_raw_and_notebook_unchanged=True,original_counts_preserved=True,decision=decision))
    from package_uploads import build as package_uploads
    package_uploads()
    print('COURSE_FINALIZED',json.dumps(decision,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
