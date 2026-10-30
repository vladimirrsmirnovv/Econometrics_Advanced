"""Two March full fits, original CUDA recipe and validation-selected hybrid history."""
import ast,gc,hashlib,io,json,platform,subprocess,time,warnings
from pathlib import Path
import holidays
import numpy as np
import pandas as pd
import xgboost as xgb
FUTURE_HYBRID_CSV=None
EXPECTED_FULL_FEATURES=None
EXPECTED_FUTURE_HYBRID_SHA256=None

def fingerprint(frame):
    return dict(columns=list(frame.columns),dtypes=[str(t) for t in frame.dtypes],rows=len(frame),
        sha256=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False,categorize=False).values.tobytes()).hexdigest())

def main():
    assert xgb.__version__=='3.1.3'
    original=list(Path('/kaggle/input').rglob('input_manifest.json'))
    derived=list(Path('/kaggle/input').rglob('research_manifest.json'))
    assert len(original)==len(derived)==1
    source=original[0].parent;data=derived[0].parent;out=Path('/kaggle/working')
    manifest=json.loads(original[0].read_text());plan=json.loads(derived[0].read_text())
    for name,digest in manifest['input_hashes'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(FUTURE_HYBRID_CSV.encode()).hexdigest()==EXPECTED_FUTURE_HYBRID_SHA256
    notebook=json.loads((source/'original_solution.ipynb').read_text());namespace={'holidays':holidays}
    for i in [10,14,16,37]:exec(compile(''.join(notebook['cells'][i]['source']),'original:'+str(i),'exec'),namespace)
    for node in ast.parse(''.join(notebook['cells'][18]['source'])).body:
        if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),'original_preparation','exec'),namespace)
        elif isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in {'COMMON_BASE_FEATURE_KWARGS','ID_COL','YEAR_COL','MONTH_COL','TIME_COL','TARGET_COL'}:
                namespace[node.targets[0].id]=ast.literal_eval(node.value)
    rename=ast.literal_eval(ast.parse(''.join(notebook['cells'][8]['source'])).body[0].value)
    raw=pd.read_csv(source/'train_2.csv').rename(columns=rename)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        features=namespace['build_retail_features_with_split'](raw,split_mode='train_val_test',min_year=2023,
            val_year=2025,val_month=2,test_year=2025,test_month=3,create_test_if_missing=True,
            expected_test_size=18657,use_val_as_test_history=True,
            base_feature_kwargs=namespace['COMMON_BASE_FEATURE_KWARGS'],verbose=True)
    base=namespace['xgboost_log_diff_features']
    full=namespace['_prepare_numeric_model_pack'](features,selected_features=base,target_mode='growth_log',
        label='xgboost_log_diff',verbose=False)['test']
    train=full['X_train'][base];test=full['X_test'][base];target=np.asarray(full['y_train'])
    actual={'train':fingerprint(train),'test':fingerprint(test),
        'effective_target':fingerprint(pd.DataFrame({'target':target}).astype('float32').where(np.isfinite(target[:,None]),np.nan))}
    assert actual==EXPECTED_FULL_FEATURES, 'Original full forecast matrices differ between local and GPU builds'
    train_meta=features['test_train_features'][['new_id','year_month_index','pto']].copy()
    test_meta=features['test_features'][['new_id','year_month_index','pto','pto_lag_1']].copy()
    assert train_meta.year_month_index.max()==25 and (test_meta.year_month_index==26).all() and test_meta.pto.isna().all()
    cols=plan['groups']['hybrid8_history']
    historical=[]
    for name in ['2025-02_train.parquet','2025-02_holdout.parquet']:
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==plan['hashes'][name]
        historical.append(pd.read_parquet(data/name,columns=['new_id','year_month_index']+cols))
    extras=pd.concat(historical,ignore_index=True).set_index(['new_id','year_month_index'])
    future=pd.read_csv(io.StringIO(FUTURE_HYBRID_CSV)).set_index(['new_id','year_month_index'])
    for col in cols:future[col]=future[col].astype('float32')
    added_train=extras.reindex(pd.MultiIndex.from_frame(train_meta[['new_id','year_month_index']]))[cols].reset_index(drop=True).set_axis(train.index)
    added_test=future.reindex(pd.MultiIndex.from_frame(test_meta[['new_id','year_month_index']]))[cols].reset_index(drop=True).set_axis(test.index)
    assert not added_test.isna().any().any() and len(added_test)==18657
    params=namespace['_clean_xgb_params'](dict(namespace['xgboost_log_diff_params']),keep_early_stopping=False)
    assert params['n_estimators']==3750 and params['device']=='cuda' and 'early_stopping_rounds' not in params
    mask=np.isfinite(target);runtime=dict(python=platform.python_version(),
        versions={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in ['xgboost','numpy','pandas','pyarrow','holidays']},
        hardware=subprocess.check_output(['nvidia-smi'],text=True))
    del features,raw,full;gc.collect()
    records=[]
    for variant in ['baseline','spatial']:
        xt=train if variant=='baseline' else pd.concat([train,added_train],axis=1)
        xp=test if variant=='baseline' else pd.concat([test,added_test],axis=1)
        start=time.perf_counter();print('FULL_CUDA_FIT',variant,xt.shape,flush=True)
        model=xgb.XGBRegressor(**params)
        model.fit(xt.loc[mask],target[mask],verbose=False)
        device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        assert device.startswith('cuda') and model.get_booster().num_boosted_rounds()==3750
        pred=namespace['inverse_forecast_target'](model.predict(xp),test_meta,mode='growth_log')
        assert len(pred)==18657 and np.isfinite(pred).all() and (pred>=0).all()
        directory=out/variant;directory.mkdir(exist_ok=True)
        name='test_xgboost_log_diff';table=pd.DataFrame({'new_id':test_meta.new_id.to_numpy(),'rto':pred})
        table.to_csv(directory/(name+'.csv'),index=False);model.save_model(directory/(name+'.ubj'))
        row=dict(model='xgboost_log_diff',variant=variant,configuration='baseline' if variant=='baseline' else 'hybrid8_history',
            forecast_month='2025-03',training_months=['2023-01','2025-02'],train_rows=len(xt),fit_rows=int(mask.sum()),
            features=list(xt.columns),parameters=params,trees=3750,early_stopping=False,
            device=device,backend='CUDA',versions=runtime['versions'],seconds=time.perf_counter()-start,
            notebook_sha256=manifest['input_hashes']['original_solution.ipynb'],raw_sha256=manifest['input_hashes']['train_2.csv'],
            original_full_feature_checks=actual,future_hybrid_sha256=EXPECTED_FUTURE_HYBRID_SHA256 if variant=='spatial' else None,
            validation_only=False,unknown_march_target=True,
            submission_sha256=hashlib.sha256((directory/(name+'.csv')).read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256((directory/(name+'.ubj')).read_bytes()).hexdigest())
        (directory/(name+'.json')).write_text(json.dumps(row,indent=2));records.append(row)
        print('MARCH_CUDA_DONE',variant,flush=True);del model;gc.collect()
    (out/'march_runtime.json').write_text(json.dumps(runtime,indent=2))
    (out/'march_checks.json').write_text(json.dumps(dict(status='passed',runs=2,original_matrix_parity=True,
        original_full_feature_checks=actual,forecast_month='2025-03',unknown_march_target=True,trees_each=3750),indent=2))
    (out/'march_metrics.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
