"""CUDA combinations, key factorial and new historical baselines. No new data upload."""
import ast,gc,hashlib,json,subprocess,time,types,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import xgboost as xgb

MODELS=['catboost_log_level','lightgbm_log_diff','lightgbm_log_level','xgboost_log_diff']
OUT=Path('/kaggle/working')

def main():
    derived=list(Path('/kaggle/input').rglob('research_manifest.json'))
    originals=list(Path('/kaggle/input').rglob('input_manifest.json'))
    assert len(derived)==len(originals)==1 and xgb.__version__=='3.1.3'
    data=derived[0].parent;source=originals[0].parent
    plan=json.loads(derived[0].read_text());original=json.loads(originals[0].read_text())
    for name,digest in original['input_hashes'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    for name,digest in plan['hashes'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==digest
    notebook=json.loads((source/'original_solution.ipynb').read_text())
    namespace=dict(holidays=holidays)
    for i in [10,14,16,20,26,37]:
        exec(compile(''.join(notebook['cells'][i]['source']),'original:'+str(i),'exec'),namespace)
    for node in ast.parse(''.join(notebook['cells'][18]['source'])).body:
        if isinstance(node,ast.FunctionDef):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'original_setup','exec'),namespace)
        elif isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in {'COMMON_BASE_FEATURE_KWARGS','ID_COL','YEAR_COL','MONTH_COL','TIME_COL','TARGET_COL'}:
                namespace[node.targets[0].id]=ast.literal_eval(node.value)
    rename=ast.literal_eval(ast.parse(''.join(notebook['cells'][8]['source'])).body[0].value)
    raw=pd.read_csv(source/'train_2.csv').rename(columns=rename)
    assert len(raw)==485082
    base=plan['base_columns'];params=plan['parameters'];records=[]
    extra_columns=list(dict.fromkeys(c for group in plan['groups'].values() for c in group))
    all_extras=pd.concat([pd.read_parquet(data/'2025-02_train.parquet',columns=['new_id','year_month_index']+extra_columns),
        pd.read_parquet(data/'2025-02_holdout.parquet',columns=['new_id','year_month_index']+extra_columns)])
    assert not all_extras.duplicated(['new_id','year_month_index']).any()
    def add_extra(meta,columns):
        joined=meta[['new_id','year_month_index']].merge(all_extras[['new_id','year_month_index']+columns],
            on=['new_id','year_month_index'],how='left',sort=False,validate='one_to_one')
        assert len(joined)==len(meta)
        return joined[columns].set_axis(meta.index)
    def fit(fold,config,xtrain,xvalid,ytrain,yvalid,meta,columns,baseline_mape=None):
        selected=base+columns;train_mask=np.isfinite(ytrain);valid_mask=np.isfinite(yvalid)
        start=time.perf_counter();model=xgb.XGBRegressor(**params)
        model.fit(xtrain.loc[train_mask,selected],np.asarray(ytrain)[train_mask],
            eval_set=[(xvalid.loc[valid_mask,selected],np.asarray(yvalid)[valid_mask])],verbose=False)
        device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        assert device.startswith('cuda')
        prediction=meta.pto_lag_1.to_numpy()*np.exp(np.asarray(model.predict(xvalid[selected]),dtype=float))
        assert len(prediction)==18657 and np.isfinite(prediction).all()
        error=prediction-meta.pto.to_numpy()
        mape=float((np.abs(error)/meta.pto.to_numpy()).mean()*100)
        row=dict(model='xgboost_log_diff',configuration=config,fold=fold,mape=mape,
            mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
            bias_pct=float((error/meta.pto.to_numpy()).mean()*100),best_iteration=int(model.best_iteration),
            seconds=time.perf_counter()-start,backend='CUDA',device=device,xgboost_version=xgb.__version__,
            parameters=params,features_added=columns,notebook_sha256=plan['notebook_sha256'],
            validation_only=True,leakage_check='passed',baseline_mape=baseline_mape,
            source_feature_checks=plan['feature_checks'],phase='followup')
        directory=OUT/'results'/fold/'xgboost_log_diff';directory.mkdir(parents=True,exist_ok=True)
        table=meta[['new_id','year_month_index','region','settlement','pto']].copy();table['prediction']=prediction
        table.to_csv(directory/(config+'.csv'),index=False)
        (directory/(config+'.json')).write_text(json.dumps(row,indent=2))
        model.save_model(directory/(config+'.ubj'));records.append(row)
        (OUT/'phase2_metrics.json').write_text(json.dumps(records,indent=2))
        print('RESULT',fold,config,mape,flush=True);del model;gc.collect()
    def build(fold,key=False,frame=None):
        year,month=map(int,fold.split('-'));r=raw.copy() if frame is None else frame.copy()
        if key:r['settlement']=r.region.astype(str)+' :: '+r.settlement.astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            full=namespace['build_retail_features_with_split'](r,split_mode='train_val',min_year=2023,
                val_year=year,val_month=month,base_feature_kwargs=namespace['COMMON_BASE_FEATURE_KWARGS'],verbose=False)
        return full
    def compact(full):
        requested={'new_id','year_month_index','pto','pto_lag_1','region','settlement'}
        for model in MODELS:
            for name in namespace[model+'_features']:
                requested.add(name[:-3] if name.endswith('_cb') else name)
                if name.endswith('_cat_cb'):requested.add(name[:-7])
        result={key:None for key in ['train_features','val_features','test_train_features','test_features']}
        for key in ['train_features','val_features']:
            result[key]=full[key][[c for c in full[key] if c in requested]].copy()
            for name in namespace['catboost_log_level_features']:
                if name not in result[key] and ('feb_mar_growth_2024' in name or 'region_march_growth' in name):result[key][name]=np.nan
        result.update(categorical_cols=full['categorical_cols'],feature_cols=full['feature_cols'])
        return result
    # Source-key leakage check, including changed current and all later outcomes.
    ids=np.random.default_rng(2026).choice(raw.new_id.unique(),160,replace=False)
    sample=raw[raw.new_id.isin(ids)]
    for fold in plan['folds']:
        year,month=map(int,fold.split('-'));changed=sample.copy()
        changed.loc[changed.year*12+changed.month>=year*12+month,'pto']*=23
        a=compact(build(fold,True,sample));b=compact(build(fold,True,changed))
        pa=namespace['_prepare_numeric_model_pack'](a,selected_features=base,target_mode='growth_log',label='xgboost_log_diff',verbose=False)['validation']
        pb=namespace['_prepare_numeric_model_pack'](b,selected_features=base,target_mode='growth_log',label='xgboost_log_diff',verbose=False)['validation']
        for part in ['X_train','X_val']:
            pd.testing.assert_frame_equal(pa[part][base],pb[part][base])
    (OUT/'phase2_checks.json').write_text(json.dumps(dict(status='passed',key_cutoffs=plan['folds'],
        extra_features=plan['feature_checks'],raw_hash=original['input_hashes']['train_2.csv']),indent=2))
    for fold in plan['folds']:
        train=pd.read_parquet(data/(fold+'_train.parquet'));valid=pd.read_parquet(data/(fold+'_holdout.parquet'))
        for config in ['knn8_history+wx_relative','hybrid8_history+within_market']:
            columns=list(dict.fromkeys(c for g in config.split('+') for c in plan['groups'][g]))
            fit(fold,config,train,valid,train.target,valid.target,valid,columns)
        del train,valid;gc.collect()
        full=build(fold,True);c=compact(full);del full;gc.collect()
        directory=OUT/'prepared'/('key_'+fold);directory.mkdir(parents=True,exist_ok=True)
        for key,name in [('train_features','train'),('val_features','holdout')]:c[key].to_parquet(directory/(name+'.parquet'),index=False)
        (directory/'manifest.json').write_text(json.dumps(dict(raw_sha256=original['input_hashes']['train_2.csv'],
            notebook_sha256=plan['notebook_sha256'],categorical_cols=c['categorical_cols'],feature_cols=c['feature_cols'],
            key='region :: original settlement',prepared_backend='Kaggle',source_untouched=True),indent=2))
        pack=namespace['_prepare_numeric_model_pack'](c,selected_features=base,target_mode='growth_log',label='xgboost_log_diff',verbose=False)['validation']
        for config,groups in [('key',[]),('key+hybrid8_history',['hybrid8_history'])]:
            columns=[col for group in groups for col in plan['groups'][group]]
            xtrain=pd.concat([pack['X_train'][base],add_extra(c['train_features'],columns)],axis=1)
            xvalid=pd.concat([pack['X_val'][base],add_extra(c['val_features'],columns)],axis=1)
            fit(fold,config,xtrain,xvalid,pack['y_train'],pack['y_val'],c['val_features'],columns)
        del c,pack,xtrain,xvalid;gc.collect()
    for fold in ['2024-02','2024-12','2024-10']:
        full=build(fold);c=compact(full);del full;gc.collect()
        directory=OUT/'prepared'/fold;directory.mkdir(parents=True,exist_ok=True)
        for key,name in [('train_features','train'),('val_features','holdout')]:c[key].to_parquet(directory/(name+'.parquet'),index=False)
        (directory/'manifest.json').write_text(json.dumps(dict(raw_sha256=original['input_hashes']['train_2.csv'],
            notebook_sha256=plan['notebook_sha256'],categorical_cols=c['categorical_cols'],feature_cols=c['feature_cols'],
            prepared_backend='Kaggle'),indent=2))
        pack=namespace['_prepare_numeric_model_pack'](c,selected_features=base,target_mode='growth_log',label='xgboost_log_diff',verbose=False)['validation']
        for config,groups in [('baseline',[])]+([('hybrid8_history',['hybrid8_history']),('knn8_history+wx_relative',['knn8_history','wx_relative'])] if fold=='2024-10' else []):
            columns=list(dict.fromkeys(col for group in groups for col in plan['groups'][group]))
            xtrain=pd.concat([pack['X_train'][base],add_extra(c['train_features'],columns)],axis=1)
            xvalid=pd.concat([pack['X_val'][base],add_extra(c['val_features'],columns)],axis=1)
            fit(fold,config,xtrain,xvalid,pack['y_train'],pack['y_val'],c['val_features'],columns)
        del c,pack,xtrain,xvalid;gc.collect()
    print('FOLLOWUP_CUDA_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
