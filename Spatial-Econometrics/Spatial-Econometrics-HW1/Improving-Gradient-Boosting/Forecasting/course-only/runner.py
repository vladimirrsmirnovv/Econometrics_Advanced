"""Course WX only; every booster retains its original full training recipe."""


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
import gc,hashlib,importlib.metadata,json,sys,time
import numpy as np
import pandas as pd
from scipy import sparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,load_recipe,matrices,NOTEBOOK_SHA256
from run_validation import get_fold
COUNTS=dict(zip(MODELS,[6500,7000,8000,3750]))
CONFIGS=['baseline','wx_knn6','wx_knn12']
EXCLUDE={'new_id','month','prev_month','year','year_month_index','days_lag_1','days_in_month'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source_columns(x,cats):
    return [c for c in x if c not in EXCLUDE and c not in cats and not c.startswith('is_')
        and not c.endswith(('_cb','_cat','_category')) and pd.api.types.is_numeric_dtype(x[c])
        and not pd.api.types.is_bool_dtype(x[c])]

def spatial_x(x,meta,columns,w,stores):
    assert x.index.equals(meta.index)
    nodes=meta.new_id.map(stores.node)
    assert nodes.notna().all()
    frame=x[columns].reset_index(drop=True).assign(node=nodes.to_numpy(),time=meta.year_month_index.to_numpy())
    grouped=frame.groupby(['time','node'],sort=True)[columns].mean()
    result=np.full((len(frame),len(columns)),np.nan,dtype=np.float32)
    for t in np.sort(frame.time.unique()):
        part=grouped.loc[t].reindex(range(w.shape[0]))
        # Literal WX: unavailable neighbour X propagates NA. No imputation,
        # renormalisation by availability, shrinkage or custom gap is introduced.
        lag=w@part.to_numpy(dtype=np.float64)
        positions=np.flatnonzero(frame.time.to_numpy()==t)
        result[positions]=lag[frame.node.to_numpy()[positions]]
    return pd.DataFrame(result,index=x.index,columns=['course_wx__'+c for c in columns])

def prepare(recipe,compact,model,config):
    assert config in CONFIGS
    xt,xv,yt,yv,cats=matrices(recipe,compact,model);columns=[];fingerprints={}
    if config!='baseline':
        columns=source_columns(xt,cats);assert columns
        stores=pd.read_parquet(HERE/'cache/stores.parquet').set_index('new_id')
        w=sparse.load_npz(HERE/'cache'/('knn'+config.removeprefix('wx_knn')+'.npz'))
        for label,x,meta in [('train',xt,compact['train_features']),('validation',xv,compact['val_features'])]:
            added=spatial_x(x,meta,columns,w,stores)
            fingerprints[label]=hashlib.sha256(pd.util.hash_pandas_object(added,index=False).values.tobytes()).hexdigest()
            if label=='train':xt=pd.concat([x,added],axis=1)
            else:xv=pd.concat([x,added],axis=1)
    return xt,xv,yt,yv,cats,columns,fingerprints

def fit(recipe,compact,model,config,threads=4):
    xt,xv,yt,yv,cats,columns,fingerprints=prepare(recipe,compact,model,config)
    params=dict(getattr(recipe,model+'_params'));count=COUNTS[model]
    if model=='catboost_log_level':
        from catboost import CatBoostRegressor,Pool
        params.update(iterations=count,thread_count=threads,allow_writing_files=False,verbose=False)
        estimator=CatBoostRegressor(**params);estimator.fit(Pool(xt,yt,cat_features=cats))
        pred=recipe._cb_inverse_log1p(estimator.predict(xv));used=estimator.tree_count_
    elif model.startswith('lightgbm'):
        import lightgbm as lgb
        params=recipe._clean_lgbm_params(params);params.update(n_estimators=count,n_jobs=threads,verbosity=-1)
        estimator=lgb.LGBMRegressor(**params);estimator.fit(xt,yt)
        pred=recipe.inverse_forecast_target(estimator.predict(xv),compact['val_features'],
            mode='growth_log' if model.endswith('log_diff') else 'direct_log1p')
        used=estimator.booster_.current_iteration()
    else:
        import xgboost as xgb
        assert xgb.__version__=='3.1.3','Original CUDA version required'
        params=recipe._clean_xgb_params(params,keep_early_stopping=False);params.update(n_estimators=count,device='cuda',n_jobs=threads)
        estimator=xgb.XGBRegressor(**params);valid=np.isfinite(yt)
        estimator.fit(xt.loc[valid],np.asarray(yt)[valid],verbose=False)
        assert json.loads(estimator.get_booster().save_config())['learner']['generic_param']['device'].startswith('cuda')
        pred=recipe.inverse_forecast_target(estimator.predict(xv),compact['val_features'],mode='growth_log')
        used=estimator.get_booster().num_boosted_rounds()
    assert used==count and len(pred)==18657 and np.isfinite(pred).all() and (pred>0).all()
    return estimator,pred,params,columns,fingerprints

def save_native(estimator,path,model):
    path.parent.mkdir(parents=True,exist_ok=True)
    if model=='catboost_log_level':path=path.with_suffix('.cbm');estimator.save_model(str(path))
    elif model.startswith('lightgbm'):path=path.with_suffix('.txt');path.write_text(estimator.booster_.model_to_string(),encoding='utf-8')
    else:path=path.with_suffix('.ubj');path.write_bytes(estimator.get_booster().save_raw(raw_format='ubj'))
    return path

def execute(model,fold,config):
    target=HERE/'results'/fold/model.replace("_", "-")/(config+'.csv');target.parent.mkdir(parents=True,exist_ok=True)
    metadata=target.with_suffix('.json')
    if metadata.exists():
        row=json.loads(metadata.read_text());assert sha(target)==row['prediction_sha256'];return row
    start=time.perf_counter();recipe=load_recipe();year,month=map(int,fold.split('-'));compact=get_fold(recipe,year,month)
    print('COURSE_FIT',model,fold,config,'trees',COUNTS[model],flush=True)
    estimator,pred,params,columns,fingerprints=fit(recipe,compact,model,config)
    anchor=compact['val_features'][['new_id','year_month_index','region','settlement','pto']].copy();anchor['prediction']=pred
    anchor.to_csv(target,index=False);native=save_native(estimator,target.parent/'models'/config.replace("_", "-"),model)
    row=dict(model=model,fold=fold,configuration=config,trees=COUNTS[model],early_stopping=False,
        parameters=params,source_regressors_for_WX=columns,WX_fingerprints=fingerprints,
        course_method='SLX explanatory spatial lags WX' if config!='baseline' else 'original baseline',
        course_weights_manifest_sha256=sha(HERE/'cache/weights_manifest.json'),notebook_sha256=NOTEBOOK_SHA256,
        source_code_sha256=sha(Path(__file__)),outer_labels_used_for_fit=False,
        model_file=native.relative_to(HERE).as_posix(),model_sha256=sha(native),prediction_sha256=sha(target),
        seconds=time.perf_counter()-start,
        environment={n:importlib.metadata.version(n) for n in ['catboost','lightgbm','xgboost','numpy','pandas','scipy']},
        **recipe._make_metrics(anchor.pto,pred))
    metadata.write_text(json.dumps(row,indent=2))
    print('COURSE_RESULT',model,fold,config,row['mape'],'seconds',round(row['seconds'],1),flush=True)
    del compact,estimator;gc.collect();return row

def march(model,config):
    out=HERE.parent/'submissions/2025-03-course';target=out/'experimental'/('test_'+model+'.csv')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.with_suffix('.json').exists():assert sha(target)==json.loads(target.with_suffix('.json').read_text())['submission_sha256'];return
    cache=HERE.parent/'cache/final_2025-03';manifest=json.loads((cache/'manifest.json').read_text())
    compact=dict(train_features=pd.read_parquet(cache/'test_train_features.parquet'),val_features=pd.read_parquet(cache/'test_features.parquet'),
        categorical_cols=manifest['categorical_cols'],feature_cols=manifest['feature_cols'],test_train_features=None,test_features=None)
    assert compact['val_features'].pto.isna().all() and compact['train_features'].year_month_index.max()==25
    recipe=load_recipe();print('COURSE_MARCH_FIT',model,config,COUNTS[model],flush=True)
    estimator,pred,params,columns,fingerprints=fit(recipe,compact,model,config)
    native=save_native(estimator,out/'models'/model.replace("_", "-"),model)
    xt,xv,_,_,cats,_,_=prepare(recipe,compact,model,config);del xt;gc.collect()
    if model=='catboost_log_level':
        from catboost import CatBoostRegressor
        restored=CatBoostRegressor();restored.load_model(str(native));raw=restored.predict(xv);replay=recipe._cb_inverse_log1p(raw)
        assert restored.tree_count_==COUNTS[model]
    elif model.startswith('lightgbm'):
        import lightgbm as lgb
        restored=lgb.Booster(model_str=native.read_text(encoding='utf-8'));assert restored.current_iteration()==COUNTS[model]
        replay=recipe.inverse_forecast_target(restored.predict(xv,num_threads=4),compact['val_features'],mode='growth_log' if model.endswith('log_diff') else 'direct_log1p')
    else:
        import xgboost as xgb
        restored=xgb.XGBRegressor();restored.load_model(native);restored.set_params(device='cuda',n_jobs=4)
        assert restored.get_booster().num_boosted_rounds()==COUNTS[model]
        replay=recipe.inverse_forecast_target(restored.predict(xv),compact['val_features'],mode='growth_log')
    np.testing.assert_allclose(replay,pred,rtol=1e-12,atol=1e-8)
    pd.DataFrame({'new_id':compact['val_features'].new_id.to_numpy(),'rto':pred}).to_csv(target,index=False)
    assert target.stat().st_size<=1000000
    info=dict(model=model,configuration=config,forecast_month='2025-03',trees=COUNTS[model],early_stopping=False,
        parameters=params,source_regressors_for_WX=columns,WX_fingerprints=fingerprints,
        native_reload_max_relative_difference=float(np.max(abs(replay/_directory_path(pred)-1))),
        model_file=native.relative_to(out).as_posix(),model_sha256=sha(native),submission_sha256=sha(target),
        actual_march_targets_available=False,notebook_sha256=NOTEBOOK_SHA256)
    target.with_suffix('.json').write_text(json.dumps(info,indent=2));print('COURSE_MARCH_DONE',model,config,flush=True)
