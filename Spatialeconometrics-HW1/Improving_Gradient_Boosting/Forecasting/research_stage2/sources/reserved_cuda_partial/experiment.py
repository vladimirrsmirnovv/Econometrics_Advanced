"""Matched rolling refits: outer labels never enter tree-count selection."""
from pathlib import Path
import argparse,contextlib,gc,hashlib,importlib.metadata,io,json,sys,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,NOTEBOOK_SHA256,load_recipe,matrices
from run_validation import get_fold

SCREEN=['baseline','previous','exp50_history','hybrid8_history','knn8_history','eigenvectors',
        'seasonal_control','seasonal_spatial','innovations_control','innovations_spatial',
        'dynamic_history','hybrid8_history+seasonal_spatial','knn8_history+eigenvectors']
NOVEL=['seasonal_control','seasonal_spatial','innovations_control','innovations_spatial',
       'dynamic_history','hybrid8_history+seasonal_spatial','knn8_history+eigenvectors']
OLD={'catboost_log_level':'exp50_history','lightgbm_log_diff':'eigenvectors',
     'lightgbm_log_level':'density_weight','xgboost_log_diff':'hybrid8_history'}
COUNTS=dict(zip(MODELS,[6500,7000,8000,3750]))

def previous_month(fold):
    year,month=map(int,fold.split('-'));date=pd.Timestamp(year,month,1)-pd.offsets.MonthBegin(1)
    return date.strftime('%Y-%m')

def read_extras(model):
    old_manifest=json.loads((HERE.parent/'research/cache/feature_manifest.json').read_text())
    old=pd.read_parquet(HERE.parent/'research/cache/features.parquet')
    new_manifest=json.loads((HERE/'cache/feature_manifest.json').read_text())
    check=json.loads((HERE/'feature_checks.json').read_text());assert check['status']=='passed'
    assert check['implementation_sha256']==hashlib.sha256((HERE/'features.py').read_bytes()).hexdigest()
    new=pd.read_parquet(HERE/'cache/features.parquet')
    table=old.merge(new,on=['new_id','year_month_index'],how='outer',validate='one_to_one')
    groups=dict(old_manifest['groups'],**new_manifest['groups'])
    if (HERE/'cache/geo_control.parquet').exists():
        geo_checks=json.loads((HERE/'geo_control_checks.json').read_text());assert geo_checks['status']=='passed'
        assert geo_checks['implementation_sha256']==hashlib.sha256((HERE/'geo_control.py').read_bytes()).hexdigest()
        geo=pd.read_parquet(HERE/'cache/geo_control.parquet')
        table=table.merge(geo,on=['new_id','year_month_index'],how='left',validate='one_to_one')
        groups['geo8_control']=[c for c in geo if c.startswith('geo8_control__')]
    if model=='catboost_log_level':
        # First-stage CatBoost used its original float64-subtraction builder.
        legacy=HERE.parent/'cache/spatial_exp50.parquet'
        if not legacy.exists():legacy=HERE.parent/'cache/final_2025-03/legacy_features.parquet'
        if legacy.exists():
            history=['wy_log_daily_lag1','wy_daily_growth_lag1','wy_daily_growth_1_3','wy_log_daily_gap_lag1']
            frame=pd.read_parquet(legacy,columns=['new_id','year_month_index']+history)
            table=table.merge(frame,on=['new_id','year_month_index'],how='left',validate='one_to_one')
            groups['exp50_history']=history
    return table.set_index(['new_id','year_month_index']),groups

def data(recipe,compact,model,config,extras,groups):
    real=OLD[model] if config=='previous' else config
    columns=[] if real in ['baseline','density_weight'] else list(dict.fromkeys(c for g in real.split('+') for c in groups[g]))
    xt,xv,yt,yv,cats=matrices(recipe,compact,model)
    if columns:
        added=[]
        for meta,x in [(compact['train_features'],xt),(compact['val_features'],xv)]:
            key=pd.MultiIndex.from_frame(meta[['new_id','year_month_index']])
            assert key.isin(extras.index).all()
            a=extras.reindex(key)[columns].reset_index(drop=True).set_axis(x.index)
            added.append(pd.concat([x,a],axis=1))
        xt,xv=added
    weight=None
    if real=='density_weight':
        stores=pd.read_parquet(HERE.parent/'research/cache/stores.parquet').set_index('new_id')
        weight=1/np.sqrt(compact['train_features'].new_id.map(stores.market_shops).to_numpy());weight/=weight.mean()
    return xt,xv,yt,yv,cats,weight,columns

def fit(recipe,compact,model,config,extras,groups,trees=None,threads=4):
    xt,xv,yt,yv,cats,weight,columns=data(recipe,compact,model,config,extras,groups)
    params=dict(getattr(recipe,model+'_params'));early=trees is None
    if model=='catboost_log_level':
        from catboost import CatBoostRegressor,Pool
        params.update(thread_count=threads,allow_writing_files=False,verbose=False)
        if trees is not None:params.update(iterations=trees)
        estimator=CatBoostRegressor(**params)
        options=dict(eval_set=Pool(xv,yv,cat_features=cats),early_stopping_rounds=50,use_best_model=True) if early else {}
        estimator.fit(Pool(xt,yt,cat_features=cats,weight=weight),**options)
        raw=estimator.predict(xv);used=estimator.get_best_iteration()+1 if early else estimator.tree_count_
        pred=recipe._cb_inverse_log1p(raw)
    elif model.startswith('lightgbm'):
        import lightgbm as lgb
        params=recipe._clean_lgbm_params(params);params.update(n_jobs=threads,verbosity=-1)
        if trees is not None:params.update(n_estimators=trees)
        estimator=lgb.LGBMRegressor(**params)
        options=dict(eval_set=[(xv,yv)],eval_metric='mae',callbacks=[lgb.early_stopping(50,verbose=False)]) if early else {}
        estimator.fit(xt,yt,sample_weight=weight,**options)
        raw=estimator.predict(xv);used=estimator.best_iteration_ if early else estimator.booster_.current_iteration()
        pred=recipe.inverse_forecast_target(raw,compact['val_features'],mode='growth_log' if model.endswith('log_diff') else 'direct_log1p')
    else:
        import xgboost as xgb
        assert xgb.__version__=='3.1.3','Matched XGBoost comparisons require the pinned CUDA runner'
        params=recipe._clean_xgb_params(params,keep_early_stopping=early);params.update(device='cuda',n_jobs=threads)
        if trees is not None:params.update(n_estimators=trees)
        mask=np.isfinite(yt);valid=np.isfinite(yv)
        estimator=xgb.XGBRegressor(**params)
        options=dict(eval_set=[(xv.loc[valid],np.asarray(yv)[valid])]) if early else {}
        estimator.fit(xt.loc[mask],np.asarray(yt)[mask],verbose=False,**options)
        device=json.loads(estimator.get_booster().save_config())['learner']['generic_param']['device']
        assert device.startswith('cuda')
        raw=estimator.predict(xv);used=estimator.best_iteration+1 if early else estimator.get_booster().num_boosted_rounds()
        pred=recipe.inverse_forecast_target(raw,compact['val_features'],mode='growth_log')
    assert np.isfinite(pred).all() and (pred>0).all() and len(pred)==18657
    return estimator,pred,int(used),params,columns

def save_model(estimator,path,model):
    path.parent.mkdir(parents=True,exist_ok=True)
    if model=='catboost_log_level':path=path.with_suffix('.cbm');estimator.save_model(str(path))
    elif model.startswith('lightgbm'):path=path.with_suffix('.txt');path.write_text(estimator.booster_.model_to_string(),encoding='utf-8')
    else:path=path.with_suffix('.ubj');path.write_bytes(estimator.get_booster().save_raw(raw_format='ubj'))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()

def execute(model,fold,config,protocol='rolling',threads=4):
    out=HERE/'results'/protocol/fold/model;out.mkdir(parents=True,exist_ok=True)
    target=out/(config+'.csv');metadata=target.with_suffix('.json')
    if metadata.exists():
        old=json.loads(metadata.read_text());assert old['prediction_sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
        print('REUSED',protocol,fold,model,config,old['mape'],flush=True);return old
    start=time.perf_counter();recipe=load_recipe();extras,groups=read_extras(model)
    inner_fold=previous_month(fold)
    year,month=map(int,inner_fold.split('-'));inner=get_fold(recipe,year,month)
    # Identical feature interventions use the same cached inner fit.
    actual=OLD[model] if config=='previous' else config
    inner_dir=HERE/'inner'/inner_fold/model;inner_dir.mkdir(parents=True,exist_ok=True)
    count_file=inner_dir/(actual+'.json');inner_csv=count_file.with_suffix('.csv')
    if count_file.exists():
        selected=json.loads(count_file.read_text());assert selected['prediction_sha256']==hashlib.sha256(inner_csv.read_bytes()).hexdigest()
        trees=selected['trees'];earlier=pd.read_csv(inner_csv)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            estimator,pred,trees,params,columns=fit(recipe,inner,model,actual,extras,groups,threads=threads)
        earlier=inner['val_features'][['new_id','year_month_index','region','settlement','pto']].copy();earlier['prediction']=pred
        earlier.to_csv(inner_csv,index=False)
        model_path,digest=save_model(estimator,inner_dir/'models'/actual,model)
        selected=dict(fold=inner_fold,model=model,configuration=actual,trees=trees,
            early_stopping=True,stopping_target_known_before=fold,parameters=params,features_added=columns,
            model_file=model_path.relative_to(HERE).as_posix(),model_sha256=digest,
            prediction_sha256=hashlib.sha256(inner_csv.read_bytes()).hexdigest(),**recipe._make_metrics(earlier.pto,pred))
        count_file.write_text(json.dumps(selected,indent=2));del estimator;gc.collect()
    del inner;gc.collect()
    year,month=map(int,fold.split('-'));outer=get_fold(recipe,year,month)
    assert earlier.year_month_index.max()<outer['val_features'].year_month_index.min()
    count=trees if protocol=='rolling' else COUNTS[model]
    print('FIT',protocol,fold,model,config,'trees',count,flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        estimator,pred,used,params,columns=fit(recipe,outer,model,actual,extras,groups,trees=count,threads=threads)
    assert used==count
    correction=None
    if config=='previous' and model=='catboost_log_level':
        signal=float(np.log(earlier.pto.to_numpy()/earlier.prediction.to_numpy()).clip(-.2,.2).mean())
        factor=float(np.exp(.5*signal));pred=pred*factor
        correction=dict(prior_fold=inner_fold,coefficient=.5,clipped_mean_log_error=signal,factor=factor,
            prior_prediction_sha256=hashlib.sha256(inner_csv.read_bytes()).hexdigest(),spatial=False)
    anchor=outer['val_features'][['new_id','year_month_index','region','settlement','pto']].copy();anchor['prediction']=pred
    anchor.to_csv(target,index=False);model_path,digest=save_model(estimator,out/'models'/config,model)
    info=dict(model=model,fold=fold,configuration=config,underlying_configuration=actual,protocol=protocol,
        trees=count,inner_selected_trees=trees,inner_fold=inner_fold,outer_target_used_for_stopping=False,
        source_inner_sha256=hashlib.sha256(count_file.read_bytes()).hexdigest(),parameters=params,features_added=columns,
        correction=correction,seconds=time.perf_counter()-start,notebook_sha256=NOTEBOOK_SHA256,
        implementation_sha256=IMPLEMENTATION_SHA256,
        feature_checks=json.loads((HERE/'feature_checks.json').read_text()),validation_only=True,
        model_file=model_path.relative_to(HERE).as_posix(),model_sha256=digest,
        prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        feature_cache_sha256=hashlib.sha256((HERE/'cache/features.parquet').read_bytes()).hexdigest(),
        environment={name:importlib.metadata.version(name) for name in ['numpy','pandas','catboost','lightgbm','xgboost','scipy','scikit-learn']},
        **recipe._make_metrics(anchor.pto,pred))
    metadata.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('RESULT',protocol,fold,model,config,round(info['mape'],6),'seconds',round(info['seconds'],1),flush=True)
    del outer,extras,estimator;gc.collect();return info

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=MODELS[:-1],choices=MODELS)
    parser.add_argument('--folds',nargs='+',default=['2024-08']);parser.add_argument('--configs',nargs='+',default=SCREEN)
    parser.add_argument('--protocol',choices=['rolling','full'],default='rolling');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    for model in args.models:
        for fold in args.folds:
            for config in args.configs:execute(model,fold,config,args.protocol,args.threads)

if __name__=='__main__':main()
