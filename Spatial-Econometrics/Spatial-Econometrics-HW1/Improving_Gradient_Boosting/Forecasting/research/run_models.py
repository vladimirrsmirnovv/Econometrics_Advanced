"""Resumable matched CPU experiments; CUDA XGBoost is handled separately."""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import ROOT, MODELS, load_recipe, fit_validation, matrices, NOTEBOOK_SHA256
from run_validation import get_fold
from spatial_features import align_features

DEFAULT_CONFIGS=['knn8_history','exp15_history','inverse75_history','binary75_history',
 'economic8_history','hybrid8_history','growth_exp50','season_exp50','multihop',
 'wx_relative','regimes','lisa','eigenvectors','hybrid_slx','hybrid_gwr',
 'within_market','density_weight']

def checks_and_data():
    manifest=json.loads((HERE/'cache/feature_manifest.json').read_text())
    checks=json.loads((HERE/'feature_checks.json').read_text())
    assert checks['status']=='passed' and checks['implementation_hashes']==manifest['implementation_hashes']
    for name,digest in manifest['implementation_hashes'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    return manifest,pd.read_parquet(HERE/'cache/features.parquet')

def fit_weighted(recipe,compact,model,weight,threads=4):
    xtrain,xvalid,ytrain,yvalid,cats=matrices(recipe,compact,model)
    params=dict(getattr(recipe,model+'_params'))
    if model=='catboost_log_level':
        from catboost import CatBoostRegressor,Pool
        params.update(thread_count=threads,allow_writing_files=False,verbose=False)
        estimator=CatBoostRegressor(**params)
        estimator.fit(Pool(xtrain,ytrain,cat_features=cats,weight=weight),
            eval_set=Pool(xvalid,yvalid,cat_features=cats),early_stopping_rounds=50,use_best_model=True)
        pred=recipe._cb_inverse_log1p(estimator.predict(xvalid));best=estimator.get_best_iteration()
    elif model.startswith('lightgbm'):
        import lightgbm as lgb
        params=recipe._clean_lgbm_params(params);params.update(n_jobs=threads,verbosity=-1)
        estimator=lgb.LGBMRegressor(**params)
        estimator.fit(xtrain,ytrain,sample_weight=weight,eval_set=[(xvalid,yvalid)],
            eval_metric='mae',callbacks=[lgb.early_stopping(50,verbose=False)])
        mode='growth_log' if model.endswith('log_diff') else 'direct_log1p'
        pred=recipe.inverse_forecast_target(estimator.predict(xvalid),compact['val_features'],mode=mode)
        best=estimator.best_iteration_
    else:raise ValueError('Use the CUDA research runner for XGBoost')
    metrics=recipe._make_metrics(compact['val_features'].pto,pred)
    metrics.update(model=model,best_iteration=int(best),features=len(xtrain.columns),threads=threads)
    return estimator,pred,metrics

def baseline(model,fold):
    directory=ROOT/'Forecasting/results'/fold
    return (json.loads((directory/(model+'.json')).read_text(encoding='utf-8')),
            pd.read_csv(directory/(model+'.csv')))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=MODELS[:-1],default=MODELS[:-1])
    parser.add_argument('--folds',nargs='+',default=['2025-02'])
    parser.add_argument('--configs',nargs='+',default=DEFAULT_CONFIGS)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    recipe=load_recipe();manifest,table=checks_and_data()
    stores=pd.read_parquet(HERE/'cache/stores.parquet').set_index('new_id')
    environment={name:importlib.metadata.version(name) for name in
                 ['pandas','numpy','catboost','lightgbm','xgboost']}
    for fold in args.folds:
        year,month=map(int,fold.split('-'));compact=get_fold(recipe,year,month)
        for model in args.models:
            original,original_predictions=baseline(model,fold)
            out=HERE/'results'/fold/model;out.mkdir(parents=True,exist_ok=True)
            if not (out/'baseline.json').exists():
                original_predictions.to_csv(out/'baseline.csv',index=False)
                (out/'baseline.json').write_text(json.dumps(dict(original,
                    configuration='baseline',reused_existing=True,backend='CPU',
                    source_result=str((ROOT/'Forecasting/results'/fold/(model+'.json')).relative_to(ROOT))),indent=2))
            for config in args.configs:
                if (out/(config+'.json')).exists():continue
                fit_compact=compact
                parts=config.split('+')
                if parts[0]=='key':
                    from key_folds import get_key_fold
                    key_check=json.loads((HERE/'key_feature_checks.json').read_text())
                    assert key_check['status']=='passed'
                    fit_compact=get_key_fold(recipe,year,month)
                    parts=parts[1:]
                # These exactly specified families have already been fitted.
                legacy={'exp50_history':'exp50_history','wx_means':'exp50_static','region_history':'region_history'}
                old=ROOT/'Forecasting/results'/fold/(model+'_'+legacy.get(config,'missing')+'.json')
                if config in legacy and old.exists():
                    row=json.loads(old.read_text(encoding='utf-8'))
                    pd.read_csv(old.with_suffix('.csv')).to_csv(out/(config+'.csv'),index=False)
                    row.update(configuration=config,reused_existing=True,backend='CPU')
                    (out/(config+'.json')).write_text(json.dumps(row,indent=2))
                    print('REUSED',fold,model,config,row['mape'],flush=True);continue
                # Union groups in a named combination; base hyperparameters stay fixed.
                extra_columns=[] if config=='density_weight' else list(dict.fromkeys(
                    column for group in parts for column in manifest['groups'][group]))
                extra_train,extra_valid=(None,None) if not extra_columns else align_features(fit_compact,table,extra_columns)
                started=time.perf_counter()
                print('START',fold,model,config,len(extra_columns),flush=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    if config=='density_weight':
                        counts=fit_compact['train_features'].new_id.map(stores.market_shops).to_numpy()
                        weight=1/np.sqrt(counts);weight/=weight.mean()
                        estimator,pred,metrics=fit_weighted(recipe,fit_compact,model,weight,args.threads)
                    else:
                        estimator,pred,metrics=fit_validation(recipe,fit_compact,model,
                            extra_train=extra_train,extra_valid=extra_valid,threads=args.threads)
                assert len(pred)==18657 and np.isfinite(pred).all() and (pred>=0).all()
                anchor=fit_compact['val_features'][['new_id','year_month_index','region','settlement','pto']].copy()
                anchor['prediction']=pred
                old_anchor=original_predictions.set_index('new_id').reindex(anchor.new_id)
                old_ape=np.abs(old_anchor.prediction.to_numpy()-anchor.pto.to_numpy())/anchor.pto.to_numpy()*100
                new_ape=np.abs(pred-anchor.pto.to_numpy())/anchor.pto.to_numpy()*100
                metrics.update(configuration=config,fold=fold,backend='CPU',seconds=time.perf_counter()-started,
                    baseline_mape=original['mape'],delta_mape=metrics['mape']-original['mape'],
                    relative_improvement_pct=(original['mape']-metrics['mape'])/original['mape']*100,
                    shops_improved_pct=float((new_ape<old_ape).mean()*100),
                    ape_change_quantiles={str(q):float(np.quantile(new_ape-old_ape,q)) for q in [.05,.25,.5,.75,.95]},
                    features_added=extra_columns,feature_version=manifest['feature_version'],
                    implementation_hashes=manifest['implementation_hashes'],leakage_check='passed',
                    parameters=dict(getattr(recipe,model+'_params')),environment=environment,
                    notebook_sha256=NOTEBOOK_SHA256,validation_only=True)
                anchor.to_csv(out/(config+'.csv'),index=False)
                (out/(config+'.json')).write_text(json.dumps(metrics,indent=2))
                # Candidate fits are reproducible; preserve light/native models.
                model_dir=out/'models';model_dir.mkdir(exist_ok=True)
                if model=='catboost_log_level':estimator.save_model(str(model_dir/(config+'.cbm')))
                else:(model_dir/(config+'.txt')).write_text(estimator.booster_.model_to_string(),encoding='utf-8')
                del estimator,extra_train,extra_valid;gc.collect()
                print('RESULT',fold,model,config,round(metrics['mape'],6),
                    'delta',round(metrics['delta_mape'],6),'seconds',round(metrics['seconds'],1),flush=True)
        del compact;gc.collect()

if __name__=='__main__':main()
