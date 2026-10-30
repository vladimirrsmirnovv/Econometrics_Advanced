"""Full-history original fits plus the configurations selected on validation."""
from pathlib import Path
import argparse,gc,hashlib,importlib.metadata,json,sys,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import ROOT,MODELS,NOTEBOOK_SHA256,load_recipe
CACHE=HERE.parent/'cache/final_2025-03'
OUT=HERE.parent/'submissions/2025-03'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',choices=MODELS[:-1],default=MODELS[:-1])
    parser.add_argument('--variants',nargs='+',choices=['baseline','spatial'],default=['baseline','spatial'])
    parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    recipe=load_recipe();manifest=json.loads((CACHE/'manifest.json').read_text())
    checks=json.loads((CACHE/'feature_checks.json').read_text());assert checks['status']=='passed'
    configs=json.loads((HERE.parent/'research/ready_configurations.json').read_text())['individual']
    versions={k:importlib.metadata.version(k) for k in ['pandas','numpy','catboost','lightgbm']}
    for model in args.models:
        requested={'new_id','year_month_index','pto','pto_lag_1'}
        for name in getattr(recipe,model+'_features'):
            requested.add(name[:-3] if name.endswith('_cb') else name)
            if name.endswith('_cat_cb'):requested.add(name[:-7])
        import pyarrow.parquet as pq
        compact={'train_features':None,'val_features':None,
            'categorical_cols':manifest['categorical_cols'],'feature_cols':manifest['feature_cols']}
        for key in ['test_train_features','test_features']:
            available=pq.read_schema(CACHE/(key+'.parquet')).names
            compact[key]=pd.read_parquet(CACHE/(key+'.parquet'),columns=[c for c in available if c in requested])
        train=compact['test_train_features'];test=compact['test_features']
        if model=='catboost_log_level':
            features=list(recipe.catboost_log_level_features)
            cats=[c for c in features if pd.api.types.is_object_dtype(train[c]) or
                pd.api.types.is_string_dtype(train[c]) or isinstance(train[c].dtype,pd.CategoricalDtype)]
            cats=list(dict.fromkeys(cats+[c for c in ['new_id','month','prev_month'] if c in features]))
            xtrain=recipe._cb_prepare_X(train,features,cats,name='march_full_train')
            xtest=recipe._cb_prepare_X(test,features,cats,name='march_forecast')
            ytrain=recipe.make_forecast_target(train,mode='direct_log1p')
        else:
            features=list(getattr(recipe,model+'_features'));cats=[]
            pack=recipe._prepare_numeric_model_pack(compact,selected_features=features,
                target_mode='growth_log' if model.endswith('log_diff') else 'direct_log1p',label=model,verbose=False)['test']
            xtrain=pack['X_train'][features];xtest=pack['X_test'][features];ytrain=pack['y_train']
            del pack;gc.collect()
        for variant in args.variants:
            directory=OUT/variant;directory.mkdir(parents=True,exist_ok=True)
            target=directory/('test_'+model+'.csv');info=target.with_suffix('.json')
            if info.exists():print('REUSED_FULL',model,variant,flush=True);continue
            spec=configs[model];params=dict(spec['full_parameters'])
            params['thread_count' if model=='catboost_log_level' else 'n_jobs']=args.threads
            extra=[];weight=None;correction=None
            if variant=='spatial':
                if model=='catboost_log_level':
                    extra=spec['features_added'];spatial=pd.read_parquet(CACHE/'legacy_features.parquet')
                elif model=='lightgbm_log_diff':
                    extra=spec['features_added'];spatial=pd.read_parquet(CACHE/'research_features.parquet')
                else:
                    counts=pd.read_parquet(CACHE/'stores.parquet').set_index('new_id').market_shops
                    weight=1/np.sqrt(train.new_id.map(counts).to_numpy());weight/=weight.mean()
                if extra:
                    lookup=spatial.set_index(['new_id','year_month_index'])
                    parts=[]
                    for meta,x in [(train,xtrain),(test,xtest)]:
                        idx=pd.MultiIndex.from_frame(meta[['new_id','year_month_index']])
                        a=lookup.reindex(idx)[extra].reset_index(drop=True).set_axis(x.index)
                        parts.append(pd.concat([x,a],axis=1))
                    fit_train,fit_test=parts
                else:fit_train,fit_test=xtrain,xtest
            else:fit_train,fit_test=xtrain,xtest
            start=time.perf_counter();print('FULL_TRAIN',model,variant,fit_train.shape,flush=True)
            if model=='catboost_log_level':
                from catboost import CatBoostRegressor,Pool
                params.update(verbose=500)
                estimator=CatBoostRegressor(**params)
                estimator.fit(Pool(fit_train,ytrain,cat_features=cats))
                raw=estimator.predict(fit_test);pred=recipe._cb_inverse_log1p(raw)
                trees=estimator.tree_count_;extension='cbm'
            else:
                from lightgbm import LGBMRegressor
                params.update(verbosity=-1)
                estimator=LGBMRegressor(**params);estimator.fit(fit_train,ytrain,sample_weight=weight)
                raw=estimator.predict(fit_test)
                pred=recipe.inverse_forecast_target(raw,test,mode='growth_log' if model.endswith('log_diff') else 'direct_log1p')
                trees=estimator.booster_.current_iteration();extension='txt'
            assert trees==spec['full_iterations'],(model,trees,spec['full_iterations'])
            if variant=='spatial' and model=='catboost_log_level':
                pure=pd.DataFrame({'new_id':test.new_id.to_numpy(),'rto':pred})
                diagnostics=OUT/'diagnostics';diagnostics.mkdir(exist_ok=True)
                pure.to_csv(diagnostics/'catboost_history_before_global_correction.csv',index=False)
                previous=HERE.parent/'results/2025-02/catboost_log_level_exp50_history.csv'
                earlier=pd.read_csv(previous)
                assert len(earlier)==18657 and earlier.new_id.is_unique
                assert json.loads(previous.with_suffix('.json').read_text())['month']==2
                signal=float(np.log(earlier.pto.to_numpy()/earlier.prediction.to_numpy()).clip(-.2,.2).mean())
                pred=pred*np.exp(.5*signal)
                correction=dict(previous_oof_month='2025-02',previous_oof_configuration='exp50_history',
                    previous_oof_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
                    clipped_mean_log_error=signal,correction_weight=.5,multiplicative_factor=float(np.exp(.5*signal)),
                    march_actuals_used=False)
            assert len(pred)==18657 and np.isfinite(pred).all() and (pred>=0).all()
            result=pd.DataFrame({'new_id':test.new_id.to_numpy(),'rto':pred})
            result.to_csv(target,index=False)
            models=OUT/'models'/variant;models.mkdir(parents=True,exist_ok=True)
            model_file=models/(model+'.'+extension)
            if model=='catboost_log_level':estimator.save_model(str(model_file))
            else:model_file.write_text(estimator.booster_.model_to_string(),encoding='utf-8')
            metadata=dict(model=model,variant=variant,configuration='baseline' if variant=='baseline' else spec['configuration'],
                forecast_month='2025-03',train_rows=len(fit_train),training_months=['2023-01','2025-02'],
                features=features+extra,parameters=params,trees=trees,early_stopping=False,
                seconds=time.perf_counter()-start,versions=versions,backend='CPU',
                notebook_sha256=NOTEBOOK_SHA256,raw_sha256=manifest['raw_sha256'],
                feature_checks=checks,correction=correction,training_weights=spec.get('training_weight') if weight is not None else None,
                model_file=str(model_file.relative_to(OUT)),model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
                submission_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),validation_only=False)
            info.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            print('FULL_DONE',model,variant,trees,len(pred),flush=True)
            del estimator,fit_train,fit_test;gc.collect()
        del xtrain,xtest,ytrain,compact,train,test;gc.collect()

if __name__=='__main__':main()
