"""Frozen configurations, tree counts selected on February, and unknown March targets."""


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
import argparse,gc,hashlib,json,sys,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from notebook_api import MODELS,load_recipe
from run_validation import get_fold
from experiment import OLD,fit,read_extras,save_model,data

def run(model,threads=4,configuration=None):
    frozen=json.loads((HERE/'frozen_final_selection.json').read_text());config=configuration or frozen['individual'][model]
    root_out=HERE.parent/'submissions/2025-03-stage2'
    out=root_out/'individual' if configuration is None else root_out/'components'/configuration.replace("_", "-")
    out.mkdir(parents=True,exist_ok=True)
    target=out/('test_'+model+'.csv')
    if target.with_suffix('.json').exists():
        info=json.loads(target.with_suffix('.json').read_text());assert info['submission_sha256']==hashlib.sha256(target.read_bytes()).hexdigest();return
    if configuration==frozen['individual'][model] and (root_out/'individual'/target.name).exists():
        original=root_out/'individual'/target.name;target.write_bytes(original.read_bytes())
        info=json.loads(original.with_suffix('.json').read_text());info['component_alias_of']='individual/'+target.name
        target.with_suffix('.json').write_text(json.dumps(info,indent=2));return
    start=time.perf_counter();recipe=load_recipe();extras,groups=read_extras(model)
    old_future=pd.read_parquet(HERE/'cache/old_future.parquet').set_index(['new_id','year_month_index'])
    for c in old_future:extras.loc[old_future.index,c]=old_future[c]
    actual=OLD[model] if config=='previous' else config
    feb=get_fold(recipe,2025,2)
    estimator,previous,trees,params,columns=fit(recipe,feb,model,actual,extras,groups,threads=threads)
    feb_predictions=feb['val_features'][['new_id','year_month_index','pto']].copy();feb_predictions['prediction']=previous
    prior=out/('february_inner_'+model+'.csv');feb_predictions.to_csv(prior,index=False)
    del estimator,feb;gc.collect()
    cache=HERE.parent/'cache/final_2025-03';manifest=json.loads((cache/'manifest.json').read_text())
    full=pd.read_parquet(cache/'test_train_features.parquet');test=pd.read_parquet(cache/'test_features.parquet')
    assert full.year_month_index.max()==25 and (test.year_month_index==26).all() and test.pto.isna().all()
    compact=dict(train_features=full,val_features=test,test_train_features=None,test_features=None,
        categorical_cols=manifest['categorical_cols'],feature_cols=manifest['feature_cols'])
    # Legacy CatBoost H4 was previously constructed for March and verified exactly.
    if model=='catboost_log_level' and 'exp50_history' in actual:
        legacy=pd.read_parquet(cache/'legacy_features.parquet').set_index(['new_id','year_month_index'])
        for c in groups['exp50_history']:extras.loc[legacy.index,c]=legacy[c]
    estimator,pred,used,params,columns=fit(recipe,compact,model,actual,extras,groups,trees=trees,threads=threads)
    correction=None
    if config=='previous' and model=='catboost_log_level':
        signal=float(np.log(feb_predictions.pto.to_numpy()/previous).clip(-.2,.2).mean());factor=float(np.exp(.5*signal))
        pred*=factor;correction=dict(source_month='2025-02',coefficient=.5,factor=factor,spatial=False)
    assert used==trees and len(pred)==18657 and np.isfinite(pred).all() and (pred>0).all()
    pd.DataFrame(dict(new_id=test.new_id.to_numpy(),rto=pred)).to_csv(target,index=False)
    models=root_out/'models'/('selected' if configuration is None else configuration);path,digest=save_model(estimator,models/model.replace("_", "-"),model)
    # Verify the actual saved native model against the submitted predictions.
    xt,xv,yt,yv,cats,weights,added=data(recipe,compact,model,actual,extras,groups);del xt,yt,yv,weights;gc.collect()
    if model=='catboost_log_level':
        from catboost import CatBoostRegressor
        restored=CatBoostRegressor();restored.load_model(str(path));assert restored.tree_count_==trees
        replay=recipe._cb_inverse_log1p(restored.predict(xv))
    elif model.startswith('lightgbm'):
        import lightgbm as lgb
        restored=lgb.Booster(model_str=path.read_text());assert restored.current_iteration()==trees
        replay=recipe.inverse_forecast_target(restored.predict(xv,num_threads=threads),test,
            mode='growth_log' if model.endswith('log_diff') else 'direct_log1p')
    else:
        import xgboost as xgb
        restored=xgb.XGBRegressor();restored.load_model(path);restored.set_params(device='cuda',n_jobs=threads)
        assert restored.get_booster().num_boosted_rounds()==trees
        replay=recipe.inverse_forecast_target(restored.predict(xv),test,mode='growth_log')
    if correction:replay*=correction['factor']
    np.testing.assert_allclose(replay,pred,rtol=1e-12,atol=1e-9)
    info=dict(model=model,configuration=config,underlying_configuration=actual,forecast_month='2025-03',
        trees=trees,tree_count_selected_on='2025-02',early_stopping_on_march=False,
        actual_march_target_available=False,full_training_months=['2023-01','2025-02'],
        train_rows=len(full),features_added=columns,parameters=params,correction=correction,
        frozen_selection_sha256=hashlib.sha256((HERE/'frozen_final_selection.json').read_bytes()).hexdigest(),
        source_inner_prediction_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
        model_file=path.relative_to(root_out).as_posix(),model_sha256=digest,
        submission_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),seconds=time.perf_counter()-start,
        original_full_baseline_preserved=True,native_reload_prediction_check=True,
        native_reload_max_relative_difference=float(np.max(abs(replay/_directory_path(pred)-1))))
    target.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print('NEW_MARCH_DONE',model,config,trees,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=MODELS[:-1]);args=parser.parse_args()
    for model in args.models:run(model)
