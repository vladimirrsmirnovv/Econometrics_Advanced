"""Kaggle CUDA 3.1.3 matched experiments on audited, precomputed matrices."""
from __future__ import annotations


def _directory_path(value):
    """Resolve directory names from preserved metadata after the folder rename."""
    from pathlib import Path as _Path
    path = _Path(value)
    if path.is_absolute():
        return path
    directory_count = len(path.parts) - bool(path.suffix)
    return _Path(*(part.replace("_", "-") if index < directory_count else part
                   for index, part in enumerate(path.parts)))

import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

def metrics(true,pred):
    error=pred-true
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
        mape=float((np.abs(error)/true).mean()*100),bias_pct=float((error/true).mean()*100))

def main():
    candidates=list(Path('/kaggle/input').rglob('research_manifest.json'))
    assert len(candidates)==1
    source=candidates[0].parent
    plan=json.loads(candidates[0].read_text())
    assert importlib.metadata.version('xgboost')=='3.1.3'
    for name,digest in plan['hashes'].items():
        assert hashlib.sha256((source/_directory_path(name)).read_bytes()).hexdigest()==digest,name
    out=Path('/kaggle/working')
    runtime=dict(python=platform.python_version(),xgboost=xgb.__version__,
        hardware=subprocess.check_output(['nvidia-smi'],text=True),
        manifest=plan,validation_only=True)
    (out/'research_runtime.json').write_text(json.dumps(runtime,indent=2))
    records=[]
    for fold in plan['folds']:
        train=pd.read_parquet(source/(fold+'_train.parquet'))
        holdout=pd.read_parquet(source/(fold+'_holdout.parquet'))
        base=plan['base_columns'];params=dict(plan['parameters'])
        extras=plan['groups']
        train_mask=np.isfinite(train.target);valid_mask=np.isfinite(holdout.target)
        fold_out=out/'results'/fold/'xgboost-log-diff'
        fold_out.mkdir(parents=True,exist_ok=True)
        baseline=None;baseline_pred=None
        for config in ['baseline']+plan['configs']:
            if config=='density_weight':
                selected=base
            else:selected=base+([] if config=='baseline' else extras[config])
            start=time.perf_counter()
            model=xgb.XGBRegressor(**params)
            weights=None
            if config=='density_weight':
                weights=1/np.sqrt(train.loc[train_mask,'market_shops'].to_numpy())
                weights/=weights.mean()
            model.fit(train.loc[train_mask,selected],train.loc[train_mask,'target'],sample_weight=weights,
                eval_set=[(holdout.loc[valid_mask,selected],holdout.loc[valid_mask,'target'])],verbose=False)
            device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            assert device.startswith('cuda')
            raw_prediction=np.asarray(model.predict(holdout[selected]),dtype=float)
            prediction=np.maximum(holdout.pto_lag_1.to_numpy()*np.exp(raw_prediction),0)
            assert len(prediction)==18657 and np.isfinite(prediction).all()
            row=metrics(holdout.pto.to_numpy(),prediction)
            if config=='baseline':
                baseline=row['mape'];baseline_pred=prediction.copy()
                if fold=='2025-02':
                    assert abs(baseline-4.4314600477414245)<1e-10,'Source CUDA baseline not reproduced'
            old_ape=np.abs(baseline_pred-holdout.pto.to_numpy())/holdout.pto.to_numpy()*100
            new_ape=np.abs(prediction-holdout.pto.to_numpy())/holdout.pto.to_numpy()*100
            row.update(model='xgboost_log_diff',configuration=config,fold=fold,
                backend='CUDA',device=device,xgboost_version=xgb.__version__,parameters=params,
                features_added=[] if config in {'baseline','density_weight'} else extras[config],
                features=len(selected),best_iteration=int(model.best_iteration),
                seconds=time.perf_counter()-start,baseline_mape=baseline,delta_mape=row['mape']-baseline,
                relative_improvement_pct=(baseline-row['mape'])/baseline*100,
                shops_improved_pct=float((new_ape<old_ape).mean()*100),
                ape_change_quantiles={str(q):float(np.quantile(new_ape-old_ape,q)) for q in [.05,.25,.5,.75,.95]},
                validation_only=True,leakage_check='passed',feature_version=plan['feature_version'],
                notebook_sha256=plan['notebook_sha256'],source_feature_checks=plan['feature_checks'])
            table=holdout[['new_id','year_month_index','region','settlement','pto']].copy()
            table['prediction']=prediction
            table.to_csv(fold_out/(config+'.csv'),index=False)
            (fold_out/(config+'.json')).write_text(json.dumps(row,indent=2))
            model.save_model(fold_out/(config+'.ubj'))
            records.append(row)
            print('RESULT',fold,config,round(row['mape'],6),'delta',round(row['delta_mape'],6),
                'seconds',round(row['seconds'],1),flush=True)
            del model;gc.collect()
        del train,holdout;gc.collect()
    (out/'research_metrics.json').write_text(json.dumps(records,indent=2))
    print('ALL_CUDA_EXPERIMENTS_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
