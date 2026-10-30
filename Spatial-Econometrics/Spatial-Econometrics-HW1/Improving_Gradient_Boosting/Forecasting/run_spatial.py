"""Matched baseline/spatial experiments on the preserved notebook holdouts."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from notebook_api import ROOT, MODELS, load_recipe, read_raw, fit_validation
from run_validation import get_fold
from spatial_features import spatial_table, align_features, attach_locations, make_weights


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--year',type=int,default=2025)
    parser.add_argument('--month',type=int,default=2)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=['xgboost_log_diff'])
    parser.add_argument('--weights',choices=['exp50','knn8','region'],default='exp50')
    parser.add_argument('--groups',nargs='+',choices=['history','static','both'],default=['history','static'])
    args=parser.parse_args()
    recipe=load_recipe()
    raw=read_raw(recipe)
    feature_cache=ROOT/f'Forecasting/cache/spatial_{args.weights}.parquet'
    metadata_cache=feature_cache.with_suffix('.json')
    if feature_cache.exists() and metadata_cache.exists():
        table=pd.read_parquet(feature_cache)
        info=json.loads(metadata_cache.read_text(encoding='utf-8'))
    else:
        table,info=spatial_table(raw,kind=args.weights)
        table.to_parquet(feature_cache,index=False)
        metadata_cache.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    del raw;gc.collect()
    compact=get_fold(recipe,args.year,args.month)
    out=ROOT/f'Forecasting/results/{args.year:04d}-{args.month:02d}'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model in args.models:
        baseline_path=out/f'{model}.json'
        if not baseline_path.exists():raise RuntimeError(f'Run baseline for {model} first: {baseline_path}')
        baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
        for group in args.groups:
            name=f'{model}_{args.weights}_{group}'
            if (out/f'{name}.json').exists():
                rows.append(json.loads((out/f'{name}.json').read_text(encoding='utf-8')))
                continue
            columns=[c for c in table.columns if (group=='both' and c.startswith(('wx_','wy_')))
                     or (group=='history' and c.startswith('wy_')) or (group=='static' and c.startswith('wx_'))]
            train_extra,valid_extra=align_features(compact,table,columns)
            started=time.perf_counter();print('EXPERIMENT',name,columns,flush=True)
            estimator,prediction,metrics=fit_validation(recipe,compact,model,
                             extra_train=train_extra,extra_valid=valid_extra)
            assert np.isfinite(prediction).all()
            metrics.update(experiment=name,year=args.year,month=args.month,
                           seconds=time.perf_counter()-started,baseline_mape=baseline['mape'],
                           delta_mape=metrics['mape']-baseline['mape'],
                           extra_features=columns,spatial_metadata=info)
            validation=compact['val_features'][['new_id','region','settlement','pto']].copy()
            validation['prediction']=prediction
            validation.to_csv(out/f'{name}.csv',index=False)
            (out/f'{name}.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
            del estimator,train_extra,valid_extra;gc.collect()
            rows.append(metrics)
            print('RESULT',json.dumps(metrics,ensure_ascii=False),flush=True)
    pd.DataFrame(rows).to_csv(out/f'spatial_{args.weights}_metrics.csv',index=False)


if __name__=='__main__':main()
