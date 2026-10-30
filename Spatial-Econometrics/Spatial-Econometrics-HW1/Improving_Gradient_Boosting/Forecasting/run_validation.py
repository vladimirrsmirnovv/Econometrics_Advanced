"""Reproduce notebook holdouts, with one original feature build per month."""
from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path


import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from notebook_api import (ROOT, NOTEBOOK_SHA256, MODELS, load_recipe, read_raw,
                          build_fold, compact_fold, fit_validation)


def get_fold(recipe, year, month):
    cache = ROOT / f'Forecasting/cache/{year:04d}-{month:02d}'
    source_hash = hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest()
    if (cache/'manifest.json').exists():
        manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['notebook_sha256']==NOTEBOOK_SHA256 and manifest['raw_sha256']==source_hash
        output = {'train_features':pd.read_parquet(cache/'train.parquet'),
                  'val_features':pd.read_parquet(cache/'holdout.parquet'),
                  'test_train_features':None,'test_features':None,
                  'categorical_cols':manifest['categorical_cols'],
                  'feature_cols':manifest['feature_cols']}
        return output
    raw=read_raw(recipe)
    print(f'Building original features for {year}-{month:02d}',flush=True)
    original=build_fold(recipe,raw,year,month)
    compact=compact_fold(recipe,original)
    manifest={'notebook_sha256':NOTEBOOK_SHA256,'raw_sha256':source_hash,
              'categorical_cols':compact['categorical_cols'],'feature_cols':compact['feature_cols'],
              'original_train_shape':list(original['train_features'].shape),
              'original_holdout_shape':list(original['val_features'].shape)}
    del original,raw
    gc.collect()
    cache.mkdir(parents=True,exist_ok=True)
    compact['train_features'].to_parquet(cache/'train.parquet',index=False)
    compact['val_features'].to_parquet(cache/'holdout.parquet',index=False)
    (cache/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return compact


def original_blends(predictions):
    # The notebook final submission is a log mean of these four candidate blends.
    # Match the original gating computation directly, without generating files.
    matrix=np.column_stack([predictions[m] for m in MODELS])
    arithmetic=matrix.mean(axis=1)
    logmean=np.expm1(np.log1p(np.maximum(matrix,0)).mean(axis=1))
    ratio=matrix.max(axis=1)/np.maximum(matrix.min(axis=1),1e-9)
    gated=np.where(ratio>1.08,np.median(matrix,axis=1),arithmetic)
    no_weak=matrix[:,[0,2,3]]
    ratio_no_weak=no_weak.max(axis=1)/np.maximum(no_weak.min(axis=1),1e-9)
    gated_no_weak=np.where(ratio_no_weak>1.08,np.median(no_weak,axis=1),no_weak.mean(axis=1))
    candidates=np.column_stack([arithmetic,logmean,gated,gated_no_weak])
    meta=np.expm1(np.log1p(np.maximum(candidates,0)).mean(axis=1))
    return {'equal':arithmetic,'log_average':logmean,'gated_1_08_all':gated,
            'gated_1_08_no_weak':gated_no_weak,'final_meta_logmean':meta}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--year',type=int,default=2025)
    parser.add_argument('--month',type=int,default=2)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    recipe=load_recipe()
    compact=get_fold(recipe,args.year,args.month)
    assert compact['train_features'].year_month_index.max()<compact['val_features'].year_month_index.min()
    if args.prepare_only:
        print('Compact feature shapes',compact['train_features'].shape,compact['val_features'].shape)
        return
    out=ROOT/f'Forecasting/results/{args.year:04d}-{args.month:02d}'
    out.mkdir(parents=True,exist_ok=True)
    predictions={}
    metrics=[]
    for model in args.models:
        started=time.perf_counter()
        if (out/f'{model}.json').exists() and (out/f'{model}.csv').exists():
            row=json.loads((out/f'{model}.json').read_text(encoding='utf-8'))
            predictions[model]=pd.read_csv(out/f'{model}.csv').prediction.to_numpy()
        else:
            print('TRAIN',model,flush=True)
            estimator,pred,row=fit_validation(recipe,compact,model)
            assert np.isfinite(pred).all() and len(pred)==18657
            row.update(year=args.year,month=args.month,seconds=time.perf_counter()-started,
                       validation_only=True,notebook_sha256=NOTEBOOK_SHA256,
                       environment={k:importlib.metadata.version(k) for k in ['pandas','numpy','catboost','lightgbm','xgboost','scikit-learn']})
            table=compact['val_features'][['new_id','year_month_index','region','settlement','pto']].copy()
            table['prediction']=pred
            table.to_csv(out/f'{model}.csv',index=False)
            (out/f'{model}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
            predictions[model]=pred
            model_dir=out/'models';model_dir.mkdir(exist_ok=True)
            if model=='catboost_log_level':estimator.save_model(str(model_dir/f'{model}.cbm'))
            elif model.startswith('lightgbm'):
                # Python handles Windows Unicode paths that native fopen misses.
                (model_dir/f'{model}.txt').write_text(estimator.booster_.model_to_string(),encoding='utf-8')
            else:(model_dir/f'{model}.ubj').write_bytes(estimator.get_booster().save_raw(raw_format='ubj'))
            del estimator
            gc.collect()
        metrics.append(row)
        print('RESULT',json.dumps(row,ensure_ascii=False),flush=True)
    if all(m in predictions for m in MODELS):
        for name,pred in original_blends(predictions).items():
            row=recipe._make_metrics(compact['val_features'].pto,pred)
            row.update(model=name,year=args.year,month=args.month,validation_only=True)
            metrics.append(row)
            table=compact['val_features'][['new_id','region','settlement','pto']].copy()
            table['prediction']=pred;table.to_csv(out/f'{name}.csv',index=False)
        print('BLENDS',json.dumps(metrics[-5:],ensure_ascii=False),flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)


if __name__=='__main__':main()
