"""Separate compound settlement-key experiment; original recipe is unchanged."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import gc
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import ROOT,load_recipe,read_raw,build_fold,compact_fold,matrices,MODELS,NOTEBOOK_SHA256

def change_keys(raw):
    raw=raw.copy()
    raw['settlement']=raw.region.astype(str)+' :: '+raw.settlement.astype(str)
    return raw

def get_key_fold(recipe,year,month):
    directory=HERE/'cache'/('key_%04d-%02d'%(year,month))
    digest=hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest()
    if (directory/'manifest.json').exists():
        info=json.loads((directory/'manifest.json').read_text())
        assert info['raw_sha256']==digest and info['notebook_sha256']==NOTEBOOK_SHA256
        output={k:None for k in ['train_features','val_features','test_features','test_train_features']}
        output.update(train_features=pd.read_parquet(directory/'train.parquet'),
            val_features=pd.read_parquet(directory/'holdout.parquet'),
            categorical_cols=info['categorical_cols'],feature_cols=info['feature_cols'])
        return output
    print('BUILD_COMPOUND_KEY',year,month,flush=True)
    raw=change_keys(read_raw(recipe))
    full=build_fold(recipe,raw,year,month,verbose=False)
    compact=compact_fold(recipe,full)
    del full,raw;gc.collect()
    directory.mkdir(parents=True,exist_ok=True)
    compact['train_features'].to_parquet(directory/'train.parquet',index=False)
    compact['val_features'].to_parquet(directory/'holdout.parquet',index=False)
    info=dict(raw_sha256=digest,notebook_sha256=NOTEBOOK_SHA256,
        key='region :: original settlement',categorical_cols=compact['categorical_cols'],
        feature_cols=compact['feature_cols'],source_untouched=True)
    (directory/'manifest.json').write_text(json.dumps(info,indent=2))
    return compact

def check():
    recipe=load_recipe();raw=read_raw(recipe)
    rng=np.random.default_rng(2026)
    sample=raw[raw.new_id.isin(rng.choice(raw.new_id.unique(),160,replace=False))]
    sample=change_keys(sample)
    periods=[]
    for year,month in [(2024,3),(2025,1),(2025,2)]:
        first=compact_fold(recipe,build_fold(recipe,sample,year,month,verbose=False))
        changed=sample.copy();future=(changed.year*12+changed.month)>=(year*12+month)
        changed.loc[future,'pto']*=23
        other=compact_fold(recipe,build_fold(recipe,changed,year,month,verbose=False))
        for model in MODELS:
            a=matrices(recipe,first,model);b=matrices(recipe,other,model)
            pd.testing.assert_frame_equal(a[0],b[0]);pd.testing.assert_frame_equal(a[1],b[1])
        periods.append('%04d-%02d'%(year,month))
    (HERE/'key_feature_checks.json').write_text(json.dumps(dict(status='passed',folds=periods,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sample_stores=160),indent=2))
    print('COMPOUND_KEY_LEAKAGE_CHECK_PASSED',flush=True)

if __name__=='__main__':check()
