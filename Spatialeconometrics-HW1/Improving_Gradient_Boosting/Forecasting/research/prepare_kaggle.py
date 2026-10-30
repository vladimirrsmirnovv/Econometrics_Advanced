"""Private derived feature matrices; same original XGBoost parameters and CUDA."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import load_recipe,matrices,NOTEBOOK_SHA256
from run_validation import get_fold
from spatial_features import align_features
from run_models import checks_and_data,DEFAULT_CONFIGS

def main():
    recipe=load_recipe();feature_manifest,table=checks_and_data()
    account=json.loads((HERE.parent/'kaggle/account.json').read_text())
    owner=account['owner'];directory=HERE/'kaggle'
    upload=directory/'upload';kernel=directory/'kernel'
    upload.mkdir(parents=True,exist_ok=True);kernel.mkdir(exist_ok=True)
    stores=pd.read_parquet(HERE/'cache/stores.parquet').set_index('new_id')
    checks=json.loads((HERE/'feature_checks.json').read_text())
    groups=feature_manifest['groups'];columns=list(dict.fromkeys(c for x in groups.values() for c in x))
    folds=['2025-02','2025-01','2024-03']
    for fold in folds:
        year,month=map(int,fold.split('-'));compact=get_fold(recipe,year,month)
        train,holdout,ytrain,yholdout,_=matrices(recipe,compact,'xgboost_log_diff')
        extra_train,extra_holdout=align_features(compact,table,columns)
        for label,x,y,extra,meta in [('train',train,ytrain,extra_train,compact['train_features']),
                                   ('holdout',holdout,yholdout,extra_holdout,compact['val_features'])]:
            frame=pd.concat([x,extra],axis=1)
            for c in ['new_id','year_month_index','region','settlement','pto','pto_lag_1']:
                if c not in frame:frame[c]=meta[c].to_numpy()
            frame['target']=np.asarray(y)
            frame['market_shops']=meta.new_id.map(stores.market_shops).to_numpy()
            frame.to_parquet(upload/(fold+'_'+label+'.parquet'),index=False)
    params=recipe._clean_xgb_params(dict(recipe.xgboost_log_diff_params),keep_early_stopping=True)
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in upload.glob('*.parquet')}
    configs=['exp50_history','wx_means']+DEFAULT_CONFIGS+['random_history','geo_sensitivity']
    manifest=dict(folds=folds,configs=configs,groups=groups,base_columns=recipe.xgboost_log_diff_features,
        parameters=params,notebook_sha256=NOTEBOOK_SHA256,hashes=hashes,feature_checks=checks,
        feature_version=feature_manifest['feature_version'],validation_only=True)
    (upload/'research_manifest.json').write_text(json.dumps(manifest,indent=2))
    dataset=owner+'/x5-spatial-research-20261009'
    (upload/'dataset-metadata.json').write_text(json.dumps(dict(id=dataset,
        title='X5 Spatial Research 20261009',licenses=[{'name':'other'}],
        description='Private authorised spatial forecasting validation matrices; no redistribution license.')))
    source=(HERE/'kaggle_runner.py').read_text(encoding='utf-8')
    install="""import subprocess,sys
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'xgboost==3.1.3','pandas==2.2.3','numpy==2.3.5','pyarrow==26.0.0'])
"""
    run="from pathlib import Path\np=Path('/kaggle/working/research_run.py')\np.write_text("+repr(source)+",encoding='utf-8')\nsubprocess.check_call([sys.executable,'-u',str(p)])\n"
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[
        dict(cell_type='code',id='step'+str(i),metadata={},execution_count=None,outputs=[],source=s)
        for i,s in enumerate([install,run])])
    (kernel/'spatial_xgboost.ipynb').write_text(json.dumps(notebook,indent=1))
    metadata=dict(id=owner+'/x5-spatial-xgboost-research-20261009',
        title='X5 Spatial XGBoost Research 20261009',code_file='spatial_xgboost.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
        machine_shape='NvidiaTeslaT4',dataset_sources=[dataset],competition_sources=[],kernel_sources=[],model_sources=[])
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (directory/'dispatch_manifest.json').write_text(json.dumps(dict(dataset=dataset,kernel=metadata['id'],
        fits=len(folds)*(len(configs)+1),upload_bytes=sum(p.stat().st_size for p in upload.iterdir())),indent=2))
    print((directory/'dispatch_manifest.json').read_text())

if __name__=='__main__':main()
