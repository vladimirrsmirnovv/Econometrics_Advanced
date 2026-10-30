"""Private code-only dispatch, reusing the two already authorised private datasets."""
from pathlib import Path
import base64,hashlib,json,sys,zlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE))
from notebook_api import load_recipe
from gpu_runner import fingerprint

def main():
    cache=HERE.parent/'cache/final_2025-03';recipe=load_recipe()
    manifest=json.loads((cache/'manifest.json').read_text())
    previous=HERE/'kaggle/dispatch_manifest.json'
    if previous.exists():expected=json.loads(previous.read_text())['original_full_expected']
    else:
        compact={k:pd.read_parquet(cache/(k+'.parquet')) for k in ['test_train_features','test_features']}
        compact.update(train_features=None,val_features=None,categorical_cols=manifest['categorical_cols'],feature_cols=manifest['feature_cols'])
        cols=recipe.xgboost_log_diff_features
        pack=recipe._prepare_numeric_model_pack(compact,selected_features=cols,target_mode='growth_log',label='xgboost_log_diff',verbose=False)['test']
        target=np.asarray(pack['y_train'])
        expected={'train':fingerprint(pack['X_train'][cols]),'test':fingerprint(pack['X_test'][cols]),
            'effective_target':fingerprint(pd.DataFrame({'target':target}).astype('float32').where(np.isfinite(target[:,None]),np.nan))}
    future=(cache/'gpu_future_hybrid.csv').read_text()
    source=(HERE/'gpu_runner.py').read_text()
    encoded=base64.b64encode(zlib.compress(future.encode())).decode()
    expression="__import__('zlib').decompress(__import__('base64').b64decode("+repr(encoded)+")).decode()"
    source=source.replace('FUTURE_HYBRID_CSV=None','FUTURE_HYBRID_CSV='+expression)
    source=source.replace('EXPECTED_FULL_FEATURES=None','EXPECTED_FULL_FEATURES='+repr(expected))
    source=source.replace('EXPECTED_FUTURE_HYBRID_SHA256=None','EXPECTED_FUTURE_HYBRID_SHA256='+repr(hashlib.sha256(future.encode()).hexdigest()))
    directory=HERE/'kaggle/kernel';directory.mkdir(parents=True,exist_ok=True)
    owner=json.loads((HERE.parent/'kaggle/account.json').read_text())['owner']
    install="import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','xgboost==3.1.3','pandas==2.2.3','numpy==2.3.5','pyarrow==26.0.0','holidays==0.106'])\n"
    run="from pathlib import Path\np=Path('/kaggle/working/march_run.py')\np.write_text("+repr(source)+",encoding='utf-8')\nsubprocess.check_call([sys.executable,'-u',str(p)])\n"
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(cell_type='code',id='step'+str(i),metadata={},execution_count=None,outputs=[],source=s) for i,s in enumerate([install,run])])
    (directory/'march_xgboost.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    metadata=dict(id=owner+'/x5-march-forecasts-20261009',title='X5 March Forecasts 20261009',code_file='march_xgboost.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
        machine_shape='NvidiaTeslaT4',dataset_sources=[owner+'/x5-xgboost-reproduction-20261009',owner+'/x5-spatial-research-20261009'],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (directory/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (HERE/'kaggle/dispatch_manifest.json').write_text(json.dumps(dict(private=True,kernel=metadata['id'],dataset_sources=metadata['dataset_sources'],
        new_dataset_upload=False,fits=2,forecast_month='2025-03',future_features_rows=18657,future_features_columns=4,
        future_features_bytes=len(future.encode()),future_features_sha256=hashlib.sha256(future.encode()).hexdigest(),
        contains_future_actual_revenue=False,original_full_expected=expected),indent=2))
    print('PRIVATE_MARCH_KERNEL_READY',metadata['id'],'fits=2; no new dataset upload')

if __name__=='__main__':main()
