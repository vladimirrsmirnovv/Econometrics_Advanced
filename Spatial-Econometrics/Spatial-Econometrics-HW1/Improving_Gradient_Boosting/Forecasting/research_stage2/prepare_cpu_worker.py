"""CPU CatBoost on Kaggle, retaining its CPU backend and the original legacy lags."""
from pathlib import Path
import argparse,ast,base64,hashlib,io,json,zipfile
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['development','full_august','lightgbm_development'],default='development');args=parser.parse_args()
    files={p.name:p.read_bytes() for p in [HERE/'features.py',HERE/'geo_control.py',HERE/'experiment.py',HERE/'cpu_driver.py',
        HERE/'selection.py',HERE.parent/'notebook_api.py',HERE.parent/'run_validation.py']}
    files['stores.parquet']=pd.read_parquet(HERE.parent/'research/cache/stores.parquet').to_parquet(index=False)
    locations=pd.read_csv(HERE.parent/'research/geography_diagnostics.csv')[['node','region','settlement_lat','settlement_lon']]
    files['locations.parquet']=locations.to_parquet(index=False)
    files['old_manifest.json']=(HERE.parent/'research/cache/feature_manifest.json').read_bytes()
    if args.task=='lightgbm_development':
        expected={}
        for model in ['lightgbm_log_diff','lightgbm_log_level']:
            frame=pd.read_csv(HERE/'results/rolling/2024-08'/model/'baseline.csv')
            expected[model]=dict(frame_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest(),
                mape=float((abs(frame.prediction-frame.pto)/frame.pto).mean()*100))
        files['expected_predictions.json']=json.dumps(expected).encode()
    source=(HERE.parent/'spatial_features.py').read_text()
    functions=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in ['make_weights','spatial_table']]
    legacy="import numpy as np\nimport pandas as pd\nfrom scipy import sparse\nfrom sklearn.neighbors import BallTree\n"
    legacy+="def attach_locations(raw):\n return STORES,LOCATIONS,{}\n"
    legacy+='\n\n'.join(ast.get_source_segment(source,node) for node in functions)
    files['legacy_source.py']=legacy.encode()
    historical=pd.read_parquet(HERE.parent/'cache/spatial_exp50.parquet').sort_values(['new_id','year_month_index']).reset_index(drop=True)
    columns=['new_id','year_month_index','wy_log_daily_lag1','wy_daily_growth_lag1','wy_daily_growth_1_3','wy_log_daily_gap_lag1']
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(historical[columns],index=False).values.tobytes()).hexdigest()
    files['expected_legacy.json']=json.dumps(dict(columns=columns,sha256=fingerprint,
        source_sha256=hashlib.sha256((HERE.parent/'spatial_features.py').read_bytes()).hexdigest())).encode()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in files.items():archive.writestr(name,value)
    launcher=(HERE/'cpu_launcher.py').read_text().replace('PAYLOAD=None','PAYLOAD='+repr(base64.b64encode(stream.getvalue()).decode()))
    launcher=launcher.replace("TASK='development'",'TASK='+repr(args.task))
    profile='kaggle' if args.task=='lightgbm_development' else ('kaggle_cpu' if args.task=='development' else 'kaggle_final')
    folder=HERE/profile/'kernel';folder.mkdir(parents=True,exist_ok=True)
    owner=json.loads((HERE.parent/'kaggle/account.json').read_text())['owner']
    install="import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','pandas==2.2.3','numpy==2.3.5','pyarrow==26.0.0','holidays==0.106','catboost==1.2.10','lightgbm==4.7.0','xgboost==3.1.3'])\n"
    run="from pathlib import Path\np=Path('/kaggle/working/cpu_stage2.py')\np.write_text("+repr(launcher)+",encoding='utf-8')\nsubprocess.check_call([sys.executable,'-u',str(p)])\n"
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(cell_type='code',id='step'+str(i),metadata={},execution_count=None,outputs=[],source=s) for i,s in enumerate([install,run])])
    (folder/'cpu_stage2.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    metadata=dict(id=owner+'/x5-catboost-final-iteration-20261009',title='X5 CatBoost Final Iteration 20261009',
        code_file='cpu_stage2.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=False,enable_internet=True,dataset_sources=[owner+'/x5-xgboost-reproduction-20261009',
        owner+'/x5-spatial-research-20261009'],kernel_sources=[owner+'/x5-spatial-final-iteration-20261009'],
        competition_sources=[],model_sources=[])
    if args.task=='full_august':metadata.update(id=owner+'/x5-full-recipe-audit-20261009',title='X5 Full Recipe Audit 20261009')
    if args.task=='lightgbm_development':
        metadata.update(id=owner+'/x5-lightgbm-final-resume-20261010',title='X5 LightGBM Final Resume 20261010')
        metadata['kernel_sources'] += [owner+'/x5-lightgbm-final-iteration-20261010']
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (HERE/profile/'dispatch_manifest.json').write_text(json.dumps(dict(private=True,new_dataset_upload=False,task=args.task,
        kernel=metadata['id'],backend='CPU CatBoost 1.2.10',payload_bytes=len(stream.getvalue()),
        original_legacy_fingerprint=fingerprint,source_datasets=metadata['dataset_sources'],
        source_kernel_outputs=metadata['kernel_sources'],files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2))
    print('PRIVATE_CPU_WORKER_READY',metadata['id'],len(stream.getvalue()))

if __name__=='__main__':main()
