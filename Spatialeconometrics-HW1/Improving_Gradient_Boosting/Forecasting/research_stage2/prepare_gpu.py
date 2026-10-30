"""Pack a private code-only kernel; reuse the two already-authorised datasets."""
from pathlib import Path
import base64,hashlib,io,json,zipfile,zlib
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    owner=json.loads((HERE.parent/'kaggle/account.json').read_text())['owner']
    files={p.name:p.read_bytes() for p in [HERE/'features.py',HERE/'experiment.py',
        HERE.parent/'notebook_api.py',HERE.parent/'run_validation.py']}
    # Geographic public centroids and an ID->market mapping for the same approved stores.
    files['stores.parquet']=pd.read_parquet(HERE.parent/'research/cache/stores.parquet').to_parquet(index=False)
    locations=pd.read_csv(HERE.parent/'research/geography_diagnostics.csv')[['node','region','settlement_lat','settlement_lon']]
    files['locations.csv']=locations.to_csv(index=False).encode()
    files['old_manifest.json']=(HERE.parent/'research/cache/feature_manifest.json').read_bytes()
    files['plan.md']=(HERE/'PLAN.md').read_bytes()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in files.items():archive.writestr(name,value)
    encoded=base64.b64encode(stream.getvalue()).decode()
    launcher=(HERE/'gpu_launcher.py').read_text().replace('PAYLOAD=None','PAYLOAD='+repr(encoded))
    folder=HERE/'kaggle/kernel';folder.mkdir(parents=True,exist_ok=True)
    install="import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','xgboost==3.1.3','pandas==2.2.3','numpy==2.3.5','pyarrow==26.0.0','holidays==0.106','catboost==1.2.10','lightgbm==4.7.0'])\n"
    run="from pathlib import Path\np=Path('/kaggle/working/stage2_launcher.py')\np.write_text("+repr(launcher)+",encoding='utf-8')\nsubprocess.check_call([sys.executable,'-u',str(p)])\n"
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(cell_type='code',id='step'+str(i),metadata={},execution_count=None,outputs=[],source=s) for i,s in enumerate([install,run])])
    (folder/'stage2.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    metadata=dict(id=owner+'/x5-spatial-final-iteration-20261009',title='X5 Spatial Final Iteration 20261009',
        code_file='stage2.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',
        dataset_sources=[owner+'/x5-xgboost-reproduction-20261009',owner+'/x5-spatial-research-20261009'],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    manifest=dict(kernel=metadata['id'],private=True,new_dataset_upload=False,payload_bytes=len(stream.getvalue()),
        files={name:hashlib.sha256(content).hexdigest() for name,content in files.items()},
        phase='development only',outer_months=['2024-06','2024-07','2024-08'],
        reserved_months=['2024-11','2024-12'],source_datasets=metadata['dataset_sources'])
    (HERE/'kaggle/dispatch_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PRIVATE_STAGE2_READY',metadata['id'],'payload',len(stream.getvalue()),flush=True)

if __name__=='__main__':main()
