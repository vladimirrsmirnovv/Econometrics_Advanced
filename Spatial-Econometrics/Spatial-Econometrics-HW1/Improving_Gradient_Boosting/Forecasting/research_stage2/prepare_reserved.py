"""Private code-only dispatch; the immutable choice must exist before dispatch."""
from pathlib import Path
import argparse,ast,base64,hashlib,io,json,zipfile
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu','cuda'],required=True)
    parser.add_argument('--cpu-group',choices=['main','level'],default='main');args=parser.parse_args()
    frozen=HERE/'frozen_final_selection.json';assert frozen.exists(),'Freeze development-only choices first'
    source=ast.parse((HERE/'kaggle_cpu/output/cpu_stage2.py').read_text())
    payload=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:files={n:z.read(n) for n in z.namelist()}
    # Only these mapping fields are read by the unchanged feature builders.
    # Removing unrelated static fields keeps the private code attachment small.
    for name,columns in [('stores.parquet',['new_id','node','market_shops']),
                         ('locations.parquet',['node','region','settlement_lat','settlement_lon'])]:
        frame=pd.read_parquet(io.BytesIO(files[name]),columns=columns)
        buffer=io.BytesIO();frame.to_parquet(buffer,index=False);files[name]=buffer.getvalue()
    for name in ['experiment.py','features.py','geo_control.py','march_forecast.py','extend_old.py']:files[name]=(HERE/name).read_bytes()
    files['frozen_final_selection.json']=frozen.read_bytes()
    files['spatial_source.txt']=(HERE.parent/'spatial_features.py').read_bytes()
    files['research_source.txt']=(HERE.parent/'research/build_features.py').read_bytes()
    for kind in ['knn8','hybrid8']:files['preserved_'+kind+'_weights.npz']=(HERE/'cache'/('preserved_'+kind+'_weights.npz')).read_bytes()
    for name in ['selection.py','cpu_driver.py','legacy_source.py']:files.pop(name,None)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    launcher=(HERE/'reserved_launcher.py').read_text().replace('PAYLOAD=None','PAYLOAD='+repr(base64.b64encode(stream.getvalue()).decode()))
    launcher=launcher.replace("BACKEND='cpu'",'BACKEND='+repr(args.backend))
    launcher=launcher.replace("CPU_GROUP='main'",'CPU_GROUP='+repr(args.cpu_group))
    if args.backend=='cpu' and args.cpu_group=='level':
        profile='research';base=HERE.parent/'research/kaggle'
    else:
        profile='kaggle_final' if args.backend=='cpu' else 'kaggle_cpu';base=HERE/profile
    folder=base/'kernel';folder.mkdir(exist_ok=True)
    if (folder/'kernel-metadata.json').exists():
        old=folder/'kernel-metadata.json';backup=HERE/'sources/dispatch'/('before_reserved_'+profile+'.json');backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():backup.write_bytes(old.read_bytes())
    install="import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','pandas==2.2.3','numpy==2.3.5','pyarrow==26.0.0','holidays==0.106','catboost==1.2.10','lightgbm==4.7.0','xgboost==3.1.3','scipy==1.16.3','scikit-learn==1.6.1'])\n"
    run="from pathlib import Path\np=Path('/kaggle/working/reserved_launcher.py')\np.write_text("+repr(launcher)+",encoding='utf-8')\nsubprocess.check_call([sys.executable,'-u',str(p)])\n"
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(cell_type='code',id='step'+str(i),metadata={},execution_count=None,outputs=[],source=s) for i,s in enumerate([install,run])])
    (folder/'reserved.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    owner=json.loads((HERE.parent/'kaggle/account.json').read_text())['owner'];tag=args.backend if args.backend=='cuda' else args.backend+'-'+args.cpu_group
    slug='x5-frozen-'+tag+'-audit-march-20261010'
    title='X5 Frozen '+('CUDA' if args.backend=='cuda' else 'CPU '+args.cpu_group.title())+' Audit March 20261010'
    metadata=dict(id=owner+'/'+slug,title=title,code_file='reserved.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=args.backend=='cuda',enable_internet=True,
        dataset_sources=[owner+'/x5-xgboost-reproduction-20261009',owner+'/x5-spatial-research-20261009'],
        kernel_sources=[owner+'/x5-spatial-final-iteration-20261009'],competition_sources=[],model_sources=[])
    if args.backend=='cuda':metadata['machine_shape']='NvidiaTeslaT4'
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (base/'reserved_dispatch_manifest.json').write_text(json.dumps(dict(kernel=metadata['id'],profile=profile,private=True,new_dataset_upload=False,
        frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),payload_bytes=len(stream.getvalue()),
        files={n:hashlib.sha256(d).hexdigest() for n,d in files.items()}),indent=2))
    print('FROZEN_PRIVATE_WORKER_READY',metadata['id'],len(stream.getvalue()))

if __name__=='__main__':main()
