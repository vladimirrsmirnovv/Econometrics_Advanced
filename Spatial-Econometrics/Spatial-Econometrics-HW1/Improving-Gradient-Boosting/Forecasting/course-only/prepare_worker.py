"""Package code and fixed course weights; reuse existing approved private X5 inputs."""


def _directory_path(value):
    """Resolve directory names from preserved metadata after the folder rename."""
    from pathlib import Path as _Path
    path = _Path(value)
    if path.is_absolute():
        return path
    directory_count = len(path.parts) - bool(path.suffix)
    return _Path(*(part.replace("_", "-") if index < directory_count else part
                   for index, part in enumerate(path.parts)))

import argparse,base64,hashlib,io,json,shutil,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
MODELS=['catboost_log_level','lightgbm_log_diff','lightgbm_log_level','xgboost_log_diff']
TAGS=['cat','ld','ll','xgb']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',choices=TAGS,required=True)
    p.add_argument('--phase',choices=['development','confirmation'],default='development')
    p.add_argument('--selected',choices=['wx_knn6','wx_knn12'],default='wx_knn6');a=p.parse_args()
    checks=json.loads((HERE/'preparation_checks.json').read_text());assert checks['status']=='passed'
    assert sha(HERE/'runner.py')==checks['implementation_sha256']
    anchor={}
    for model in [MODELS[TAGS.index(a.tag)]]:
        path=HERE.parent/'research-stage2/results/full/2024-11'/model.replace("_", "-")/'baseline.csv'
        frame=pd.read_csv(path).sort_values('new_id');meta=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==meta['prediction_sha256'] and meta['protocol']=='full'
        assert meta['trees']==dict(zip(MODELS,[6500,7000,8000,3750]))[model]
        if not anchor:anchor.update(ids=frame.new_id.to_numpy(),truth=frame.pto.to_numpy())
        else:np.testing.assert_array_equal(frame.new_id,anchor['ids']);np.testing.assert_array_equal(frame.pto,anchor['truth'])
        anchor[model]=frame.prediction.to_numpy()
    mem=io.BytesIO();np.savez_compressed(mem,**anchor);anchor_bytes=mem.getvalue()
    frame_path=HERE/'expected_march_frames.json'
    if not frame_path.exists():
        fps={}
        for key in ['test_train_features','test_features']:
            frame=pd.read_parquet(HERE.parent/'cache/final_2025-03'/(key+'.parquet'))
            fps[key]=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()
        frame_path.write_text(json.dumps(fps,indent=2))
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in [HERE/'runner.py',HERE.parent/'notebook_api.py',HERE.parent/'run_validation.py',
                     HERE/'preparation_checks.json',frame_path]:z.write(path,path.name)
        for name in ['knn6.npz','knn12.npz','stores.parquet','weights_manifest.json']:z.write(HERE/'cache'/_directory_path(name),'cache/'+name)
        z.writestr('expected_baseline.npz',anchor_bytes)
    source=(HERE/'worker.py').read_text().replace('PAYLOAD=None','PAYLOAD='+repr(base64.b64encode(payload.getvalue()).decode()))
    source=source.replace("TAG='cat'",'TAG='+repr(a.tag)).replace("PHASE='development'",'PHASE='+repr(a.phase)).replace("SELECTED='wx_knn6'",'SELECTED='+repr(a.selected))
    install="%pip install -q numpy==2.3.5 pandas==2.2.3 pyarrow==26.0.0 holidays==0.106 catboost==1.2.10 lightgbm==4.7.0 xgboost==3.1.3 scipy==1.16.3 scikit-learn==1.6.1\n"
    launch="from pathlib import Path\nimport subprocess,sys\nPath('/tmp/course_worker.py').write_text("+repr(source)+")\nsubprocess.run([sys.executable,'-u','/tmp/course_worker.py'],check=True)\n"
    cell=lambda text:dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=text.splitlines(True))
    nb=dict(cells=[cell(install),cell(launch)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
    active=HERE/('kaggle_'+a.tag);kernel=active/'kernel';kernel.mkdir(parents=True,exist_ok=True)
    meta=dict(id='vladimirsmirnovve/x5-course-wx-'+a.tag+'-20261010',title='X5 Course WX '+a.tag.upper()+' 20261010',
        code_file='course.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=a.tag=='xgb',enable_internet=True,
        dataset_sources=['vladimirsmirnovve/x5-xgboost-reproduction-20261009','vladimirsmirnovve/x5-spatial-research-20261009'],
        kernel_sources=['vladimirsmirnovve/x5-spatial-final-iteration-20261009'],competition_sources=[],model_sources=[])
    if a.tag=='xgb':meta['machine_shape']='NvidiaTeslaT4'
    (kernel/'course.ipynb').write_text(json.dumps(nb),encoding='utf-8')
    assert (kernel/'course.ipynb').stat().st_size<1000000,'Kaggle source limit'
    (kernel/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    info=dict(tag=a.tag,phase=a.phase,selected=a.selected if a.phase=='confirmation' else None,
        model=MODELS[TAGS.index(a.tag)],private=True,new_dataset=False,inputs=meta['dataset_sources'],
        runner_sha256=sha(HERE/'runner.py'),worker_sha256=sha(HERE/'worker.py'),plan_sha256=sha(HERE/'PLAN.md'),
        payload_sha256=hashlib.sha256(payload.getvalue()).hexdigest(),notebook_sha256=sha(kernel/'course.ipynb'),
        original_recipe_sha256=sha(HERE.parent/'baseline/original_solution.ipynb'),backend='CUDA' if a.tag=='xgb' else 'CPU')
    (active/'dispatch_manifest.json').write_text(json.dumps(info,indent=2))
    saved=HERE/'sources'/(a.phase+'_'+a.tag);saved.mkdir(parents=True,exist_ok=True)
    for path in [kernel/'course.ipynb',kernel/'kernel-metadata.json',active/'dispatch_manifest.json']:shutil.copyfile(path,saved/path.name)
    print(json.dumps(info))
if __name__=='__main__':main()
