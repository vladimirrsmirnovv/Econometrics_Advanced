"""New original fold caches and development-only rolling CUDA experiments."""


def _directory_path(value):
    """Resolve directory names from preserved metadata after the folder rename."""
    from pathlib import Path as _Path
    path = _Path(value)
    if path.is_absolute():
        return path
    directory_count = len(path.parts) - bool(path.suffix)
    return _Path(*(part.replace("_", "-") if index < directory_count else part
                   for index, part in enumerate(path.parts)))

import base64,gc,hashlib,io,json,os,shutil,sys,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
PAYLOAD=None

def main():
    root=Path('/tmp/x5_stage2_project');parent=root/'Forecasting';here=parent/'research-stage2'
    here.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
        for name in ['features.py','experiment.py','plan.md']:(here/_directory_path(name)).write_bytes(archive.read(name))
        for name in ['notebook_api.py','run_validation.py']:(parent/_directory_path(name)).write_bytes(archive.read(name))
        stores=pd.read_parquet(io.BytesIO(archive.read('stores.parquet')))
        locations=pd.read_csv(io.BytesIO(archive.read('locations.csv')))
        old_manifest=json.loads(archive.read('old_manifest.json'))
    source_manifest=list(Path('/kaggle/input').rglob('input_manifest.json'))
    research_manifest=list(Path('/kaggle/input').rglob('research_manifest.json'))
    assert len(source_manifest)==len(research_manifest)==1
    source=source_manifest[0].parent;derived=research_manifest[0].parent
    raw_dir=root/'Data/Shops';raw_dir.mkdir(parents=True,exist_ok=True)
    baseline=parent/'baseline';baseline.mkdir(exist_ok=True)
    (raw_dir/'train_2.csv').symlink_to(source/'train_2.csv')
    (baseline/'original_solution.ipynb').symlink_to(source/'original_solution.ipynb')
    sys.path.insert(0,str(parent));sys.path.insert(0,str(here))
    from notebook_api import load_recipe,read_raw
    from run_validation import get_fold
    from features import compute
    from experiment import SCREEN,execute
    recipe=load_recipe();raw=read_raw(recipe)
    new,info=compute(raw,stores,locations)
    changed=raw.copy();index=(changed.year-2023)*12+changed.month-1
    changed.loc[index>=19,'pto']*=37
    other,_=compute(changed,stores,locations)
    pd.testing.assert_frame_equal(new.loc[new.year_month_index<=19].reset_index(drop=True),
        other.loc[other.year_month_index<=19].reset_index(drop=True))
    cache=here/'cache';cache.mkdir(exist_ok=True)
    new.to_parquet(cache/'features.parquet',index=False)
    info['implementation_sha256']=hashlib.sha256((here/'features.py').read_bytes()).hexdigest()
    (cache/'feature_manifest.json').write_text(json.dumps(info,indent=2))
    (here/'feature_checks.json').write_text(json.dumps(dict(status='passed',gpu_cutoff=19,
        implementation_sha256=info['implementation_sha256'],features=len(new.columns)-2),indent=2))
    old_dir=parent/'research/cache';old_dir.mkdir(parents=True,exist_ok=True)
    plan=json.loads(research_manifest[0].read_text());columns=list(dict.fromkeys(c for g in plan['groups'].values() for c in g))
    history=pd.concat([pd.read_parquet(derived/_directory_path(name),columns=['new_id','year_month_index']+columns)
        for name in ['2025-02_train.parquet','2025-02_holdout.parquet']],ignore_index=True)
    assert not history.duplicated(['new_id','year_month_index']).any()
    history.to_parquet(old_dir/'features.parquet',index=False);stores.to_parquet(old_dir/'stores.parquet',index=False)
    (old_dir/'feature_manifest.json').write_text(json.dumps(old_manifest,indent=2))
    del raw,new,other,changed,history;gc.collect()
    # Prepare all inputs, but do not fit/read aggregate errors for reserved months.
    for fold in ['2024-05','2024-06','2024-07','2024-08','2024-10','2024-11','2024-12']:
        year,month=map(int,fold.split('-'));compact=get_fold(recipe,year,month);del compact;gc.collect()
        print('PREPARED',fold,flush=True)
    output=Path('/kaggle/working')
    with zipfile.ZipFile(output/'prepared_stage2.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in (parent/'cache').rglob('*'):
            if p.is_file():archive.write(p,p.relative_to(parent/'cache').as_posix())
        archive.write(cache/'features.parquet','stage2_features.parquet')
    print('PREPARED_ARCHIVE_READY',flush=True)
    for fold in ['2024-08','2024-06','2024-07']:
        for config in SCREEN:execute('xgboost_log_diff',fold,config,'rolling',threads=4)
    for config in ['baseline','previous']:execute('xgboost_log_diff','2024-08',config,'full',threads=4)
    with zipfile.ZipFile(output/'cuda_development.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in [here/'results',here/'inner']:
            for p in folder.rglob('*'):
                if p.is_file():archive.write(p,p.relative_to(here).as_posix())
    (output/'stage2_gpu_checks.json').write_text(json.dumps(dict(status='passed',xgboost_version='3.1.3',
        outer_months=['2024-06','2024-07','2024-08'],reserved_months_not_evaluated=['2024-11','2024-12'],
        source_datasets_unchanged=True,original_folds_built_with_preserved_recipe=True,
        outer_targets_never_used_for_stopping=True,
        archives={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in output.glob('*.zip')}),indent=2))
    print('PHASE1_DONE',flush=True)

if __name__=='__main__':main()
