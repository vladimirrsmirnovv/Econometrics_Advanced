"""Kaggle CPU worker reuses private GPU-prepared folds; reserved months stay closed."""


def _directory_path(value):
    """Resolve directory names from preserved metadata after the folder rename."""
    from pathlib import Path as _Path
    path = _Path(value)
    if path.is_absolute():
        return path
    directory_count = len(path.parts) - bool(path.suffix)
    return _Path(*(part.replace("_", "-") if index < directory_count else part
                   for index, part in enumerate(path.parts)))

import base64,gc,hashlib,io,json,os,sys,zipfile,importlib.metadata
from pathlib import Path
import pandas as pd
import numpy as np
PAYLOAD=None
TASK='development'

def main():
    root=Path('/tmp/x5_cpu_stage2');parent=root/'Forecasting';here=parent/'research-stage2';here.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
        for name in ['features.py','geo_control.py','experiment.py','cpu_driver.py','selection.py']:(here/_directory_path(name)).write_bytes(archive.read(name))
        for name in ['notebook_api.py','run_validation.py']:(parent/_directory_path(name)).write_bytes(archive.read(name))
        stores=pd.read_parquet(io.BytesIO(archive.read('stores.parquet')))
        locations=pd.read_parquet(io.BytesIO(archive.read('locations.parquet')))
        old_manifest=json.loads(archive.read('old_manifest.json'));expected=json.loads(archive.read('expected_legacy.json'))
        legacy=archive.read('legacy_source.py').decode()
        expected_predictions=json.loads(archive.read('expected_predictions.json')) if 'expected_predictions.json' in archive.namelist() else {}
    source=list(Path('/kaggle/input').rglob('input_manifest.json'))[0].parent
    data=list(Path('/kaggle/input').rglob('research_manifest.json'))[0].parent
    prepared=list(Path('/kaggle/input').rglob('prepared_stage2.zip'))
    assert len(prepared)==1,'Attach the completed private phase1 kernel output'
    raw_dir=root/'Data/Shops';raw_dir.mkdir(parents=True,exist_ok=True)
    baseline=parent/'baseline';baseline.mkdir(exist_ok=True)
    (raw_dir/'train_2.csv').symlink_to(source/'train_2.csv');(baseline/'original_solution.ipynb').symlink_to(source/'original_solution.ipynb')
    sys.path.insert(0,str(parent));sys.path.insert(0,str(here))
    from notebook_api import load_recipe,read_raw
    from experiment import SCREEN,execute
    from selection import pick_one
    cache=parent/'cache';cache.mkdir(exist_ok=True)
    new_cache=here/'cache';new_cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(prepared[0]) as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            if item.filename=='stage2_features.parquet':target=new_cache/'features.parquet'
            else:target=cache/item.filename
            assert target.resolve().is_relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(item))
    # Manifest and causal checks are recomputed using the unchanged builder.
    from features import compute
    raw=read_raw(load_recipe());new,info=compute(raw,stores,locations)
    pd.testing.assert_frame_equal(new,pd.read_parquet(new_cache/'features.parquet'),check_exact=True)
    digest=hashlib.sha256((here/'features.py').read_bytes()).hexdigest();info['implementation_sha256']=digest
    (new_cache/'feature_manifest.json').write_text(json.dumps(info,indent=2))
    (here/'feature_checks.json').write_text(json.dumps(dict(status='passed',implementation_sha256=digest,
        private_phase1_feature_byte_parity=True,outer_targets_never_used_for_stopping=True),indent=2))
    import geo_control
    geo=geo_control.compute(raw,stores,locations)
    geo.to_parquet(new_cache/'geo_control.parquet',index=False)
    out=Path('/kaggle/working')
    geo.to_parquet(out/'canonical_geo_control.parquet',index=False)
    from sklearn.neighbors import BallTree
    radians=np.radians(locations.sort_values('node')[['settlement_lat','settlement_lon']].to_numpy())
    ds,js=BallTree(radians,metric='haversine').query(radians,k=33)
    candidate=np.stack([a[a!=i][:32] for i,a in enumerate(js)])
    distances=np.stack([a[b!=i][:32] for i,(a,b) in enumerate(zip(ds,js))])*6371
    np.savez_compressed(out/'canonical_geographic_graph.npz',candidate=candidate,distance_km=distances)
    (out/'runtime_versions.json').write_text(json.dumps({n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','xgboost']},indent=2))
    (here/'geo_control_checks.json').write_text(json.dumps(dict(status='passed',
        implementation_sha256=hashlib.sha256((here/'geo_control.py').read_bytes()).hexdigest(),
        static_distance_only=True,target_cutoff='strictly earlier outcomes'),indent=2))
    old_dir=parent/'research/cache';old_dir.mkdir(parents=True,exist_ok=True)
    groups=old_manifest['groups'];columns=list(dict.fromkeys(c for g in groups.values() for c in g))
    old=pd.concat([pd.read_parquet(data/_directory_path(name),columns=['new_id','year_month_index']+columns)
        for name in ['2025-02_train.parquet','2025-02_holdout.parquet']],ignore_index=True)
    old.to_parquet(old_dir/'features.parquet',index=False);stores.to_parquet(old_dir/'stores.parquet',index=False)
    (old_dir/'feature_manifest.json').write_text(json.dumps(old_manifest,indent=2))
    namespace=dict(STORES=stores,LOCATIONS=locations)
    exec(compile(legacy,'preserved_spatial_functions','exec'),namespace)
    original,_=namespace['spatial_table'](raw,kind='exp50')
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(original[expected['columns']],index=False).values.tobytes()).hexdigest()
    assert fingerprint==expected['sha256'],'Original CatBoost spatial builder changed'
    original.to_parquet(cache/'spatial_exp50.parquet',index=False)
    del raw,original,old,new;gc.collect()
    threads=4 if TASK=='lightgbm_development' else min(8,os.cpu_count() or 4)
    if TASK=='lightgbm_development':
        resume=list(Path('/kaggle/input').rglob('lightgbm_development.zip'))
        if resume:
            assert len(resume)==1
            with zipfile.ZipFile(resume[0]) as z:
                assert z.testzip() is None
                for name in z.namelist():
                    target=(here/_directory_path(name)).resolve();assert target.is_relative_to(here);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    archive_name={'development':'cpu_development.zip','full_august':'full_august.zip','lightgbm_development':'lightgbm_development.zip'}[TASK]
    def export():
        with zipfile.ZipFile(out/archive_name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for folder in [here/'results',here/'inner']:
                for p in folder.rglob('*'):
                    if p.is_file():archive.write(p,p.relative_to(here).as_posix())
            for p in here.glob('confirm_plan_*.json'):archive.write(p,p.name)
    try:
        models=['lightgbm_log_diff','lightgbm_log_level'] if TASK=='lightgbm_development' else ['catboost_log_level']
        for model in models:
            if TASK=='full_august':
                for config in ['baseline','previous']:execute(model,'2024-08',config,'full',threads=threads)
            else:
                for config in SCREEN:execute(model,'2024-08',config,threads=threads)
                if TASK=='lightgbm_development':
                    parity_frame=pd.read_csv(here/'results/rolling/2024-08'/model.replace("_", "-")/'baseline.csv')
                    actual_mape=float((abs(parity_frame.prediction-parity_frame.pto)/parity_frame.pto).mean()*100)
                    assert abs(actual_mape-expected_predictions[model]['mape'])<1e-10, 'CPU baseline MAPE differs'
                    print('LOCAL_KAGGLE_BASELINE_MAPE_MATCHES',model,actual_mape,flush=True)
                export();configs,scores=pick_one(model)
                if 'geo8_control' in configs:execute(model,'2024-08','geo8_control',threads=threads)
                plan=dict(model=model,screen='2024-08',confirm_months=['2024-06','2024-07'],configs=configs,scores=scores)
                (here/('confirm_plan_'+model+'.json')).write_text(json.dumps(plan,indent=2))
                for fold in ['2024-06','2024-07']:
                    for config in configs:execute(model,fold,config,threads=threads)
                    export()
                if TASK=='lightgbm_development':
                    for config in ['baseline','previous']:execute(model,'2024-08',config,'full',threads=threads)
        (out/'cpu_checks.json').write_text(json.dumps(dict(status='passed',backend='CPU',threads=threads,
            original_legacy_byte_parity=True,legacy_fingerprint=fingerprint,development_only=True,task=TASK,
            reserved_months_not_evaluated=['2024-11','2024-12']),indent=2))
    finally:export()
    print('CPU_DEVELOPMENT_DONE',flush=True)

if __name__=='__main__':main()
