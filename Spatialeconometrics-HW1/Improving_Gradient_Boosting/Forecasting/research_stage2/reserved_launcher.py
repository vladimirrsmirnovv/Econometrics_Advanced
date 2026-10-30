"""Evaluate a frozen choice, then train March; never select using reserved errors."""
import base64,gc,hashlib,io,json,os,sys,zipfile,warnings,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
PAYLOAD=None
BACKEND='cpu'
CPU_GROUP='main'

def main():
    root=Path('/tmp/x5_reserved_'+BACKEND);parent=root/'Forecasting';here=parent/'research_stage2';here.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
        for name in z.namelist():
            if name.endswith('.py') or name=='frozen_final_selection.json':
                target=(parent if name in ['notebook_api.py','run_validation.py'] else here)/name;target.write_bytes(z.read(name))
        stores=pd.read_parquet(io.BytesIO(z.read('stores.parquet')));locations=pd.read_parquet(io.BytesIO(z.read('locations.parquet')))
        old_manifest=json.loads(z.read('old_manifest.json'));expected=json.loads(z.read('expected_legacy.json'))
        spatial_source=z.read('spatial_source.txt').decode();research_source=z.read('research_source.txt').decode()
        from scipy import sparse
        old_graphs={kind:sparse.load_npz(io.BytesIO(z.read('preserved_'+kind+'_weights.npz'))) for kind in ['knn8','hybrid8']}
    source=list(Path('/kaggle/input').rglob('input_manifest.json'))[0].parent
    derived=list(Path('/kaggle/input').rglob('research_manifest.json'))[0].parent
    prepared=list(Path('/kaggle/input').rglob('prepared_stage2.zip'));assert len(prepared)==1
    raw_dir=root/'Data/Shops';raw_dir.mkdir(parents=True,exist_ok=True);baseline=parent/'baseline';baseline.mkdir(exist_ok=True)
    (raw_dir/'train_2.csv').symlink_to(source/'train_2.csv');(baseline/'original_solution.ipynb').symlink_to(source/'original_solution.ipynb')
    sys.path.insert(0,str(parent));sys.path.insert(0,str(here))
    cache=parent/'cache';cache.mkdir(exist_ok=True);new_cache=here/'cache';new_cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(prepared[0]) as z:
        assert z.testzip() is None
        for name in z.namelist():
            target=new_cache/'features.parquet' if name=='stage2_features.parquet' else cache/name
            assert target.resolve().is_relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    from notebook_api import read_raw,load_recipe,compact_fold,NOTEBOOK_SHA256,MODELS
    from features import compute
    from experiment import execute
    recipe=load_recipe();raw=read_raw(recipe);new,manifest=compute(raw,stores,locations)
    pd.testing.assert_frame_equal(new,pd.read_parquet(new_cache/'features.parquet'),check_exact=True)
    digest=hashlib.sha256((here/'features.py').read_bytes()).hexdigest();manifest['implementation_sha256']=digest
    (new_cache/'feature_manifest.json').write_text(json.dumps(manifest,indent=2))
    (here/'feature_checks.json').write_text(json.dumps(dict(status='passed',implementation_sha256=digest,canonical_feature_exact=True),indent=2))
    import geo_control
    geo_control.compute(raw,stores,locations).to_parquet(new_cache/'geo_control.parquet',index=False)
    from sklearn.neighbors import BallTree
    radians=np.radians(locations.sort_values('node')[['settlement_lat','settlement_lon']].to_numpy())
    distances,neighbours=BallTree(radians,metric='haversine').query(radians,k=33)
    np.savez_compressed(Path('/kaggle/working')/('canonical_graph_'+BACKEND+'.npz'),radians=radians,distances_rad=distances,neighbours=neighbours)
    (here/'geo_control_checks.json').write_text(json.dumps(dict(status='passed',implementation_sha256=hashlib.sha256((here/'geo_control.py').read_bytes()).hexdigest()),indent=2))
    old_dir=parent/'research/cache';old_dir.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(c for group in old_manifest['groups'].values() for c in group))
    old=pd.concat([pd.read_parquet(derived/name,columns=['new_id','year_month_index']+columns)
        for name in ['2025-02_train.parquet','2025-02_holdout.parquet']],ignore_index=True)
    old.to_parquet(old_dir/'features.parquet',index=False);stores.to_parquet(old_dir/'stores.parquet',index=False)
    (old_dir/'feature_manifest.json').write_text(json.dumps(old_manifest,indent=2))
    from extend_old import preserved_functions,extend
    namespace=preserved_functions(spatial_source,research_source,stores,locations)
    original_family=namespace['weight_family']
    namespace['weight_family']=lambda locations,kind,profile:old_graphs[kind] if kind in old_graphs else original_family(locations,kind,profile)
    legacy,_=namespace['spatial_table'](raw,kind='exp50')
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(legacy[expected['columns']],index=False).values.tobytes()).hexdigest()
    assert fingerprint==expected['sha256'];legacy.to_parquet(cache/'spatial_exp50.parquet',index=False)
    future,checks=extend(raw,stores,locations,old,old_manifest['groups'],namespace)
    future.to_parquet(new_cache/'old_future.parquet',index=False)
    frozen=json.loads((here/'frozen_final_selection.json').read_text())
    models=[MODELS[-1]] if BACKEND=='cuda' else (MODELS[:2] if CPU_GROUP=='main' else [MODELS[2]])
    del raw,new,legacy,old,future;gc.collect()
    out=Path('/kaggle/working');tag=BACKEND if BACKEND=='cuda' else BACKEND+'_'+CPU_GROUP;archive_name='reserved_'+tag+'.zip'
    def export():
        with zipfile.ZipFile(out/archive_name,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for folder,base,prefix in [(here/'results',here,''),(here/'inner',here,''),
                (parent/'submissions/2025-03-stage2',parent/'submissions/2025-03-stage2','march/')]:
                for p in folder.rglob('*'):
                    if p.is_file():z.write(p,prefix+p.relative_to(base).as_posix())
    try:
        for model in models:
            for fold in frozen['reserved_months']:
                for config in dict.fromkeys(['baseline','previous',frozen['individual'][model]]):execute(model,fold,config,threads=4)
                full_configs=['baseline','previous'] if fold=='2024-11' or model==MODELS[-1] else ['baseline']
                for config in full_configs:execute(model,fold,config,'full',threads=4)
                export()
        # Recreate the exact original March split on this host, before any fit.
        full_cache=cache/'final_2025-03';full_cache.mkdir(exist_ok=True)
        raw=read_raw(recipe)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            original=recipe.build_retail_features_with_split(raw,split_mode='train_val_test',min_year=2023,
                val_year=2025,val_month=2,test_year=2025,test_month=3,create_test_if_missing=True,
                expected_test_size=18657,use_val_as_test_history=True,base_feature_kwargs=recipe.COMMON_BASE_FEATURE_KWARGS,verbose=False)
        compact=compact_fold(recipe,original);del original;gc.collect()
        for key in ['test_train_features','test_features']:compact[key].to_parquet(full_cache/(key+'.parquet'),index=False)
        assert compact['test_train_features'].year_month_index.max()==25 and compact['test_features'].pto.isna().all()
        (full_cache/'manifest.json').write_text(json.dumps(dict(categorical_cols=compact['categorical_cols'],feature_cols=compact['feature_cols'],
            notebook_sha256=NOTEBOOK_SHA256,raw_sha256=hashlib.sha256((raw_dir/'train_2.csv').read_bytes()).hexdigest()),indent=2))
        fingerprints={key:hashlib.sha256(pd.util.hash_pandas_object(compact[key],index=False).values.tobytes()).hexdigest() for key in ['test_train_features','test_features']}
        extra=raw[(raw.year==2025)&(raw.month==2)].copy();extra['month']=3;extra['pto']=np.nan
        legacy,_=namespace['spatial_table'](pd.concat([raw,extra],ignore_index=True),kind='exp50')
        legacy[expected['columns']].to_parquet(full_cache/'legacy_features.parquet',index=False)
        del raw,extra,legacy,compact;gc.collect()
        from march_forecast import run
        composition=frozen['ensemble']['composition']
        for model in models:
            run(model,threads=4)
            extra_configs=[]
            if composition in ['baseline','cautious_new']:extra_configs=['baseline']
            elif composition=='previous':extra_configs=['previous']
            elif composition=='mandatory_mixed':extra_configs=['previous' if model==MODELS[-1] else 'baseline']
            for config in extra_configs:run(model,threads=4,configuration=config)
            export()
        check=dict(status='passed',backend=BACKEND,models=models,original_legacy_exact=True,old_future_checks=checks,
            frozen_selection_sha256=hashlib.sha256((here/'frozen_final_selection.json').read_bytes()).hexdigest(),
            canonical_features_sha256=hashlib.sha256((new_cache/'features.parquet').read_bytes()).hexdigest(),
            original_march_frame_fingerprints=fingerprints,
            environment={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','catboost','lightgbm','xgboost']})
        (out/('reserved_'+tag+'_checks.json')).write_text(json.dumps(check,indent=2))
    finally:export()
    print('FROZEN_RESERVED_AND_MARCH_DONE',BACKEND,flush=True)

if __name__=='__main__':main()
