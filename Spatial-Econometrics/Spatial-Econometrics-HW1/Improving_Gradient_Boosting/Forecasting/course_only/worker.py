"""Private course-only fixed-count worker; existing approved inputs only."""
import base64,gc,hashlib,io,json,sys,zipfile,warnings
from pathlib import Path
import numpy as np
import pandas as pd
PAYLOAD=None
TAG='cat'
PHASE='development'
SELECTED='wx_knn6'

def main():
    root=Path('/tmp/x5_course_'+TAG);parent=root/'Forecasting';here=parent/'course_only';here.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
        for name in z.namelist():
            target=parent/name if name in ['notebook_api.py','run_validation.py'] else here/name
            assert target.resolve().is_relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    source=list(Path('/kaggle/input').rglob('input_manifest.json'));assert len(source)==1;source=source[0].parent
    raw_dir=root/'Data/Shops';raw_dir.mkdir(parents=True,exist_ok=True);baseline=parent/'baseline';baseline.mkdir(exist_ok=True)
    (raw_dir/'train_2.csv').symlink_to(source/'train_2.csv');(baseline/'original_solution.ipynb').symlink_to(source/'original_solution.ipynb')
    prepared=list(Path('/kaggle/input').rglob('prepared_stage2.zip'));assert len(prepared)==1
    cache=parent/'cache';cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(prepared[0]) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.startswith(('2024-08/','2024-11/','2024-12/')):
                target=(cache/name).resolve();assert target.is_relative_to(cache);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    sys.path.insert(0,str(parent));sys.path.insert(0,str(here))
    from notebook_api import MODELS,load_recipe,read_raw,compact_fold
    from run_validation import get_fold
    from runner import prepare,execute,march
    model=MODELS[{'cat':0,'ld':1,'ll':2,'xgb':3}[TAG]]
    expected=json.loads((here/'preparation_checks.json').read_text());assert expected['status']=='passed'
    assert hashlib.sha256((here/'runner.py').read_bytes()).hexdigest()==expected['implementation_sha256']
    assert hashlib.sha256((here/'cache/weights_manifest.json').read_bytes()).hexdigest()==expected['weights_manifest_sha256']
    output=Path('/kaggle/working');archive=output/(PHASE+'_'+TAG+'.zip')
    def export():
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for folder,base,prefix in [(here/'results',here,''),(parent/'submissions/2025-03-course',parent/'submissions/2025-03-course','march/')]:
                for path in folder.rglob('*'):
                    if path.is_file():z.write(path,prefix+path.relative_to(base).as_posix())
    parity=None
    try:
        if PHASE=='development':
            recipe=load_recipe();compact=get_fold(recipe,2024,11)
            for config in ['wx_knn6','wx_knn12']:
                xt,xv,_,_,_,columns,hashes=prepare(recipe,compact,model,config)
                assert columns==expected['november_expected'][model][config]['columns']
                assert hashes==expected['november_expected'][model][config]['WX_fingerprints'],(model,config,hashes)
                del xt,xv;gc.collect()
            del compact;gc.collect()
            execute(model,'2024-11','baseline')
            actual=pd.read_csv(here/'results/2024-11'/model/'baseline.csv').sort_values('new_id')
            with np.load(here/'expected_baseline.npz') as anchor:
                np.testing.assert_array_equal(actual.new_id,anchor['ids'])
                np.testing.assert_array_equal(actual.pto,anchor['truth'])
                np.testing.assert_allclose(actual.prediction,anchor[model],rtol=1e-12,atol=1e-8)
                parity=dict(max_relative_difference=float(np.max(abs(actual.prediction/anchor[model]-1))),
                    mape=float((abs(actual.prediction-actual.pto)/actual.pto).mean()*100))
            print('COURSE_BASELINE_PARITY_PASSED',model,parity,flush=True);export()
            for fold in ['2024-08','2024-11','2024-12']:
                for config in ['wx_knn6','wx_knn12']:execute(model,fold,config);export()
        else:
            assert PHASE=='confirmation' and SELECTED in ['wx_knn6','wx_knn12']
            for fold in ['2025-01','2025-02']:
                for config in ['baseline',SELECTED]:execute(model,fold,config);export()
            recipe=load_recipe();raw=read_raw(recipe)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
                original=recipe.build_retail_features_with_split(raw,split_mode='train_val_test',min_year=2023,
                    val_year=2025,val_month=2,test_year=2025,test_month=3,create_test_if_missing=True,
                    expected_test_size=18657,use_val_as_test_history=True,base_feature_kwargs=recipe.COMMON_BASE_FEATURE_KWARGS,verbose=False)
            compact=compact_fold(recipe,original);del original,raw;gc.collect()
            full=cache/'final_2025-03';full.mkdir(exist_ok=True)
            expected_frames=json.loads((here/'expected_march_frames.json').read_text())
            for key in ['test_train_features','test_features']:
                digest=hashlib.sha256(pd.util.hash_pandas_object(compact[key],index=False).values.tobytes()).hexdigest()
                assert digest==expected_frames[key],key
                compact[key].to_parquet(full/(key+'.parquet'),index=False)
            (full/'manifest.json').write_text(json.dumps({'categorical_cols':compact['categorical_cols'],'feature_cols':compact['feature_cols']}))
            del compact;gc.collect();march(model,SELECTED);export()
        result=dict(status='passed',tag=TAG,model=model,phase=PHASE,selected=SELECTED if PHASE=='confirmation' else None,
            baseline_parity=parity,original_training_counts_preserved=True,private_existing_datasets_only=True,
            runner_sha256=hashlib.sha256((here/'runner.py').read_bytes()).hexdigest(),
            weights_manifest_sha256=expected['weights_manifest_sha256'])
        (output/(PHASE+'_'+TAG+'_checks.json')).write_text(json.dumps(result,indent=2))
    finally:export()
    print('COURSE_PHASE_DONE',PHASE,TAG,flush=True)

if __name__=='__main__':main()
