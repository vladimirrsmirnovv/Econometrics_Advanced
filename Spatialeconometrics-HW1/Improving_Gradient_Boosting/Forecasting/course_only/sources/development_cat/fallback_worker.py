"""Identical CPU CatBoost fallback, isolated from the submitted remote worker."""
import argparse,gc,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import runner
from notebook_api import load_recipe
from run_validation import get_fold
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','confirmation'],default='development')
    p.add_argument('--selected',choices=['wx_knn6','wx_knn12'],default='wx_knn6');a=p.parse_args()
    model='catboost_log_level';work=HERE/'local_cat';work.mkdir(exist_ok=True);(work/'cache').mkdir(exist_ok=True)
    for name in ['stores.parquet','weights_manifest.json','knn6.npz','knn12.npz']:shutil.copyfile(HERE/'cache'/name,work/'cache'/name)
    runner.HERE=work
    recipe=load_recipe();expected=json.loads((HERE/'preparation_checks.json').read_text())
    compact=get_fold(recipe,2024,11)
    for config in ['wx_knn6','wx_knn12']:
        xt,xv,_,_,_,cols,hashes=runner.prepare(recipe,compact,model,config)
        assert hashes==expected['november_expected'][model][config]['WX_fingerprints']
        assert cols==expected['november_expected'][model][config]['columns'];del xt,xv;gc.collect()
    del compact;gc.collect();parity=None
    if a.phase=='development':
        runner.execute(model,'2024-11','baseline')
        old=pd.read_csv(HERE.parent/'research_stage2/results/full/2024-11'/model/'baseline.csv').sort_values('new_id')
        new=pd.read_csv(work/'results/2024-11'/model/'baseline.csv').sort_values('new_id')
        np.testing.assert_array_equal(old.new_id,new.new_id);np.testing.assert_array_equal(old.pto,new.pto)
        np.testing.assert_allclose(old.prediction,new.prediction,rtol=1e-12,atol=1e-8)
        parity=dict(max_relative_difference=float(np.max(abs(new.prediction/old.prediction-1))),mape=float((abs(new.pto-new.prediction)/new.pto).mean()*100))
        print('COURSE_LOCAL_BASELINE_PARITY_PASSED',parity,flush=True)
        for fold in ['2024-08','2024-11','2024-12']:
            for config in ['wx_knn6','wx_knn12']:runner.execute(model,fold,config)
    else:
        for fold in ['2025-01','2025-02']:
            for config in ['baseline',a.selected]:runner.execute(model,fold,config)
        # March uses the preserved, already fingerprinted original compact split.
        runner.HERE=HERE;runner.march(model,a.selected);runner.HERE=work
    output=work/'output';output.mkdir(exist_ok=True)
    with zipfile.ZipFile(output/(a.phase+'_cat.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in (work/'results').rglob('*'):
            if path.is_file() and (a.phase=='development' or '2025-' in path.as_posix()):z.write(path,path.relative_to(work).as_posix())
        if a.phase=='confirmation':
            out=HERE.parent/'submissions/2025-03-course'
            for path in [out/'experimental/test_catboost_log_level.csv',out/'experimental/test_catboost_log_level.json',out/'models/catboost_log_level.cbm']:
                z.write(path,'march/'+path.relative_to(out).as_posix())
    check=dict(status='passed',tag='cat',phase=a.phase,model=model,selected=a.selected if a.phase=='confirmation' else None,
        runner_sha256=runner.sha(HERE/'runner.py'),weights_manifest_sha256=expected['weights_manifest_sha256'],
        baseline_parity=parity,original_training_counts_preserved=True,private_existing_datasets_only=True,
        execution_location='Windows CPU fallback; no remote upload')
    (output/(a.phase+'_cat_checks.json')).write_text(json.dumps(check,indent=2))
    print('COURSE_LOCAL_PHASE_DONE',a.phase,flush=True)
if __name__=='__main__':main()
