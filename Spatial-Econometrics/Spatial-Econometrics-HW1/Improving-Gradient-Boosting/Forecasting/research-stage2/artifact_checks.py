"""Validate the completed native models, submissions and preserved sources."""
from pathlib import Path
import argparse,gc,hashlib,json,subprocess,zipfile
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
FORECAST=HERE.parent
REPO=FORECAST.parent
OUT=FORECAST/'submissions/2025-03-stage2'
from preserved_submissions import check_previous

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def native_count(path):
    if path.suffix=='.cbm':
        from catboost import CatBoostRegressor
        estimator=CatBoostRegressor();estimator.load_model(str(path));count=estimator.tree_count_
    elif path.suffix=='.txt':
        import lightgbm as lgb
        estimator=lgb.Booster(model_str=path.read_text(encoding='utf-8'));count=estimator.current_iteration()
    else:
        import xgboost as xgb
        estimator=xgb.Booster();estimator.load_model(bytearray(path.read_bytes()));count=estimator.num_boosted_rounds()
    del estimator;gc.collect()
    return int(count)

def main(validation_only=False):
    frozen_sha=sha(HERE/'frozen_final_selection.json')
    ids=pd.read_csv(FORECAST/'submissions/2025-03/baseline/test_catboost_log_level.csv').new_id.to_numpy()
    native={};csv_rows=[];metrics=[]
    counts_path=HERE/'native_count_checks.json'
    counts=json.loads(counts_path.read_text()) if counts_path.exists() else {}
    def checked_count(path,digest):
        key=path.relative_to(REPO).as_posix()
        if key not in counts or counts[key]['sha256']!=digest:counts[key]=dict(sha256=digest,trees=native_count(path))
        return counts[key]['trees']
    for root,inner in [(HERE/'results',False),(HERE/'inner',True)]:
        for path in root.rglob('*.json'):
            row=json.loads(path.read_text());prediction=path.with_suffix('.csv')
            assert sha(prediction)==row['prediction_sha256'],path
            frame=pd.read_csv(prediction)
            assert len(frame)==18657 and frame.new_id.is_unique and np.isfinite(frame.prediction).all()
            mape=float((abs(frame.prediction-frame.pto)/frame.pto).mean()*100)
            assert abs(mape-row['mape'])<1e-9,path
            model=HERE/row['model_file'];assert sha(model)==row['model_sha256'],model
            if str(model) not in native:native[str(model)]=checked_count(model,row['model_sha256'])
            assert native[str(model)]>=row['trees'] if inner else native[str(model)]==row['trees'],path
            if not inner:
                assert row['outer_target_used_for_stopping'] is False
                assert row['inner_fold']<row['fold']
            metrics.append(dict(file=path.relative_to(HERE).as_posix(),mape=mape,trees=row['trees'],native_trees=native[str(model)],inner=inner))
    counts_path.write_text(json.dumps(counts,indent=2))
    if validation_only:
        (HERE/'development_integrity.json').write_text(json.dumps(dict(status='passed',prediction_sets=len(metrics),native_models=len(native)),indent=2))
        print('VALIDATION_INTEGRITY_PASSED',len(metrics),len(native),flush=True)
        return
    for path in OUT.rglob('*.csv'):
        if path.name.startswith('february_inner_'):continue
        frame=pd.read_csv(path)
        assert list(frame)==['new_id','rto'] and len(frame)==18657 and frame.new_id.is_unique,path
        np.testing.assert_array_equal(np.sort(frame.new_id),np.sort(ids))
        assert np.isfinite(frame.rto).all() and (frame.rto>0).all() and path.stat().st_size<=1000000,path
        info_path=path.with_suffix('.json')
        if info_path.exists():
            row=json.loads(info_path.read_text())
            assert row['submission_sha256']==sha(path) and row['frozen_selection_sha256']==frozen_sha
            assert row['actual_march_target_available'] is False and row['early_stopping_on_march'] is False
            assert row['native_reload_prediction_check'] is True
            model=OUT/row['model_file'];assert sha(model)==row['model_sha256']
            if str(model) not in native:native[str(model)]=checked_count(model,row['model_sha256'])
            assert native[str(model)]==row['trees']
        csv_rows.append(dict(file=path.relative_to(OUT).as_posix(),rows=len(frame),bytes=path.stat().st_size,sha256=sha(path)))
    march=pd.read_parquet(FORECAST/'cache/final_2025-03/test_features.parquet',columns=['pto','year_month_index'])
    assert march.pto.isna().all() and (march.year_month_index==26).all()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(OUT/'test.csv')==sha(OUT/manifest['source'])
    with zipfile.ZipFile(OUT/'march_submissions_stage2.zip') as archive:
        assert archive.testzip() is None
        for row in csv_rows:assert hashlib.sha256(archive.read(row['file'])).hexdigest()==row['sha256']
    previous_integrity=check_previous()
    assert previous_integrity['status']=='passed'
    for row in manifest['references']:
        assert sha(OUT/row['file'])==row['sha256']
        expected=next(r['expected_sha256'] for r in previous_integrity['files'] if r['file']==row['source'])
        assert row['sha256']==expected
    expected={FORECAST/'baseline/original_solution.ipynb':'15a74a6b30c47fb9821b7d8ad5574868b7a793b744c7ce1997d1919aaa9caeb5',
              REPO/'Data/Shops/train_2.csv':'f1422045d80c673b28d3abcef91feb78308499eb511d4f5090a81adbebd44338'}
    for path,digest in expected.items():assert sha(path)==digest,path
    diff=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=REPO,text=True).strip()
    assert not diff,'Tracked original files changed: '+diff
    result=dict(status='passed',csv_files=csv_rows,native_models_checked=len(native),
        validated_prediction_sets=len(metrics),native_tree_counts_and_sha256=True,
        all_march_targets_unknown=True,original_notebook_and_raw_data_unchanged=True,
        previous_submission_references_match_original_manifest=True,
        damaged_previous_loose_files=previous_integrity['damaged_loose_files'],
        original_submission_directory_modified_by_stage2=False,
        tracked_original_files_unchanged=True,archive_crc_and_hashes=True,
        frozen_selection_sha256=frozen_sha,competition_submissions_sent=False)
    (HERE/'artifact_checks.json').write_text(json.dumps(result,indent=2))
    counts_path.write_text(json.dumps(counts,indent=2))
    pd.DataFrame(metrics).to_csv(HERE/'prediction_integrity.csv',index=False)
    print('FINAL_ARTIFACT_CHECKS_PASSED',len(native),len(metrics),len(csv_rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validation-only',action='store_true');args=parser.parse_args()
    main(args.validation_only)
