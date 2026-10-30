"""Compare the actual March models and their already saved February predictions.

No new fit, tuning, tree-count selection or rewrite of submitted CSVs occurs.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS
from run_validation import original_blends

def main():
    out=HERE.parent/'submissions/2025-03-stage2'
    old=HERE.parent/'submissions/2025-03'
    reference=pd.read_csv(out/'reference_baseline/test_catboost_log_level.csv').new_id.to_numpy()
    feb_lag=pd.read_parquet(HERE.parent/'cache/final_2025-03/test_features.parquet',columns=['new_id','pto_lag_1']).set_index('new_id').reindex(reference).pto_lag_1.to_numpy()
    results=[];baseline={};new={};truth=None;march_components={}
    for model in MODELS:
        basefile=HERE.parent/'results/2025-02'/(model+('_cuda' if model=='xgboost_log_diff' else '')+'.csv')
        b=pd.read_csv(basefile).set_index('new_id').reindex(reference)
        n=pd.read_csv(out/'individual'/('february_inner_'+model+'.csv')).set_index('new_id').reindex(reference)
        np.testing.assert_array_equal(b.pto.to_numpy(),n.pto.to_numpy())
        truth=b.pto.to_numpy() if truth is None else truth
        baseline[model]=b.prediction.to_numpy();new[model]=n.prediction.to_numpy()
        info=json.loads((out/'individual'/('test_'+model+'.json')).read_text())
        old_info=json.loads((old/'baseline'/('test_'+model+'.json')).read_text())
        r=pd.read_csv(out/'reference_baseline'/('test_'+model+'.csv')).set_index('new_id').reindex(reference).rto.to_numpy()
        p=pd.read_csv(out/'individual'/('test_'+model+'.csv')).set_index('new_id').reindex(reference).rto.to_numpy()
        march_components[model]=p
        assert len(p)==18657 and np.isfinite(p).all() and (p>0).all()
        results.append(dict(model=model,original_full_trees=old_info['trees'],new_full_trees=info['trees'],
            february_baseline_mape=float(np.mean(abs(baseline[model]-truth)/truth)*100),
            february_new_mape=float(np.mean(abs(new[model]-truth)/truth)*100),
            march_median_change_to_baseline_percent=float(np.median(p/r-1)*100),
            march_median_growth_from_february_percent=float(np.median(p/feb_lag-1)*100),
            baseline_march_median_growth_from_february_percent=float(np.median(r/feb_lag-1)*100),
            unchanged_baseline_reference_bytes=(out/'reference_baseline'/('test_'+model+'.csv')).read_bytes()==(old/'baseline'/('test_'+model+'.csv')).read_bytes(),
            march_submission_sha256=hashlib.sha256((out/'individual'/('test_'+model+'.csv')).read_bytes()).hexdigest()))
    blends=[]
    for variant,components in [('baseline',baseline),('new',new)]:
        for rule,prediction in original_blends(components).items():
            blends.append(dict(variant=variant,rule=rule,february_mape=float(np.mean(abs(prediction-truth)/truth)*100)))
    pd.DataFrame(results).to_csv(HERE/'march_regression_model_diagnostics.csv',index=False)
    pd.DataFrame(blends).to_csv(HERE/'march_regression_february_blends.csv',index=False)
    import xgboost as xgb
    native=xgb.Booster();native.load_model(bytearray((out/'models/selected/xgboost_log_diff.ubj').read_bytes()))
    assert native.num_boosted_rounds()==1
    gate_matrix=np.column_stack([march_components[m] for m in [MODELS[0],MODELS[2],MODELS[3]]])
    gate=gate_matrix.max(axis=1)/gate_matrix.min(axis=1)>1.08
    report=dict(baseline_references_identical=all(r['unchanged_baseline_reference_bytes'] for r in results),
        models=results,february_blends=blends,new_xgboost_native_tree_count=native.num_boosted_rounds(),
        new_xgboost_base_score=json.loads(native.save_config())['learner']['learner_model_param']['base_score'],
        new_xgboost_total_tree_nodes=len(native.trees_to_dataframe()),
        march_new_ensemble_median_gate_share=float(gate.mean()),
        march_new_ensemble_xgboost_smallest_share=float((gate_matrix[:,2]==gate_matrix.min(axis=1)).mean()),
        february_is_stopping_data_not_independent_validation=True,
        source_of_submitted_score_mapping='User screenshots; approximate order, not confirmed exact file mapping',
        screenshot_scores_observed=[76.99,89.68,83.24,89.92],
        model_score_mapping_confirmed=False,
        no_submitted_csvs_modified=True)
    (HERE/'march_regression_diagnostics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
