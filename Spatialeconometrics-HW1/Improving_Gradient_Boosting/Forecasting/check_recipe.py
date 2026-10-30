"""Read-only input checks and small, meaningful feature leakage checks."""
from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path


import hashlib
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

from notebook_api import ROOT, NOTEBOOK_SHA256, load_recipe, read_raw, build_fold, compact_fold, matrices, MODELS
from run_validation import original_blends
from spatial_features import spatial_table


def main():
    recipe = load_recipe()
    raw = read_raw(recipe)
    result = {'notebook_sha256': NOTEBOOK_SHA256,
              'raw_sha256': hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest(),
              'rows':len(raw), 'stores':int(raw.new_id.nunique()),
              'months':int(raw[['year','month']].drop_duplicates().shape[0]),
              'first_rows':raw.head().to_dict(orient='records'), 'checks':[]}
    assert len(raw)==485082 and raw.new_id.nunique()==18657
    result['checks'].append('raw keys, dimensions and target completeness')
    locations=raw[['new_id','region','settlement']].drop_duplicates()
    ambiguous=locations.groupby('settlement').region.nunique()
    result['same_settlement_label_multiple_regions'] = int((ambiguous>1).sum())
    result['stores_with_ambiguous_settlement_label'] = int(locations.settlement.isin(ambiguous[ambiguous>1].index).sum())
    ids=np.random.default_rng(2026).choice(raw.new_id.unique(),size=160,replace=False)
    small=raw[raw.new_id.isin(ids)].copy()
    normal=build_fold(recipe,small,2025,2,verbose=False)
    compact=compact_fold(recipe,normal)
    perturbed=small.copy()
    future=(perturbed.year==2025)&(perturbed.month==2)
    for column in ['pto','avg_promo_items_per_receipt','avg_items_per_receipt','avg_cancellations','working_hours_per_day']:
        perturbed.loc[future,column] *= 17
    altered=compact_fold(recipe,build_fold(recipe,perturbed,2025,2,verbose=False))
    for model in MODELS:
        a=matrices(recipe,compact,model)
        b=matrices(recipe,altered,model)
        pd.testing.assert_frame_equal(a[0],b[0])
        pd.testing.assert_frame_equal(a[1],b[1])
    result['checks'].append('all four selected feature matrices invariant to holdout targets and current receipt/hour values')
    all_three=compact_fold(recipe,build_fold(recipe,small,2025,2,split_mode='train_val_test',verbose=False))
    for key in ['train_features','val_features']:
        pd.testing.assert_frame_equal(compact[key],all_three[key])
    result['checks'].append('validation-only features equal the notebook three-part split')
    # Check the adapted blend against the notebook functions, including both
    # sides of the 1.08 disagreement threshold and the three-model variant.
    notebook=json.loads((ROOT/'Forecasting/baseline/original_solution.ipynb').read_text(encoding='utf-8'))
    for index in [64,67,76]:
        for node in ast.parse(''.join(notebook['cells'][index]['source'])).body:
            if isinstance(node,ast.FunctionDef):
                exec(compile(ast.Module(body=[node],type_ignores=[]),'original_blend_reference','exec'),recipe.__dict__)
    rng=np.random.default_rng(314)
    pred={m:rng.lognormal(18,0.17,60) for m in MODELS}
    fast=original_blends(pred)
    results={m.upper()+'_RESULT':{'val_predictions_df':pd.DataFrame({'new_id':np.arange(60),'rto_true':np.full(60,1e8),'rto_pred':pred[m]}),
             'full_predictions_df':pd.DataFrame({'new_id':np.arange(60),'rto_pred':pred[m]})} for m in MODELS}
    tmp=ROOT/'tmp/forecasting/blend_checks';tmp.mkdir(parents=True,exist_ok=True)
    ref_log=recipe.make_log_average_blend(results,output_dir=tmp/'log',submission_filename='submission.csv')
    ref_gate=recipe.make_uncertainty_gated_blend(results,ratio_threshold=1.08,output_dir=tmp/'gate',submission_filename='submission.csv')
    no_weak={k:v for k,v in results.items() if k!='LIGHTGBM_LOG_DIFF_RESULT'}
    ref_gate3=recipe.make_uncertainty_gated_blend(no_weak,ratio_threshold=1.08,output_dir=tmp/'gate3',submission_filename='submission.csv')
    for key,ref in [('log_average',ref_log),('gated_1_08_all',ref_gate),('gated_1_08_no_weak',ref_gate3)]:
        np.testing.assert_allclose(fast[key],ref['submission_df'].rto,rtol=1e-14)
    equal=pd.DataFrame({'new_id':np.arange(60),'rto':fast['equal']});equal.to_csv(tmp/'equal.csv',index=False)
    groups={'equal':[tmp/'equal.csv'],'log':[tmp/'log/submission.csv'],'gate':[tmp/'gate/submission.csv'],'gate3':[tmp/'gate3/submission.csv']}
    ref_meta=recipe.make_meta_logmean_submission(groups,output_path=tmp/'meta.csv',verbose=False)
    np.testing.assert_allclose(fast['final_meta_logmean'],ref_meta['submission_df'].rto,rtol=1e-14)
    result['checks'].append('four candidate blends and final meta blend equal original notebook functions')
    spatial,spatial_info=spatial_table(small)
    changed=small.copy()
    future=((changed.year-2023)*12+changed.month-1)>=14
    changed.loc[future,'pto']*=23
    altered_spatial,_=spatial_table(changed.sample(frac=1,random_state=314))
    # Every row through March 2024 must ignore March 2024 and later targets.
    history_columns=[c for c in spatial if c.startswith('wy_')]+['new_id','year_month_index']
    a=spatial.loc[spatial.year_month_index<=14,history_columns].reset_index(drop=True)
    b=altered_spatial.loc[altered_spatial.year_month_index<=14,history_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    result['checks'].append('spatial revenue features invariant to row order and current/future targets through holdout')
    result['spatial_sample_metadata']=spatial_info
    result['sample_sizes']={'train':len(compact['train_features']),'holdout':len(compact['val_features'])}
    out=ROOT/'Forecasting/results';out.mkdir(parents=True,exist_ok=True)
    (out/'recipe_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='first_rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
