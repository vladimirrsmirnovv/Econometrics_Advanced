"""Whole-family, early and late time-cutoff checks, including learnt forecasters."""
from __future__ import annotations
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build_features import compute, load_recipe, read_raw

def main():
    raw=read_raw(load_recipe())
    reference,info,_=compute(raw)
    numeric=[c for c in reference if c not in ['new_id','year_month_index']]
    results=[]
    for year,month in [(2024,3),(2025,1),(2025,2)]:
        cutoff=(year-2023)*12+month-1
        perturbed=raw.copy()
        future=(perturbed.year-2023)*12+perturbed.month-1>=cutoff
        perturbed.loc[future,'pto']=np.random.default_rng(31).uniform(1e4,1e10,future.sum())
        altered,_,_=compute(perturbed)
        before=reference.loc[reference.year_month_index<=cutoff,numeric]
        after=altered.loc[altered.year_month_index<=cutoff,numeric]
        np.testing.assert_allclose(before,after,rtol=0,atol=1e-6,equal_nan=True)
        results.append(dict(fold=f'{year}-{month:02}',target_and_all_future_replaced=True,
            train_and_holdout_feature_rows_checked=len(before),features=len(numeric),passed=True))
        del altered,perturbed;gc.collect()
        print('PASSED',results[-1],flush=True)
    # Row order is important for within-store shifts; verify arbitrary raw order.
    shuffled,_,_=compute(raw.sample(frac=1,random_state=42))
    pd.testing.assert_frame_equal(reference,shuffled,check_exact=False,rtol=0,atol=1e-6)
    result=dict(status='passed',time_tests=results,row_permutation_passed=True,
        feature_version=info['feature_version'],implementation_hashes=info['implementation_hashes'],
        native_float_tolerance=1e-6,training_cutoff='strictly earlier realised outcomes')
    (HERE/'feature_checks.json').write_text(json.dumps(result,indent=2))
    print('ALL_FEATURE_FAMILIES_PASSED',flush=True)

if __name__=='__main__':main()
