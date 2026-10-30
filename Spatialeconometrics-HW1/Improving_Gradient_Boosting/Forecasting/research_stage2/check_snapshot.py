"""Assess an optimisation without assuming that retrospective feature slicing is safe."""
from pathlib import Path
import gc,json,sys
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import load_recipe,matrices
from run_validation import get_fold

def main():
    recipe=load_recipe();cache=HERE.parent/'cache/final_2025-03'
    source=pd.read_parquet(cache/'test_train_features.parquet')
    manifest=json.loads((cache/'manifest.json').read_text());results=[]
    for fold in ['2024-09','2024-10','2024-12','2025-01','2025-02']:
        year,month=map(int,fold.split('-'));time=(year-2023)*12+month-1
        original=get_fold(recipe,year,month)
        candidate=dict(original,train_features=source[source.year_month_index<time].copy(),
            val_features=source[source.year_month_index==time].copy())
        for model in ['catboost_log_level','lightgbm_log_diff','lightgbm_log_level','xgboost_log_diff']:
            before=matrices(recipe,original,model);after=matrices(recipe,candidate,model)
            passed=True;reason=None
            try:
                for i in range(4):
                    a=before[i].reset_index(drop=True);b=after[i].reset_index(drop=True)
                    pd.testing.assert_equal(a,b) if hasattr(pd.testing,'assert_equal') else (
                        pd.testing.assert_frame_equal(a,b,check_exact=True) if isinstance(a,pd.DataFrame) else
                        pd.testing.assert_series_equal(a,b,check_exact=True))
            except AssertionError as e:passed=False;reason=str(e)[:600]
            results.append(dict(fold=fold,model=model,byte_equal=passed,reason=reason))
        del original,candidate;gc.collect()
    (HERE/'snapshot_diagnostics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
