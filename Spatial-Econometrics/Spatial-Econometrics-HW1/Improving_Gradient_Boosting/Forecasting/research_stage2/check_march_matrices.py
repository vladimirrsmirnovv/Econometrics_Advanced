"""Confirm the outer-forecast wrapper matches the notebook final matrix preparation."""
from pathlib import Path
import gc,json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,load_recipe,matrices

def main():
    cache=HERE.parent/'cache/final_2025-03';recipe=load_recipe();manifest=json.loads((cache/'manifest.json').read_text())
    full=pd.read_parquet(cache/'test_train_features.parquet');test=pd.read_parquet(cache/'test_features.parquet')
    original=dict(train_features=None,val_features=None,test_train_features=full,test_features=test,
        categorical_cols=manifest['categorical_cols'],feature_cols=manifest['feature_cols'])
    outer=dict(original,train_features=full,val_features=test,test_train_features=None,test_features=None)
    results=[]
    for model in MODELS[1:]:
        features=getattr(recipe,model+'_features');mode='growth_log' if model.endswith('log_diff') else 'direct_log1p'
        before=recipe._prepare_numeric_model_pack(original,selected_features=features,target_mode=mode,label=model,verbose=False)['test']
        after=matrices(recipe,outer,model)
        pd.testing.assert_frame_equal(before['X_train'][features],after[0],check_exact=True)
        pd.testing.assert_frame_equal(before['X_test'][features],after[1],check_exact=True)
        pd.testing.assert_series_equal(before['y_train'],after[2],check_exact=True)
        assert len(after[1])==18657 and np.isnan(np.asarray(after[3],dtype=float)).all()
        results.append(dict(model=model,original_final_matrices_exact=True,march_targets_all_nan=True))
        del before,after;gc.collect()
    (HERE/'march_matrix_checks.json').write_text(json.dumps(dict(status='passed',checks=results),indent=2))
    print('MARCH_MATRICES_MATCH_ORIGINAL_FINAL',flush=True)

if __name__=='__main__':main()
