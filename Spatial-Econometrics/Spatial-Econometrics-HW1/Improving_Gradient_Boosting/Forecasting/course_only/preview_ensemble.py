"""Read completed validation CSVs while the separate March fit is still running."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from manage import MODELS,load,mape,sha
from run_validation import original_blends

def main():
    rows=[]
    for fold in ['2025-01','2025-02']:
        base={};changed={};ids=None;truth=None
        for model in MODELS:
            if model=='catboost_log_level':
                path=HERE/'local_cat/results'/fold/model/'baseline.csv'
                meta=json.loads(path.with_suffix('.json').read_text())
                assert meta['trees']==6500 and not meta['early_stopping']
                assert sha(path)==meta['prediction_sha256']
                b=pd.read_csv(path).sort_values('new_id')
                assert abs(mape(b.pto,b.prediction)-meta['mape'])<1e-10
            else:b,_=load(fold,model,'baseline')
            if ids is None:ids=b.new_id.to_numpy();truth=b.pto.to_numpy()
            np.testing.assert_array_equal(ids,b.new_id.to_numpy())
            np.testing.assert_array_equal(truth,b.pto.to_numpy())
            base[model]=b.prediction.to_numpy()
            if model=='lightgbm_log_diff':
                n,_=load(fold,model,'wx_knn12')
                np.testing.assert_array_equal(ids,n.new_id.to_numpy())
                np.testing.assert_array_equal(truth,n.pto.to_numpy())
                changed[model]=n.prediction.to_numpy()
            else:changed[model]=base[model]
        before=mape(truth,original_blends(base)['final_meta_logmean'])
        after=mape(truth,original_blends(changed)['final_meta_logmean'])
        rows.append(dict(fold=fold,baseline=before,only_lightgbm_diff_WX=after,gain=before-after))
    pending=not (HERE/'imports/confirmation_cat.json').exists()
    print(json.dumps(dict(validation_preview=rows,march_artifacts_still_pending=pending),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
