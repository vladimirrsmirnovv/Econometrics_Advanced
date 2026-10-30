"""Replay one actual remote validation fit on the original local CPU backend."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from notebook_api import load_recipe
from run_validation import get_fold
from experiment import fit

def main():
    directory=HERE/'results/rolling/2024-08/catboost-log-level'
    row=json.loads((directory/'baseline.json').read_text());remote=pd.read_csv(directory/'baseline.csv')
    recipe=load_recipe();compact=get_fold(recipe,2024,8)
    start=time.perf_counter();model,pred,trees,params,columns=fit(recipe,compact,'catboost_log_level','baseline',None,{},trees=row['trees'],threads=4)
    np.testing.assert_allclose(pred,remote.prediction,rtol=1e-12,atol=1e-9)
    local_mape=float((abs(pred-remote.pto.to_numpy())/remote.pto.to_numpy()).mean()*100)
    result=dict(status='passed',backend='CPU',trees=trees,local_mape=local_mape,kaggle_mape=row['mape'],
        absolute_mape_difference=abs(local_mape-row['mape']),maximum_relative_prediction_difference=float(np.max(abs(pred/remote.prediction.to_numpy()-1))),
        exact_float_predictions=bool(np.array_equal(pred,remote.prediction.to_numpy())),seconds=time.perf_counter()-start)
    (HERE/'catboost_platform_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
