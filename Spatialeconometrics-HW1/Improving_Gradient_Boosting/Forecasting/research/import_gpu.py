"""Verify downloaded CUDA predictions before adding them to the shared ledger."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent

def main():
    source=HERE/'kaggle/output'
    records=json.loads((source/'research_metrics.json').read_text())
    assert len(records)==66
    checked=[]
    for row in records:
        fold,config=row['fold'],row['configuration']
        directory=source/'results'/fold/'xgboost_log_diff'
        path=directory/(config+'.csv')
        prediction=pd.read_csv(path)
        reference=pd.read_parquet(HERE/'kaggle/upload'/(fold+'_holdout.parquet'),columns=['new_id','pto'])
        assert len(prediction)==18657 and prediction.new_id.is_unique
        np.testing.assert_array_equal(prediction.new_id,reference.new_id)
        np.testing.assert_allclose(prediction.pto,reference.pto,rtol=1e-14)
        assert np.isfinite(prediction.prediction).all() and (prediction.prediction>=0).all()
        mape=float((abs(prediction.prediction-prediction.pto)/prediction.pto).mean()*100)
        assert abs(mape-row['mape'])<1e-10
        assert row['backend']=='CUDA' and row['xgboost_version']=='3.1.3'
        assert row['leakage_check']=='passed'
        target=HERE/'results'/fold/'xgboost_log_diff'
        target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target/path.name)
        shutil.copy2(directory/(config+'.json'),target/(config+'.json'))
        models=target/'models';models.mkdir(exist_ok=True)
        shutil.copy2(directory/(config+'.ubj'),models/(config+'.ubj'))
        checked.append(dict(fold=fold,configuration=config,mape=mape))
    archive=HERE/'kaggle/phase_1';archive.mkdir(exist_ok=True)
    for name in ['research_metrics.json','research_runtime.json']:
        shutil.copy2(source/name,archive/name)
    (archive/'import_checks.json').write_text(json.dumps(dict(status='passed',runs=len(checked),checks=checked),indent=2))
    print('CUDA_IMPORT_PASSED',len(checked))

if __name__=='__main__':main()
