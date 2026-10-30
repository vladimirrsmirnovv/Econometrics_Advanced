"""Import validated follow-up fits and portable source feature caches."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import ROOT,NOTEBOOK_SHA256

def main():
    source=HERE/'kaggle/output';checks=json.loads((source/'phase2_checks.json').read_text())
    assert checks['status']=='passed'
    digest=hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest()
    for directory in (source/'prepared').iterdir():
        info=json.loads((directory/'manifest.json').read_text())
        assert info['raw_sha256']==digest and info['notebook_sha256']==NOTEBOOK_SHA256
        train=pd.read_parquet(directory/'train.parquet');holdout=pd.read_parquet(directory/'holdout.parquet')
        assert len(holdout)==18657 and holdout.new_id.is_unique
        assert train.year_month_index.max()<holdout.year_month_index.min()
        destination=HERE/'cache'/directory.name if directory.name.startswith('key_') else ROOT/'Forecasting/cache'/directory.name
        if destination.exists():raise FileExistsError(str(destination))
        destination.mkdir(parents=True)
        for name in ['train.parquet','holdout.parquet','manifest.json']:shutil.copy2(directory/name,destination/name)
    records=json.loads((source/'phase2_metrics.json').read_text());assert len(records)==17
    for row in records:
        fold,config=row['fold'],row['configuration'];directory=source/'results'/fold/'xgboost_log_diff'
        table=pd.read_csv(directory/(config+'.csv')).sort_values('new_id')
        assert len(table)==18657 and table.new_id.is_unique and np.isfinite(table.prediction).all()
        mape=float((abs(table.prediction-table.pto)/table.pto).mean()*100)
        assert abs(mape-row['mape'])<1e-10 and row['backend']=='CUDA' and row['xgboost_version']=='3.1.3'
        target=HERE/'results'/fold/'xgboost_log_diff';target.mkdir(parents=True,exist_ok=True)
        for ext in ['csv','json']:shutil.copy2(directory/(config+'.'+ext),target/(config+'.'+ext))
        model_dir=target/'models';model_dir.mkdir(exist_ok=True)
        shutil.copy2(directory/(config+'.ubj'),model_dir/(config+'.ubj'))
    archive=HERE/'kaggle/phase_2';archive.mkdir(exist_ok=True)
    for name in ['phase2_metrics.json','phase2_checks.json']:shutil.copy2(source/name,archive/name)
    print('CUDA_FOLLOWUP_IMPORT_PASSED',len(records))

if __name__=='__main__':main()
