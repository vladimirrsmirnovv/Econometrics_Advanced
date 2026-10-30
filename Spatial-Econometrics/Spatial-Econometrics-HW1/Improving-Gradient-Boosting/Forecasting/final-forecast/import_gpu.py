"""Check CUDA full fits and exact original March reproduction before importing."""


def _directory_path(value):
    """Resolve directory names from preserved metadata after the folder rename."""
    from pathlib import Path as _Path
    path = _Path(value)
    if path.is_absolute():
        return path
    directory_count = len(path.parts) - bool(path.suffix)
    return _Path(*(part.replace("_", "-") if index < directory_count else part
                   for index, part in enumerate(path.parts)))

from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import NOTEBOOK_SHA256

def main():
    source=HERE/'kaggle/output';out=HERE.parent/'submissions/2025-03'
    checks=json.loads((source/'march_checks.json').read_text());assert checks['status']=='passed' and checks['runs']==2
    original=HERE.parent/'kaggle/results/3.1.3/test_xgboost_log_diff.csv'
    saved=pd.read_csv(original,float_precision='round_trip').sort_values('new_id')
    generated=pd.read_csv(source/'baseline/test_xgboost_log_diff.csv',float_precision='round_trip').sort_values('new_id')
    np.testing.assert_array_equal(saved.new_id,generated.new_id)
    np.testing.assert_allclose(saved.rto,generated.rto,rtol=1e-12,atol=1e-6)
    relative=float(np.max(np.abs(saved.rto.to_numpy()-generated.rto.to_numpy())/saved.rto.to_numpy()))
    for variant in ['baseline','spatial']:
        directory=source/variant.replace("_", "-");row=json.loads((directory/'test_xgboost_log_diff.json').read_text())
        assert row['device'].startswith('cuda') and row['versions']['xgboost']=='3.1.3' and row['trees']==3750
        assert row['notebook_sha256']==NOTEBOOK_SHA256 and row['unknown_march_target'] and not row['early_stopping']
        csv=directory/'test_xgboost_log_diff.csv';model=directory/'test_xgboost_log_diff.ubj'
        assert hashlib.sha256(csv.read_bytes()).hexdigest()==row['submission_sha256']
        assert hashlib.sha256(model.read_bytes()).hexdigest()==row['model_sha256']
        table=pd.read_csv(csv);assert list(table)==['new_id','rto'] and len(table)==18657 and table.new_id.is_unique
        assert np.isfinite(table.rto).all() and (table.rto>=0).all()
        destination=out/variant.replace("_", "-");destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(csv,destination/csv.name)
        models=out/'models'/variant.replace("_", "-");models.mkdir(parents=True,exist_ok=True)
        shutil.copy2(model,models/'xgboost_log_diff.ubj')
        row.update(model_file=str((models/'xgboost_log_diff.ubj').relative_to(out)),
            baseline_reproduction_max_relative_difference=relative)
        (destination/'test_xgboost_log_diff.json').write_text(json.dumps(row,indent=2))
    archive=out/'gpu-audit';archive.mkdir(exist_ok=True)
    for name in ['march_runtime.json','march_checks.json','march_metrics.json']:
        shutil.copy2(source/_directory_path(name),archive/_directory_path(name))
    (archive/'original_march_parity.json').write_text(json.dumps(dict(status='passed',
        original_csv_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        original_csv_byte_identical=original.read_bytes()==(source/'baseline/test_xgboost_log_diff.csv').read_bytes(),
        max_relative_difference=relative,tolerance_relative=1e-12),indent=2))
    print('FULL_CUDA_IMPORT_PASSED; original March max relative difference',relative)

if __name__=='__main__':main()
