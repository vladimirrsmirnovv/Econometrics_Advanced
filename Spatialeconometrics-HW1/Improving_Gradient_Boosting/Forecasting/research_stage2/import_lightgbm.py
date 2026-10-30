"""Import only common-feature LightGBM fits; validate source CPU metric parity."""
from pathlib import Path
import hashlib,json,zipfile,shutil
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    folder=HERE/'kaggle/output';check=json.loads((folder/'cpu_checks.json').read_text())
    assert check['status']=='passed' and check['task']=='lightgbm_development' and check['original_legacy_byte_parity']
    archive=folder/'lightgbm_development.zip';parity=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for model in ['lightgbm_log_diff','lightgbm_log_level']:
            name='results/rolling/2024-08/'+model+'/baseline.csv'
            old=pd.read_csv(HERE/'precanonical_windows'/name);new=pd.read_csv(z.open(name))
            np.testing.assert_array_equal(old.new_id,new.new_id);np.testing.assert_allclose(old.prediction,new.prediction,rtol=1e-12,atol=1e-9)
            old_mape=float((abs(old.prediction-old.pto)/old.pto).mean()*100);new_mape=float((abs(new.prediction-new.pto)/new.pto).mean()*100)
            assert abs(old_mape-new_mape)<1e-10
            parity.append(dict(model=model,local_mape=old_mape,kaggle_mape=new_mape,
                maximum_relative_prediction_difference=float(np.max(abs(old.prediction/new.prediction-1)))))
        for name in z.namelist():
            target=(HERE/name).resolve();assert target.is_relative_to(HERE)
            data=z.read(name)
            if target.exists():assert target.read_bytes()==data,name
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    pd.testing.assert_frame_equal(pd.read_parquet(HERE/'cache/geo_control.parquet'),pd.read_parquet(folder/'canonical_geo_control.parquet'),check_exact=True)
    (HERE/'lightgbm_import_checks.json').write_text(json.dumps(dict(status='passed',cpu_metric_parity=parity,
        original_local_fits_preserved=True,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),runtime=json.loads((folder/'runtime_versions.json').read_text())),indent=2))
    print('COMMON_FEATURE_LIGHTGBM_IMPORTED',flush=True)

if __name__=='__main__':main()
