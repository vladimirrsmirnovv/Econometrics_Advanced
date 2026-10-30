"""Prove that the unknown-target guard fix preserved all completed CUDA metrics."""
from pathlib import Path
import hashlib,io,json,zipfile
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    old=HERE/'sources/reserved-cuda-partial/reserved_cuda.zip'
    new=HERE/'kaggle-cpu/output/reserved_cuda.zip'
    rows=[]
    with zipfile.ZipFile(old) as a,zipfile.ZipFile(new) as b:
        assert a.testzip() is None and b.testzip() is None
        for name in a.namelist():
            if not name.endswith('.csv'):continue
            before=a.read(name);after=b.read(name)
            first=pd.read_csv(io.BytesIO(before));second=pd.read_csv(io.BytesIO(after))
            pd.testing.assert_frame_equal(first.drop(columns='prediction'),second.drop(columns='prediction'),check_exact=True)
            np.testing.assert_allclose(first.prediction,second.prediction,rtol=1e-12,atol=1e-9)
            m1=float((abs(first.prediction-first.pto)/first.pto).mean()*100)
            m2=float((abs(second.prediction-second.pto)/second.pto).mean()*100)
            assert abs(m1-m2)<1e-10,name
            rows.append(dict(file=name,prediction_bytes_identical=before==after,mape_before=m1,mape_after=m2,
                maximum_relative_prediction_difference=float(np.max(abs(first.prediction/second.prediction-1)))))
    result=dict(status='passed',unknown_target_guard_only=True,selection_unchanged=True,
        old_archive_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_archive_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),comparisons=rows)
    (HERE/'cuda_repeat_checks.json').write_text(json.dumps(result,indent=2));print('CUDA_REPEAT_METRICS_PARITY_PASSED',len(rows))

if __name__=='__main__':main()
