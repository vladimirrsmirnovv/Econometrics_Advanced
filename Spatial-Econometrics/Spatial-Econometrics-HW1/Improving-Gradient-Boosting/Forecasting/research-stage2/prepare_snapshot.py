"""Reuse a tested feature snapshot; independently verify new cutoffs on Kaggle."""
from pathlib import Path
import hashlib,json
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    checks=json.loads((HERE/'snapshot_diagnostics.json').read_text())
    assert len(checks)==20 and all(r['byte_equal'] for r in checks)
    source=HERE.parent/'cache/final_2025-03';manifest=json.loads((source/'manifest.json').read_text())
    full=pd.read_parquet(source/'test_train_features.parquet')
    for fold in ['2024-05','2024-06','2024-07','2024-08','2024-11']:
        folder=HERE.parent/'cache'/fold
        if (folder/'manifest.json').exists():continue
        year,month=map(int,fold.split('-'));time=(year-2023)*12+month-1
        train=full[full.year_month_index<time].copy();valid=full[full.year_month_index==time].copy()
        assert len(valid)==18657 and train.year_month_index.max()<time
        folder.mkdir(exist_ok=True)
        train.to_parquet(folder/'train.parquet',index=False);valid.to_parquet(folder/'holdout.parquet',index=False)
        info=dict(raw_sha256=manifest['raw_sha256'],notebook_sha256=manifest['notebook_sha256'],
            categorical_cols=manifest['categorical_cols'],feature_cols=manifest['feature_cols'],
            prepared_backend='snapshot optimisation; existing 20 model-matrix comparisons exact',
            source_snapshot_sha256=hashlib.sha256((source/'test_train_features.parquet').read_bytes()).hexdigest(),
            new_cutoff_gpu_parity_pending=True)
        (folder/'manifest.json').write_text(json.dumps(info,indent=2))
        print('SNAPSHOT_PREPARED',fold,len(train),flush=True)

if __name__=='__main__':main()
