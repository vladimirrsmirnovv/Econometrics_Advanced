"""Verify archive hashes, paths, original fold provenance and feature parity."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import argparse,hashlib,json,shutil,sys,zipfile
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent

def safe_extract(archive,destination):
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in z.infolist():
            target=(destination/item.filename).resolve()
            assert target.is_relative_to(destination) and not item.is_dir()
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(z.read(item)).hexdigest(),str(target)
            else:target.write_bytes(z.read(item))

def main():
    output=HERE/'kaggle/output';checks=json.loads((output/'stage2_gpu_checks.json').read_text())
    assert checks['status']=='passed' and checks['xgboost_version']=='3.1.3'
    for name,entry in checks['archives'].items():
        path=output/name;assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    prepared=HERE/'cache/imported';safe_extract(output/'prepared_stage2.zip',prepared)
    local=pd.read_parquet(HERE/'cache/features.parquet');gpu=pd.read_parquet(prepared/'stage2_features.parquet')
    differences=[]
    for c in local.columns[2:]:
        a=local[c].to_numpy();b=gpu[c].to_numpy()
        mask=~((a==b)|(np.isnan(a)&np.isnan(b)))
        if mask.any():differences.append(dict(feature=c,rows=int(mask.sum()),
            stores=int(local.loc[mask,'new_id'].nunique()),maximum_difference=float(np.nanmax(abs(a-b)))))
    if differences:
        # Use the same immutable input artifact for every learner. Never silently
        # compare models trained on two tie-sensitive neighbour implementations.
        backup=HERE/'cache/precanonical_features.parquet'
        if not backup.exists():shutil.copy2(HERE/'cache/features.parquet',backup)
        shutil.copy2(prepared/'stage2_features.parquet',HERE/'cache/features.parquet')
        pd.DataFrame(differences).to_csv(HERE/'geometry_platform_difference.csv',index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(HERE/'cache/features.parquet'),gpu,check_exact=True)
    source=hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest()
    for folder in prepared.iterdir():
        if not folder.is_dir():continue
        manifest=json.loads((folder/'manifest.json').read_text());assert manifest['raw_sha256']==source
        target=HERE.parent/'cache'/folder.name
        target.mkdir(exist_ok=True)
        for path in folder.iterdir():
            destination=target/path.name
            if destination.exists():
                if path.suffix=='.parquet':
                    original=pd.read_parquet(path);snapshot=pd.read_parquet(destination)
                    pd.testing.assert_frame_equal(snapshot[list(original.columns)],original,check_exact=True)
                else:
                    existing=json.loads(destination.read_text());assert existing['raw_sha256']==source
            else:shutil.copy2(path,destination)
    safe_extract(output/'cuda_development.zip',HERE)
    for info in (HERE/'results').rglob('*.json'):
        row=json.loads(info.read_text())
        if row['model']!='xgboost_log_diff':continue
        assert row['environment']['xgboost']=='3.1.3' and row['outer_target_used_for_stopping'] is False
        assert row['prediction_sha256']==hashlib.sha256(info.with_suffix('.csv').read_bytes()).hexdigest()
    result=dict(status='passed',feature_cpu_gpu_byte_parity=not bool(differences),
        canonical_feature_artifact_sha256=hashlib.sha256((HERE/'cache/features.parquet').read_bytes()).hexdigest(),
        original_local_spatial_fits_require_replacement=bool(differences),original_raw_hash=source,
        caches_imported=[p.name for p in prepared.iterdir() if p.is_dir()],archives_verified=True)
    (HERE/'gpu_import_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
