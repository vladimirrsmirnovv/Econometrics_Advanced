"""Import post-freeze forecasts and verify their original March inputs."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import pandas as pd
HERE=Path(__file__).resolve().parent

def main(only_backend=None):
    frozen_hash=hashlib.sha256((HERE/'frozen_final_selection.json').read_bytes()).hexdigest();checks=[]
    for backend,profile in [('cpu_main','kaggle_final'),('cpu_level','research'),('cuda','kaggle_cpu')]:
        if only_backend is not None and backend!=only_backend:continue
        folder=HERE.parent/'research/kaggle/output' if profile=='research' else HERE/profile/'output'
        row=json.loads((folder/('reserved_'+backend+'_checks.json')).read_text())
        assert row['status']=='passed' and row['frozen_selection_sha256']==frozen_hash and row['original_legacy_exact']
        assert row['canonical_features_sha256']==hashlib.sha256((HERE/'cache/features.parquet').read_bytes()).hexdigest()
        for key,digest in row['original_march_frame_fingerprints'].items():
            frame=pd.read_parquet(HERE.parent/'cache/final_2025-03'/(key+'.parquet'))
            assert hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()==digest,key
        archive=folder/('reserved_'+backend+'.zip')
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():
                base=HERE.parent/'submissions/2025-03-stage2' if name.startswith('march/') else HERE
                relative=name[6:] if name.startswith('march/') else name;target=(base/relative).resolve();assert target.is_relative_to(base.resolve())
                data=z.read(name)
                if target.exists():assert target.read_bytes()==data,name
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        checks.append(dict(backend=backend,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),checks=row))
    target=HERE/'reserved_import_checks.json'
    if target.exists():
        earlier=json.loads(target.read_text())['archives']
        checks+= [row for row in earlier if row['backend'] not in {r['backend'] for r in checks}]
    status='passed' if len({r['backend'] for r in checks})==3 else 'partial'
    target.write_text(json.dumps(dict(status=status,archives=checks),indent=2));print('RESERVED_FORECASTS_IMPORTED',status)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu_main','cpu_level','cuda']);args=parser.parse_args()
    main(args.backend)
