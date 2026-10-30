"""Import completed private CPU workers while retaining duplicate native artifacts."""


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
import hashlib,json,zipfile
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    records=[]
    for profile,name in [('kaggle_cpu','cpu_development.zip'),('kaggle_final','full_august.zip')]:
        folder=HERE/profile.replace("_", "-")/'output';check=json.loads((folder/'cpu_checks.json').read_text())
        assert check['status']=='passed' and check['backend']=='CPU' and check['original_legacy_byte_parity']
        archive=folder/_directory_path(name)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            # Inner fits are deterministic but CatBoost native files include
            # training timestamps/IDs. Keep the first complete inner artifact.
            duplicate={n.rsplit('.',1)[0] for n in z.namelist() if n.startswith('inner/') and n.endswith('.json') and (HERE/n).exists()}
            for stem in duplicate:
                before=json.loads((HERE/(stem+'.json')).read_text());after=json.loads(z.read(stem+'.json'))
                assert before['trees']==after['trees'] and before['parameters']==after['parameters']
                pd.testing.assert_frame_equal(pd.read_csv(HERE/(stem+'.csv')),pd.read_csv(z.open(stem+'.csv')),check_exact=True)
            skip=set()
            for stem in duplicate:
                row=json.loads(z.read(stem+'.json'));skip.update([stem+'.json',stem+'.csv',row['model_file']])
            for name_in in z.namelist():
                destination=(HERE/_directory_path(name_in)).resolve();assert destination.is_relative_to(HERE.resolve())
                if name_in in skip:continue
                data=z.read(name_in)
                if destination.exists():assert destination.read_bytes()==data,name_in
                else:destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        records.append(dict(profile=profile,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),checks=check,duplicate_inner_artifacts=len(duplicate)))
    (HERE/'cpu_import_checks.json').write_text(json.dumps(dict(status='passed',archives=records),indent=2))
    print('CPU_ARTIFACTS_IMPORTED',flush=True)

if __name__=='__main__':main()
