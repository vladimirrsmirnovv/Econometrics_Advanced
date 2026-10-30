"""Read original submission bytes, recovering from their verified archive if needed.

The original directory is read-only in this workflow. A damaged loose file is
reported and its matching archived bytes are used only for new reference copies.
"""
from pathlib import Path
import hashlib,json,zipfile

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'submissions/2025-03'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_previous(relative):
    manifest=json.loads((OLD/'submission_manifest.json').read_text())
    expected=next(row['sha256'] for row in manifest['files'] if row['file']==relative)
    current=(OLD/relative).read_bytes()
    if digest(current)==expected:return current
    with zipfile.ZipFile(OLD/'march_submissions.zip') as archive:
        assert archive.testzip() is None
        data=archive.read(relative)
    assert digest(data)==expected,relative
    return data

def check_previous():
    manifest=json.loads((OLD/'submission_manifest.json').read_text())
    rows=[]
    for row in manifest['files']:
        relative=row['file'];current=(OLD/relative).read_bytes();data=read_previous(relative)
        rows.append(dict(file=relative,expected_sha256=row['sha256'],
            loose_file_sha256=digest(current),loose_file_bytes=len(current),
            loose_file_matches_manifest=digest(current)==row['sha256'],
            reference_sha256=digest(data),
            reference_source='loose_file' if current==data else 'verified_original_archive'))
    result=dict(status='passed',original_loose_files_modified_by_stage2=False,
        damaged_loose_files=[r['file'] for r in rows if not r['loose_file_matches_manifest']],files=rows)
    (HERE/'previous_submission_integrity.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    print(json.dumps(check_previous(),indent=2))
