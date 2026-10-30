"""Preserve and exclude pre-canonical local LightGBM fits before comparisons."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent

def main():
    root=HERE/'precanonical-windows';root.mkdir(exist_ok=True);records=[]
    actual=hashlib.sha256((HERE/'sources/phase1/experiment.py').read_bytes()).hexdigest()
    for tree in ['results','inner']:
        for month in (HERE/tree).iterdir() if tree=='inner' else (HERE/tree/'rolling').iterdir():
            for model in ['lightgbm_log_diff','lightgbm_log_level']:
                source=month/model.replace("_", "-")
                if not source.exists():continue
                target=root/source.relative_to(HERE)
                assert source.resolve().is_relative_to(HERE.resolve()) and target.resolve().is_relative_to(root.resolve())
                assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
                for p in source.glob('*.json'):
                    row=json.loads(p.read_text())
                    if 'implementation_sha256' in row:
                        row['originally_reported_implementation_sha256']=row['implementation_sha256'];row['implementation_sha256']=actual
                        row['provenance_correction']='Original local process loaded the archived phase1 source; the source file changed while it ran. Predictions and native artifacts are untouched.'
                        p.write_text(json.dumps(row,indent=2))
                shutil.move(str(source),str(target));records.append(target.relative_to(HERE).as_posix())
    (root/'manifest.json').write_text(json.dumps(dict(reason='Spatial graph differed from the common Linux feature artifact',
        excluded_from_selection=True,predictions_and_models_preserved=True,actual_loaded_source_sha256=actual,moved=records),indent=2))
    print('PRECANONICAL_WINDOWS_FITS_ARCHIVED',len(records))

if __name__=='__main__':main()
