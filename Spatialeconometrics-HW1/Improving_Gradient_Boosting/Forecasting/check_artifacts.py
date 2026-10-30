"""Verify saved prediction keys, metrics and preservation of source inputs."""
from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path


import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from notebook_api import ROOT, NOTEBOOK, NOTEBOOK_SHA256


def main():
    assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == NOTEBOOK_SHA256
    original = Path('C:/Users/smirn/Downloads/final_solution (1).ipynb')
    if original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest() == NOTEBOOK_SHA256
    raw_hash = hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest()
    expected = json.loads((ROOT / 'Forecasting/results/recipe_checks.json').read_text(encoding='utf-8'))
    assert raw_hash == expected['raw_sha256']
    # Verify that publication changes do not modify shared research inputs.
    for args in [['git', 'diff', '--name-only'], ['git', 'diff', '--cached', '--name-only']]:
        assert not subprocess.check_output(args + ['--', 'Analysis', 'Data', 'Export', 'Paper', 'Parsing', 'Runs', 'Scripts', 'Source', 'src', 'Task'], cwd=REPOSITORY_ROOT).strip()
    checked = []
    results = ROOT / 'Forecasting/results'
    for folder in sorted(p for p in results.iterdir() if p.is_dir()):
        base = pd.read_csv(folder / 'catboost_log_level.csv').set_index('new_id').sort_index()
        assert len(base) == 18657 and base.index.is_unique and (base.pto > 0).all()
        blend_metrics = pd.read_csv(folder / 'metrics.csv').set_index('model')
        for path in sorted(folder.glob('*.csv')):
            table = pd.read_csv(path)
            if 'prediction' not in table.columns:
                continue
            assert len(table) == 18657 and table.new_id.is_unique
            table = table.set_index('new_id').sort_index()
            pd.testing.assert_index_equal(table.index, base.index)
            np.testing.assert_allclose(table.pto, base.pto, rtol=0, atol=0)
            assert np.isfinite(table.prediction).all() and (table.prediction >= 0).all()
            mape = float((100 * (table.prediction - table.pto).abs() / table.pto).mean())
            if path.with_suffix('.json').exists():
                metrics = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                saved_mape = metrics['mape']
            else:
                saved_mape = blend_metrics.loc[path.stem, 'mape']
            np.testing.assert_allclose(mape, saved_mape, rtol=0, atol=1e-10)
            checked.append(str(path.relative_to(results)))
    result = dict(status='passed', notebook_sha256=NOTEBOOK_SHA256, raw_sha256=raw_hash,
                  tracked_files_unchanged=True, prediction_files=len(checked),
                  stores_per_file=18657, checked=checked)
    (results / 'artifact_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'checked'}, indent=2))


if __name__ == '__main__':
    main()
