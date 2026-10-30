"""Verify relocation and final saved artifacts without fitting models."""
from pathlib import Path
import ast
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FORECASTING = HERE / 'Forecasting'
sys.path.insert(0, str(FORECASTING))
from notebook_api import NOTEBOOK, NOTEBOOK_SHA256, load_recipe, resource_path
import spatial_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(NOTEBOOK) == NOTEBOOK_SHA256
    assert sha(FORECASTING / 'report/Прогноз_РТО_финальный_отчёт.tex') == (
        'a90ca9c08c15b13210c9a468332da959549c78ebe7061e9a8b89ca7611f45008')
    assert sha(FORECASTING / 'report/output/pdf/Прогноз_РТО_финальный_отчёт.pdf') == (
        'da53ae13148be96b7e77f31e81f683a0de04588325730486cf68d54ee9724695')
    inputs = ['Data/Shops/train_2.csv', 'Data/settlement_panel.parquet',
              'Parsing/geonames-settlement-coordinates/result/settlement_coordinates.csv',
              'Source/Seminars/Seminar4/municipalities_models.R', 'Paper/main.tex']
    for name in inputs:
        assert resource_path(name).is_file(), name
    recipe = load_recipe()
    assert hasattr(recipe, 'rename_cols')
    assert spatial_features.location_key('Москва', 'Москва')
    parsed = 0
    for path in HERE.rglob('*.py'):
        if any(part in {'.venv', '__pycache__', 'cache', 'tmp', 'output', 'upload'}
               for part in path.relative_to(HERE).parts):
            continue
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        parsed += 1
    checked = []
    reference_ids = None
    for batch in ('2025-03-course-final-ten', '2025-03-course-final-five'):
        files = sorted((FORECASTING / 'submissions' / batch).glob('*/test.csv'))
        assert len(files) == (10 if batch.endswith('ten') else 5), batch
        for path in files:
            frame = pd.read_csv(path)
            assert list(frame.columns) == ['new_id', 'rto'], path
            assert len(frame) == 18657 and frame.new_id.is_unique, path
            assert np.isfinite(frame.rto).all() and (frame.rto > 0).all(), path
            if reference_ids is None:
                reference_ids = frame.new_id.to_numpy()
            else:
                np.testing.assert_array_equal(frame.new_id.to_numpy(), reference_ids)
            checked.append(path.relative_to(HERE).as_posix())
    result = dict(status='passed', original_notebook_unchanged=True,
                  final_report_source_and_pdf_unchanged=True, shared_inputs_resolved=inputs,
                  python_files_parsed=parsed, final_forecast_csvs_checked=len(checked),
                  rows_per_forecast=18657, new_models_trained=0, checked_forecasts=checked)
    (HERE / 'publication_checks.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'checked_forecasts'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
