"""Compare the adapter predictions with the notebook's original CPU validation fit.

This optional check fits one model. It truncates the original function immediately
after validation metrics, before the separate full-history submission fit.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from notebook_api import ROOT, NOTEBOOK, load_recipe, matrices
from run_validation import get_fold


def main():
    recipe = load_recipe()
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    tree = ast.parse(''.join(notebook['cells'][37]['source']))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'run_xgboost_log_diff')
    end = next(i for i, node in enumerate(function.body)
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'val_metrics' for t in node.targets))
    function.name = 'original_validation_only'
    function.body = function.body[:end + 1] + ast.parse('return val_pred, val_metrics').body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, 'original_xgboost_validation', 'exec'), recipe.__dict__)
    compact = get_fold(recipe, 2025, 2)
    train, valid, ytrain, yvalid, _ = matrices(recipe, compact, 'xgboost_log_diff')
    params = dict(recipe.xgboost_log_diff_params, device='cpu', n_jobs=4)
    prediction, metrics = recipe.original_validation_only(
        params=params, features=recipe.xgboost_log_diff_features,
        val_train_df=train, val_test_df=valid, val_y_train=ytrain, val_y_test=yvalid,
        full_train_df=train, full_test_df=valid, full_y_train=ytrain,
        val_test_meta_df=compact['val_features'], full_test_meta_df=compact['val_features'],
        save_files=False, verbose=False)
    base = ROOT / 'Forecasting/results/2025-02'
    saved = pd.read_csv(base / 'xgboost_log_diff.csv')
    np.testing.assert_array_equal(saved.new_id, compact['val_features'].new_id)
    np.testing.assert_allclose(prediction, saved.prediction, rtol=1e-12, atol=1e-6)
    local = json.loads((base / 'xgboost_log_diff.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(metrics['mape'], local['mape'], rtol=0, atol=1e-10)
    result = dict(status='passed',
                  reference='original notebook validation function on CPU; full-fit part skipped',
                  original_function_mape=metrics['mape'], local_adapter_mape=local['mape'],
                  max_prediction_difference_from_csv=float(np.max(np.abs(prediction - saved.prediction))))
    (ROOT / 'Forecasting/results/xgboost_adapter_check.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
