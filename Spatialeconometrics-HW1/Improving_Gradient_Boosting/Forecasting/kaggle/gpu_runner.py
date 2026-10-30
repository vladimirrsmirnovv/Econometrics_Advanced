"""Kaggle entry point: run only the original XGBoost validation and final fits."""
from __future__ import annotations

import ast
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import xgboost

EXPECTED_EFFECTIVE_TARGETS = None


def fingerprint(frame):
    return dict(columns=list(frame.columns), dtypes=[str(t) for t in frame.dtypes],
                rows=len(frame), sha256=hashlib.sha256(
                    pd.util.hash_pandas_object(frame, index=False, categorize=False).values.tobytes()).hexdigest())


def main():
    inputs = list(Path('/kaggle/input').rglob('input_manifest.json'))
    assert len(inputs) == 1, 'Expected exactly one reproduction input package'
    source = inputs[0].parent
    out = Path('/kaggle/working')
    manifest = json.loads(inputs[0].read_text())
    for name, digest in manifest['input_hashes'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    hardware = subprocess.check_output(['nvidia-smi'], text=True)
    runtime = dict(python=platform.python_version(), gpu=hardware,
                   versions={name: importlib.metadata.version(name) for name in
                             ['xgboost', 'pandas', 'numpy', 'pyarrow', 'holidays']},
                   input_manifest=manifest)
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    print('RUNTIME', runtime['python'], runtime['versions'], hardware, flush=True)
    notebook = json.loads((source / 'original_solution.ipynb').read_text())
    namespace = dict(holidays=holidays)
    # Reuse code definitions and parameter registries only. Installation cells,
    # markdown and stored outputs from the provided notebook are never executed.
    for index in [10, 14, 16, 37]:
        exec(compile(''.join(notebook['cells'][index]['source']),
                     f'original_solution:cell_{index}', 'exec'), namespace)
    rename = ast.literal_eval(ast.parse(''.join(notebook['cells'][8]['source'])).body[0].value)
    for node in ast.parse(''.join(notebook['cells'][18]['source'])).body:
        if isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'original_preparation', 'exec'), namespace)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'COMMON_BASE_FEATURE_KWARGS', 'ID_COL', 'YEAR_COL',
                                      'MONTH_COL', 'TIME_COL', 'TARGET_COL'}:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
    raw = pd.read_csv(source / 'train_2.csv').rename(columns=rename)
    assert len(raw) == 485082 and raw.new_id.nunique() == 18657
    with __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', pd.errors.PerformanceWarning)
        features = namespace['build_retail_features_with_split'](
            raw, split_mode='train_val_test', min_year=2023,
            val_year=2025, val_month=2, test_year=2025, test_month=3,
            create_test_if_missing=True, expected_test_size=18657,
            use_val_as_test_history=True,
            base_feature_kwargs=namespace['COMMON_BASE_FEATURE_KWARGS'], verbose=True)
    del raw
    gc.collect()
    selected = namespace['xgboost_log_diff_features']
    pack = namespace['_prepare_numeric_model_pack'](
        features, selected_features=selected, target_mode='growth_log',
        label='xgboost_log_diff', verbose=True)
    validation, final = pack['validation'], pack['test']
    expected = json.loads((source / 'expected_validation_features.json').read_text())
    actual = {name: fingerprint(frame) for name, frame in {
        'train': validation['X_train'][selected], 'holdout': validation['X_val'][selected],
        'ytrain': pd.DataFrame({'target': validation['y_train']}),
        'yholdout': pd.DataFrame({'target': validation['y_val']})}.items()}
    # Linux and Windows may differ in the last float64 bits of np.log. XGBoost
    # stores labels in float32; compare the effective finite values and drop mask.
    effective = {name: fingerprint(frame.astype('float32').where(np.isfinite(frame), np.nan))
                 for name, frame in {'ytrain': pd.DataFrame({'target': validation['y_train']}),
                                     'yholdout': pd.DataFrame({'target': validation['y_val']})}.items()}
    pd.DataFrame({'target': validation['y_train']}).to_parquet(out / 'training_labels_float64.parquet', index=False)
    passed = all(actual[k] == expected[k] for k in ['train', 'holdout']) and effective == EXPECTED_EFFECTIVE_TARGETS
    (out / 'feature_checks.json').write_text(json.dumps(dict(expected=expected, actual=actual,
        effective_targets_expected=EXPECTED_EFFECTIVE_TARGETS, effective_targets_actual=effective,
        passed=passed), indent=2))
    assert passed, 'GPU features or effective labels differ from the reproduced local baseline'
    print('FEATURE_PARITY_PASSED', flush=True)
    # Release unselected columns before fitting; metadata keeps original anchors.
    meta_columns = ['new_id', 'year', 'month', 'year_month_index', 'region', 'settlement',
                    'pto', 'pto_lag_1', 'opening_date_category', 'trading_area_category',
                    'population_city_type', 'lag1_rto_size_bin', 'month_name', 'holiday_month_type']
    val_meta = features['val_features'][[c for c in meta_columns if c in features['val_features']]].copy()
    test_meta = features['test_features'][[c for c in meta_columns if c in features['test_features']]].copy()
    del features
    gc.collect()
    params = dict(namespace['xgboost_log_diff_params'])
    assert params['device'] == 'cuda' and params['n_estimators'] == 3750
    result = namespace['run_xgboost_log_diff'](
        params=params, features=selected,
        val_train_df=validation['X_train'], val_test_df=validation['X_val'],
        val_y_train=validation['y_train'], val_y_test=validation['y_val'],
        full_train_df=final['X_train'], full_test_df=final['X_test'], full_y_train=final['y_train'],
        val_test_meta_df=val_meta, full_test_meta_df=test_meta,
        output_dir=str(out / 'outputs_xgboost_log_diff'),
        submission_filename='test_xgboost_log_diff.csv',
        final_iterations_strategy='params', save_files=True, verbose=True)
    configuration = json.loads(result['val_model'].get_booster().save_config())
    actual_device = configuration['learner']['generic_param']['device']
    assert actual_device.startswith('cuda'), 'XGBoost silently fell back to CPU'
    metrics = result['val_metrics']
    metrics.update(device=actual_device, notebook_sha256=manifest['input_hashes']['original_solution.ipynb'],
                   original_saved_mape=4.4314600477414245,
                   delta_from_saved_mape=metrics['mape'] - 4.4314600477414245,
                   runtime_versions=runtime['versions'])
    (out / 'gpu_metrics.json').write_text(json.dumps(metrics, indent=2))
    for label in ['val', 'full']:
        result[label + '_model'].save_model(out / (label + '_model.ubj'))
    print('GPU_RESULT', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
