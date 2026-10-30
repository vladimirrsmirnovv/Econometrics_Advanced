"""Validate downloaded CUDA predictions and compare the unchanged ensemble."""
from __future__ import annotations

import json
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from notebook_api import ROOT, MODELS, load_recipe, matrices
from run_validation import original_blends, get_fold


def main():
    output = HERE / 'output'
    gpu_metrics = json.loads(next(output.rglob('gpu_metrics.json')).read_text())
    feature_checks = json.loads(next(output.rglob('feature_checks.json')).read_text())
    assert feature_checks['passed']
    assert gpu_metrics['device'].startswith('cuda')
    gpu = pd.read_csv(next(output.rglob('xgboost_log_diff_validation_predictions.csv'))).set_index('new_id').sort_index()
    assert len(gpu) == 18657 and gpu.index.is_unique
    assert np.isfinite(gpu.rto_pred).all() and (gpu.rto_pred >= 0).all()
    directory = ROOT / 'Forecasting/results/2025-02'
    base = {}
    anchor = pd.read_csv(directory / 'catboost_log_level.csv').set_index('new_id').sort_index()
    pd.testing.assert_index_equal(gpu.index, anchor.index)
    np.testing.assert_allclose(gpu.rto_true, anchor.pto, rtol=0, atol=1e-7)
    recipe = load_recipe()
    _, _, local_labels, _, _ = matrices(recipe, get_fold(recipe, 2025, 2), 'xgboost_log_diff')
    remote_labels = pd.read_parquet(next(output.rglob('training_labels_float64.parquet'))).target
    np.testing.assert_array_equal(np.isfinite(local_labels), np.isfinite(remote_labels))
    np.testing.assert_allclose(local_labels, remote_labels, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.asarray(local_labels, dtype='float32'),
                               np.asarray(remote_labels, dtype='float32'), rtol=0, atol=0, equal_nan=True)
    label_difference = float(np.nanmax(np.abs(np.asarray(local_labels) - np.asarray(remote_labels))))
    actual_metrics = recipe._make_metrics(anchor.pto, gpu.rto_pred)
    np.testing.assert_allclose(actual_metrics['mape'], gpu_metrics['mape'], atol=1e-10, rtol=0)
    for model in MODELS:
        table = pd.read_csv(directory / (model + '.csv')).set_index('new_id').sort_index()
        pd.testing.assert_index_equal(table.index, anchor.index)
        np.testing.assert_allclose(table.pto, anchor.pto, rtol=0, atol=0)
        base[model] = table.prediction.to_numpy()
    base['xgboost_log_diff'] = gpu.rto_pred.to_numpy()
    variants = original_blends(base)
    records = []
    for name, prediction in variants.items():
        metrics = recipe._make_metrics(anchor.pto, prediction)
        metrics.update(model='CUDA_' + name, year=2025, month=2, validation_only=True)
        records.append(metrics)
        table = anchor[['region', 'settlement', 'pto']].copy()
        table['prediction'] = prediction
        table.reset_index().to_csv(directory / ('CUDA_' + name + '.csv'), index=False)
        (directory / ('CUDA_' + name + '.json')).write_text(json.dumps(metrics, indent=2))
    changed = dict(base)
    spatial = pd.read_csv(directory / 'catboost_log_level_exp50_history.csv').set_index('new_id').sort_index()
    pd.testing.assert_index_equal(spatial.index, anchor.index)
    changed['catboost_log_level'] = spatial.prediction.to_numpy()
    improved = original_blends(changed)['final_meta_logmean']
    improved_metrics = recipe._make_metrics(anchor.pto, improved)
    base_mape = records[-1]['mape']
    improved_metrics.update(model='CUDA_final_meta_replace_catboost_history', year=2025, month=2,
                            validation_only=True, baseline_mape=base_mape,
                            delta_mape=improved_metrics['mape'] - base_mape)
    records.append(improved_metrics)
    for label, prediction, metrics in [('xgboost_log_diff_cuda', gpu.rto_pred.to_numpy(), gpu_metrics),
                                      ('CUDA_final_meta_replace_catboost_history', improved, improved_metrics)]:
        metrics = dict(metrics, model=label, year=2025, month=2, validation_only=True)
        table = anchor[['region', 'settlement', 'pto']].copy()
        table['prediction'] = prediction
        table.reset_index().to_csv(directory / (label + '.csv'), index=False)
        (directory / (label + '.json')).write_text(json.dumps(metrics, indent=2))
    final_path = next(output.rglob('test_xgboost_log_diff.csv'))
    final = pd.read_csv(final_path).set_index('new_id').sort_index()
    pd.testing.assert_index_equal(final.index, anchor.index)
    assert np.isfinite(final.rto).all() and (final.rto >= 0).all()
    saved = HERE / 'results'
    saved.mkdir(exist_ok=True)
    shutil.copyfile(final_path, saved / final_path.name)
    for name in ['gpu_metrics.json', 'runtime.json', 'feature_checks.json']:
        shutil.copyfile(next(output.rglob(name)), saved / name)
    historical = json.loads((ROOT / 'Forecasting/baseline/saved_notebook_metrics.json').read_text())['models']['xgboost_log_diff']
    historical_differences = {name: gpu_metrics[name] - value for name, value in historical.items()}
    historical_match = all(np.isclose(gpu_metrics[name], value, rtol=1e-10, atol=1e-10)
                           for name, value in historical.items())
    comparison = dict(gpu_mape=gpu_metrics['mape'], saved_notebook_mape=4.4314600477414245,
                      gpu_best_iteration=gpu_metrics['best_iteration'],
                      saved_notebook_best_iteration=440,
                      delta_from_saved_mape=gpu_metrics['mape'] - 4.4314600477414245,
                      reproduced_cuda_ensemble_mape=base_mape,
                      catboost_history_replacement_mape=improved_metrics['mape'],
                      replacement_delta_mape=improved_metrics['mape'] - base_mape,
                      validation_stores=len(gpu), forecast_stores=len(final),
                      original_public_score_verified=False, feature_parity_passed=True,
                      max_training_label_difference_float64=label_difference,
                      effective_float32_labels_identical=True,
                      xgboost_version=gpu_metrics['runtime_versions']['xgboost'],
                      historical_metrics_match=bool(historical_match),
                      historical_metric_differences=historical_differences)
    (saved / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    pd.DataFrame(records).to_csv(saved / 'ensemble_metrics.csv', index=False)
    archive = saved / gpu_metrics['runtime_versions']['xgboost']
    archive.mkdir(exist_ok=True)
    for path in saved.iterdir():
        if path.is_file():
            shutil.copyfile(path, archive / path.name)
    shutil.copyfile(next(output.rglob('xgboost_log_diff_validation_predictions.csv')),
                    archive / 'xgboost_log_diff_validation_predictions.csv')
    model_hashes = {}
    for name in ['val_model.ubj', 'full_model.ubj']:
        source = next(output.rglob(name))
        shutil.copyfile(source, archive / name)
        model_hashes[name] = hashlib.sha256((archive / name).read_bytes()).hexdigest()
    (archive / 'model_hashes.json').write_text(json.dumps(model_hashes, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
