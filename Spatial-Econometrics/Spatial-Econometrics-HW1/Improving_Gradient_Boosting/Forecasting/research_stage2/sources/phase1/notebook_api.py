"""Reuse the preserved notebook recipe without executing installation or output cells."""
from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path
import types
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'Forecasting/baseline/original_solution.ipynb'
NOTEBOOK_SHA256 = '15a74a6b30c47fb9821b7d8ad5574868b7a793b744c7ce1997d1919aaa9caeb5'
MODELS = ['catboost_log_level', 'lightgbm_log_diff', 'lightgbm_log_level', 'xgboost_log_diff']


def load_recipe():
    assert hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() == NOTEBOOK_SHA256
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    recipe = types.ModuleType('preserved_x5_recipe')
    namespace = recipe.__dict__
    import holidays
    namespace['holidays'] = holidays
    for index in [10, 14, 16, 20, 26, 37]:
        source = ''.join(notebook['cells'][index]['source'])
        exec(compile(source, f'{NOTEBOOK.name}:cell_{index}', 'exec'), namespace)
    rename_node = ast.parse(''.join(notebook['cells'][8]['source'])).body[0]
    recipe.rename_cols = ast.literal_eval(rename_node.value)
    setup = ast.parse(''.join(notebook['cells'][18]['source']))
    for node in setup.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'COMMON_BASE_FEATURE_KWARGS', 'ID_COL', 'YEAR_COL', 'MONTH_COL', 'TIME_COL', 'TARGET_COL'}:
                namespace[name] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module(body=[node], type_ignores=[]), f'{NOTEBOOK.name}:cell_18', 'exec'), namespace)
    return recipe


def read_raw(recipe):
    frame = pd.read_csv(ROOT / 'Data/Shops/train_2.csv').rename(columns=recipe.rename_cols)
    assert not frame.duplicated(['new_id', 'year', 'month']).any()
    assert frame.pto.notna().all()
    return frame


def build_fold(recipe, raw, year, month, *, split_mode='train_val', verbose=True):
    kwargs = dict(split_mode=split_mode, min_year=2023, val_year=year, val_month=month,
                  base_feature_kwargs=recipe.COMMON_BASE_FEATURE_KWARGS, verbose=verbose)
    if split_mode == 'train_val_test':
        kwargs.update(test_year=2025, test_month=3, expected_test_size=raw.new_id.nunique())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        output = recipe.build_retail_features_with_split(raw, **kwargs)
    return output


def compact_fold(recipe, output):
    requested = {'new_id', 'year_month_index', 'pto', 'pto_lag_1', 'region', 'settlement'}
    for model in MODELS:
        for name in getattr(recipe, model + '_features'):
            requested.add(name[:-3] if name.endswith('_cb') else name)
            if name.endswith('_cat_cb'):
                requested.add(name[:-7])
    compact = {key: None for key in ['train_features', 'val_features', 'test_train_features', 'test_features']}
    for key in compact:
        frame = output.get(key)
        if frame is not None:
            compact[key] = frame[[c for c in frame.columns if c in requested]].copy()
            # Early backtests have no observed March 2024 yet. The original
            # generator omits these unavailable anchors instead of emitting NA.
            # Preserve their absence as NA, never use the later March target.
            for name in recipe.catboost_log_level_features:
                if name not in compact[key] and ('feb_mar_growth_2024' in name or 'region_march_growth' in name):
                    compact[key][name] = np.nan
    compact['categorical_cols'] = output['categorical_cols']
    compact['feature_cols'] = output['feature_cols']
    return compact


def matrices(recipe, compact, model):
    features = getattr(recipe, model + '_features')
    if model == 'catboost_log_level':
        train, valid = compact['train_features'], compact['val_features']
        cats = [c for c in features if (pd.api.types.is_object_dtype(train[c])
                or pd.api.types.is_string_dtype(train[c]) or isinstance(train[c].dtype, pd.CategoricalDtype))]
        cats += [c for c in ['new_id', 'month', 'prev_month'] if c in features and c not in cats]
        xtrain = recipe._cb_prepare_X(train, features, cats, name='validation_train')
        xvalid = recipe._cb_prepare_X(valid, features, cats, name='validation_holdout')
        ytrain = recipe.make_forecast_target(train, mode='direct_log1p')
        yvalid = recipe.make_forecast_target(valid, mode='direct_log1p')
        return xtrain, xvalid, ytrain, yvalid, cats
    mode = 'growth_log' if model.endswith('log_diff') else 'direct_log1p'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        pack = recipe._prepare_numeric_model_pack(compact, selected_features=features,
                target_mode=mode, label=model, verbose=False)
    valid = pack['validation']
    return valid['X_train'][features], valid['X_val'][features], valid['y_train'], valid['y_val'], []


def fit_validation(recipe, compact, model, *, extra_train=None, extra_valid=None, threads=4):
    """Notebook validation fit; skip the separate full-history submission fit."""
    xtrain, xvalid, ytrain, yvalid, cats = matrices(recipe, compact, model)
    if extra_train is not None:
        assert extra_train.index.equals(xtrain.index) and extra_valid.index.equals(xvalid.index)
        xtrain = pd.concat([xtrain, extra_train], axis=1)
        xvalid = pd.concat([xvalid, extra_valid], axis=1)
    params = dict(getattr(recipe, model + '_params'))
    changes = {}
    if model == 'catboost_log_level':
        from catboost import CatBoostRegressor, Pool
        params.update(thread_count=threads, allow_writing_files=False, verbose=100)
        estimator = CatBoostRegressor(**params)
        estimator.fit(Pool(xtrain, ytrain, cat_features=cats),
                      eval_set=Pool(xvalid, yvalid, cat_features=cats),
                      early_stopping_rounds=50, use_best_model=True)
        prediction = recipe._cb_inverse_log1p(estimator.predict(xvalid))
        best_iteration = estimator.get_best_iteration()
    elif model.startswith('lightgbm'):
        import lightgbm as lgb
        params = recipe._clean_lgbm_params(params)
        params['n_jobs'] = threads
        estimator = lgb.LGBMRegressor(**params)
        # Exactly the notebook target filtering; its current LightGBM path keeps
        # missing growth targets. We retain that recipe for baseline reproduction.
        estimator.fit(xtrain, ytrain, eval_set=[(xvalid, yvalid)], eval_metric='mae',
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        raw_prediction = estimator.predict(xvalid, num_iteration=estimator.best_iteration_)
        mode = 'growth_log' if model.endswith('log_diff') else 'direct_log1p'
        prediction = recipe.inverse_forecast_target(raw_prediction, compact['val_features'], mode=mode)
        best_iteration = estimator.best_iteration_
    else:
        from xgboost import XGBRegressor
        params = recipe._clean_xgb_params(params, keep_early_stopping=True)
        changes['device'] = {'original': params.get('device'), 'local': 'cpu'}
        params.update(device='cpu', n_jobs=threads)
        train_mask = np.isfinite(np.asarray(ytrain, dtype=float))
        valid_mask = np.isfinite(np.asarray(yvalid, dtype=float))
        estimator = XGBRegressor(**params)
        estimator.fit(xtrain.loc[train_mask], ytrain.loc[train_mask],
                      eval_set=[(xvalid.loc[valid_mask], yvalid.loc[valid_mask])], verbose=100)
        prediction = recipe.inverse_forecast_target(estimator.predict(xvalid), compact['val_features'], mode='growth_log')
        best_iteration = estimator.best_iteration
    metrics = recipe._make_metrics(compact['val_features'].pto.to_numpy(), prediction)
    metrics.update(model=model, best_iteration=int(best_iteration), features=len(xtrain.columns),
                   train_rows=len(xtrain), holdout_rows=len(xvalid), device_changes=changes,
                   threads=threads)
    del xtrain, xvalid, ytrain, yvalid
    gc.collect()
    return estimator, prediction, metrics
