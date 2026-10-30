"""Resolve validated configurations without running final competition training."""


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
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import load_recipe

def describe(model,configuration=None):
    chosen=json.loads((HERE/'best_model_configs.json').read_text())
    configuration=configuration or chosen['components'][model]
    groups=json.loads((HERE/'cache/feature_manifest.json').read_text())['groups']
    spec=dict(model=model,configuration=configuration,
        original_parameters_source='Forecasting/baseline/original_solution.ipynb',
        original_feature_registry=model+'_features',
        full_iterations=chosen['final_tree_counts'][model],
        validation_early_stopping=50 if model!='xgboost_log_diff' else 100,
        forecast_cutoff='Feature time t uses only realised outcomes before t',
        full_training_required=True,submission_created=False)
    recipe=load_recipe()
    spec.update(original_parameters=dict(getattr(recipe,model+'_params')),
        original_features=list(getattr(recipe,model+'_features')),
        target='growth_log' if model.endswith('log_diff') else 'direct_log1p',
        inverse_target='previous_month_pto * exp(prediction)' if model.endswith('log_diff') else 'expm1(prediction)')
    full_params=dict(spec['original_parameters'])
    if model=='xgboost_log_diff':
        full_params=recipe._clean_xgb_params(full_params,keep_early_stopping=False)
        full_params.update(device='cuda',n_jobs=4,n_estimators=spec['full_iterations'])
    elif model.startswith('lightgbm'):
        full_params=recipe._clean_lgbm_params(full_params)
        full_params.update(n_jobs=4,n_estimators=spec['full_iterations'])
    else:full_params.update(iterations=spec['full_iterations'],thread_count=4,allow_writing_files=False)
    spec.update(full_parameters=full_params,full_fit_eval_set=None,
        full_fit_early_stopping=False,full_fit_use_original_preprocessing=True)
    if model=='xgboost_log_diff':spec.update(xgboost_version='3.1.3',device='cuda')
    if configuration=='density_weight':
        spec.update(base_configuration='baseline',training_weight='1 / sqrt(market_shop_count), normalised to mean 1',
            validation_weights=None,changes_training_objective=True,implementation='run_models.py',features_added=[])
    elif configuration.startswith('exp50_history+residual_'):
        spec.update(base_configuration='exp50_history',postprocessing=configuration.split('+',1)[1],
            earlier_honest_oof_required=True,previous_oof_configuration='exp50_history',correction_weight=.5,
            implementation='combine_residuals.py',feature_builder='Forecasting/spatial_features.py',
            base_validation_runner='Forecasting/run_spatial.py',
            next_forecast_previous_oof_month='2025-02',
            clipping_log_error=[-.2,.2],
            features_added=['wy_log_daily_lag1','wy_daily_growth_lag1','wy_daily_growth_1_3','wy_log_daily_gap_lag1'])
    elif configuration.startswith('residual_'):
        spec.update(base_configuration='baseline',postprocessing=configuration,
            earlier_honest_oof_required=True,correction_weight=.5,
            implementation='residual_correction.py')
    elif configuration.startswith(('sem_','sdem_')):
        spec.update(base_configuration='baseline',postprocessing=configuration,
            economic_model=configuration.split('_')[0],economy_weight=.1 if 'blend' in configuration else 1,
            implementation='sem_forecasts.py',lambda_grid=[0,.3,.6],weights='knn8')
    else:
        parts=configuration.split('+') if configuration!='baseline' else []
        key=bool(parts and parts[0]=='key')
        if key:parts=parts[1:]
        spec.update(compound_settlement_key=key,feature_groups=parts,
            features_added=list(dict.fromkeys(c for group in parts for c in groups[group])),
            feature_version='research_v1',implementation='build_features.py',
            key_implementation='key_folds.py' if key else None)
        if model!='xgboost_log_diff' and configuration in ['exp50_history','wx_means']:
            spec.update(implementation='Forecasting/spatial_features.py',legacy_feature_builder='spatial_table(kind="exp50")',
                features_added=['wy_log_daily_lag1','wy_daily_growth_lag1','wy_daily_growth_1_3','wy_log_daily_gap_lag1']
                if configuration=='exp50_history' else ['wx_log_'+c for c in
                    ['population_size','number_of_households','pedestrian_traffic_per_hour','car_traffic_per_hour','grocery_stores_500m','pyaterochka_stores_500m']],
                precision_note='Preserve the previously validated legacy subtraction before float32 conversion; see legacy_feature_parity.json')
    # Keep the selection-stage recipe, but report completed full fits when present.
    variant='baseline' if configuration=='baseline' else 'spatial'
    submission=HERE.parent/'submissions/2025-03'/variant.replace("_", "-")/('test_'+model+'.csv')
    metadata=submission.with_suffix('.json')
    if metadata.exists() and submission.exists():
        actual=json.loads(metadata.read_text(encoding='utf-8'))
        if actual['configuration']==configuration:
            assert actual['submission_sha256']==hashlib.sha256(submission.read_bytes()).hexdigest()
            spec.update(full_training_required=False,submission_created=True,
                full_training_forecast_month=actual['forecast_month'],
                submission_file=submission.resolve().as_posix(),
                trained_model_file=(submission.parents[1]/_directory_path(actual['model_file'])).resolve().as_posix(),
                full_training_metadata=metadata.resolve().as_posix())
    return spec

if __name__=='__main__':
    configs=json.loads((HERE/'best_model_configs.json').read_text())
    resolved=dict(individual={model:describe(model) for model in configs['components']},
        ensemble={model:describe(model,config) for model,config in configs.get('ensemble_components',configs['components']).items()},
        ensemble_recipe=dict(reference_rule=configs.get('reference_ensemble_rule','final_meta_logmean'),
            recommended_rule=configs.get('recommended_ensemble_rule','final_meta_logmean'),
            implementation='Forecasting/run_validation.py:original_blends',
            gate_ratio=1.08,gate_above='median of four model forecasts',gate_below='arithmetic mean of four model forecasts',
            reference_meta='log1p mean of equal, log_average, gated_1_08_all, gated_1_08_no_weak',
            selection_months=configs['primary_months'],submission_created=False))
    final_directory=HERE.parent/'submissions/2025-03/ensemble'
    ensemble_files=dict(baseline_submission_file=final_directory/'baseline_final_meta_logmean.csv',
        spatial_meta_submission_file=final_directory/'spatial_final_meta_logmean.csv',
        recommended_submission_file=final_directory/('spatial_'+resolved['ensemble_recipe']['recommended_rule']+'.csv'))
    if all(p.exists() for p in ensemble_files.values()):
        resolved['ensemble_recipe'].update(submission_created=True,
            **{key:path.resolve().as_posix() for key,path in ensemble_files.items()})
    (HERE/'ready_configurations.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2))
    print(json.dumps(resolved,ensure_ascii=False,indent=2))
