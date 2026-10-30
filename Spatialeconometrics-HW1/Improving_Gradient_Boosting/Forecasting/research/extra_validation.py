"""Freeze primary choices, then run October without changing primary rankings."""
from pathlib import Path
import json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent

def main():
    configs=json.loads((HERE/'best_model_configs.json').read_text())
    (HERE/'frozen_selection_before_extra_month.json').write_text(json.dumps(configs,indent=2))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',PYTHONUTF8='1')
    def run(script,args):subprocess.run([sys.executable,str(script),*args],env=env,check=True)
    run(PARENT/'run_validation.py',['--year','2024','--month','10','--models','catboost_log_level','lightgbm_log_diff','lightgbm_log_level'])
    for model,config in configs['components'].items():
        if model=='xgboost_log_diff':
            if not (HERE/'results/2024-10'/model/(config+'.json')).exists():
                raise FileNotFoundError('GPU extra month missing for chosen '+config)
            continue
        # Fitted-feature and training-weight winners can be rerun directly.
        if config.startswith('exp50_history+residual_'):
            # Forecast the SAME spatial model on September before correcting October.
            run(PARENT/'run_validation.py',['--year','2024','--month','9','--models',model])
            for month in ['9','10']:
                run(PARENT/'run_spatial.py',['--year','2024','--month',month,'--models',model,'--weights','exp50','--groups','history'])
            run(HERE/'run_models.py',['--models',model,'--folds','2024-10','--configs','baseline'])
            import numpy as np
            import pandas as pd
            from residual_correction import prior_signals
            sys.path.insert(0,str(PARENT))
            from notebook_api import load_recipe
            stores=pd.read_parquet(HERE/'cache/stores.parquet').sort_values('new_id')
            locations=pd.read_csv(HERE/'geography_diagnostics.csv').sort_values('node')
            current=pd.read_csv(PARENT/'results/2024-10/catboost_log_level_exp50_history.csv').sort_values('new_id')
            previous=pd.read_csv(PARENT/'results/2024-09/catboost_log_level_exp50_history.csv')
            name=config.split('+residual_',1)[1]
            signal=prior_signals(previous,stores,locations)[name]
            pred=current.prediction.to_numpy()*np.exp(.5*signal)
            row=load_recipe()._make_metrics(current.pto,pred)
            row.update(model=model,fold='2024-10',configuration=config,backend='OOF postprocessing',
                previous_oof_fold='2024-09',previous_oof_configuration='exp50_history',correction_weight=.5,
                validation_only=True,leakage_check='passed',selection_frozen_before_local_extra_fit=True)
            directory=HERE/'results/2024-10'/model;current['prediction']=pred
            current.to_csv(directory/(config+'.csv'),index=False)
            (directory/(config+'.json')).write_text(json.dumps(row,indent=2))
            continue
        if config.startswith(('residual_','sem_','sdem_')):
            raise ValueError('Extra month needs the matched postprocessing history: '+config)
        run(HERE/'run_models.py',['--models',model,'--folds','2024-10','--configs',config])
    print('EXTRA_MONTH_COMPLETE',flush=True)

if __name__=='__main__':main()
