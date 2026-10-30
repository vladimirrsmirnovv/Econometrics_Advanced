"""Feature/error interaction with honest prior OOF of the SAME CatBoost version."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import load_recipe
from residual_correction import prior_signals,PREVIOUS

def main():
    recipe=load_recipe();stores=pd.read_parquet(HERE/'cache/stores.parquet').sort_values('new_id')
    locations=pd.read_csv(HERE/'geography_diagnostics.csv').sort_values('node');records=[]
    for fold,previous_fold in PREVIOUS.items():
        assert previous_fold<fold
        directory=HERE/'results'/fold/'catboost-log-level'
        current=pd.read_csv(directory/'exp50_history.csv').sort_values('new_id')
        if previous_fold=='2025-01':previous=pd.read_csv(HERE/'results'/previous_fold/'catboost-log-level/exp50_history.csv')
        else:
            path=HERE.parent/'results'/previous_fold/'catboost_log_level_exp50_history.csv'
            meta=json.loads(path.with_suffix('.json').read_text())
            assert meta['year']==int(previous_fold[:4]) and meta['month']==int(previous_fold[-2:])
            assert meta['extra_features']==['wy_log_daily_lag1','wy_daily_growth_lag1','wy_daily_growth_1_3','wy_log_daily_gap_lag1']
            previous=pd.read_csv(path)
        signals=prior_signals(previous,stores,locations)
        for name in ['global','region','spatial_exp50','spatial_knn8','spatial_region_gap']:
            config='exp50_history+residual_'+name
            pred=current.prediction.to_numpy()*np.exp(.5*signals[name])
            row=recipe._make_metrics(current.pto,pred)
            row.update(model='catboost_log_level',fold=fold,configuration=config,backend='OOF postprocessing',
                validation_only=True,leakage_check='passed',previous_oof_fold=previous_fold,
                previous_oof_configuration='exp50_history',correction_weight=.5,
                current_targets_used_for_correction=False,feature_version='legacy_history')
            table=current.copy();table['prediction']=pred;table.to_csv(directory/(config+'.csv'),index=False)
            (directory/(config+'.json')).write_text(json.dumps(row,indent=2));records.append(row)
    (HERE/'combined_residual_checks.json').write_text(json.dumps(dict(status='passed',runs=len(records),
        rule='Strictly earlier OOF from the same spatial CatBoost configuration'),indent=2))
    print(pd.DataFrame(records).pivot(index='configuration',columns='fold',values='mape').to_string())

if __name__=='__main__':main()
