"""Describe feature extrapolation; this diagnostic never changes frozen choices."""
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    frame=pd.read_parquet(HERE/'cache/features.parquet');rows=[]
    for month,time in [('2024-06',17),('2024-07',18),('2024-08',19),('2024-11',22),('2024-12',23),('2025-03',26)]:
        for feature in [c for c in frame if c.startswith(('seasonal_control__','seasonal_spatial__'))]:
            train=frame.loc[frame.year_month_index<time,feature].dropna().to_numpy()
            future=frame.loc[frame.year_month_index==time,feature].dropna().to_numpy()
            low,high=np.quantile(train,[.01,.99])
            rows.append(dict(forecast_month=month,feature=feature,train_nonmissing=len(train),validation_nonmissing=len(future),
                train_min=float(train.min()),train_max=float(train.max()),train_p01=float(low),train_p99=float(high),
                validation_median=float(np.median(future)),outside_training_minmax_share=float(np.mean((future<train.min())|(future>train.max()))),
                outside_training_p01_p99_share=float(np.mean((future<low)|(future>high)))))
    support=pd.DataFrame(rows);support.to_csv(HERE/'seasonal_feature_support.csv',index=False)
    bias=[]
    for fold in ['2024-11','2024-12']:
        for config in ['baseline','previous','seasonal_spatial']:
            path=HERE/'results/rolling'/fold/'lightgbm_log_level'/(config+'.csv')
            if not path.exists():continue
            f=pd.read_csv(path);error=100*(f.prediction-f.pto)/f.pto
            bias.append(dict(fold=fold,configuration=config,mape=float(abs(error).mean()),
                mean_signed_percent_error=float(error.mean()),median_signed_percent_error=float(error.median()),
                share_underpredicted=float((error<0).mean())))
    pd.DataFrame(bias).to_csv(HERE/'seasonal_error_direction.csv',index=False)
    print('SEASONAL_SUPPORT_DIAGNOSTIC_SAVED',len(rows),len(bias))

if __name__=='__main__':main()
