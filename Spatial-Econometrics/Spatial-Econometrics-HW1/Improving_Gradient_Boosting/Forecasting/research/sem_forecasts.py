"""Small forecast-only pooled SEM/SDEM, profiled on strictly historical outcomes."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import ROOT,MODELS,load_recipe
from spatial_features import make_weights
FOLDS=['2024-03','2025-01','2025-02']

def forecast(log_monthly,static,weights,t):
    # This routine physically truncates outcomes, making future values inaccessible.
    history=log_monthly[:,:t]
    n=history.shape[0]
    growth=np.column_stack([np.full(n,np.nan),np.diff(history,axis=1)])
    def x(s):
        own=np.column_stack([np.ones(n),growth[:,s-1],
            np.nanmean(growth[:,max(1,s-3):s],axis=1),
            growth[:,s-12] if s>=13 else np.zeros(n),static,
            np.full(n,np.sin(2*np.pi*(s%12+1)/12)),np.full(n,np.cos(2*np.pi*(s%12+1)/12))])
        return np.nan_to_num(own)
    periods=list(range(2,t));historic=[x(s) for s in periods];current=x(t)
    y=[growth[:,s] for s in periods];result={};details=[]
    for family in ['sem','sdem']:
        xx=[a if family=='sem' else np.column_stack([a,weights@a[:,1:4]]) for a in historic]
        current_x=current if family=='sem' else np.column_stack([current,weights@current[:,1:4]])
        scores=[]
        for lam in [0,.3,.6]:
            transform=sparse.eye(n,format='csc')-lam*weights
            z=np.vstack([transform@a for a in xx]);target=np.concatenate([transform@a for a in y])
            # Columns scaled from past covariates only; small ridge handles collinearity.
            scales=np.maximum(np.sqrt(np.mean(z*z,axis=0)),1e-8);zs=z/scales
            penalty=np.eye(z.shape[1])*1e-3;penalty[0,0]=0
            coefficients=np.linalg.solve(zs.T@zs+penalty,zs.T@target)/scales
            residual=target-z@coefficients;sse=float(residual@residual)
            factor=splu(transform.tocsc())
            logdet=float(np.log(np.abs(factor.U.diagonal())).sum())
            profile=len(periods)*logdet-len(target)/2*np.log(sse/len(target))
            scores.append((profile,lam,coefficients))
        score,lam,coef=max(scores,key=lambda row:row[0])
        result[family]=np.clip(current_x@coef,-.7,.7)
        details.append(dict(family=family,time=t,training_times=periods,lambda_grid=[0,.3,.6],
            selected_lambda=lam,profile_likelihood=score,forecast='X_t beta; expected future innovation zero',
            assumptions='Pooled independent-month Gaussian innovation profile; short-panel predictive approximation'))
    return result,details

def main():
    recipe=load_recipe()
    wanted=[k for k,v in recipe.rename_cols.items() if v in ['new_id','year','month','pto']]
    raw=pd.read_csv(resource_path('Data/Shops/train_2.csv'),usecols=wanted).rename(columns=recipe.rename_cols)
    stores=pd.read_parquet(HERE/'cache/stores.parquet').sort_values('new_id')
    locations=pd.read_csv(HERE/'geography_diagnostics.csv').sort_values('node')
    raw=raw.merge(stores[['new_id','node']],on='new_id',validate='many_to_one');raw['time']=(raw.year-2023)*12+raw.month-1
    mean=raw.groupby(['node','time']).pto.mean().unstack().reindex(locations.node)
    monthly=np.log(mean.to_numpy());static=stores.groupby('node')[['population_log','competition_log']].median().reindex(locations.node).to_numpy()
    weights,_=make_weights(locations,'knn8');metadata=[];records=[]
    for fold in FOLDS:
        year,month=map(int,fold.split('-'));t=(year-2023)*12+month-1
        predicted,details=forecast(monthly,static,weights,t);metadata+=details
        changed=monthly.copy();changed[:,t:]+=100
        control,_=forecast(changed,static,weights,t)
        for family in predicted:np.testing.assert_array_equal(predicted[family],control[family])
        for model in MODELS:
            directory=HERE/'results'/fold/model;base=pd.read_csv(directory/'baseline.csv').sort_values('new_id')
            np.testing.assert_array_equal(base.new_id,stores.new_id)
            lag=raw[raw.time==t-1].set_index('new_id').reindex(stores.new_id).pto.to_numpy()
            for family,values in predicted.items():
                economic=lag*np.exp(values[stores.node.to_numpy()])
                for config,pred in [(family+'_standalone',economic),
                                    (family+'_blend_0p1',np.expm1(.9*np.log1p(base.prediction.to_numpy())+.1*np.log1p(economic)))]:
                    row=recipe._make_metrics(base.pto,pred)
                    row.update(fold=fold,model=model,configuration=config,backend='Economic postprocessing',
                        economy_weight=1 if 'standalone' in config else .1,validation_only=True,
                        leakage_check='passed',source='sem_forecasts.py',parameters={'W':'knn8','lambda_grid':[0,.3,.6]},
                        limitation='Forecast approximation; no causal interpretation; short pooled panel')
                    table=base.copy();table['prediction']=pred;table.to_csv(directory/(config+'.csv'),index=False)
                    (directory/(config+'.json')).write_text(json.dumps(row,indent=2));records.append(row)
    (HERE/'sem_checks.json').write_text(json.dumps(dict(status='passed',runs=len(records),cutoffs=FOLDS,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),estimation=metadata),indent=2))
    print(pd.DataFrame(records).pivot_table(index='configuration',columns='model',values='mape').to_string())

if __name__=='__main__':main()
