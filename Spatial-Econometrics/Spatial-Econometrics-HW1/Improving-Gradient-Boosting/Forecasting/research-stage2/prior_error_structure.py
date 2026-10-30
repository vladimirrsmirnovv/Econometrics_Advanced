"""Describe prior residual geography without fitting corrections or using March."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy import sparse
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS

def moran(values,w):
    z=values-values.mean();return float(len(z)/(w.sum())*(z@(w@z))/(z@z))

def main():
    old=HERE.parent/'research';choice=json.loads((old/'best_model_configs.json').read_text())['components']
    stores=pd.read_parquet(old/'cache/stores.parquet').set_index('new_id')
    w=sparse.load_npz(HERE/'cache/preserved_knn8_weights.npz');rows=[];vectors={}
    for fold in ['2024-03','2024-10','2025-01','2025-02']:
        for model in MODELS:
            for variant,config in [('baseline','baseline'),('previous',choice[model])]:
                frame=pd.read_csv(old/'results'/fold/model.replace("_", "-")/(config+'.csv'));frame['node']=frame.new_id.map(stores.node)
                frame['log_error']=np.log(frame.pto/frame.prediction)
                market=frame.groupby('node').agg(log_error=('log_error','mean'),region=('region','first'),shops=('new_id','size')).reindex(range(w.shape[0]))
                assert not market.isna().any().any()
                regional=market.groupby('region').apply(lambda f:np.average(f.log_error,weights=f.shops),include_groups=False)
                error=market.log_error.to_numpy();demeaned=error-market.region.map(regional).to_numpy()
                rows.append(dict(fold=fold,model=model,variant=variant,mean_store_log_error=float(frame.log_error.mean()),
                    market_moran_raw=moran(error,w),market_moran_region_removed=moran(demeaned,w),
                    region_mean_log_error_sd=float(regional.std()),description_only=True))
                vectors[(fold,model,variant)]=demeaned
    persistence=[]
    for model in MODELS:
        for variant in ['baseline','previous']:
            a=vectors[('2025-01',model,variant)];b=vectors[('2025-02',model,variant)]
            persistence.append(dict(model=model,variant=variant,months='2025-01 -> 2025-02',
                correlation_region_removed=float(np.corrcoef(a,b)[0,1]),description_only=True))
    pd.DataFrame(rows).to_csv(HERE/'prior_residual_spatial_structure.csv',index=False)
    pd.DataFrame(persistence).to_csv(HERE/'prior_residual_persistence.csv',index=False)
    print(pd.DataFrame(rows).groupby('variant')[['market_moran_raw','market_moran_region_removed']].mean().to_string())
    print(pd.DataFrame(persistence).to_string(index=False))

if __name__=='__main__':main()
