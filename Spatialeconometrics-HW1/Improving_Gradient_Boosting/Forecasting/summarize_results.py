"""Measured holdout metrics, matched blends, and descriptive residual Moran."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from notebook_api import ROOT, MODELS, load_recipe, read_raw
from run_validation import original_blends
from spatial_features import attach_locations, make_weights


def moran(value,weights):
    mask=np.isfinite(value)
    w=weights[mask][:,mask]
    y=value[mask]-np.mean(value[mask])
    return float(len(y)/w.sum()*(y@(w@y))/(y@y))


def main():
    recipe=load_recipe()
    raw=read_raw(recipe)
    stores,locations,geo=attach_locations(raw)
    weights,_=make_weights(locations)
    summary=[];diagnostics=[]
    root=ROOT/'Forecasting/results'
    for directory in sorted(root.glob('20??-??')):
        model_metrics=[]
        for p in sorted(directory.glob('*.json')):
            row=json.loads(p.read_text(encoding='utf-8'))
            if 'mape' in row:
                summary.append({'holdout':directory.name,'name':row.get('experiment',row.get('model')),
                    'mape':row['mape'],'baseline_mape':row.get('baseline_mape'),
                    'delta_mape':row.get('delta_mape'),'best_iteration':row.get('best_iteration')})
                model_metrics.append(row)
        base={}
        for name in MODELS:
            path=directory/f'{name}.csv'
            if path.exists():
                table=pd.read_csv(path).sort_values('new_id').reset_index(drop=True)
                base[name]=table.prediction.to_numpy()
        if len(base)==4:
            true=table.pto.to_numpy()
            base_meta=original_blends(base)['final_meta_logmean']
            base_mape=recipe._make_metrics(true,base_meta)['mape']
            summary.append({'holdout':directory.name,'name':'CPU_final_meta_baseline','mape':base_mape})
            for kind,group in [('exp50','history'),('exp50','static'),('region','history'),('knn8','history')]:
                candidates={}
                for name in MODELS:
                    path=directory/f'{name}_{kind}_{group}.csv'
                    if path.exists():candidates[name]=pd.read_csv(path).sort_values('new_id').prediction.to_numpy()
                if len(candidates)==4:
                    pred=original_blends(candidates)['final_meta_logmean']
                    mape=recipe._make_metrics(true,pred)['mape']
                    summary.append({'holdout':directory.name,'name':'CPU_final_meta_'+kind+'_'+group,
                       'mape':mape,'baseline_mape':base_mape,'delta_mape':mape-base_mape})
                for name,pred in candidates.items():
                    changed=dict(base);changed[name]=pred
                    combo=original_blends(changed)['final_meta_logmean']
                    mape=recipe._make_metrics(true,combo)['mape']
                    summary.append({'holdout':directory.name,'name':'CPU_final_meta_replace_'+name+'_'+kind+'_'+group,
                       'mape':mape,'baseline_mape':base_mape,'delta_mape':mape-base_mape})
        for p in directory.glob('*.csv'):
            if 'metrics' in p.name:continue
            table=pd.read_csv(p)
            if not {'new_id','pto','prediction'}.issubset(table):continue
            table=table.merge(stores[['new_id','node']],on='new_id',validate='one_to_one')
            table['log_error']=np.log(table.prediction/table.pto)
            residual=table.groupby('node').log_error.mean().reindex(range(len(locations))).to_numpy()
            diagnostics.append({'holdout':directory.name,'name':p.stem,'residual_moran':moran(residual,weights),
                                'scale':'mean log(prediction/actual) by settlement','inference':'descriptive; no p-value'})
    pd.DataFrame(summary).to_csv(root/'experiment_summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(root/'residual_moran.csv',index=False)
    (root/'geography.json').write_text(json.dumps(geo,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print('\nRESIDUAL MORAN\n',pd.DataFrame(diagnostics).to_string(index=False))


if __name__=='__main__':main()
