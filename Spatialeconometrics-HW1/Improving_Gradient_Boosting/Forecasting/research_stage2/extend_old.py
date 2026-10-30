"""Extend only the four old transfer families to March using known February history."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree,NearestNeighbors
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

def preserved_functions(spatial_source,research_source,stores,locations):
    namespace=dict(np=np,pd=pd,sparse=sparse,BallTree=BallTree,NearestNeighbors=NearestNeighbors)
    namespace.update(STORES=stores,LOCATIONS=locations)
    namespace['attach_locations']=lambda raw:(stores,locations,{})
    for source,names in [(spatial_source,['make_weights','spatial_table']),(research_source,['normalise','weight_family'])]:
        tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node],type_ignores=[]),'preserved:'+node.name,'exec'),namespace)
    return namespace

def extend(raw,stores,locations,historical,groups,namespace):
    frame=raw.merge(stores[['new_id','node']],on='new_id',validate='many_to_one')
    frame['time']=(frame.year-2023)*12+frame.month-1
    frame=frame.sort_values(['time','new_id'])
    n=len(locations);count=int(frame.time.max())+1
    means=frame.groupby(['node','time']).pto.mean().unstack('time').reindex(index=range(n),columns=range(count))
    times=np.arange(count)
    days=pd.to_datetime(dict(year=2023+times//12,month=times%12+1,day=np.ones(count,int))).dt.days_in_month.to_numpy()
    logdaily=np.log(means.to_numpy()/days)
    change=np.column_stack([np.full(n,np.nan),np.diff(logdaily,axis=1)])
    static=['population_size','number_of_households','pedestrian_traffic_per_hour','car_traffic_per_hour','grocery_stores_500m','pyaterochka_stores_500m']
    first=frame.drop_duplicates('new_id')
    values=np.log1p(first.groupby('node')[static].median().reindex(range(n)).clip(lower=0).to_numpy())
    values=np.where(np.isfinite(values),values,np.nanmedian(values,axis=0))
    profile=(values-values.mean(axis=0))/np.maximum(values.std(axis=0),1e-8)
    graphs={kind:namespace['weight_family'](locations,kind,profile) for kind in ['exp50','knn8','hybrid8']}
    def at(t):
        previous=frame[frame.time==t-1].sort_values('new_id')
        nodes=previous.node.to_numpy();own=np.log(previous.pto.to_numpy()/days[t-1])
        out=pd.DataFrame(dict(new_id=previous.new_id.to_numpy(),year_month_index=t))
        for kind,w in graphs.items():
            group=kind+'_history';peer=w@logdaily[:,t-1]
            out[group+'__level_lag1']=peer[nodes].astype('float32')
            out[group+'__daily_growth_lag1']=(w@change[:,t-1])[nodes].astype('float32')
            out[group+'__growth_1_3']=((w@logdaily[:,t-1]-w@logdaily[:,t-3])/2)[nodes].astype('float32')
            out[group+'__own_peer_gap_lag1']=(own-out[group+'__level_lag1'].to_numpy()).astype('float32')
        return out
    check=at(count-1).sort_values('new_id').reset_index(drop=True)
    expected=historical[historical.year_month_index==count-1].sort_values('new_id').reset_index(drop=True)
    columns=[c for g in ['exp50_history','knn8_history','hybrid8_history'] for c in groups[g]]
    pd.testing.assert_frame_equal(check[columns],expected[columns],check_exact=True)
    future=at(count)
    for c in groups['eigenvectors']:future[c]=expected[c].to_numpy()
    assert count==26 and len(future)==18657 and not future[columns+groups['eigenvectors']].isna().any().any()
    return future,dict(status='passed',future_time=count,historical_time=count-1,
        historical_byte_parity=True,future_actuals_used=False,groups=['exp50_history','knn8_history','hybrid8_history','eigenvectors'])

def main():
    from notebook_api import read_raw,load_recipe
    stores=pd.read_parquet(HERE.parent/'research/cache/stores.parquet')
    locations=pd.read_csv(HERE.parent/'research/geography_diagnostics.csv')
    groups=json.loads((HERE.parent/'research/cache/feature_manifest.json').read_text())['groups']
    historical=pd.read_parquet(HERE.parent/'research/cache/features.parquet')
    namespace=preserved_functions((HERE.parent/'spatial_features.py').read_text(),
        (HERE.parent/'research/build_features.py').read_text(),stores,locations)
    raw=read_raw(load_recipe());future,checks=extend(raw,stores,locations,historical,groups,namespace)
    future.to_parquet(HERE/'cache/old_future.parquet',index=False)
    (HERE/'old_future_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
