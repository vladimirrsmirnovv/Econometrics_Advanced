"""Distance-only ablation for the exact directed adaptive graph design."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree
HERE=Path(__file__).resolve().parent

def compute(raw,stores,locations,forecast_time=26):
    stores=stores.sort_values('new_id');locations=locations.sort_values('node')
    ids=stores.new_id.to_numpy();nodes=stores.node.to_numpy();n=len(locations)
    raw=raw.copy();raw['time']=(raw.year-2023)*12+raw.month-1
    times=np.arange(forecast_time+1)
    days=pd.to_datetime(dict(year=2023+times//12,month=times%12+1,day=np.ones(len(times),int))).dt.days_in_month.to_numpy()
    panel=raw.pivot(index='new_id',columns='time',values='pto').reindex(index=ids,columns=times).to_numpy()
    membership=sparse.csr_matrix((np.ones(len(ids)),(nodes,np.arange(len(ids)))),shape=(n,len(ids)))
    counts=np.asarray(membership.sum(axis=1)).ravel()
    logdaily=np.log((membership@panel)/counts[:,None]/days)
    own=np.log(panel/days)
    growth=np.column_stack([np.full(n,np.nan),np.diff(logdaily,axis=1)])
    radians=np.radians(locations[['settlement_lat','settlement_lon']].to_numpy())
    distance,nearest=BallTree(radians,metric='haversine').query(radians,k=min(n,33))
    candidate=np.stack([js[js!=i][:32] for i,js in enumerate(nearest)])
    ds=np.stack([d[js!=i][:32] for i,(js,d) in enumerate(zip(nearest,distance))])*6371
    score=np.exp(-ds/50);selected=np.argsort(-score,axis=1,kind='stable')[:,:8]
    neighbours=np.take_along_axis(candidate,selected,axis=1)
    weights=np.take_along_axis(score,selected,axis=1)
    w=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(n),8),neighbours.ravel())),shape=(n,n))
    w=sparse.diags(1/np.asarray(w.sum(axis=1)).ravel())@w
    assert (w.diagonal()==0).all()
    arrays={name:np.full((len(ids),len(times)),np.nan) for name in ['level_lag1','daily_growth_lag1','growth_1_3','own_peer_gap_lag1']}
    for t in times[1:]:
        peer=w@logdaily[:,t-1]
        arrays['level_lag1'][:,t]=peer[nodes];arrays['own_peer_gap_lag1'][:,t]=own[:,t-1]-peer[nodes]
        if t>=2:arrays['daily_growth_lag1'][:,t]=(w@growth[:,t-1])[nodes]
        if t>=3:arrays['growth_1_3'][:,t]=(w@(logdaily[:,t-1]-logdaily[:,t-3]))[nodes]/2
    table=pd.DataFrame(dict(new_id=np.repeat(ids,len(times)),year_month_index=np.tile(times,len(ids))))
    for name,a in arrays.items():table['geo8_control__'+name]=a.ravel().astype('float32')
    return table

def main():
    sys.path.insert(0,str(HERE.parent));from notebook_api import read_raw,load_recipe
    raw=read_raw(load_recipe());stores=pd.read_parquet(HERE.parent/'research/cache/stores.parquet')
    locations=pd.read_csv(HERE.parent/'research/geography_diagnostics.csv')
    table=compute(raw,stores,locations)
    for cutoff in [17,19,22]:
        changed=raw.copy();t=(changed.year-2023)*12+changed.month-1;changed.loc[t>=cutoff,'pto']*=37
        other=compute(changed,stores,locations)
        pd.testing.assert_frame_equal(table.loc[table.year_month_index<=cutoff].reset_index(drop=True),other.loc[other.year_month_index<=cutoff].reset_index(drop=True),check_exact=True)
    pd.testing.assert_frame_equal(table,compute(raw.sample(frac=1,random_state=1027),stores,locations),check_exact=True)
    dynamic=pd.read_parquet(HERE/'cache/features.parquet')
    columns=['dynamic_history__'+c.split('__')[1] for c in table if c.startswith('geo8_control__')]
    pd.testing.assert_frame_equal(table.loc[table.year_month_index<=4].reset_index(drop=True),
        dynamic.loc[dynamic.year_month_index<=4,['new_id','year_month_index']+columns].rename(columns={c:c.replace('dynamic_history','geo8_control') for c in columns}).reset_index(drop=True),check_exact=True)
    table.to_parquet(HERE/'cache/geo_control.parquet',index=False)
    (HERE/'geo_control_checks.json').write_text(json.dumps(dict(status='passed',future_current_replacement=True,
        row_shuffle=True,early_period_exact_adaptive_parity=True,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('GEOGRAPHIC_ABLATION_CHECKED',flush=True)

if __name__=='__main__':main()
