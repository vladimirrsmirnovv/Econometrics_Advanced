"""Causal seasonal, innovation and adaptive geographic lag features (stage2_v1)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree

VERSION='stage2_v1'

def normalise(w):
    w=w.tocsr(copy=True);w.setdiag(0);w.eliminate_zeros()
    total=np.asarray(w.sum(axis=1)).ravel()
    w=(sparse.diags(np.divide(1.,total,out=np.zeros_like(total),where=total>0))@w).tocsr()
    assert np.all(w.diagonal()==0)
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel()[total>0],1,atol=1e-12)
    return w

def compute(raw,stores,locations,forecast_time=26):
    stores=stores.sort_values('new_id').reset_index(drop=True)
    locations=locations.sort_values('node').reset_index(drop=True)
    n=len(locations);ids=stores.new_id.to_numpy();nodes=stores.node.to_numpy()
    raw=raw.copy();raw['time']=(raw.year-2023)*12+raw.month-1
    assert not raw.duplicated(['new_id','time']).any()
    # Every operation below is either strictly lagged or uses an initial static map.
    panel=raw.pivot(index='new_id',columns='time',values='pto').reindex(index=ids,columns=range(forecast_time+1)).to_numpy()
    times=np.arange(forecast_time+1)
    days=pd.to_datetime(dict(year=2023+times//12,month=times%12+1,day=np.ones(len(times),int))).dt.days_in_month.to_numpy()
    own=np.log(panel/days);own_growth=np.column_stack([np.full(len(ids),np.nan),np.diff(own,axis=1)])
    membership=sparse.csr_matrix((np.ones(len(ids)),(nodes,np.arange(len(ids)))),shape=(n,len(ids)))
    counts=np.asarray(membership.sum(axis=1)).ravel()
    markets=np.log((membership@panel)/counts[:,None]/days)
    growth=np.column_stack([np.full(n,np.nan),np.diff(markets,axis=1)])
    region_codes,_=pd.factorize(locations.region,sort=True)
    reg=sparse.csr_matrix((counts,(region_codes,np.arange(n))))
    reg= sparse.diags(1/np.asarray(reg.sum(axis=1)).ravel())@reg
    regional=(reg@growth)[region_codes]
    global_growth=counts@growth/counts.sum()
    innovation=growth-regional
    reliability=counts/(counts+3.)
    stable=innovation*reliability[:,None]
    radians=np.radians(locations[['settlement_lat','settlement_lon']].to_numpy())
    distance,nearest=BallTree(radians,metric='haversine').query(radians,k=min(n,33))
    # Remove own index even where identical coordinates make tie order ambiguous.
    candidate=np.stack([js[js!=i][:32] for i,js in enumerate(nearest)])
    candidate_dist=np.stack([ds[js!=i][:32] for i,(js,ds) in enumerate(zip(nearest,distance))])*6371
    rows=np.repeat(np.arange(n),8);knn=normalise(sparse.csr_matrix((np.ones(n*8),(rows,candidate[:,:8].ravel())),shape=(n,n)))
    columns={};arrays={};diagnostics=[];previous_edges=None
    def add(group,name,a):
        c=group+'__'+name;columns.setdefault(group,[]).append(c);arrays[c]=np.asarray(a,dtype='float32')
    shape=(len(ids),len(times))
    buffers={g:{k:np.full(shape,np.nan) for k in names} for g,names in {
        'seasonal_control':['own_same_month','region_same_month','national_same_month'],
        'seasonal_spatial':['peer_same_month','peer_same_month_innovation','own_peer_season_gap'],
        'innovations_control':['own_region_gap_lag1','regional_growth_lag1','national_growth_lag1','regional_growth_mean3'],
        'innovations_spatial':['peer_innovation_lag1','peer_innovation_mean3','peer_innovation_sd3','own_peer_innovation_gap'],
        'dynamic_history':['level_lag1','daily_growth_lag1','growth_1_3','own_peer_gap_lag1']}.items()}
    for t in times:
        if t>=13:
            previous_year=t-12
            buffers['seasonal_control']['own_same_month'][:,t]=own_growth[:,previous_year]
            buffers['seasonal_control']['region_same_month'][:,t]=regional[nodes,previous_year]
            buffers['seasonal_control']['national_same_month'][:,t]=global_growth[previous_year]
            peer=knn@growth[:,previous_year]
            buffers['seasonal_spatial']['peer_same_month'][:,t]=peer[nodes]
            buffers['seasonal_spatial']['peer_same_month_innovation'][:,t]=(knn@stable[:,previous_year])[nodes]
            buffers['seasonal_spatial']['own_peer_season_gap'][:,t]=own_growth[:,previous_year]-peer[nodes]
        if t>=2:
            past=slice(max(1,t-3),t)
            recent=np.mean(stable[:,past],axis=1);peer=knn@stable[:,t-1]
            buffers['innovations_control']['own_region_gap_lag1'][:,t]=own_growth[:,t-1]-regional[nodes,t-1]
            buffers['innovations_control']['regional_growth_lag1'][:,t]=regional[nodes,t-1]
            buffers['innovations_control']['national_growth_lag1'][:,t]=global_growth[t-1]
            buffers['innovations_control']['regional_growth_mean3'][:,t]=np.mean(regional[:,past],axis=1)[nodes]
            buffers['innovations_spatial']['peer_innovation_lag1'][:,t]=peer[nodes]
            buffers['innovations_spatial']['peer_innovation_mean3'][:,t]=(knn@recent)[nodes]
            dispersion=np.sqrt(np.maximum(knn@(recent**2)-(knn@recent)**2,0))
            buffers['innovations_spatial']['peer_innovation_sd3'][:,t]=dispersion[nodes]
            buffers['innovations_spatial']['own_peer_innovation_gap'][:,t]=own_growth[:,t-1]-regional[nodes,t-1]-peer[nodes]
        if t>=1:
            geo=np.exp(-candidate_dist/50)
            if t>=5:
                history=innovation[:,max(1,t-6):t]
                centered=history-history.mean(axis=1,keepdims=True)
                norm=np.sqrt(np.sum(centered**2,axis=1))
                correlation=np.einsum('ij,ikj->ik',centered,centered[candidate])/np.maximum(norm[:,None]*norm[candidate],1e-12)
                score=geo*(1+np.maximum(correlation,0)) # fixed, bounded adaptation
            else:score=geo
            selected=np.argsort(-score,axis=1,kind='stable')[:,:8]
            js=np.take_along_axis(candidate,selected,axis=1)
            values=np.take_along_axis(score,selected,axis=1)
            w=normalise(sparse.csr_matrix((values.ravel(),(rows,js.ravel())),shape=(n,n)))
            peer=w@markets[:,t-1]
            buffers['dynamic_history']['level_lag1'][:,t]=peer[nodes]
            buffers['dynamic_history']['own_peer_gap_lag1'][:,t]=own[:,t-1]-peer[nodes]
            if t>=2:buffers['dynamic_history']['daily_growth_lag1'][:,t]=(w@growth[:,t-1])[nodes]
            if t>=3:buffers['dynamic_history']['growth_1_3'][:,t]=(w@(markets[:,t-1]-markets[:,t-3]))[nodes]/2
            overlap=np.nan if previous_edges is None else np.mean([len(set(a)&set(b))/8 for a,b in zip(js,previous_edges)])
            diagnostics.append(dict(time=int(t),edges=int(w.nnz),mean_neighbour_overlap=float(overlap),
                maximum_graph_information_time=int(t-1),zero_diagonal=True,row_normalised=True))
            previous_edges=js
    for group,values in buffers.items():
        for name,array in values.items():add(group,name,array)
    out=pd.DataFrame({'new_id':np.repeat(ids,len(times)),'year_month_index':np.tile(times,len(ids))})
    for c,a in arrays.items():out[c]=a.ravel()
    out=out.replace([np.inf,-np.inf],np.nan)
    groups=dict(columns)
    groups['seasonal_spatial']=columns['seasonal_control']+columns['seasonal_spatial']
    groups['innovations_spatial']=columns['innovations_control']+columns['innovations_spatial']
    info=dict(version=VERSION,groups=groups,graph_diagnostics=diagnostics,
        annual_reference='growth in calendar month t-12, not t-1-12',
        count_shrink=3,adaptive_history=6,adaptive_minimum_transitions=4,
        adaptive_candidates=32,adaptive_selected=8,adaptive_score='exp(-distance/50)*(1+max(past_correlation,0))',
        target_cutoff='strictly less than feature time',nodes=n,stores=len(ids),rows=len(out))
    return out,info

def main():
    import sys,hashlib
    here=Path(__file__).resolve().parent;sys.path.insert(0,str(here.parent))
    from notebook_api import read_raw,load_recipe
    raw=read_raw(load_recipe());stores=pd.read_parquet(here.parent/'research/cache/stores.parquet')
    locations=pd.read_csv(here.parent/'research/geography_diagnostics.csv')
    table,info=compute(raw,stores,locations)
    tests=[]
    for cutoff in [17,19,22,26]:
        changed=raw.copy();time=(changed.year-2023)*12+changed.month-1
        changed.loc[time>=cutoff,'pto']*=37
        alternative,_=compute(changed,stores,locations)
        pd.testing.assert_frame_equal(table.loc[table.year_month_index<=cutoff].reset_index(drop=True),
            alternative.loc[alternative.year_month_index<=cutoff].reset_index(drop=True))
        tests.append(dict(cutoff=cutoff,current_and_future_targets_changed=True,passed=True))
    shuffled,_=compute(raw.sample(frac=1,random_state=1026),stores,locations)
    pd.testing.assert_frame_equal(table,shuffled)
    cache=here/'cache';cache.mkdir(exist_ok=True)
    table.to_parquet(cache/'features.parquet',index=False)
    info['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (cache/'feature_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (here/'feature_checks.json').write_text(json.dumps(dict(status='passed',tests=tests,row_order=True,
        implementation_sha256=info['implementation_sha256'],features=len(table.columns)-2),indent=2))
    print('FEATURES_CHECKED',len(table),len(table.columns)-2,flush=True)

if __name__=='__main__':main()
