"""Settlement spatial features for all stores, using strictly earlier revenue."""
from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path


import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree

ROOT=Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:sys.path.insert(0,str(REPOSITORY_ROOT))
from src.names import region_key, split_settlement, settlement_base_name, settlement_type_class


def location_key(region, settlement):
    try:
        name,typ=split_settlement(settlement)
        return region_key(region)+'|'+settlement_base_name(name)+'|'+settlement_type_class(typ)
    except ValueError:
        return region_key(region)+'|unparsed|'+str(settlement).strip().lower()


def attach_locations(raw):
    stores=raw[['new_id','region','settlement']].drop_duplicates().copy()
    assert not stores.new_id.duplicated().any()
    stores['location_key']=[location_key(r,s) for r,s in zip(stores.region,stores.settlement)]
    coords=pd.read_csv(resource_path('Parsing/geonames-settlement-coordinates/result/settlement_coordinates.csv'),
                      usecols=['region','settlement','settlement_lat','settlement_lon'])
    coords['location_key']=[location_key(r,s) for r,s in zip(coords.region,coords.settlement)]
    unique=coords[['location_key','settlement_lat','settlement_lon']].drop_duplicates()
    ambiguous=unique.location_key.duplicated(keep=False)
    rejected=int(unique.loc[ambiguous,'location_key'].nunique())
    unique=unique.loc[~ambiguous].set_index('location_key')
    panel=pd.read_parquet(resource_path('Data/settlement_panel.parquet'),
          columns=['shop_region','settlement_name','settlement_type_class','latitude','longitude']).drop_duplicates()
    panel['location_key']=[region_key(r)+'|'+settlement_base_name(n)+'|'+t
        for r,n,t in zip(panel.shop_region,panel.settlement_name,panel.settlement_type_class)]
    overrides=panel.set_index('location_key')[['latitude','longitude']].rename(
          columns={'latitude':'settlement_lat','longitude':'settlement_lon'})
    assert not overrides.index.duplicated().any()
    unique=pd.concat([unique.loc[~unique.index.isin(overrides.index)],overrides])
    locations=stores[['location_key','region','settlement']].drop_duplicates('location_key').sort_values('location_key')
    locations=locations.join(unique,on='location_key').reset_index(drop=True)
    locations['node']=np.arange(len(locations))
    stores=stores.merge(locations[['location_key','node']],on='location_key',validate='many_to_one')
    info={'locations':len(locations),'coordinate_locations':int(locations.settlement_lat.notna().sum()),
          'coordinate_stores':int(stores.location_key.isin(locations.loc[locations.settlement_lat.notna(),'location_key']).sum()),
          'stores':len(stores),'ambiguous_coordinate_keys_rejected':rejected,
          'geometry':'settlement centroids; full competition store set',
          'linkage':'normalized region + settlement name + settlement type; never old new_id',
          'research_coordinate_overrides':len(overrides)}
    return stores,locations,info


def make_weights(locations,kind='exp50'):
    valid=locations.settlement_lat.notna()&locations.settlement_lon.notna()
    nodes=locations.loc[valid,'node'].to_numpy()
    coordinates=np.radians(locations.loc[valid,['settlement_lat','settlement_lon']].to_numpy())
    tree=BallTree(coordinates,metric='haversine')
    rows=[];cols=[];values=[]
    if kind=='exp50':
        neighbours,distances=tree.query_radius(coordinates,r=300/6371,return_distance=True,sort_results=True)
        nearest,nearest_idx=tree.query(coordinates,k=min(2,len(nodes)))
        for i,(js,ds) in enumerate(zip(neighbours,distances)):
            mask=js!=i;js=js[mask];ds=ds[mask]
            if not len(js):
                js=nearest_idx[i][nearest_idx[i]!=i]
                ds=nearest[i][nearest_idx[i]!=i]
            rows.extend([int(nodes[i])]*len(js));cols.extend(nodes[js].tolist())
            values.extend(np.exp(-ds*6371/50).tolist())
        matrix=sparse.csr_matrix((values,(rows,cols)),shape=(len(locations),len(locations)))
    elif kind=='region':
        # Non-geographic control: equal weights on other markets in the region.
        groups=locations.loc[valid].groupby('region').node
        for _,members in groups:
            js=members.to_numpy()
            for node in js:
                peers=js[js!=node]
                rows.extend([int(node)]*len(peers));cols.extend(peers.tolist());values.extend([1.0]*len(peers))
        matrix=sparse.csr_matrix((values,(rows,cols)),shape=(len(locations),len(locations)))
    elif kind=='knn8':
        _,nearest_idx=tree.query(coordinates,k=min(9,len(nodes)))
        for i,js in enumerate(nearest_idx):
            js=js[js!=i][:8]
            rows.extend([int(nodes[i])]*len(js));cols.extend(nodes[js].tolist());values.extend([1.0]*len(js))
        matrix=sparse.csr_matrix((values,(rows,cols)),shape=(len(locations),len(locations)))
        matrix=matrix.maximum(matrix.T)
    else:raise ValueError(kind)
    totals=np.asarray(matrix.sum(axis=1)).ravel()
    matrix=sparse.diags(np.divide(1,totals,out=np.zeros_like(totals),where=totals>0))@matrix
    assert np.all(matrix.diagonal()==0)
    supported=valid.to_numpy()&(totals>0)
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel()[supported],1,atol=1e-12)
    return matrix.tocsr(),supported


def spatial_table(raw,*,kind='exp50'):
    stores,locations,info=attach_locations(raw)
    weights,coordinate_valid=make_weights(locations,kind)
    frame=raw.merge(stores[['new_id','node']],on='new_id',validate='many_to_one')
    frame['time']=(frame.year-2023)*12+frame.month-1
    n=len(locations)
    static_columns=['population_size','number_of_households','pedestrian_traffic_per_hour',
                    'car_traffic_per_hour','grocery_stores_500m','pyaterochka_stores_500m']
    shop_static=frame.drop_duplicates('new_id')
    static=shop_static.groupby('node')[static_columns].median().reindex(range(n))
    wx=weights@np.log1p(static.clip(lower=0).to_numpy())
    wx[~coordinate_valid]=np.nan
    statics=pd.DataFrame(wx,columns=['wx_log_'+x for x in static_columns])
    statics['node']=np.arange(n)
    means=frame.groupby(['node','time']).pto.mean().unstack('time').reindex(range(n))
    times=np.arange(int(frame.time.max())+1)
    means=means.reindex(columns=times)
    days=pd.to_datetime({'year':2023+times//12,'month':1+times%12,'day':np.ones(len(times),dtype=int)}).dt.days_in_month.to_numpy()
    logdaily=np.log(means.to_numpy()/days)
    peer_log=weights@logdaily
    peer_growth=weights@np.column_stack([np.full(n,np.nan),np.diff(logdaily,axis=1)])
    peer_log[~coordinate_valid]=np.nan;peer_growth[~coordinate_valid]=np.nan
    records=[]
    for t in times:
        lag1=peer_log[:,t-1] if t>=1 else np.full(n,np.nan)
        growth=peer_growth[:,t-1] if t>=2 else np.full(n,np.nan)
        growth3=(peer_log[:,t-1]-peer_log[:,t-3])/2 if t>=3 else np.full(n,np.nan)
        records.append(pd.DataFrame({'node':np.arange(n),'year_month_index':t,
                      'wy_log_daily_lag1':lag1,'wy_daily_growth_lag1':growth,
                      'wy_daily_growth_1_3':growth3}))
    dynamic=pd.concat(records,ignore_index=True)
    output=frame[['new_id','node','time']].rename(columns={'time':'year_month_index'})
    output=output.merge(statics,on='node',validate='many_to_one').merge(dynamic,
                    on=['node','year_month_index'],validate='many_to_one')
    prior=frame.sort_values(['new_id','time'])
    prior['store_prior_log_daily']=prior.groupby('new_id').pto.shift(1)
    prior['store_prior_log_daily']=np.log(prior.store_prior_log_daily/days[np.maximum(prior.time.to_numpy()-1,0)])
    output=output.merge(prior[['new_id','time','store_prior_log_daily']].rename(columns={'time':'year_month_index'}),
                    on=['new_id','year_month_index'],validate='one_to_one')
    output['wy_log_daily_gap_lag1']=output.store_prior_log_daily-output.wy_log_daily_lag1
    output=output.drop(columns=['node','store_prior_log_daily']).sort_values(['new_id','year_month_index']).reset_index(drop=True)
    numeric=[c for c in output.columns if c not in ['new_id','year_month_index']]
    output[numeric]=output[numeric].astype('float32')
    info.update(weights=kind,nnz=int(weights.nnz),features=numeric,
                missing_coordinate_policy='NaN spatial features; retain every store',
                revenue_cutoff='all revenue features at t use only times <= t-1')
    return output,info


def align_features(compact,table,columns):
    indexed=table.set_index(['new_id','year_month_index'])
    assert not indexed.index.duplicated().any()
    parts=[]
    for key in ['train_features','val_features']:
        frame=compact[key]
        index=pd.MultiIndex.from_frame(frame[['new_id','year_month_index']])
        aligned=indexed.reindex(index)[columns].reset_index(drop=True)
        aligned.index=frame.index
        assert len(aligned)==len(frame)
        parts.append(aligned)
    return parts
