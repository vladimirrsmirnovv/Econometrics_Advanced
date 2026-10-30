"""Market graph feature families using strictly available historical information."""
from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.neighbors import BallTree, NearestNeighbors

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import ROOT, load_recipe, read_raw
from spatial_features import attach_locations, make_weights
from economic_forecasts import expanding_forecasts

STATIC=['population_size','number_of_households','pedestrian_traffic_per_hour',
        'car_traffic_per_hour','grocery_stores_500m','pyaterochka_stores_500m']
FEATURE_VERSION='research_v1'

def normalise(matrix):
    matrix=matrix.tocsr(copy=True)
    matrix.setdiag(0);matrix.eliminate_zeros()
    totals=np.asarray(matrix.sum(axis=1)).ravel()
    inverse=np.divide(1.,totals,out=np.zeros_like(totals),where=totals>0)
    matrix=(sparse.diags(inverse)@matrix).tocsr()
    assert (matrix.diagonal()==0).all()
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel()[totals>0],1,atol=1e-12)
    return matrix

def weight_family(locations,kind,profile):
    if kind in {'exp50','knn8','region'}:
        return make_weights(locations,kind)[0]
    n=len(locations)
    radians=np.radians(locations[['settlement_lat','settlement_lon']].to_numpy())
    assert np.isfinite(radians).all()
    tree=BallTree(radians,metric='haversine')
    rows=[];cols=[];values=[]
    if kind=='economic8':
        neighbours=NearestNeighbors(n_neighbors=9).fit(profile).kneighbors(return_distance=False)
        # kneighbors(X=None) excludes self already; explicitly remove it anyway.
        for i,js in enumerate(neighbours):
            js=js[js!=i][:8]
            rows.extend([i]*len(js));cols.extend(js);values.extend([1.]*len(js))
    elif kind=='hybrid8':
        distances,neighbours=tree.query(radians,k=min(33,n))
        for i,(js,ds) in enumerate(zip(neighbours,distances)):
            keep=js!=i;js=js[keep];ds=ds[keep]*6371
            economic=np.square(profile[js]-profile[i]).mean(axis=1)
            score=np.exp(-ds/50-economic/2)
            best=np.argsort(score)[-8:]
            rows.extend([i]*len(best));cols.extend(js[best]);values.extend(score[best])
    else:
        radius=150 if kind=='exp15' else (300 if kind=='exp100' else 75)
        neighbours,distances=tree.query_radius(radians,r=radius/6371,return_distance=True,sort_results=True)
        fallback_d,fallback_j=tree.query(radians,k=2)
        for i,(js,ds) in enumerate(zip(neighbours,distances)):
            keep=js!=i;js=js[keep];ds=ds[keep]*6371
            if not len(js):
                js=fallback_j[i,1:];ds=fallback_d[i,1:]*6371
            value=(np.exp(-ds/(15 if kind=='exp15' else 100)) if kind.startswith('exp')
                   else (1/(1+ds) if kind=='inverse75' else np.ones(len(js))))
            rows.extend([i]*len(js));cols.extend(js);values.extend(value)
    matrix=sparse.csr_matrix((values,(rows,cols)),shape=(n,n))
    if kind=='economic8':matrix=matrix.maximum(matrix.T)
    return normalise(matrix)

def shift(value,lag):
    out=np.full_like(value,np.nan,dtype=float)
    if lag<value.shape[1]:out[:,lag:]=value[:,:-lag]
    return out

def rolling(value,window,operation='mean'):
    out=np.full_like(value,np.nan,dtype=float)
    for t in range(value.shape[1]):
        part=value[:,max(0,t-window+1):t+1]
        count=np.isfinite(part).sum(axis=1)
        if operation=='mean':
            out[:,t]=np.divide(np.nansum(part,axis=1),count,
                out=np.full(len(value),np.nan),where=count>0)
        else:
            average=np.divide(np.nansum(part,axis=1),count,
                out=np.zeros(len(value)),where=count>0)
            variance=np.nansum((part-average[:,None])**2,axis=1)
            out[:,t]=np.sqrt(np.divide(variance,count,out=np.full(len(value),np.nan),where=count>0))
    return out

def compute(raw):
    stores,locations,info=attach_locations(raw)
    frame=raw.merge(stores[['new_id','node']],on='new_id',validate='many_to_one').copy()
    frame['time']=(frame.year-2023)*12+frame.month-1
    frame=frame.sort_values(['time','new_id'])
    n=len(locations);periods=int(frame.time.max())+1;times=np.arange(periods)
    means=frame.groupby(['node','time']).pto.mean().unstack('time').reindex(index=range(n),columns=times)
    assert np.isfinite(means.to_numpy()).all() and (means.to_numpy()>0).all()
    logmonthly=np.log(means.to_numpy())
    days=pd.to_datetime({'year':2023+times//12,'month':1+times%12,'day':np.ones(periods,dtype=int)}).dt.days_in_month.to_numpy()
    logdaily=np.log(means.to_numpy()/days)
    growth=np.column_stack([np.full(n,np.nan),np.diff(logmonthly,axis=1)])
    dailygrowth=np.column_stack([np.full(n,np.nan),np.diff(logdaily,axis=1)])
    first=frame.drop_duplicates('new_id')
    static=np.log1p(first.groupby('node')[STATIC].median().reindex(range(n)).clip(lower=0).to_numpy())
    static=np.where(np.isfinite(static),static,np.nanmedian(static,axis=0))
    profile=(static-static.mean(axis=0))/np.maximum(static.std(axis=0),1e-8)
    graphs={kind:weight_family(locations,kind,profile) for kind in
            ['exp50','knn8','exp15','exp100','inverse75','binary75','economic8','hybrid8']}
    permutation=np.random.default_rng(42).permutation(n)
    graphs['random']=graphs['exp50'][permutation][:,permutation]
    groups={};columns={};values={}
    row_nodes=frame.node.to_numpy();row_times=frame.time.to_numpy()
    def add(group,name,array):
        full=group+'__'+name
        if array.ndim==1:array=np.repeat(array[:,None],periods,axis=1)
        values[full]=np.asarray(array[row_nodes,row_times],dtype='float32')
        columns.setdefault(group,[]).append(full)
    for kind in ['exp50','knn8','exp15','inverse75','binary75','economic8','hybrid8','random']:
        w=graphs[kind];peer=w@logdaily;change=w@dailygrowth
        group=kind+'_history'
        add(group,'level_lag1',shift(peer,1))
        add(group,'daily_growth_lag1',shift(change,1))
        add(group,'growth_1_3',(shift(peer,1)-shift(peer,3))/2)
        # Own-shop level gap is added below, not own-market gap.
    w=graphs['exp50'];peer_growth=w@growth
    group='growth_exp50'
    add(group,'monthly_growth_lag1',shift(peer_growth,1))
    add(group,'monthly_growth_lag2',shift(peer_growth,2))
    add(group,'monthly_growth_lag3',shift(peer_growth,3))
    add(group,'acceleration',shift(peer_growth,1)-shift(peer_growth,2))
    add(group,'growth_mean3',shift(rolling(peer_growth,3),1))
    group='season_exp50'
    add(group,'level_lag12',w@shift(logdaily,12))
    add(group,'year_growth_lag1',w@(shift(logdaily,1)-shift(logdaily,13)))
    add(group,'level_mean3',w@shift(rolling(logdaily,3),1))
    add(group,'growth_volatility6',w@shift(rolling(growth,6,'std'),1))
    month_sin=np.sin(2*np.pi*(times%12+1)/12)[None,:]
    month_cos=np.cos(2*np.pi*(times%12+1)/12)[None,:]
    add(group,'growth_month_sin',shift(peer_growth,1)*month_sin)
    add(group,'growth_month_cos',shift(peer_growth,1)*month_cos)
    second=(graphs['knn8']@graphs['knn8']).tocsr()
    second.setdiag(0)
    second=second-second.multiply(graphs['knn8'].astype(bool))
    second=normalise(second)
    group='multihop'
    add(group,'second_level_lag1',second@shift(logdaily,1))
    add(group,'second_growth_lag1',second@shift(growth,1))
    add(group,'near_far_level_lag1',(graphs['exp15']-graphs['exp100'])@shift(logdaily,1))
    add(group,'near_far_growth_lag1',(graphs['exp15']-graphs['exp100'])@shift(growth,1))
    group='wx_relative'
    wx=w@static
    for i,name in enumerate(STATIC):
        add('wx_means',name,wx[:,i])
        add(group,'gap_'+name,static[:,i]-wx[:,i])
        add(group,'sd_'+name,np.sqrt(np.maximum(w@(static[:,i]**2)-wx[:,i]**2,0)))
    group='regimes'
    add(group,'growth_pop',shift(peer_growth,1)*profile[:,0,None])
    add(group,'growth_density',shift(peer_growth,1)*profile[:,5,None])
    add(group,'level_pop',(w@shift(logdaily,1))*profile[:,0,None])
    distances,indices=BallTree(np.radians(locations[['settlement_lat','settlement_lon']]),metric='haversine').query(
        np.radians(locations[['settlement_lat','settlement_lon']]),k=2)
    add(group,'log_nearest_km',np.log1p(distances[:,1]*6371))
    add(group,'log_neighbour_count',np.log1p(np.diff(graphs['exp50'].indptr)))
    group='lisa'
    for name,data in [('level',logdaily),('growth',growth)]:
        mean=np.nanmean(data,axis=0);sd=np.nanstd(data,axis=0)
        z=(data-mean)/np.maximum(sd,1e-8)
        lag=w@z
        add(group,name+'_moran_lag1',shift(z*lag,1))
        add(group,name+'_quadrant_lag1',shift((z>=0).astype(float)*2+(lag>=0).astype(float),1))
    symmetric=(graphs['knn8']+graphs['knn8'].T)/2
    def operator(v):
        centered=v-v.mean()
        result=symmetric@centered
        return result-result.mean()
    eigenvalues,eigenvectors=eigsh(LinearOperator((n,n),matvec=operator,dtype=float),
        k=8,which='LA',v0=np.linspace(-1,1,n),tol=1e-7)
    for i in range(8):
        vector=eigenvectors[:,i]
        if vector[np.argmax(np.abs(vector))]<0:vector=-vector
        add('eigenvectors','eigen_'+str(i+1),vector*np.sqrt(n))
    forecasts,iv_metadata=expanding_forecasts(logmonthly,static,w,graphs['knn8'])
    for name,prediction in forecasts.items():
        group='hybrid_'+name
        add(group,'market_growth_forecast',prediction)
        add(group,'forecast_vs_recent_growth',prediction-shift(growth,1))
    duplicates=locations.duplicated(['settlement_lat','settlement_lon'],keep=False).to_numpy()
    invalid=(~locations.settlement_lat.between(41,82) | ~locations.settlement_lon.between(19,181)).to_numpy()
    doubtful=duplicates|invalid
    clean=w.copy().tolil()
    clean[doubtful,:]=0;clean[:,doubtful]=0
    clean=normalise(clean.tocsr())
    for name,data in [('level_lag1',shift(logdaily,1)),('growth_lag1',shift(dailygrowth,1)),
                      ('growth_1_3',(shift(logdaily,1)-shift(logdaily,3))/2)]:
        array=clean@data;array[doubtful]=np.nan
        add('geo_sensitivity',name,array)
    table=frame[['new_id','time']].rename(columns={'time':'year_month_index'}).copy()
    for name,array in values.items():table[name]=array
    store_log=np.log(frame.pto.to_numpy()/days[row_times])
    own_prior=pd.Series(store_log,index=frame.index).groupby(frame.new_id).shift(1).to_numpy()
    for kind in ['exp50','knn8','exp15','inverse75','binary75','economic8','hybrid8','random']:
        group=kind+'_history'
        name=group+'__own_peer_gap_lag1'
        table[name]=(own_prior-table[group+'__level_lag1'].to_numpy()).astype('float32')
        columns[group].append(name)
    name='geo_sensitivity__own_peer_gap_lag1'
    table[name]=(own_prior-table['geo_sensitivity__level_lag1'].to_numpy()).astype('float32')
    columns['geo_sensitivity'].append(name)
    own_monthly=np.log(frame.pto).groupby(frame.new_id).diff().to_numpy()
    own_previous=pd.Series(own_monthly,index=frame.index).groupby(frame.new_id).shift(1).to_numpy()
    table['growth_exp50__own_peer_growth_gap']=(own_previous-table['growth_exp50__monthly_growth_lag1'].to_numpy()).astype('float32')
    columns['growth_exp50'].append('growth_exp50__own_peer_growth_gap')
    columns['regimes']+=columns['exp50_history']
    # Shop-level peers within the settlement: exclude own shop explicitly.
    totals=frame.groupby(['node','time']).pto.transform('sum')
    count=frame.groupby(['node','time']).pto.transform('size')
    loo=(totals-frame.pto)/(count-1)
    loo=np.log(loo/days[row_times])
    loo_prior=loo.groupby(frame.new_id).shift(1)
    loo_growth=loo.groupby(frame.new_id).diff().groupby(frame.new_id).shift(1)
    table['within_market__level_lag1']=loo_prior.astype('float32')
    table['within_market__growth_lag1']=loo_growth.astype('float32')
    table['within_market__own_peer_gap']=(own_prior-loo_prior.to_numpy()).astype('float32')
    columns['within_market']=['within_market__level_lag1','within_market__growth_lag1','within_market__own_peer_gap']
    table=table.sort_values(['new_id','year_month_index']).reset_index(drop=True)
    feature_names=[c for c in table if c not in ['new_id','year_month_index']]
    table[feature_names]=table[feature_names].replace([np.inf,-np.inf],np.nan)
    counts=first.groupby('node').size().reindex(range(n))
    stores=stores.merge(pd.DataFrame({'node':range(n),'market_shops':counts.to_numpy(),
        'population_log':static[:,0],'competition_log':static[:,4],
        'nearest_market_km':distances[:,1]*6371,'doubtful_coordinate':doubtful}),
        on='node',validate='many_to_one')
    info.update(feature_version=FEATURE_VERSION,groups=columns,
        weights={k:dict(nnz=int(m.nnz),zero_diagonal=True,row_normalised=True) for k,m in graphs.items()},
        eigenvalues=eigenvalues.tolist(),doubtful_coordinate_markets=int(doubtful.sum()),
        doubtful_coordinate_shops=int(stores.doubtful_coordinate.sum()),
        doubtful_policy='Sensitivity only: aliases/outside broad geographic bounds; source unchanged',
        iv_forecasts=iv_metadata,cutoff='Every target-dependent feature at t uses only outcomes < t',
        static_policy='First available shop record, median within market; no target similarity',
        source_hash=hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest())
    info['implementation_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),HERE/'economic_forecasts.py']}
    return table,info,stores

def main():
    raw=read_raw(load_recipe())
    table,info,stores=compute(raw)
    cache=HERE/'cache';cache.mkdir(exist_ok=True)
    table.to_parquet(cache/'features.parquet',index=False)
    stores.to_parquet(cache/'stores.parquet',index=False)
    (cache/'feature_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(table),features=len(table.columns)-2,groups={k:len(v) for k,v in info['groups'].items()},
        dubious_markets=info['doubtful_coordinate_markets'],joint_IV_stable=sum(x['stable_joint_forecast'] for x in info['iv_forecasts'])),indent=2))

if __name__=='__main__':main()
