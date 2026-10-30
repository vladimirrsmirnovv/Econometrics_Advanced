"""Original full-history March split and strictly historical spatial extensions."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import gc,hashlib,json,sys,warnings
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'research'))
from notebook_api import ROOT,NOTEBOOK_SHA256,load_recipe,read_raw,compact_fold
from spatial_features import spatial_table,attach_locations
from build_features import STATIC,weight_family
OUT=HERE.parent/'cache/final_2025-03'
HYBRID=['hybrid8_history__level_lag1','hybrid8_history__daily_growth_lag1',
        'hybrid8_history__growth_1_3','hybrid8_history__own_peer_gap_lag1']
HISTORY=['wy_log_daily_lag1','wy_daily_growth_lag1','wy_daily_growth_1_3','wy_log_daily_gap_lag1']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def hybrid_next(history,forecast_time):
    stores,locations,_=attach_locations(history)
    frame=history.merge(stores[['new_id','node']],on='new_id',validate='many_to_one')
    frame['time']=(frame.year-2023)*12+frame.month-1
    assert frame.time.max()==forecast_time-1
    frame=frame.sort_values(['time','new_id']);n=len(locations)
    first=frame.drop_duplicates('new_id')
    static=np.log1p(first.groupby('node')[STATIC].median().reindex(range(n)).clip(lower=0).to_numpy())
    static=np.where(np.isfinite(static),static,np.nanmedian(static,axis=0))
    profile=(static-static.mean(axis=0))/np.maximum(static.std(axis=0),1e-8)
    weights=weight_family(locations,'hybrid8',profile)
    times=np.arange(forecast_time)
    means=frame.groupby(['node','time']).pto.mean().unstack('time').reindex(index=range(n),columns=times)
    assert np.isfinite(means.to_numpy()).all()
    days=pd.to_datetime({'year':2023+times//12,'month':1+times%12,'day':1}).dt.days_in_month.to_numpy()
    logdaily=np.log(means.to_numpy()/days)
    peer=weights@logdaily
    change=weights@np.column_stack([np.full(n,np.nan),np.diff(logdaily,axis=1)])
    last=frame[frame.time==forecast_time-1].sort_values('new_id')
    nodes=last.node.to_numpy();level=peer[nodes,-1].astype('float32')
    result=pd.DataFrame({'new_id':last.new_id.to_numpy(),'year_month_index':forecast_time,
        HYBRID[0]:level,HYBRID[1]:change[nodes,-1].astype('float32'),
        HYBRID[2]:((peer[nodes,-1]-peer[nodes,-3])/2).astype('float32'),
        HYBRID[3]:(np.log(last.pto.to_numpy()/days[-1])-level).astype('float32')})
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    recipe=load_recipe();raw=read_raw(recipe)
    raw_hash=digest(resource_path('Data/Shops/train_2.csv'))
    if not (OUT/'manifest.json').exists():
        print('BUILD_ORIGINAL_FULL_SPLIT',flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            features=recipe.build_retail_features_with_split(raw,split_mode='train_val_test',min_year=2023,
                val_year=2025,val_month=2,test_year=2025,test_month=3,create_test_if_missing=True,
                expected_test_size=18657,use_val_as_test_history=True,
                base_feature_kwargs=recipe.COMMON_BASE_FEATURE_KWARGS,verbose=True)
        compact=compact_fold(recipe,features)
        assert len(compact['test_train_features'])==485082 and len(compact['test_features'])==18657
        assert compact['test_train_features'].year_month_index.max()==25
        assert (compact['test_features'].year_month_index==26).all() and compact['test_features'].pto.isna().all()
        for key in ['train_features','val_features','test_train_features','test_features']:
            compact[key].to_parquet(OUT/(key+'.parquet'),index=False)
        manifest=dict(raw_sha256=raw_hash,notebook_sha256=NOTEBOOK_SHA256,
            categorical_cols=compact['categorical_cols'],feature_cols=compact['feature_cols'],
            training_months=['2023-01','2025-02'],forecast_month='2025-03',full_rows=485082,shops=18657,
            split_mode='train_val_test',create_test_if_missing=True,use_val_as_test_history=True)
        manifest['files']={p.name:digest(p) for p in OUT.glob('*features.parquet')}
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        del features,compact;gc.collect()
    else:
        manifest=json.loads((OUT/'manifest.json').read_text())
        assert manifest['raw_sha256']==raw_hash and manifest['notebook_sha256']==NOTEBOOK_SHA256
    # Validate the new forecast extension against a previously measured holdout.
    time=(raw.year-2023)*12+raw.month-1
    feature_cache=ROOT/'Forecasting/research/cache'
    cached=pd.read_parquet(feature_cache/'features.parquet',columns=['new_id','year_month_index']+HYBRID+
        json.loads((feature_cache/'feature_manifest.json').read_text())['groups']['eigenvectors'])
    before=hybrid_next(raw[time<25],25).sort_values('new_id').reset_index(drop=True)
    expected=cached[cached.year_month_index==25].sort_values('new_id').reset_index(drop=True)
    np.testing.assert_array_equal(before.new_id,expected.new_id)
    np.testing.assert_allclose(before[HYBRID],expected[HYBRID],rtol=0,atol=1e-6,equal_nan=True)
    future=hybrid_next(raw,26)
    eigen=[c for c in cached if c.startswith('eigenvectors__')]
    future=future.merge(expected[['new_id']+eigen],on='new_id',validate='one_to_one')
    table=pd.concat([cached,future],ignore_index=True)
    table.to_parquet(OUT/'research_features.parquet',index=False)
    # Preserve the legacy subtraction precision used by the selected CatBoost.
    future_raw=raw[(raw.year==2025)&(raw.month==2)].copy()
    future_raw['month']=3;future_raw['pto']=np.nan
    extended=pd.concat([raw,future_raw],ignore_index=True)
    legacy,legacy_meta=spatial_table(extended,kind='exp50')
    reference=pd.read_parquet(ROOT/'Forecasting/cache/spatial_exp50.parquet')
    cols=['new_id','year_month_index']+HISTORY
    pd.testing.assert_frame_equal(legacy.loc[legacy.year_month_index<26,cols].reset_index(drop=True),
        reference[cols].reset_index(drop=True),check_exact=True)
    changed=extended.copy();changed.loc[changed.month.eq(3)&changed.year.eq(2025),'pto']=1e12
    altered,_=spatial_table(changed,kind='exp50')
    pd.testing.assert_frame_equal(legacy[cols],altered[cols],check_exact=True)
    legacy[cols].to_parquet(OUT/'legacy_features.parquet',index=False)
    stores=pd.read_parquet(feature_cache/'stores.parquet')
    stores.to_parquet(OUT/'stores.parquet',index=False)
    checks=dict(status='passed',forecast_month='2025-03',future_target_unknown=True,
        historical_legacy_features_exact=True,hybrid_extension_matches_measured_february=True,
        hybrid_extension_max_difference=float(np.nanmax(np.abs(before[HYBRID].to_numpy()-expected[HYBRID].to_numpy()))),
        future_target_perturbation_legacy_passed=True,training_uses_only_times_below_26=True,
        static_eigenvectors_reused_exactly=True,
        spatial_implementation_sha256=digest(ROOT/'Forecasting/spatial_features.py'),
        research_implementation_sha256=digest(ROOT/'Forecasting/research/build_features.py'))
    (OUT/'feature_checks.json').write_text(json.dumps(checks,indent=2))
    future[['new_id','year_month_index']+HYBRID].to_csv(OUT/'gpu_future_hybrid.csv',index=False)
    print('MARCH_FEATURES_READY',json.dumps(checks),flush=True)

if __name__=='__main__':main()
