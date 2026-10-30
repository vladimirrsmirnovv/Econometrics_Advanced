"""Shared results, group errors, component selection and honest past-only weights."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import hashlib,importlib.metadata,itertools,json,sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,load_recipe,ROOT,NOTEBOOK_SHA256
from run_validation import original_blends
FOLDS=['2024-03','2025-01','2025-02']

def prediction(fold,model,config):
    return pd.read_csv(HERE/'results'/fold/model.replace("_", "-")/(config+'.csv')).sort_values('new_id').reset_index(drop=True)

def main():
    recipe=load_recipe();records=[]
    raw_digest=hashlib.sha256((resource_path('Data/Shops/train_2.csv')).read_bytes()).hexdigest()
    cpu_environment={name:importlib.metadata.version(name) for name in ['pandas','numpy','catboost','lightgbm','xgboost']}
    for p in sorted((HERE/'results').glob('*/*/*.json')):
        r=json.loads(p.read_text());fold=p.parts[-3];model=p.parts[-2];config=p.stem
        r.update(fold=fold,model=model,configuration=config)
        r.setdefault('notebook_sha256',NOTEBOOK_SHA256);r.setdefault('raw_sha256',raw_digest)
        base_backend='CUDA XGBoost 3.1.3' if model=='xgboost_log_diff' else 'CPU '+cpu_environment['catboost' if model.startswith('catboost') else 'lightgbm']
        r['underlying_booster_backend']='None: standalone econometric forecast' if config.endswith('_standalone') else base_backend
        r['comparison_baseline_backend']=base_backend
        baseline_path=p.parent/'baseline.json'
        if not baseline_path.exists():continue
        b=json.loads(baseline_path.read_text())
        table=pd.read_csv(p.with_suffix('.csv')).sort_values('new_id')
        base=pd.read_csv(p.parent/'baseline.csv').sort_values('new_id')
        np.testing.assert_array_equal(table.new_id,base.new_id)
        np.testing.assert_allclose(table.pto,base.pto,rtol=1e-14)
        ape=abs(table.prediction.to_numpy()-table.pto.to_numpy())/table.pto.to_numpy()*100
        old=abs(base.prediction.to_numpy()-base.pto.to_numpy())/base.pto.to_numpy()*100
        assert len(table)==18657 and table.new_id.is_unique and np.isfinite(table.prediction).all()
        assert abs(ape.mean()-r['mape'])<1e-9,str(p)
        r.update(baseline_mape=b['mape'],delta_mape=r['mape']-b['mape'],
            prediction_sha256=hashlib.sha256(p.with_suffix('.csv').read_bytes()).hexdigest(),
            baseline_prediction_sha256=hashlib.sha256((p.parent/'baseline.csv').read_bytes()).hexdigest(),
            relative_improvement_pct=(b['mape']-r['mape'])/b['mape']*100,
            shops_improved_pct=float((ape<old).mean()*100),prediction_file=str(p.with_suffix('.csv').relative_to(HERE)),
            ape_change_p05=float(np.quantile(ape-old,.05)),ape_change_p25=float(np.quantile(ape-old,.25)),
            ape_change_p50=float(np.quantile(ape-old,.5)),ape_change_p75=float(np.quantile(ape-old,.75)),
            ape_change_p95=float(np.quantile(ape-old,.95)))
        for k in list(r):
            if isinstance(r[k],(dict,list)):r[k]=json.dumps(r[k],ensure_ascii=False)
        records.append(r)
    ledger=pd.DataFrame(records);ledger.to_csv(HERE/'all_experiments.csv',index=False,encoding='utf-8-sig')
    primary=ledger[ledger.fold.isin(FOLDS)]
    pivot=primary.pivot(index=['model','configuration'],columns='fold',values='mape')
    pivot['months']=pivot[FOLDS].notna().sum(axis=1)
    pivot['available_month_mean_mape']=pivot[FOLDS].mean(axis=1)
    pivot['mean_mape']=pivot['available_month_mean_mape'].where(pivot.months==3)
    pivot['comparison']='three_months' if len(pivot)==0 else np.where(pivot.months==3,'three_months','screening_only')
    pivot=pivot.sort_values(['model','mean_mape'])
    pivot.to_csv(HERE/'model_comparison.csv',encoding='utf-8-sig')
    def intervention(config):
        if config=='baseline':return 'baseline'
        if config=='key':return 'technical_key'
        if config.startswith('key+'):return 'spatial_and_key'
        if config=='density_weight':return 'training_weights'
        if '+residual_' in config:return 'spatial_and_error_correction'
        if config.startswith('residual_'):return 'error_correction'
        if config.startswith(('sem_','sdem_')):return 'economic_forecast'
        return 'spatial_features'
    effects=pivot.reset_index();effects['intervention']=effects.configuration.map(intervention)
    effects=effects[effects.months==3].sort_values('mean_mape').groupby(['model','intervention'],as_index=False).first()
    effects.to_csv(HERE/'intervention_comparison.csv',index=False,encoding='utf-8-sig')
    choices={};past_choices={}
    for model in MODELS:
        d=pivot.loc[model];complete=d[d.months==3]
        assert 'baseline' in complete.index
        chosen=complete.mean_mape.idxmin();choices[model]=chosen
        earlier=d.dropna(subset=FOLDS[:2]);past_choices[model]=earlier[FOLDS[:2]].mean(axis=1).idxmin()
    configs=dict(primary_months=FOLDS,selection='Equal-month mean MAPE; original baseline always eligible',
        components=choices,past_only_components=past_choices,
        full_training_not_run=True,model_comparison_sha_source='all_experiments.csv',
        model_comparison_sha256=hashlib.sha256((HERE/'model_comparison.csv').read_bytes()).hexdigest(),
        final_tree_counts=dict(catboost_log_level=6500,lightgbm_log_diff=7000,lightgbm_log_level=8000,xgboost_log_diff=3750),
        xgboost=dict(version='3.1.3',device='cuda'),
        limitation='Primary folds used in candidate selection; apparent gains need the extra historical check.')
    (HERE/'best_model_configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    outputs=HERE/'ensemble';outputs.mkdir(exist_ok=True)
    blend_records=[];residual_correlations=[]
    predictions={}
    # Bounded component interaction search. XGB alternatives have an extra GPU month already.
    pools={}
    for model in MODELS:
        complete=pivot.loc[model].query('months==3').sort_values('mean_mape')
        if model=='xgboost_log_diff':pool=['baseline','hybrid8_history','knn8_history+wx_relative']
        else:pool=list(dict.fromkeys(['baseline']+(['exp50_history'] if model=='catboost_log_level' else [])+complete.head(2).index.tolist()))
        pools[model]=pool
    grid=[];vectors={};truth={}
    for fold in FOLDS:
        truth[fold]=prediction(fold,MODELS[0],'baseline').pto.to_numpy()
        vectors[fold]={(m,c):prediction(fold,m,c).prediction.to_numpy() for m in MODELS for c in pools[m]}
    for combination in itertools.product(*[pools[m] for m in MODELS]):
        selection=dict(zip(MODELS,combination));row=dict(components=selection)
        for fold in FOLDS:
            pred=original_blends({m:vectors[fold][(m,c)] for m,c in selection.items()})['final_meta_logmean']
            row[fold]=float((abs(pred-truth[fold])/truth[fold]).mean()*100)
        row['mean_mape']=float(np.mean([row[f] for f in FOLDS]));grid.append(row)
    grid=sorted(grid,key=lambda row:row['mean_mape'])
    grid_choices=grid[0]['components']
    (HERE/'ensemble_component_grid.json').write_text(json.dumps(dict(pools=pools,runs=len(grid),
        best=grid[0],results=grid,limitation='Selected on the same three months; extra month needed'),indent=2))
    configs['ensemble_components']=grid_choices
    (HERE/'best_model_configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    for fold in FOLDS:
        reference=prediction(fold,MODELS[0],'baseline');true=reference.pto.to_numpy()
        configurations={'baseline':{m:'baseline' for m in MODELS},'best_components':choices,
            'best_component_combination':grid_choices}
        for replaced in MODELS:
            configurations['replace_'+replaced]={m:choices[m] if m==replaced else 'baseline' for m in MODELS}
        predictions[fold]={}
        for name,selection in configurations.items():
            vectors={m:prediction(fold,m,c).prediction.to_numpy() for m,c in selection.items()}
            pred=original_blends(vectors)['final_meta_logmean']
            predictions[fold][name]=pred
            row=recipe._make_metrics(true,pred);row.update(fold=fold,configuration=name,components=selection,
                original_blend_unchanged=True,validation_only=True)
            blend_records.append(row)
            table=reference.copy();table['prediction']=pred;table.to_csv(outputs/(fold+'_'+name+'.csv'),index=False)
            if name in ['baseline','best_components']:
                for blend,variant in original_blends(vectors).items():
                    if blend=='final_meta_logmean':continue
                    details=recipe._make_metrics(true,variant)
                    details.update(fold=fold,configuration=name+'__'+blend,components=selection,
                        original_blend_rule=blend,validation_only=True)
                    blend_records.append(details)
                    table=reference.copy();table['prediction']=variant
                    table.to_csv(outputs/(fold+'_'+name+'__'+blend+'.csv'),index=False)
        for label,selection in [('baseline',{m:'baseline' for m in MODELS}),('best_components',choices)]:
            errors=np.column_stack([(prediction(fold,m,c).prediction.to_numpy()-true)/true for m,c in selection.items()])
            correlation=pd.DataFrame(errors,columns=MODELS).corr()
            for a,b in itertools.combinations(MODELS,2):residual_correlations.append(dict(fold=fold,configuration=label,model_a=a,model_b=b,error_correlation=correlation.loc[a,b]))
    # Every component and its weight is chosen using earlier folds only.
    logp=[];targets=[]
    for fold in FOLDS[:2]:
        frames=[prediction(fold,m,past_choices[m]) for m in MODELS]
        logp.append(np.column_stack([np.log1p(f.prediction.to_numpy()) for f in frames]));targets.append(frames[0].pto.to_numpy())
    logp=np.vstack(logp);targets=np.concatenate(targets)
    def loss(w):return np.mean(abs(np.expm1(logp@w)-targets)/targets)
    optimum=minimize(loss,np.full(4,.25),method='SLSQP',bounds=[(0,1)]*4,
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],options={'maxiter':200,'ftol':1e-10})
    assert optimum.success
    weights=optimum.x/optimum.x.sum()
    for fold in FOLDS:
        frames=[prediction(fold,m,past_choices[m]) for m in MODELS]
        pred=np.expm1(np.column_stack([np.log1p(f.prediction.to_numpy()) for f in frames])@weights)
        row=recipe._make_metrics(frames[0].pto,pred)
        row.update(fold=fold,configuration='past_only_weighted_logblend',components=past_choices,
            weights=weights.tolist(),weight_training_months=FOLDS[:2],
            independent_weight_evaluation=(fold==FOLDS[-1]),validation_only=True)
        blend_records.append(row);table=frames[0].copy();table['prediction']=pred
        table.to_csv(outputs/(fold+'_past_only_weighted_logblend.csv'),index=False)
    (outputs/'metrics.json').write_text(json.dumps(blend_records,ensure_ascii=False,indent=2))
    pd.DataFrame(blend_records).to_csv(HERE/'ensemble_comparison.csv',index=False,encoding='utf-8-sig')
    eligible_blends=pd.DataFrame(blend_records)
    eligible_blends=eligible_blends[eligible_blends.configuration.str.startswith('best_components')]
    blend_means=eligible_blends.groupby('configuration').mape.mean()
    blend_choice=blend_means.idxmin()
    configs.update(reference_ensemble_rule='final_meta_logmean',
        recommended_ensemble_rule=blend_choice.split('__',1)[1] if '__' in blend_choice else 'final_meta_logmean',
        recommended_ensemble_primary_mean_mape=float(blend_means.loc[blend_choice]),
        blend_selection_months=FOLDS,
        blend_selection_limitation='Chosen on primary validation months, not the additional October check')
    (HERE/'best_model_configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    pd.DataFrame(residual_correlations).to_csv(HERE/'error_correlations.csv',index=False)
    # Group diagnostics are descriptive; current-target bins are never model inputs.
    stores=pd.read_parquet(HERE/'cache/stores.parquet').sort_values('new_id').reset_index(drop=True)
    groups={'region':stores.region,'market_size':pd.cut(stores.market_shops,[0,1,10,50,1e9],labels=['1','2-10','11-50','51+']),
        'population':pd.cut(np.expm1(stores.population_log),[-1,10000,50000,500000,np.inf],labels=['<10k','10-50k','50-500k','500k+']),
        'isolation_km':pd.cut(stores.nearest_market_km,[-1,5,25,np.inf],labels=['<5','5-25','25+']),
        'competition':pd.cut(np.expm1(stores.competition_log),[-1,2,10,np.inf],labels=['0-2','3-10','11+']),
        'coordinate_flag':stores.doubtful_coordinate.astype(str)}
    group_records=[]
    for fold in FOLDS:
        for model in MODELS:
            candidate=prediction(fold,model,choices[model]);base=prediction(fold,model,'baseline')
            ape=abs(candidate.prediction-candidate.pto)/candidate.pto*100
            old=abs(base.prediction-base.pto)/base.pto*100
            group_map=dict(groups,observed_rto_quintile=pd.qcut(candidate.pto,5,labels=False,duplicates='drop'))
            for family,labels in group_map.items():
                data=pd.DataFrame(dict(group=labels,ape=ape,old=old,better=ape<old))
                for label,part in data.groupby('group',observed=True):
                    group_records.append(dict(fold=fold,model=model,configuration=choices[model],group_family=family,
                        group=str(label),shops=len(part),baseline_mape=part.old.mean(),mape=part.ape.mean(),
                        delta_mape=part.ape.mean()-part.old.mean(),shops_improved_pct=part.better.mean()*100))
    pd.DataFrame(group_records).to_csv(HERE/'group_effects.csv',index=False,encoding='utf-8-sig')
    # Paired regional bootstrap keeps stores in the same region together.
    cluster_records=[];rng=np.random.default_rng(20261009)
    for fold in FOLDS:
        reference=prediction(fold,MODELS[0],'baseline');true=reference.pto.to_numpy()
        difference=abs(predictions[fold]['best_components']-true)/true*100-abs(predictions[fold]['baseline']-true)/true*100
        clustered=pd.DataFrame(dict(region=reference.region,delta=difference)).groupby('region').delta.agg(['sum','size'])
        draws=rng.integers(0,len(clustered),size=(1000,len(clustered)))
        simulated=clustered['sum'].to_numpy()[draws].sum(axis=1)/clustered['size'].to_numpy()[draws].sum(axis=1)
        cluster_records.append(dict(fold=fold,delta_mape=float(difference.mean()),
            cluster_bootstrap_p025=float(np.quantile(simulated,.025)),cluster_bootstrap_p975=float(np.quantile(simulated,.975)),
            clusters=len(clustered),unit='region',replicates=1000,
            limitation='Conditional on selected configurations; does not account for searching many candidates or repeated seeds'))
    pd.DataFrame(cluster_records).to_csv(HERE/'ensemble_cluster_uncertainty.csv',index=False)
    from spatial_features import make_weights
    locations=pd.read_csv(HERE/'geography_diagnostics.csv').sort_values('node')
    w,_=make_weights(locations,'exp50');moran_records=[]
    for fold in FOLDS:
        for model in MODELS:
            for config in ['baseline',choices[model]]:
                frame=prediction(fold,model,config)
                errors=pd.DataFrame(dict(node=stores.node,error=(frame.prediction-frame.pto)/frame.pto))
                market=errors.groupby('node').error.mean().reindex(locations.node).to_numpy()
                z=market-market.mean();value=float(len(z)/w.sum()*(z@(w@z))/(z@z))
                moran_records.append(dict(fold=fold,model=model,configuration=config,
                    moran_I_market_percentage_error=value,weights='exp50',
                    descriptive_only=True,significance_test_run=False))
    pd.DataFrame(moran_records).to_csv(HERE/'residual_spatial_structure.csv',index=False)
    (HERE/'result_checks.json').write_text(json.dumps(dict(status='passed',prediction_files=len(records),
        rows_per_file=18657,metrics_recomputed=True,ids_and_targets_matched=True),indent=2))
    print('CHECKED',len(records),'BEST_COMPONENTS',choices,'PAST_ONLY_WEIGHTS',dict(zip(MODELS,weights)))
    print(pd.DataFrame(blend_records).pivot(index='configuration',columns='fold',values='mape').to_string())

if __name__=='__main__':main()
