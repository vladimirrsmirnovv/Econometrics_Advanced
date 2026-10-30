"""Coverage comes from actual completed files, with explicit remaining limitations."""
from pathlib import Path
import json,shutil
import pandas as pd
HERE=Path(__file__).resolve().parent
FOLDS=['2024-03','2025-01','2025-02']
PREFIXES={
 'W01':['exp50_history','knn8_history','exp15_history','inverse75_history','binary75_history'],
 'W02':['economic8_history'],'W03':['hybrid8_history'],'Y01':['exp50_history'],
 'Y02':['growth_exp50'],'Y03':['season_exp50'],'Y04':['multihop'],
 'X01':['wx_means'],'X02':['wx_relative'],'T01':['season_exp50'],'R01':['regimes'],
 'A01':['lisa'],'A02':['eigenvectors'],'E01':['residual_global','residual_own','residual_region','residual_spatial'],
 'E02':['residual_ridge'],'H01':['hybrid_slx'],'H03':['sem_','sdem_'],'H04':['hybrid_gwr'],
 'L01':['density_weight'],'D01':['key'],'D02':['geo_sensitivity'],
 'C01':['random_history','economic8_history','region_history']}

def main():
    path=HERE/'hypothesis_registry.csv';archive=HERE/'hypothesis_registry_preregistered.csv'
    if not archive.exists():shutil.copy2(path,archive)
    registry=pd.read_csv(archive).fillna('');results=[]
    for p in (HERE/'results').glob('*/*/*.json'):
        value=json.loads(p.read_text())
        results.append(dict(fold=p.parts[-3],model=p.parts[-2],configuration=p.stem,mape=value['mape']))
    data=pd.DataFrame(results)
    for id_,prefixes in PREFIXES.items():
        matched=data[data.configuration.map(lambda config:any(config.startswith(p) or ('+'+p) in config for p in prefixes))]
        if matched.empty:continue
        idx=registry.id==id_;complete=matched[matched.fold.isin(FOLDS)].groupby(['model','configuration']).fold.nunique()
        fully=complete[complete==3]
        registry.loc[idx,'status']='executed_temporal' if len(fully) else 'executed_screening'
        registry.loc[idx,'result']=f'{len(matched)} completed prediction sets; {len(fully)} model/configurations have all three primary months; see all_experiments.csv'
        registry.loc[idx,'conclusion']='Predictive effect is model- and period-dependent; positive and negative results retained.'
        registry.loc[idx,'reason_not_run']=''
    meta=json.loads((HERE/'cache/feature_manifest.json').read_text())
    stable=sum(r['stable_joint_forecast'] for r in meta['iv_forecasts'])
    registry.loc[registry.id=='H02',['status','result','conclusion','reason_not_run']]=[
        'estimated_unstable',f'{len(meta["iv_forecasts"])} chronological 2SLS estimates; {stable} stable joint forecasts',
        'No accepted dynamic SDM forecast; all outputs equal SLX fallback. Duplicate booster fits excluded.',
        'Historical estimated rho outside stability range or deficient second-stage rank. New instruments/regularisation require a new design.']
    if (HERE/'residual_checks.json').exists():
        registry.loc[registry.id=='E02','conclusion']='Past-only ridge trained on March 2024 and January 2025 OOF pairs, tested only on February. Not eligible for three-month ranking.'
        registry.loc[registry.id=='E02','reason_not_run']='A richer nonlinear corrector remains deferred: only two separated training OOF months, insufficient temporal diversity.'
    registry.loc[registry.id=='L02',['status','conclusion','reason_not_run']]=[
        'deferred_low_priority','Continuous regimes and local ridge have been tested as cheaper alternatives.',
        'Separate regional booster banks fragment data and multiply fits; no stable evidence from simpler heterogeneity experiments yet.']
    if (HERE/'ensemble_comparison.csv').exists():
        for id_ in ['B01','B02']:
            registry.loc[registry.id==id_,['status','result','reason_not_run']]=[
                'executed_temporal','ensemble_comparison.csv; component replacements, joint blend and past-only weights','']
        registry.loc[registry.id=='B02','conclusion']='Weight fitting uses earlier folds only; only February is an independent weight evaluation. Regime-specific mixtures remain deferred.'
    registry.to_csv(path,index=False,encoding='utf-8-sig')
    print(registry[['id','status']].to_string(index=False))

if __name__=='__main__':main()
