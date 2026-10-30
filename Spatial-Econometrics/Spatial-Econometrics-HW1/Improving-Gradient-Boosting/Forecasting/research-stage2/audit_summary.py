"""Test one frozen ensemble and report alternatives without audit reselection."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS
from selection import load,blend

def main():
    frozen=json.loads((HERE/'frozen_final_selection.json').read_text());folds=frozen['reserved_months']
    # Completeness is checked before reading any aggregate reserved outcome.
    for fold in folds:
        for model in MODELS:
            for config in dict.fromkeys(['baseline','previous',frozen['individual'][model]]):assert (HERE/'results/rolling'/fold/model.replace("_", "-")/(config+'.json')).exists()
            assert (HERE/'results/full'/fold/model.replace("_", "-")/'baseline.json').exists()
    individual=[];ensembles=[];paired=[];regions=[];markets=[]
    stores=pd.read_parquet(HERE.parent/'research/cache/stores.parquet').set_index('new_id')
    for fold in folds:
        sets={k:{} for k in ['baseline','previous','mandatory_mixed','new','cautious_new','original_full','mandatory_mixed_full']};truth=None;anchor=None
        for model in MODELS:
            b,br=load(fold,model,'baseline');p,pr=load(fold,model,'previous');n,nr=load(fold,model,frozen['individual'][model]);o,orr=load(fold,model,'baseline','full')
            if truth is None:truth=b.pto.to_numpy();anchor=b[['new_id','region','settlement']].copy()
            for f in [p,n,o]:np.testing.assert_array_equal(f.new_id,b.new_id);np.testing.assert_allclose(f.pto,truth)
            for label,row in [('original_full',orr),('original_rolling',br),('previous_rolling',pr),('frozen_new',nr)]:
                individual.append(dict(fold=fold,model=model,variant=label,configuration=row['configuration'],mape=row['mape'],trees=row['trees']))
            sets['baseline'][model]=b.prediction.to_numpy();sets['previous'][model]=p.prediction.to_numpy();sets['new'][model]=n.prediction.to_numpy()
            sets['mandatory_mixed'][model]=(p if model==MODELS[-1] else b).prediction.to_numpy()
            sets['cautious_new'][model]=(b.prediction.to_numpy()+n.prediction.to_numpy())*.5;sets['original_full'][model]=o.prediction.to_numpy()
            sets['mandatory_mixed_full'][model]=load(fold,model,'previous','full')[0].prediction.to_numpy() if model==MODELS[-1] else o.prediction.to_numpy()
            frame=anchor.copy();frame['node']=frame.new_id.map(stores.node);frame['base_ape']=100*abs(b.prediction.to_numpy()-truth)/truth;frame['new_ape']=100*abs(n.prediction.to_numpy()-truth)/truth
            for group,target in [('region',regions),('node',markets)]:
                part=frame.groupby(group).agg(shops=('new_id','size'),baseline_mape=('base_ape','mean'),new_mape=('new_ape','mean')).reset_index()
                part['delta']=part.new_mape-part.baseline_mape;part['fold']=fold;part['model']=model;target.extend(part.to_dict('records'))
        selected=None;rolling_base=None;full_base=None
        for composition,components in sets.items():
            predictions=blend(components)
            for rule,pred in predictions.items():
                ensembles.append(dict(fold=fold,composition=composition,rule=rule,mape=float((abs(pred-truth)/truth).mean()*100)))
            if composition==frozen['ensemble']['composition']:selected=predictions[frozen['ensemble']['rule']]
            if composition=='baseline':rolling_base=predictions['final_meta_logmean']
            if composition=='original_full':full_base=predictions['final_meta_logmean']
        frame=anchor.copy();frame['fold']=fold
        for label,pred in [('selected',selected),('rolling_baseline',rolling_base),('original_full',full_base)]:frame[label+'_ape']=100*abs(pred-truth)/truth
        frame['delta_rolling']=frame.selected_ape-frame.rolling_baseline_ape;frame['delta_full']=frame.selected_ape-frame.original_full_ape;paired.append(frame)
    pd.DataFrame(individual).to_csv(HERE/'reserved_model_comparison.csv',index=False)
    pd.DataFrame(ensembles).to_csv(HERE/'reserved_ensemble_comparison.csv',index=False)
    full_rows=[]
    for fold in ['2024-08']+folds:
        full_sets={'original_full':{}}
        if all((HERE/'results/full'/fold/m.replace("_", "-")/'previous.json').exists() for m in MODELS):full_sets['previous_full']={}
        full_sets['mandatory_mixed_full']={};truth=None
        for model in MODELS:
            base,_=load(fold,model,'baseline','full');truth=base.pto.to_numpy() if truth is None else truth
            full_sets['original_full'][model]=base.prediction.to_numpy()
            if 'previous_full' in full_sets:full_sets['previous_full'][model]=load(fold,model,'previous','full')[0].prediction.to_numpy()
            full_sets['mandatory_mixed_full'][model]=load(fold,model,'previous','full')[0].prediction.to_numpy() if model==MODELS[-1] else base.prediction.to_numpy()
        for composition,components in full_sets.items():
            for rule,pred in blend(components).items():full_rows.append(dict(fold=fold,composition=composition,rule=rule,mape=float((abs(pred-truth)/truth).mean()*100)))
    pd.DataFrame(full_rows).to_csv(HERE/'full_recipe_ensemble_comparison.csv',index=False)
    pd.DataFrame(regions).to_csv(HERE/'reserved_region_comparison.csv',index=False);pd.DataFrame(markets).to_csv(HERE/'reserved_market_comparison.csv',index=False)
    paired=pd.concat(paired,ignore_index=True);paired.to_parquet(HERE/'reserved_paired_errors.parquet',index=False)
    monthly=paired.groupby('fold')[['selected_ape','rolling_baseline_ape','original_full_ape','delta_rolling','delta_full']].mean()
    passed=bool((monthly.delta_rolling<0).all() and monthly.delta_rolling.mean()<=-.01 and monthly.delta_full.mean()<=0)
    rng=np.random.default_rng(20261010);cluster=paired.groupby('region').agg(count=('new_id','size'),rolling=('delta_rolling','sum'),full=('delta_full','sum'))
    draw=rng.integers(0,len(cluster),size=(2000,len(cluster)));count=cluster['count'].to_numpy()[draw].sum(axis=1)
    uncertainty={label:np.quantile(cluster[key].to_numpy()[draw].sum(axis=1)/count,[.025,.5,.975]).tolist() for label,key in [('versus_rolling','rolling'),('versus_original_full','full')]}
    decision=dict(status='passed' if passed else 'fallback',frozen_choice=frozen['ensemble'],reserved_months=folds,
        monthly=monthly.reset_index().to_dict('records'),mean_delta_rolling=float(monthly.delta_rolling.mean()),mean_delta_original_full=float(monthly.delta_full.mean()),
        criterion='Both reserved months improve rolling baseline; mean delta <= -0.01 MAPE pp; mean MAPE does not exceed original full baseline',
        recommendation='frozen_new_ensemble' if passed else 'original_full_final_meta_logmean',
        no_audit_reselection=True,regional_paired_bootstrap_95_percent=uncertainty,
        bootstrap_limit='Fixed selection; does not account for candidate search or uncertainty across future months')
    (HERE/'final_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
