"""Explicit development-only selection; reserved outcomes never read here."""
from pathlib import Path
import argparse,hashlib,itertools,json,sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from notebook_api import MODELS
from run_validation import original_blends
from experiment import NOVEL
DEVELOPMENT=['2024-06','2024-07','2024-08']

def load(fold,model,config,protocol='rolling'):
    path=HERE/'results'/protocol/fold/model.replace("_", "-")/(config+'.csv')
    frame=pd.read_csv(path).sort_values('new_id')
    info=json.loads(path.with_suffix('.json').read_text())
    assert info['prediction_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(frame)==18657 and frame.new_id.is_unique and np.isfinite(frame.prediction).all()
    return frame,info

def pick_one(model):
    candidates=[];scores=[]
    for config in NOVEL:
        _,row=load('2024-08',model,config);candidates.append((row['mape'],config))
        scores.append(dict(model=model,configuration=config,mape=row['mape']))
    picked=[c for _,c in sorted(candidates)[:2]]
    configs=list(dict.fromkeys(['baseline','previous','exp50_history','hybrid8_history','knn8_history','eigenvectors']+picked))
    if any('seasonal_spatial' in c for c in picked):configs+=['seasonal_control','seasonal_spatial']
    if 'innovations_spatial' in picked:configs+=['innovations_control']
    if 'dynamic_history' in picked:configs+=['geo8_control']
    return list(dict.fromkeys(configs)),scores

def screen():
    picks={};scores=[]
    for model in MODELS:
        picks[model],part=pick_one(model);scores+=part
    (HERE/'screen_selection.json').write_text(json.dumps(dict(screen_month='2024-08',
        rule='Two strongest predeclared new configurations plus matched ablations and transfers',configs=picks),indent=2))
    pd.DataFrame(scores).to_csv(HERE/'screen_new_methods.csv',index=False)
    print(json.dumps(picks,indent=2))

def blend(components):
    rules=original_blends(components)
    matrix=np.column_stack([components[m] for m in MODELS])
    rules['trimmed_mean']=np.sort(matrix,axis=1)[:,1:3].mean(axis=1)
    return rules

def robust(delta):
    delta=np.asarray(delta)
    eligible=delta.mean()<=-.01 and np.sum(delta<0)>=2 and delta.max()<=.10
    return bool(eligible),float(delta.mean()+.25*delta.std())

def freeze():
    screen_plan=json.loads((HERE/'screen_selection.json').read_text())['configs']
    chosen={};rows=[];details={}
    for model in MODELS:
        baseline=np.array([load(fold,model,'baseline')[1]['mape'] for fold in DEVELOPMENT])
        candidates=[];details[model]=[]
        for config in screen_plan[model]:
            measured=np.array([load(fold,model,config)[1]['mape'] for fold in DEVELOPMENT]);delta=measured-baseline
            eligible,score=robust(delta)
            row=dict(model=model,configuration=config,mean_mape=float(measured.mean()),
                mean_delta=float(delta.mean()),worst_delta=float(delta.max()),months_won=int((delta<0).sum()),
                eligible=eligible or config=='baseline',robust_score=score,
                **{fold:float(m) for fold,m in zip(DEVELOPMENT,measured)})
            rows.append(row);details[model].append(row)
            if eligible or config=='baseline':candidates.append((score,config))
        chosen[model]=min(candidates)[1] if candidates else 'baseline'
    pd.DataFrame(rows).to_csv(HERE/'development_model_comparison.csv',index=False)
    ensemble_rows=[];ensemble_predictions={};source_files={}
    for fold in DEVELOPMENT:
        sets={key:{} for key in ['baseline','previous','mandatory_mixed','new','cautious_new']}
        truth=None
        for model in MODELS:
            b,_=load(fold,model,'baseline');p,_=load(fold,model,'previous');n,_=load(fold,model,chosen[model])
            truth=b.pto.to_numpy() if truth is None else truth
            np.testing.assert_allclose(truth,n.pto)
            sets['baseline'][model]=b.prediction.to_numpy();sets['previous'][model]=p.prediction.to_numpy()
            sets['mandatory_mixed'][model]=(p if model=='xgboost_log_diff' else b).prediction.to_numpy()
            sets['new'][model]=n.prediction.to_numpy()
            sets['cautious_new'][model]=(b.prediction.to_numpy()+n.prediction.to_numpy())*.5
        base=blend(sets['baseline'])['final_meta_logmean']
        base_mape=float((abs(base-truth)/truth).mean()*100)
        for composition,components in sets.items():
            for rule,pred in blend(components).items():
                mape=float((abs(pred-truth)/truth).mean()*100)
                ensemble_rows.append(dict(fold=fold,composition=composition,rule=rule,mape=mape,baseline_mape=base_mape,delta=mape-base_mape))
    frame=pd.DataFrame(ensemble_rows);frame.to_csv(HERE/'development_ensemble_comparison.csv',index=False)
    shortlist=[(0.,'baseline','final_meta_logmean')]
    for (composition,rule),part in frame.groupby(['composition','rule']):
        eligible,score=robust(part.delta)
        if eligible:shortlist.append((score,composition,rule))
    selection=min(shortlist)[1:] if shortlist else ('baseline','final_meta_logmean')
    frozen=dict(development_months=DEVELOPMENT,reserved_months=['2024-11','2024-12'],
        individual=chosen,ensemble=dict(composition=selection[0],rule=selection[1]),
        selection_rule='mean delta <= -0.01; wins >=2/3; worst delta <=0.10; minimise mean+0.25*std',
        fallback_if_reserved_check_fails=dict(composition='baseline',rule='final_meta_logmean',protocol='original full'),
        individual_candidates=screen_plan,leaderboard_used_for_selection=False,
        reserved_acceptance_rule='Both reserved months improve rolling baseline; mean delta <= -0.01 MAPE pp; mean MAPE does not exceed original full baseline',
        plan_sha256=hashlib.sha256((HERE/'PLAN.md').read_bytes()).hexdigest(),
        canonical_features_sha256=hashlib.sha256((HERE/'cache/features.parquet').read_bytes()).hexdigest(),
        development_model_metrics_sha256=hashlib.sha256((HERE/'development_model_comparison.csv').read_bytes()).hexdigest(),
        development_ensemble_metrics_sha256=hashlib.sha256((HERE/'development_ensemble_comparison.csv').read_bytes()).hexdigest())
    path=HERE/'frozen_final_selection.json'
    if path.exists():assert json.loads(path.read_text())==frozen,'Never overwrite a frozen selection'
    else:path.write_text(json.dumps(frozen,indent=2))
    provenance=HERE/'freeze_provenance.json'
    if not provenance.exists():provenance.write_text(json.dumps(dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reserved_results_opened=False),indent=2))
    print(json.dumps(frozen,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['screen','freeze']);args=parser.parse_args()
    screen() if args.phase=='screen' else freeze()
