"""Import checked fits, freeze development choice, and compare fixed course WX."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,zipfile
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,load_recipe,NOTEBOOK_SHA256
from run_validation import original_blends
from runner import COUNTS
TAGS=['cat','ld','ll','xgb'];DEVELOPMENT=['2024-08','2024-11','2024-12'];CONFIRM=['2025-01','2025-02']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mape(y,p):return float(np.mean(abs(np.asarray(p)-np.asarray(y))/np.asarray(y))*100)
def write_json(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def fit_path(fold,model,config):return HERE/'results'/fold/model/(config+'.csv')
def load(fold,model,config):
    p=fit_path(fold,model,config);info=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    assert sha(p)==info['prediction_sha256']
    f=pd.read_csv(p).sort_values('new_id');assert len(f)==18657 and f.new_id.is_unique
    assert np.isfinite(f[['pto','prediction']]).all().all() and (f[['pto','prediction']]>0).all().all()
    assert info['trees']==COUNTS[model] and info['notebook_sha256']==NOTEBOOK_SHA256
    assert abs(mape(f.pto,f.prediction)-info['mape'])<1e-10
    return f,info
def reuse_baselines():
    rows=[]
    for fold in ['2024-08','2024-12']:
        for model in MODELS:
            old=HERE.parent/'research_stage2/results/full'/fold/model/'baseline.csv'
            info=json.loads(old.with_suffix('.json').read_text());assert sha(old)==info['prediction_sha256']
            assert info['protocol']=='full' and info['trees']==COUNTS[model] and info['configuration']=='baseline'
            native=HERE.parent/'research_stage2'/info['model_file'];assert sha(native)==info['model_sha256']
            target=fit_path(fold,model,'baseline');target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==old.read_bytes()
            else:target.write_bytes(old.read_bytes())
            copied=dict(info,reused_full_original_fit=True,source_metadata_sha256=sha(old.with_suffix('.json')),
                model_file='../research_stage2/'+info['model_file'],early_stopping=False)
            write_json(target.with_suffix('.json'),copied);frame,checked=load(fold,model,'baseline')
            rows.append(dict(fold=fold,model=model,mape=checked['mape'],csv_sha256=sha(target),source=str(old)))
    write_json(HERE/'baseline_reuse_checks.json',dict(status='passed',fits=rows,original_files_modified=False))
    print('REUSED_ORIGINAL_FULL_BASELINES',len(rows),flush=True)
def import_run(tag,phase,local=False):
    active=HERE/(('local_'+tag) if local else ('kaggle_'+tag));output=active/'output';archive=output/(phase+'_'+tag+'.zip')
    check=json.loads((output/(phase+'_'+tag+'_checks.json')).read_text())
    model=MODELS[TAGS.index(tag)];assert check['status']=='passed' and check['model']==model and check['phase']==phase
    dispatch=json.loads((HERE/'sources'/(phase+'_'+tag)/'dispatch_manifest.json').read_text())
    assert check['runner_sha256']==dispatch['runner_sha256']==sha(HERE/'runner.py')
    assert check['original_training_counts_preserved'] and check['private_existing_datasets_only']
    if phase=='development':assert check['baseline_parity']['max_relative_difference']<1e-12
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.startswith('march/'):
                root=HERE.parent/'submissions/2025-03-course';relative=name.removeprefix('march/')
            else:root=HERE;relative=name
            target=(root/relative).resolve();assert target.is_relative_to(root.resolve())
            data=z.read(name)
            if target.exists():assert target.read_bytes()==data,name
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    fits=[('2024-11','baseline')]+[(f,c) for f in DEVELOPMENT for c in ['wx_knn6','wx_knn12']] if phase=='development' else [(f,c) for f in CONFIRM for c in ['baseline',check['selected']]]
    verified=[]
    for fold,config in fits:
        frame,meta=load(fold,model,config);native=HERE/meta['model_file'];assert sha(native)==meta['model_sha256']
        assert meta['early_stopping'] is False and not meta['outer_labels_used_for_fit']
        assert meta['source_code_sha256']==dispatch['runner_sha256']
        assert meta['environment']['numpy']=='2.3.5' and meta['environment']['pandas']=='2.2.3'
        if config!='baseline':assert meta['source_regressors_for_WX'] and meta['WX_fingerprints']
        verified.append(dict(fold=fold,model=model,configuration=config,mape=meta['mape'],trees=meta['trees']))
    if phase=='confirmation':
        out=HERE.parent/'submissions/2025-03-course';p=out/'experimental'/('test_'+model+'.csv');meta=json.loads(p.with_suffix('.json').read_text())
        assert sha(p)==meta['submission_sha256'] and sha(out/meta['model_file'])==meta['model_sha256']
        assert meta['trees']==COUNTS[model] and meta['configuration']==check['selected'] and meta['native_reload_max_relative_difference']<1e-12
    result=dict(status='passed',archive_sha256=sha(archive),tag=tag,phase=phase,checks=check,verified_fits=verified,
        execution_location=check.get('execution_location','Kaggle'),archive_file=archive.relative_to(HERE).as_posix())
    write_json(HERE/'imports'/(phase+'_'+tag+'.json'),result)
    print('COURSE_IMPORTED',phase,tag,json.dumps(verified),flush=True)
def select(model):
    rows=[]
    for config in ['baseline','wx_knn6','wx_knn12']:
        part=[];sources={}
        for fold in DEVELOPMENT:
            frame,info=load(fold,model,config);part.append(info['mape']);sources[fold]=info['prediction_sha256']
        rows.append(dict(model=model,configuration=config,mean_mape=float(np.mean(part)),
            months=dict(zip(DEVELOPMENT,part)),sources=sources))
    best_wx=min(rows[1:],key=lambda r:(r['mean_mape'],r['configuration']))
    choice=dict(model=model,selected_wx=best_wx['configuration'],development_winner=min(rows,key=lambda r:(r['mean_mape'],r['configuration']))['configuration'],
        development_gain=rows[0]['mean_mape']-best_wx['mean_mape'],development_eligible=best_wx['mean_mape']<rows[0]['mean_mape'],
        rule='Lower development mean than baseline AND lower MAPE on each January and February confirmation; else baseline',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),runner_sha256=sha(HERE/'runner.py'),plan_sha256=sha(HERE/'PLAN.md'),
        confirmation_metrics_used_for_selection=False,candidates=rows)
    target=HERE/'selection'/(model+'.json')
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        assert previous['selected_wx']==choice['selected_wx'] and previous['candidates']==rows
        choice=previous
    else:
        assert not any(fit_path(f,model,best_wx['configuration']).exists() for f in CONFIRM),'Choice must precede confirmation'
        write_json(target,choice)
    print('FROZEN_COURSE_WX',model,choice['selected_wx'],'development gain',choice['development_gain'],flush=True)
    return choice
def inspect():
    rows=[]
    for p in sorted((HERE/'results').glob('*/*/*.json')):
        info=json.loads(p.read_text(encoding='utf-8'));f,_=load(info['fold'],info['model'],info['configuration'])
        rows.append({k:info[k] for k in ['model','fold','configuration','trees','mape']})
    pd.DataFrame(rows).to_csv(HERE/'current_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['baselines','import','select','inspect'])
    p.add_argument('--tag',choices=TAGS);p.add_argument('--phase',choices=['development','confirmation']);p.add_argument('--model',choices=MODELS)
    p.add_argument('--local',action='store_true')
    a=p.parse_args()
    if a.action=='baselines':reuse_baselines()
    elif a.action=='import':assert a.tag and a.phase;import_run(a.tag,a.phase,a.local)
    elif a.action=='select':assert a.model;select(a.model)
    else:inspect()
if __name__=='__main__':main()
