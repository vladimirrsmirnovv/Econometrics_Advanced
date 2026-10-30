"""Sequential follow-ups after the screen; resume individual saved fits."""
from pathlib import Path
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
env=dict(os.environ,OPENBLAS_NUM_THREADS='4',PYTHONUTF8='1')

def run(script,args):
    print('DISPATCH',script,args,flush=True)
    subprocess.run([sys.executable,str(script),*args],env=env,check=True)

def main():
    while not (HERE/'results/2025-02/catboost-log-level/density_weight.json').exists():time.sleep(15)
    for fold in ['2024-02','2024-12']:
        year,month=fold.split('-')
        run(PARENT/'run_validation.py',['--year',year,'--month',month,'--models','catboost_log_level'])
    run(HERE/'run_models.py',['--models','catboost_log_level','lightgbm_log_diff','lightgbm_log_level',
        '--folds','2024-02','2024-12','--configs','baseline'])
    # Key-only is a distinct technical intervention, paired with the spatial history.
    run(HERE/'run_models.py',['--models','catboost_log_level','--folds','2024-03','2025-01','2025-02',
        '--configs','key','key+exp50_history'])
    run(HERE/'run_models.py',['--models','catboost_log_level','--folds','2025-02',
        '--configs','exp50_history+wx_relative','random_history','geo_sensitivity'])
    # One additional representative of a separate mechanism, chosen before its old folds are examined.
    screen=HERE/'results/2025-02/catboost-log-level'
    candidates=['lisa','eigenvectors','hybrid_slx','hybrid_gwr','within_market','season_exp50']
    configs=sorted(candidates,key=lambda c:json.loads((screen/(c+'.json')).read_text())['mape'])[:2]
    (HERE/'cat_secondary_selection.json').write_text(json.dumps(dict(screen_month='2025-02',
        candidates=candidates,selected=configs,rule='Two strongest distinct remaining mechanisms by screen MAPE'),indent=2))
    run(HERE/'run_models.py',['--models','catboost_log_level','--folds','2024-03','2025-01','--configs',*configs])
    print('CPU_FOLLOWUPS_COMPLETE',flush=True)

if __name__=='__main__':main()
