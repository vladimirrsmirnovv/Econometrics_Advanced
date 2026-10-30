"""Resumable screen followed by predeclared historical confirmation per model."""
from pathlib import Path
import argparse,json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment import SCREEN,execute
from selection import pick_one

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',required=True);args=parser.parse_args()
    for model in args.models:
        for config in SCREEN:execute(model,'2024-08',config)
        configs,scores=pick_one(model)
        path=HERE/('confirm_plan_'+model+'.json')
        plan=dict(model=model,screen='2024-08',confirm_months=['2024-06','2024-07'],configs=configs,scores=scores)
        if path.exists():assert json.loads(path.read_text())==plan
        else:path.write_text(json.dumps(plan,indent=2))
        print('CONFIRM_PLAN',model,configs,flush=True)
        for fold in ['2024-06','2024-07']:
            for config in configs:execute(model,fold,config)
    print('CPU_DEVELOPMENT_DONE',args.models,flush=True)

if __name__=='__main__':main()
