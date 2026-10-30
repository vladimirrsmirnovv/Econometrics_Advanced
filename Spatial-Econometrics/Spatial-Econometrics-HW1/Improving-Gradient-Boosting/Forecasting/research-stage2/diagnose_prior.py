"""Read-only diagnostics of first-stage choices and the reported leaderboard."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,load_recipe
from run_validation import original_blends

def main():
    old=HERE.parent/'research'
    selected=json.loads((old/'best_model_configs.json').read_text())['components']
    records=[];blend_rows=[];trees=[]
    for fold in ['2024-03','2024-10','2025-01','2025-02']:
        base={};modified={};truth=None
        for model in MODELS:
            directory=old/'results'/fold/model.replace("_", "-")
            before=pd.read_csv(directory/'baseline.csv').sort_values('new_id')
            after=pd.read_csv(directory/(selected[model]+'.csv')).sort_values('new_id')
            np.testing.assert_array_equal(before.new_id,after.new_id)
            if truth is None:truth=before.pto.to_numpy()
            else:np.testing.assert_allclose(truth,before.pto)
            bm=float((abs(before.prediction-before.pto)/before.pto).mean()*100)
            am=float((abs(after.prediction-after.pto)/after.pto).mean()*100)
            records.append(dict(fold=fold,model=model,configuration=selected[model],baseline_mape=bm,mape=am,delta_mape=am-bm))
            base[model]=before.prediction.to_numpy();modified[model]=after.prediction.to_numpy()
            if fold=='2025-02':
                prior=json.loads((directory/'baseline.json').read_text())
                march=json.loads((HERE.parent/'submissions/2025-03/baseline'/('test_'+model+'.json')).read_text())
                validation_trees=int(prior['best_iteration'])+(model in ['catboost_log_level','xgboost_log_diff'])
                trees.append(dict(model=model,february_validation_trees=validation_trees,march_full_trees=march['trees'],ratio=march['trees']/validation_trees))
        mixed=dict(base);mixed['xgboost_log_diff']=modified['xgboost_log_diff']
        for name,components in [('baseline',base),('previous',modified),('mandatory_mixed',mixed)]:
            for rule,prediction in original_blends(components).items():
                blend_rows.append(dict(fold=fold,composition=name,rule=rule,mape=float((abs(prediction-truth)/truth).mean()*100)))
    pd.DataFrame(records).to_csv(HERE/'prior_month_effects.csv',index=False)
    pd.DataFrame(blend_rows).to_csv(HERE/'prior_mixed_ensembles.csv',index=False)
    pd.DataFrame(trees).to_csv(HERE/'training_protocol_difference.csv',index=False)
    recipe=load_recipe();anchors=[c for c in recipe.catboost_log_level_features if 'feb_mar_growth_2024' in c or 'region_march_growth' in c]
    availability={}
    for fold,name in [('2024-03','holdout'),('2025-02','holdout'),('final_2025-03','test_features')]:
        frame=pd.read_parquet(HERE.parent/'cache'/fold/(name+'.parquet'))
        availability[fold]={c:float(frame[c].isna().mean()) for c in anchors if c in frame}
    (HERE/'march_anchor_availability.json').write_text(json.dumps(dict(na_fraction=availability,
        reason='March 2024 was not known before its own forecast; it is known before March 2025',
        original_features_unchanged=True),indent=2))
    board=pd.DataFrame(dict(model=MODELS,before=[90.55,90.32,90.57,90.45],after=[90.52,90.31,90.46,90.46]))
    board['delta_score']=board.after-board.before
    for key in ['before','after']:
        board[key+'_mape_approx']=100*(1-np.sqrt(board[key]/100))
        board[key+'_mape_lower']=100*(1-np.sqrt((board[key]+.005)/100))
        board[key+'_mape_upper']=100*(1-np.sqrt((board[key]-.005)/100))
    board['delta_mape_approx']=board.after_mape_approx-board.before_mape_approx
    board.to_csv(HERE/'march_leaderboard_reported.csv',index=False)
    (HERE/'leaderboard_provenance.json').write_text(json.dumps(dict(source='User screenshots and explicit before/after ordering, 2026-10-09',
        individual_ids_before=[167812607,167812856,167812879,167812903],
        individual_ids_after=[167812841,167812869,167812891,167812913],
        ensemble_scores=dict(original=90.78,previous_equal=90.78,previous_gated_1_08_all=90.79),
        actual_march_outcomes_available=False,used_for_new_selection=False,score_to_mape_conversion=True,
        metric_source='Full competition conditions provided directly by user, 2026-10-09',
        metric='Unweighted store MAPE in percent',
        score_formula='100 * ((100 - min(MAPE, 100)) / 100) ** 2',
        inverse_for_positive_score='100 * (1 - sqrt(score / 100))',
        scores_rounded_decimals=2,current_leaderboard_subset_unknown=True,
        competition_note='During competition scores cover a subset; final score covers all stores. Current displayed subset status is unknown.',
        example_typo='The provided example 80% MAPE -> 20 points conflicts with the formula; the formula gives 4 points.'),indent=2))
    print(pd.DataFrame(trees).to_string(index=False))
    print(pd.DataFrame(blend_rows).groupby(['composition','rule']).mape.mean().round(6).to_string())

if __name__=='__main__':main()
