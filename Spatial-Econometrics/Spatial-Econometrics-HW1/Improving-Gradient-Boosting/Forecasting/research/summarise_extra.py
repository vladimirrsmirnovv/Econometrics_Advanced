"""Extra-month diagnostics do not change the primary component selection."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS,load_recipe
from run_validation import original_blends

def main():
    frozen=json.loads((HERE/'frozen_selection_before_extra_month.json').read_text())
    choices=frozen['components'];recipe=load_recipe();rows=[];base_vectors={};best_vectors={};reference=None
    for model in MODELS:
        directory=HERE/'results/2024-10'/model.replace("_", "-")
        base=pd.read_csv(directory/'baseline.csv').sort_values('new_id')
        selected=pd.read_csv(directory/(choices[model]+'.csv')).sort_values('new_id')
        np.testing.assert_array_equal(base.new_id,selected.new_id)
        np.testing.assert_allclose(base.pto,selected.pto,rtol=1e-14)
        assert len(base)==18657 and base.new_id.is_unique
        if reference is None:reference=base
        np.testing.assert_array_equal(base.new_id,reference.new_id)
        np.testing.assert_allclose(base.pto,reference.pto,rtol=1e-14)
        true=base.pto.to_numpy();old=recipe._make_metrics(true,base.prediction)['mape']
        metrics=recipe._make_metrics(true,selected.prediction)
        rows.append(dict(fold='2024-10',model=model,configuration=choices[model],baseline_mape=old,
            mape=metrics['mape'],delta_mape=metrics['mape']-old,selection_frozen_before_local_extra_fit=True))
        base_vectors[model]=base.prediction.to_numpy();best_vectors[model]=selected.prediction.to_numpy()
    baseline_blends=original_blends(base_vectors)
    for label,vectors in [('baseline_ensemble',base_vectors),('selected_ensemble',best_vectors)]:
        for rule,pred in original_blends(vectors).items():
            metrics=recipe._make_metrics(true,pred)
            old=recipe._make_metrics(true,baseline_blends[rule])['mape']
            configuration=label if rule=='final_meta_logmean' else label+'__'+rule
            metrics.update(fold='2024-10',model='ensemble',configuration=configuration,
                baseline_mape=old,delta_mape=metrics['mape']-old,original_blend_rule=rule,
                components_frozen_before_local_extra_fit=True,
                rule_selection_months='2024-03,2025-01,2025-02')
            rows.append(metrics);table=reference.copy();table['prediction']=pred
            table.to_csv(HERE/'ensemble'/('2024-10_'+configuration+'.csv'),index=False)
    pd.DataFrame(rows).to_csv(HERE/'extra_month_comparison.csv',index=False,encoding='utf-8-sig')
    (HERE/'extra_month_checks.json').write_text(json.dumps(dict(status='passed',month='2024-10',
        selection_months=['2024-03','2025-01','2025-02'],selection_unchanged=True,
        limitation='XGBoost extra month was available before the last CPU validations, but did not enter the automatic three-month ranking'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
