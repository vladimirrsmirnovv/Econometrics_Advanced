"""Next-month bias correction from genuinely earlier forecast errors only."""
from pathlib import Path
import hashlib
import json
import sys
import shutil
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS
from spatial_features import make_weights

PREVIOUS={'2024-03':'2024-02','2025-01':'2024-12','2025-02':'2025-01'}

def prior_signals(previous,stores,locations):
    """No current targets or predictions are accepted by this function."""
    earlier=previous.set_index('new_id').reindex(stores.new_id)
    assert earlier.prediction.notna().all() and (earlier.prediction>0).all()
    error=np.log(earlier.pto.to_numpy()/earlier.prediction.to_numpy()).clip(-.2,.2)
    panel=stores[['new_id','region','node']].copy();panel['error']=error
    market=panel.groupby('node').error.agg(['mean','count']).reindex(locations.node)
    global_mean=float(error.mean())
    shrink=market['count'].to_numpy()/(market['count'].to_numpy()+5)
    market_error=global_mean+shrink*(market['mean'].to_numpy()-global_mean)
    regional=panel.groupby('region').error.agg(['mean','count'])
    regional_error=global_mean+regional['count']/(regional['count']+20)*(regional['mean']-global_mean)
    result={'global':np.full(len(stores),global_mean),'own':error,
        'region':stores.region.map(regional_error).to_numpy()}
    for kind in ['exp50','knn8']:
        weights,_=make_weights(locations,kind)
        result['spatial_'+kind]=(weights@market_error)[stores.node.to_numpy()]
    result['spatial_region_gap']=result['spatial_exp50']-result['region']
    return result

def main():
    stores=pd.read_parquet(HERE/'cache/stores.parquet').sort_values('new_id')
    locations=pd.read_csv(HERE/'geography_diagnostics.csv').sort_values('node')
    records=[];second_stage={}
    for fold,previous_fold in PREVIOUS.items():
        for model in MODELS:
            directory=HERE/'results'/fold/model.replace("_", "-")
            previous_path=HERE/'results'/previous_fold/model.replace("_", "-")/'baseline.csv'
            if not previous_path.exists():
                legacy=HERE.parent/'results'/previous_fold/(model+'.csv')
                if legacy.exists():
                    previous_path.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(legacy,previous_path)
                    info=json.loads(legacy.with_suffix('.json').read_text())
                    info.update(configuration='baseline',backend='CPU',reused_existing=True)
                    previous_path.with_suffix('.json').write_text(json.dumps(info,indent=2))
            current_path=directory/'baseline.csv'
            if not previous_path.exists() or not current_path.exists():
                raise FileNotFoundError(str(previous_path))
            previous=pd.read_csv(previous_path);current=pd.read_csv(current_path).sort_values('new_id')
            assert previous.year_month_index.max()<current.year_month_index.min()
            signals=prior_signals(previous,stores,locations)
            np.testing.assert_array_equal(current.new_id,stores.new_id)
            base_mape=float((abs(current.prediction-current.pto)/current.pto).mean()*100)
            second_stage[(fold,model)]=(current,signals)
            for name,signal in signals.items():
                config='residual_'+name
                pred=current.prediction.to_numpy()*np.exp(.5*signal)
                ape=abs(pred-current.pto.to_numpy())/current.pto.to_numpy()*100
                original_ape=abs(current.prediction-current.pto)/current.pto*100
                error=pred-current.pto.to_numpy()
                row=dict(fold=fold,model=model,configuration=config,mape=float(ape.mean()),
                    mae=float(abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
                    bias_pct=float((error/current.pto.to_numpy()).mean()*100),
                    baseline_mape=base_mape,delta_mape=float(ape.mean()-base_mape),
                    relative_improvement_pct=float((base_mape-ape.mean())/base_mape*100),
                    shops_improved_pct=float((ape<original_ape.to_numpy()).mean()*100),
                    ape_change_quantiles={str(q):float(np.quantile(ape-original_ape,q)) for q in [.05,.25,.5,.75,.95]},
                    previous_oof_fold=previous_fold,correction_weight=.5,
                    clipping_log_error=[-.2,.2],market_shrink_prior_count=5,
                    baseline_file_sha256=hashlib.sha256(current_path.read_bytes()).hexdigest(),
                    previous_oof_file_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                    backend='OOF postprocessing',validation_only=True,leakage_check='passed',
                    current_targets_used_for_correction=False)
                table=current.copy();table['prediction']=pred
                table.to_csv(directory/(config+'.csv'),index=False)
                (directory/(config+'.json')).write_text(json.dumps(row,indent=2))
                records.append(row)
    # A deliberately small learned corrector: earlier OOF pairs train, February tests.
    # Two separated training months are weak coverage, explicitly recorded below.
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    def design(fold,signals):
        month=int(fold[-2:])
        return np.column_stack([signals[name] for name in sorted(signals)]+[
            stores.population_log.to_numpy(),stores.competition_log.to_numpy(),
            np.log1p(stores.market_shops.to_numpy()),np.log1p(stores.nearest_market_km.to_numpy()),
            np.full(len(stores),np.sin(2*np.pi*month/12)),np.full(len(stores),np.cos(2*np.pi*month/12))])
    for model in MODELS:
        xs=[];ys=[]
        for fold in ['2024-03','2025-01']:
            frame,signals=second_stage[(fold,model)]
            xs.append(design(fold,signals));ys.append(np.log(frame.pto.to_numpy()/frame.prediction.to_numpy()).clip(-.2,.2))
        x=np.vstack(xs);y=np.concatenate(ys);scaler=StandardScaler().fit(x)
        estimator=Ridge(alpha=100).fit(scaler.transform(x),y)
        frame,signals=second_stage[('2025-02',model)]
        correction=estimator.predict(scaler.transform(design('2025-02',signals))).clip(-.2,.2)
        pred=frame.prediction.to_numpy()*np.exp(.5*correction)
        error=pred-frame.pto.to_numpy()
        row=dict(fold='2025-02',model=model,configuration='residual_ridge_past_only',
            mape=float((abs(error)/frame.pto.to_numpy()).mean()*100),
            mae=float(abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
            bias_pct=float((error/frame.pto.to_numpy()).mean()*100),
            backend='OOF learned postprocessing',validation_only=True,leakage_check='passed',
            second_stage_training_months=['2024-03','2025-01'],test_month='2025-02',
            ridge_alpha=100,correction_weight=.5,
            limitation='Only two separated training OOF months; February-only candidate, excluded from three-month component ranking')
        directory=HERE/'results/2025-02'/model.replace("_", "-");table=frame.copy();table['prediction']=pred
        table.to_csv(directory/(row['configuration']+'.csv'),index=False)
        (directory/(row['configuration']+'.json')).write_text(json.dumps(row,indent=2));records.append(row)
    (HERE/'residual_checks.json').write_text(json.dumps(dict(status='passed',runs=len(records),
        rule='Previous strictly earlier temporal OOF only; fixed 0.5 shrinkage; current outcomes inaccessible to prior_signals',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(pd.DataFrame(records).pivot_table(index='configuration',columns='model',values='delta_mape').to_string())

if __name__=='__main__':main()
