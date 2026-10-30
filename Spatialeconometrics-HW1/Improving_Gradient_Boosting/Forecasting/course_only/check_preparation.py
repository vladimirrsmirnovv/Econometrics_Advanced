"""Check WX algebra, ID invariance and unchanged targets before dispatch."""
from pathlib import Path
import gc,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import sparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from notebook_api import MODELS,load_recipe,matrices
from run_validation import get_fold
from runner import prepare,spatial_x

def main():
    # A manual three-market example with two shops in the first market.
    stores=pd.DataFrame({'node':[0,0,1,2]},index=pd.Index([10,11,12,13],name='new_id'))
    meta=pd.DataFrame({'new_id':[10,11,12,13],'year_month_index':[7]*4,'pto':[99,101,203,304]})
    x=pd.DataFrame({'a':[2.,4.,8.,14.]});w=sparse.csr_matrix([[0,.5,.5],[1,0,0],[0,1,0]])
    result=spatial_x(x,meta,['a'],w,stores)
    np.testing.assert_array_equal(result.iloc[:,0],[11.,11.,3.,8.])
    changed=meta.copy();changed['pto']*=1e6
    pd.testing.assert_frame_equal(result,spatial_x(x,changed,['a'],w,stores),check_exact=True)
    permutation=[2,0,3,1]
    shuffled=spatial_x(x.iloc[permutation],meta.iloc[permutation],['a'],w,stores).sort_index()
    pd.testing.assert_frame_equal(result,shuffled,check_exact=True)
    recipe=load_recipe();compact=get_fold(recipe,2024,11);expected={}
    for model in MODELS:
        base=matrices(recipe,compact,model);expected[model]={}
        for config in ['wx_knn6','wx_knn12']:
            xt,xv,yt,yv,cats,columns,fingerprints=prepare(recipe,compact,model,config)
            pd.testing.assert_frame_equal(xt[base[0].columns],base[0],check_exact=True)
            pd.testing.assert_frame_equal(xv[base[1].columns],base[1],check_exact=True)
            pd.testing.assert_series_equal(yt,base[2],check_exact=True)
            pd.testing.assert_series_equal(yv,base[3],check_exact=True)
            expected[model][config]=dict(columns=columns,WX_fingerprints=fingerprints)
            del xt,xv;gc.collect()
        del base;gc.collect()
    row=dict(status='passed',manual_WX_algebra=True,shop_ID_permutation=True,
        additional_features_ignore_current_target=True,original_X_and_targets_unchanged=True,
        implementation_sha256=hashlib.sha256((HERE/'runner.py').read_bytes()).hexdigest(),
        weights_manifest_sha256=hashlib.sha256((HERE/'cache/weights_manifest.json').read_bytes()).hexdigest(),
        november_expected=expected)
    (HERE/'preparation_checks.json').write_text(json.dumps(row,indent=2));print('COURSE_WX_PREPARATION_PASSED', {m:len(expected[m]['wx_knn6']['columns']) for m in MODELS},flush=True)

if __name__=='__main__':main()
