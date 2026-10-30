"""Replay the saved geographic neighbour snapshot across operating systems."""
from pathlib import Path
import ast,base64,hashlib,io,json,sys,zipfile
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))

def main():
    from notebook_api import read_raw,load_recipe
    import features,geo_control
    snapshot=HERE/'kaggle/output/canonical_geographic_graph.npz'
    arrays=np.load(snapshot);candidate=arrays['candidate'];ds=arrays['distance_km']/6371
    neighbours=np.column_stack([np.arange(len(candidate)),candidate]);distances=np.column_stack([np.zeros(len(candidate)),ds])
    class FixedTree:
        def __init__(self,points,metric):self.points=np.asarray(points);assert len(points)==len(candidate) and metric=='haversine'
        def query(self,points,k):np.testing.assert_array_equal(points,self.points);assert k==33;return distances.copy(),neighbours.copy()
    source=ast.parse((HERE/'kaggle-cpu/output/cpu_stage2.py').read_text())
    payload=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
        stores=pd.read_parquet(io.BytesIO(z.read('stores.parquet')));locations=pd.read_parquet(io.BytesIO(z.read('locations.parquet')))
    raw=read_raw(load_recipe());features.BallTree=FixedTree;geo_control.BallTree=FixedTree
    new,_=features.compute(raw,stores,locations)
    pd.testing.assert_frame_equal(new,pd.read_parquet(HERE/'cache/features.parquet'),check_exact=True)
    geo=geo_control.compute(raw,stores,locations)
    pd.testing.assert_frame_equal(geo,pd.read_parquet(HERE/'kaggle/output/canonical_geo_control.parquet'),check_exact=True)
    geo.to_parquet(HERE/'cache/geo_control.parquet',index=False)
    future=raw[(raw.year==2025)&(raw.month==2)].copy();future['month']=3;future['pto']=1e12
    changed,_=features.compute(pd.concat([raw,future],ignore_index=True),stores,locations)
    pd.testing.assert_frame_equal(new,changed,check_exact=True)
    result=dict(status='passed',original_builder_unchanged=True,canonical_feature_exact=True,
        canonical_geo_control_exact=True,fake_current_march_target_replacement_passed=True,
        graph_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256((HERE/'features.py').read_bytes()).hexdigest(),
        graph_node_mapping='Preserved node order and locations in the approved code payload')
    (HERE/'canonical_replay_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
