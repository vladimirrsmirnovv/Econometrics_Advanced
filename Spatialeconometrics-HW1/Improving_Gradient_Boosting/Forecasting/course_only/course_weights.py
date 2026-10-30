"""Seminar 4 fixed symmetric kNN6/12 weights in its specified projection."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent
PROJECTION='+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=105 +x_0=18500000 +y_0=0 +ellps=krass +units=m +towgs84=28,-130,-95,0,0,0,0 +no_defs'

def main():
    from pyproj import Transformer
    root=REPOSITORY_ROOT;source=root/'Source/Seminars/Seminar4/municipalities_models.R'
    text=source.read_text(encoding='utf-8');assert PROJECTION in text
    assert 'make.sym.nb(knn2nb(knearneigh(coords, k = 6)))' in text
    assert 'make.sym.nb(knn2nb(knearneigh(coords, k = 12)))' in text
    loc=pd.read_csv(HERE.parent/'research/geography_diagnostics.csv').sort_values('node').reset_index(drop=True)
    np.testing.assert_array_equal(loc.node,np.arange(len(loc)))
    trans=Transformer.from_crs('EPSG:4326',PROJECTION,always_xy=True)
    x,y=trans.transform(loc.settlement_lon.to_numpy(),loc.settlement_lat.to_numpy())
    coords=np.column_stack([x,y]);assert np.isfinite(coords).all()
    distances=cdist(coords,coords);np.fill_diagonal(distances,np.inf)
    # A stable full sort specifies ties consistently on every host.
    near=np.argsort(distances,axis=1,kind='stable')[:,:12]
    folder=HERE/'cache';folder.mkdir(exist_ok=True);rows=[]
    for k in [6,12]:
        a=sparse.csr_matrix((np.ones(len(loc)*k),(np.repeat(np.arange(len(loc)),k),near[:,:k].ravel())),shape=(len(loc),len(loc)))
        a=a.maximum(a.T);a.setdiag(0);a.eliminate_zeros()
        w=(sparse.diags(1/np.asarray(a.sum(axis=1)).ravel())@a).tocsr()
        assert np.all(w.diagonal()==0);np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(),1,atol=1e-12)
        path=folder/('knn'+str(k)+'.npz');sparse.save_npz(path,w)
        rows.append(dict(k=k,edges=w.nnz,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            graph_symmetrized_before_row_normalization=True,zero_diagonal=True,row_normalized=True))
    loc[['node','settlement_lat','settlement_lon']].assign(x_m=x,y_m=y).to_parquet(folder/'projected_coordinates.parquet',index=False)
    stores=pd.read_parquet(HERE.parent/'research/cache/stores.parquet',columns=['new_id','node'])
    stores.to_parquet(folder/'stores.parquet',index=False)
    manifest=dict(source=source.relative_to(root).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        projection=PROJECTION,nodes=len(loc),shops=len(stores),weights=rows,
        tie_rule='Ascending node for exactly equal projected Euclidean distance')
    (folder/'weights_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
