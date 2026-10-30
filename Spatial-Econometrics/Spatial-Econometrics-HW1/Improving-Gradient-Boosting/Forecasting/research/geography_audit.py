"""Coordinate consistency and sensitivity diagnostics; no source-coordinate edits."""
from __future__ import annotations
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from notebook_api import ROOT,load_recipe
from spatial_features import attach_locations

def contains(points,ring):
    ring=np.asarray(ring,dtype=float)
    result=np.zeros(len(points),dtype=bool)
    x,y=points[:,0],points[:,1]
    for first,last in zip(ring,np.roll(ring,1,axis=0)):
        a,b=first; c,d=last
        if b==d:continue
        result ^= ((b>y)!=(d>y)) & (x < (c-a)*(y-b)/(d-b)+a)
    return result

def main():
    recipe=load_recipe()
    wanted=[c for c in recipe.rename_cols if recipe.rename_cols[c] in {'new_id','region','settlement'}]
    raw=pd.read_csv(resource_path('Data/Shops/train_2.csv'),usecols=wanted).rename(columns=recipe.rename_cols)
    stores,locations,info=attach_locations(raw)
    points=locations[['settlement_lon','settlement_lat']].to_numpy()
    distances,_=BallTree(np.radians(points[:,::-1]),metric='haversine').query(np.radians(points[:,::-1]),k=2)
    nearest=distances[:,1]*6371
    geometry=json.loads((resource_path('Data/Maps/russia_regions_simplified.geojson')).read_text(encoding='utf-8'))
    observed=np.full(len(points),'',dtype=object)
    for feature in geometry['features']:
        g=feature['geometry'];polygons=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
        for rings in polygons:
            outer=np.asarray(rings[0])
            mask=(points[:,0]>=outer[:,0].min())&(points[:,0]<=outer[:,0].max())&(
                points[:,1]>=outer[:,1].min())&(points[:,1]<=outer[:,1].max())
            if not mask.any():continue
            selected=np.flatnonzero(mask);inside=contains(points[mask],rings[0])
            for hole in rings[1:]:inside&=~contains(points[mask],hole)
            observed[selected[inside]]=feature['properties']['shapeName']
    locations['mapped_polygon']=observed
    locations['nearest_market_km']=nearest
    locations['coordinate_alias']=locations.duplicated(['settlement_lat','settlement_lon'],keep=False)
    majority={}
    for region,group in locations.groupby('region'):
        names=group.loc[group.mapped_polygon!='','mapped_polygon']
        majority[region]=names.mode().iloc[0] if len(names) else ''
    locations['dominant_polygon_for_region']=locations.region.map(majority)
    locations['polygon_consistency_outlier']=(locations.mapped_polygon!='')&(
        locations.mapped_polygon!=locations.dominant_polygon_for_region)
    locations.to_csv(HERE/'geography_diagnostics.csv',index=False)
    result=dict(**info,nearest_distance_km_quantiles={str(q):float(np.quantile(nearest,q)) for q in [0,.01,.1,.5,.9,.99,1]},
        coordinate_alias_markets=int(locations.coordinate_alias.sum()),
        uncovered_markets=int((observed=='').sum()),polygon_consistency_outliers=int(locations.polygon_consistency_outlier.sum()),
        limitations=['Simplified 83-region map is a consistency diagnostic, not independent geocoding.',
            'Dominant polygons inferred by region label; not an official Russian-English name crosswalk.',
            'Border points and territories absent from the map are not automatically coordinate errors.',
            'Same-centroid aliases may represent genuine co-located markets; sensitivity only.',
            'Shop-level distances are unavailable; all geometry is settlement-level.'],
        source_coordinates_changed=False,geometry_source='Data/Maps/russia_regions_simplified.geojson')
    (HERE/'geography_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
