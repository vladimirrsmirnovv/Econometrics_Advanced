"""Read-only audit; only this Audit directory and repository tmp/audit are written.

Run from the repository root with Python >=3.12 and pandas, numpy, openpyxl,
pyarrow installed. No models or research inputs are replaced.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if (ROOT / "tmp/audit/deps").exists():
    sys.path.insert(0, str(ROOT / "tmp/audit/deps"))

import numpy as np
import pandas as pd

from src import paths, panel, population, regions, settlements, shops_clean, geography
from src import rosstat_inflation, rosstat_population, rosstat_unemployment, rosstat_wages

OUT = Path(__file__).resolve().parent
TEMP = ROOT / "tmp/audit/checks"
TEMP.mkdir(parents=True, exist_ok=True)
RESULT = {"checks": [], "limitations": [], "evidence": {}}


def record(name, operation):
    try:
        details = operation()
        RESULT["checks"].append({"name": name, "status": "passed", "details": details})
    except Exception as exc:
        RESULT["checks"].append({"name": name, "status": "failed", "error": str(exc)})


def compare_frame(actual, expected, key):
    actual = actual.sort_values(key).reset_index(drop=True)
    expected = expected.sort_values(key).reset_index(drop=True)
    assert len(actual) == len(expected), (len(actual), len(expected))
    for column in expected:
        left, right = actual[column], expected[column]
        np.testing.assert_array_equal(left.isna(), right.isna(), err_msg=column)
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            np.testing.assert_allclose(left.to_numpy(float, na_value=np.nan),
                                       right.to_numpy(float, na_value=np.nan),
                                       equal_nan=True, rtol=1e-10, atol=1e-8, err_msg=column)
        else:
            np.testing.assert_array_equal(left[left.notna()].astype(str),
                                          right[right.notna()].astype(str), err_msg=column)
    return {"rows": len(actual), "columns": len(expected.columns)}


def compare_csv(actual, path, key, codes):
    return compare_frame(actual, pd.read_csv(path, dtype={c: str for c in codes}), key)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8").split("\0")
originals = {p: sha(ROOT / p) for p in tracked if p and (ROOT / p).is_file()
             and not p.startswith("Audit/")}
saved = pd.read_parquet(paths.PANEL)
points = saved.drop_duplicates("settlement_id").sort_values("settlement_id")
record("Existing panel assertions", lambda: panel.check_panel(saved))
RESULT["evidence"]["panel"] = {
    "rows": len(saved), "columns": len(saved.columns), "settlements": len(points),
    "regions": points.region_okato_code.nunique(), "stores": int(points.store_count.sum()),
    "months": saved.period.nunique(), "start": str(saved.period.min()),
    "end": str(saved.period.max()), "population_sources": points.population_source.value_counts().to_dict(),
    "missing_counts": {c: int(n) for c, n in saved.isna().sum().items() if n},
}

record("Inflation XLSX -> stored CSV", lambda: compare_csv(
    rosstat_inflation.build_consumer_price_index(), paths.INFLATION_CSV,
    ["okato_code", "year", "month"], ["okato_code"]))
record("Wages XLSX -> stored CSV", lambda: compare_csv(
    rosstat_wages.build_wages(), paths.WAGES_CSV,
    ["okato_code", "year", "month"], ["okato_code"]))
record("Monthly unemployment XLSX -> stored CSV", lambda: compare_csv(
    rosstat_unemployment.build_monthly_unemployment(), paths.UNEMPLOYMENT_CSV,
    ["okato_code", "year", "month"], ["okato_code"]))
record("Annual unemployment XLSX -> stored CSV", lambda: compare_csv(
    rosstat_unemployment.build_annual_unemployment(), paths.UNEMPLOYMENT_ANNUAL_CSV,
    ["okato_code", "year"], ["okato_code"]))
record("Population XLSX -> stored CSV", lambda: compare_csv(
    rosstat_population.build_population(), paths.POPULATION_CSV,
    ["oktmo_code"], ["oktmo_code", "region_okato_code"]))

shop_regions = regions.build_shop_regions()
shop_regions.to_csv(TEMP / "regions.csv", index=False)
paths.REGIONS = TEMP / "regions.csv"  # Only this audit process; original code is untouched.
shops = shops_clean.read_shops()
record("Raw store panel assertions", lambda: shops_clean.check_panel(shops))
stores = shops_clean.add_settlement_keys(shops_clean.build_stores(shops))
store_months = shops[["store_id", "year", "month", *shops_clean.MONTHLY_COLUMNS]]
stores = stores.merge(shops_clean.flag_revenue_jumps(store_months), on="store_id")
stores = settlements.add_share_indicators(
    stores[~stores.settlement_type_class.isin(settlements.NOT_A_SETTLEMENT)])
places = settlements.assign_ids(settlements.aggregate_settlements(stores))
stores = stores.merge(places[[*settlements.KEY_COLUMNS, "settlement_key"]],
                      on=settlements.KEY_COLUMNS, validate="many_to_one")
monthly = settlements.aggregate_months(store_months, stores)
selected = places[places.is_selected]
static_columns = [c for c in selected if c in points]
record("Raw stores -> settlement IDs and static features", lambda: compare_frame(
    selected[static_columns], points[static_columns], ["settlement_id"]))
nominal_columns = [c for c in monthly if c in saved]
record("Raw revenue -> stored monthly settlement aggregates", lambda: compare_frame(
    monthly[monthly.settlement_key.isin(points.settlement_key)][nominal_columns],
    saved[nominal_columns], ["settlement_key", "year", "month"]))
bulletin = population.load_bulletin()
matched = population.apply_manual_links(population.match_automatically(places, bulletin))
matched["population_rosstat_2024"] = matched.oktmo_code.map(
    bulletin.set_index("oktmo_code").population_total).astype("Int64")
pop_columns = ["settlement_key", "oktmo_code", "population_rosstat_2024"]
record("Population matching and manual links -> stored panel", lambda: compare_frame(
    matched[matched.is_selected][pop_columns], points[pop_columns], ["settlement_key"]))

regional = panel.load_regional_months()
reg_cols = [c for c in regional if c in saved]
reg_check = saved[reg_cols].drop_duplicates().sort_values(panel.MONTH_KEY)
record("Regional indices and real wages -> stored panel", lambda: compare_frame(
    regional[regional.region_okato_code.isin(points.region_okato_code)][reg_cols],
    reg_check, panel.MONTH_KEY))


def check_transforms():
    np.testing.assert_allclose(saved.revenue_per_store_real_rub,
                               saved.revenue_per_store_rub / saved.food_price_index, rtol=1e-12)
    np.testing.assert_allclose(saved.log_revenue_per_store_real,
                               np.log(saved.revenue_per_store_real_rub), rtol=1e-12)
    np.testing.assert_allclose(saved.revenue_total_rub,
                               saved.revenue_per_store_rub * saved.store_count, rtol=1e-12)
    assert saved.loc[saved.time_index == 1, "food_price_index"].eq(1).all()
    cols = ["population", "households_median", "foot_traffic_per_hour_median",
            "car_traffic_per_hour_median", "checkouts_mean", "share_large_area",
            "share_new_stores", "grocery_competitors_500m_mean", "pyaterochka_stores_500m_mean",
            "pickup_points_100m_mean", "medical_and_pharmacies_300m_mean"]
    assert saved[cols].notna().all().all()
    assert saved[cols[:5]].gt(0).all().all()
    assert saved.groupby("settlement_id")[cols].nunique().eq(1).all().all()
    return {"deflator_base": "2023-01", "all_model_inputs_complete": True}


record("Deflation, logarithms, aggregation identities, model inputs", check_transforms)

# Independent spherical-distance calculation. R spdep uses its own longlat distance
# routine, so this is a numerical cross-check, not an exact replacement of the R W.
lat = np.radians(points.latitude.to_numpy(float))
lon = np.radians(points.longitude.to_numpy(float))
a = np.sin((lat[:, None] - lat[None, :]) / 2)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2)**2
dist = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
np.fill_diagonal(dist, np.inf)
nearest = np.argmin(dist, axis=1)
connected = dist <= 300
connected[np.arange(len(points)), nearest] = True
W = np.where(connected, np.exp(-dist / 50), 0)
W /= W.sum(axis=1, keepdims=True)


def check_weights():
    np.testing.assert_allclose(W.sum(axis=1), 1)
    assert np.all(np.diag(W) == 0) and np.all(W >= 0) and np.isfinite(W).all()
    y = saved.groupby("settlement_id", sort=True).log_revenue_per_store_real.mean().to_numpy()
    z = y-y.mean()
    I = float(z @ W @ z / (z @ z))
    assert abs(I - 0.58161509) < 0.001
    return {"moran_independent_spherical": I, "moran_existing_R_log": 0.58161509,
            "mean_neighbours": float(connected.sum(axis=1).mean()),
            "isolates_within_100km": int((dist <= 100).sum(axis=1).astype(bool).size - np.count_nonzero((dist <= 100).sum(axis=1))),
            "id_order": "ascending settlement_id", "distance_method": "haversine, radius 6371 km"}


record("Independent W invariants and global Moran numerical cross-check", check_weights)

# Test whether two-way FE and region x month remove the observed spatial pattern.
Y = saved.pivot(index="settlement_id", columns="time_index", values="log_revenue_per_store_real").to_numpy()
X = np.stack([np.log(saved.pivot(index="settlement_id", columns="time_index", values="real_wage_rub").to_numpy()),
              saved.pivot(index="settlement_id", columns="time_index", values="unemployment_rate_percent").to_numpy()],axis=2)
def within(v):
    return v-v.mean(axis=0,keepdims=True)-v.mean(axis=1,keepdims=True)+v.mean(axis=(0,1),keepdims=True)
YW, XW = within(Y), within(X)
coef = np.linalg.lstsq(XW.reshape(-1, 2), YW.ravel(), rcond=None)[0]
fe_residual = YW - XW @ coef
region_residual = Y-Y.mean(axis=1,keepdims=True)
for code in points.region_okato_code.unique():
    mask = points.region_okato_code.eq(code).to_numpy()
    region_residual[mask] -= region_residual[mask].mean(axis=0,keepdims=True)
def monthly_moran(residual):
    z=residual-residual.mean(axis=0,keepdims=True)
    return ((z*(W@z)).sum(axis=0)/(z*z).sum(axis=0))
RESULT["evidence"]["independent_panel_residuals"] = {
    "twoways_FE_coefficients": coef.tolist(),
    "FE_mean_Moran": float(monthly_moran(fe_residual).mean()),
    "region_month_mean_Moran": float(monthly_moran(region_residual).mean()),
    "region_month_range": [float(monthly_moran(region_residual).min()),float(monthly_moran(region_residual).max())],
    "note": "Independent numpy calculation; spherical W, not a rerun of spatialreg/splm."}

RESULT["evidence"]["unavailable_inputs"] = {
    "Data/Interim_directory": (ROOT/"Data/Interim").exists(),
    "GeoNames_diagnostics": paths.GEONAMES_DIAGNOSTICS.exists(),
    "Source/Qualifying/train.csv": (ROOT/"Source/Qualifying/train.csv").exists(),
    "Utils_directory": (ROOT/"Utils").exists(),
    "Proposal/README.md": (ROOT/"Proposal/README.md").exists(),
    "Rules_directory": (ROOT/"Rules").exists(),
}
for script in ["Parsing/geonames_settlement_coordinates/build_settlement_coordinates.py",
               "Parsing/geonames_settlement_coordinates/build_settlement_geographic_features.py"]:
    process = subprocess.run([sys.executable, str(ROOT/script)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    RESULT["evidence"][script] = {"returncode": process.returncode, "stderr": process.stderr[-3000:]}

# Reproduce the actual step 5 failure without creating/changing Data/Interim.
missing_parent = TEMP / "nonexistent_parent" / "regions.csv"
assert not missing_parent.parent.exists()
audit_region_path = paths.REGIONS
paths.REGIONS = missing_parent
try:
    regions.main()
except OSError as exc:
    RESULT["evidence"]["step5_missing_directory"] = str(exc)
finally:
    paths.REGIONS = audit_region_path
try:
    geography.load_geonames_matches()
except FileNotFoundError as exc:
    RESULT["evidence"]["step8_missing_input"] = str(exc)

# Compare published geographic coordinates by settlement key, never old new_id.
geo = pd.read_csv(paths.GEONAMES_FEATURES)
from src.names import split_settlement, settlement_base_name, settlement_type_class
geo["region_okato_code"] = geo.source_region.map(shop_regions.set_index("shop_region").okato_code)
parsed = geo.source_settlement.map(geography.parse_or_none)
geo = geo[parsed.notna() & geo.region_okato_code.notna()].copy()
geo["settlement_base_name"] = parsed.dropna().str[0].map(settlement_base_name)
geo["settlement_type_class"] = parsed.dropna().str[1].map(settlement_type_class)
geo["settlement_key"] = geo[settlements.KEY_COLUMNS].agg(":".join, axis=1)
candidates = points[["settlement_key", "latitude", "longitude", "coordinate_source"]].merge(
    geo[["settlement_key", "settlement_lat", "settlement_lon"]], on="settlement_key", how="left")
candidates["coordinate_match"] = np.isclose(candidates.latitude, candidates.settlement_lat, rtol=0, atol=1e-8) & np.isclose(candidates.longitude, candidates.settlement_lon, rtol=0, atol=1e-8)
matched_coordinates = candidates.groupby("settlement_key").coordinate_match.any()
def check_coordinates():
    manual = points.coordinate_source.eq("manual")
    assert matched_coordinates.reindex(points.loc[~manual,"settlement_key"]).all()
    corrections = pd.read_csv(paths.MANUAL_COORDINATES)
    corrections = corrections[corrections.decision.eq("replace")]
    actual = points.loc[manual,["settlement_key", "latitude", "longitude"]]
    compare_frame(actual, corrections[actual.columns], ["settlement_key"])
    # Distances from the original pipeline are checked from its own published centers.
    centers = geography.add_region_centers(points.copy())
    np.testing.assert_allclose(centers.distance_to_moscow_km, points.distance_to_moscow_km, rtol=1e-12)
    np.testing.assert_allclose(centers.distance_to_region_center_km, points.distance_to_region_center_km, rtol=1e-12)
    return {"published_geonames_matches": int((~manual).sum()), "manual_replacements": int(manual.sum()),
            "linkage": "region + normalized settlement name + settlement type"}
record("Published geographic coordinates, manual corrections, distances", check_coordinates)

RESULT["evidence"]["paper_export_identical"] = sha(next((ROOT/"Paper").glob("*.pdf"))) == sha(next((ROOT/"Export").glob("*.pdf")))
RESULT["limitations"] = [
    "Rscript unavailable in this environment: no R estimators/permutation tests rerun.",
    "Missing GeoNames diagnostics prevents full geography matching provenance verification.",
    "No original X5 forecasting notebook, test file or submission sample present.",
]
RESULT["checks"].append({"name": "Original tracked file hashes unchanged",
                         "status": "passed" if all(sha(ROOT/p)==h for p,h in originals.items()) else "failed",
                         "details": {"files": len(originals)}})
RESULT["environment"] = {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__}
(OUT/"verification.json").write_text(json.dumps(RESULT, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
print(json.dumps(RESULT, ensure_ascii=False, indent=2, default=str))
