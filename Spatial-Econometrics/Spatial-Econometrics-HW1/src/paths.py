"""Пути к исходным, промежуточным и итоговым файлам конвейера."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "Data"
INTERIM = DATA / "Interim"
MANUAL = DATA / "Manual"
RUNS = PROJECT_ROOT / "Runs"

SHOPS_RAW = DATA / "Shops" / "train_2.csv"

INFLATION_RAW = DATA / "Inflation" / "ipc_RF_fo_sub_08-2026.xlsx"
INFLATION_CSV = DATA / "Inflation" / "consumer_price_index_by_region.csv"

WAGES_RAW = DATA / "Wages" / "tab2-zpl_06-2026.xlsx"
WAGES_CSV = DATA / "Wages" / "nominal_wage_by_region.csv"

UNEMPLOYMENT_RAW = DATA / "Unemployment" / "trud_3_15-s.xlsx"
UNEMPLOYMENT_CSV = DATA / "Unemployment" / "unemployment_rate_monthly_by_region.csv"
UNEMPLOYMENT_ANNUAL_CSV = (
    DATA / "Unemployment" / "unemployment_rate_annual_by_region.csv"
)

POPULATION_RAW = DATA / "Population" / "BUL_MO_2024.xlsx"
POPULATION_CSV = DATA / "Population" / "urban_settlement_population_2024.csv"

GEONAMES_FEATURES = (
    PROJECT_ROOT
    / "Parsing"
    / 'geonames-settlement-coordinates'
    / "result"
    / "settlement_geographic_features.csv"
)
GEONAMES_DIAGNOSTICS = (
    PROJECT_ROOT
    / "Parsing"
    / 'geonames-settlement-coordinates'
    / "tmp"
    / "geonames_ru"
    / "service_outputs"
    / "settlement_coordinates_diagnostics.csv"
)

REGIONS = INTERIM / "regions.csv"
STORES = INTERIM / "stores.parquet"
STORE_MONTHS = INTERIM / "store_months.parquet"
SETTLEMENTS = INTERIM / "settlements.parquet"
SETTLEMENT_MONTHS = INTERIM / "settlement_months.parquet"
SETTLEMENT_GEOGRAPHY = INTERIM / "settlement_geography.parquet"
SETTLEMENT_POPULATION = INTERIM / "settlement_population.parquet"

MANUAL_POPULATION_LINKS = MANUAL / "population_links.csv"
MANUAL_COORDINATES = MANUAL / "coordinates.csv"

PANEL = DATA / "settlement_panel.parquet"
