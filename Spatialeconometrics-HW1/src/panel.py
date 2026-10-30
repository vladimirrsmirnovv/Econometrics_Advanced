"""Шаг 10. Итоговая панель «населенный пункт - месяц» для анализа в R.

Выручка переводится в цены января 2023 года региональным индексом цен на
продовольственные товары, зарплата - общим индексом потребительских цен.
Численность населения берется из бюллетеня Росстата, а для сел и станиц,
которых в бюллетене нет, - медианой из данных сети.
"""

import numpy as np
import pandas as pd

from src import paths

FIRST_PERIOD = (2023, 1)
LAST_PERIOD = (2025, 2)
MONTH_KEY = ["region_okato_code", "year", "month"]


def price_index(prices: pd.DataFrame, column: str) -> pd.Series:
    """Цепной индекс цен с базой январь 2023 года = 1."""
    ordered = prices.sort_values(MONTH_KEY)
    growth = (ordered[column] / 100).where(
        (ordered.year * 12 + ordered.month) > FIRST_PERIOD[0] * 12 + FIRST_PERIOD[1],
        1.0,
    )
    return growth.groupby(ordered.region_okato_code).cumprod()


def in_study_period(table: pd.DataFrame) -> pd.Series:
    """Отмечает месяцы с января 2023 по февраль 2025 года."""
    period = table.year * 12 + table.month
    first = FIRST_PERIOD[0] * 12 + FIRST_PERIOD[1]
    last = LAST_PERIOD[0] * 12 + LAST_PERIOD[1]
    return period.between(first, last)


def load_regional_months() -> pd.DataFrame:
    """Собирает помесячные региональные показатели Росстата."""
    code = {"okato_code": str}
    renamed = {"okato_code": "region_okato_code"}
    prices = pd.read_csv(paths.INFLATION_CSV, dtype=code).rename(columns=renamed)
    prices = prices[in_study_period(prices)].copy()
    prices["food_price_index"] = price_index(prices, "cpi_food")
    prices["consumer_price_index"] = price_index(prices, "cpi_all_items")

    wages = pd.read_csv(paths.WAGES_CSV, dtype=code).rename(columns=renamed)
    unemployment = pd.read_csv(paths.UNEMPLOYMENT_CSV, dtype=code).rename(
        columns=renamed
    )
    annual = pd.read_csv(paths.UNEMPLOYMENT_ANNUAL_CSV, dtype=code).rename(
        columns={
            "okato_code": "region_okato_code",
            "unemployment_rate_percent": "unemployment_rate_annual_percent",
        }
    )

    regional = prices[
        MONTH_KEY + ["region_name", "food_price_index", "consumer_price_index"]
    ]
    regional = regional.merge(
        wages[MONTH_KEY + ["nominal_wage_rub"]], on=MONTH_KEY, how="left"
    )
    regional = regional.merge(
        unemployment[MONTH_KEY + ["unemployment_rate_percent"]],
        on=MONTH_KEY,
        how="left",
    )
    regional = regional.merge(
        annual[["region_okato_code", "year", "unemployment_rate_annual_percent"]],
        on=["region_okato_code", "year"],
        how="left",
    )
    regional["real_wage_rub"] = (
        regional.nominal_wage_rub / regional.consumer_price_index
    )
    return regional


def load_settlements() -> pd.DataFrame:
    """Собирает постоянные характеристики отобранных населенных пунктов."""
    geography = pd.read_parquet(paths.SETTLEMENT_GEOGRAPHY)
    population = pd.read_parquet(paths.SETTLEMENT_POPULATION)
    settlements = geography[geography.is_selected].merge(
        population, on="settlement_key"
    )

    has_rosstat = settlements.population_rosstat_2024.notna()
    settlements["population"] = settlements.population_rosstat_2024.astype(
        "Float64"
    ).where(has_rosstat, settlements.population_reported_median)
    settlements["population_source"] = np.where(
        has_rosstat, "rosstat_2024", "shops_median"
    )
    return settlements


def build_panel() -> pd.DataFrame:
    """Соединяет населенные пункты, их выручку и региональные показатели."""
    settlements = load_settlements()
    months = pd.read_parquet(paths.SETTLEMENT_MONTHS)
    panel = settlements.merge(months, on="settlement_key", validate="one_to_many")
    panel = panel.merge(
        load_regional_months(), on=MONTH_KEY, how="left", validate="many_to_one"
    )

    panel["revenue_per_store_real_rub"] = (
        panel.revenue_per_store_rub / panel.food_price_index
    )
    panel["log_revenue_per_store_real"] = np.log(panel.revenue_per_store_real_rub)
    panel["period"] = pd.to_datetime(dict(year=panel.year, month=panel.month, day=1))
    panel["time_index"] = (
        (panel.year - FIRST_PERIOD[0]) * 12 + panel.month - FIRST_PERIOD[1] + 1
    )
    return (
        panel[PANEL_COLUMNS]
        .sort_values(["settlement_id", "period"])
        .reset_index(drop=True)
    )


PANEL_COLUMNS = [
    "settlement_id",
    "settlement_key",
    "settlement_name",
    "settlement_type_class",
    "shop_region",
    "region_name",
    "region_okato_code",
    "oktmo_code",
    "year",
    "month",
    "period",
    "time_index",
    "log_revenue_per_store_real",
    "revenue_per_store_real_rub",
    "revenue_per_store_rub",
    "revenue_median_store_rub",
    "revenue_total_rub",
    "food_price_index",
    "latitude",
    "longitude",
    "coordinate_source",
    "distance_to_moscow_km",
    "distance_to_region_center_km",
    "region_center_name",
    "population",
    "population_source",
    "population_rosstat_2024",
    "population_reported_median",
    "households_median",
    "foot_traffic_per_hour_median",
    "car_traffic_per_hour_median",
    "grocery_competitors_500m_mean",
    "pyaterochka_stores_500m_mean",
    "pickup_points_100m_mean",
    "medical_and_pharmacies_300m_mean",
    "schools_300m_mean",
    "public_transport_stops_300m_mean",
    "checkouts_mean",
    "working_hours_per_day_mean",
    "store_count",
    "share_large_area",
    "share_small_area",
    "share_new_stores",
    "share_old_stores",
    "share_alcohol_license",
    "stores_with_revenue_jump",
    "nominal_wage_rub",
    "real_wage_rub",
    "consumer_price_index",
    "unemployment_rate_percent",
    "unemployment_rate_annual_percent",
    "items_per_check_mean",
    "cancellations_mean",
    "promo_items_per_check_mean",
]

REQUIRED_COMPLETE = [
    "log_revenue_per_store_real",
    "latitude",
    "longitude",
    "population",
    "distance_to_moscow_km",
    "distance_to_region_center_km",
    "food_price_index",
    "nominal_wage_rub",
    "unemployment_rate_percent",
    "unemployment_rate_annual_percent",
    "grocery_competitors_500m_mean",
    "pyaterochka_stores_500m_mean",
    "checkouts_mean",
]


def check_panel(panel: pd.DataFrame) -> None:
    """Проверяет баланс панели, полноту переменных и корректность точек."""
    periods = panel.period.nunique()
    settlements = panel.settlement_id.nunique()
    assert periods == 26, "ожидалось 26 месяцев"
    assert len(panel) == periods * settlements, "панель не сбалансирована"
    assert not panel.duplicated(["settlement_id", "period"]).any()

    missing = panel[REQUIRED_COMPLETE].isna().sum()
    assert missing.eq(0).all(), f"пропуски: {missing[missing > 0].to_dict()}"
    assert panel.revenue_per_store_real_rub.gt(0).all()
    assert panel.food_price_index.between(0.9, 1.5).all()

    points = panel.drop_duplicates("settlement_id")
    assert not points.duplicated(["latitude", "longitude"]).any(), "совпадающие точки"
    assert points.store_count.ge(3).all()


def main() -> None:
    panel = build_panel()
    check_panel(panel)
    panel.to_parquet(paths.PANEL, index=False)
    print(
        f"{paths.PANEL.name}: {panel.settlement_id.nunique()} населенных пунктов "
        f"x {panel.period.nunique()} месяцев = {len(panel)} строк, "
        f"{len(panel.columns)} колонок"
    )


if __name__ == "__main__":
    main()
