"""Шаг 7. Агрегирование магазинов до уровня населенного пункта.

Один населенный пункт в исходных данных бывает записан по-разному:
«Малаховка пгт.» и «Малаховка рп», «Большие Вяземы» с буквой е и с е с точками.
Поэтому единица наблюдения задается тройкой «субъект - нормализованное
название - класс типа», где пгт, рп, гп и дп относятся к одному классу,
а город и село - к разным («Топки г» и «Топки с» - разные места).

Правила агрегирования зависят от природы признака. Зашумленные непрерывные
признаки сворачиваются медианой, устойчивой к мультипликативному шуму.
Счетные признаки окружения - средним, категориальные - долей.
"""

import pandas as pd

from src import paths
from src.names import NOT_A_SETTLEMENT

MIN_STORES = 3

MEDIAN_FEATURES = [
    "population_reported",
    "households",
    "foot_traffic_per_hour",
    "car_traffic_per_hour",
]

MEAN_FEATURES = [
    "pickup_points_100m",
    "medical_and_pharmacies_300m",
    "schools_300m",
    "public_transport_stops_300m",
    "grocery_competitors_500m",
    "pyaterochka_stores_500m",
    "checkouts",
    "working_hours_per_day",
]

KEY_COLUMNS = ["region_okato_code", "settlement_base_name", "settlement_type_class"]


def add_share_indicators(stores: pd.DataFrame) -> pd.DataFrame:
    """Превращает категории магазина в индикаторы для расчета долей."""
    return stores.assign(
        is_large_area=stores.sales_area_category.isin(["Большой", "Очень большой"]),
        is_small_area=stores.sales_area_category.eq("Маленький"),
        is_new_store=stores.store_age_category.eq("Новый"),
        is_old_store=stores.store_age_category.eq("Открыт давно"),
    )


def most_common_name(names: pd.Series) -> str:
    """Выбирает самое частое написание, при равенстве - первое по алфавиту."""
    counts = names.str.rstrip(".").value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def aggregate_settlements(stores: pd.DataFrame) -> pd.DataFrame:
    """Сворачивает постоянные характеристики магазинов до населенного пункта."""
    grouped = stores.groupby(KEY_COLUMNS)
    settlements = grouped.agg(
        shop_region=("shop_region", "first"),
        settlement_name=("shop_settlement", most_common_name),
        source_settlement_names=(
            "shop_settlement",
            lambda names: "; ".join(sorted(set(names))),
        ),
        store_count=("store_id", "size"),
        stores_with_revenue_jump=("has_revenue_jump", "sum"),
        share_large_area=("is_large_area", "mean"),
        share_small_area=("is_small_area", "mean"),
        share_new_stores=("is_new_store", "mean"),
        share_old_stores=("is_old_store", "mean"),
        share_alcohol_license=("has_alcohol_license", "mean"),
    )
    medians = grouped[MEDIAN_FEATURES].median().add_suffix("_median")
    means = grouped[MEAN_FEATURES].mean().add_suffix("_mean")
    observed = grouped.population_reported.count().rename(
        "stores_with_reported_population"
    )
    return pd.concat([settlements, medians, means, observed], axis=1).reset_index()


def assign_ids(settlements: pd.DataFrame) -> pd.DataFrame:
    """Присваивает устойчивый ключ и порядковый номер населенного пункта."""
    settlements = settlements.sort_values(["region_okato_code", "settlement_base_name"])
    settlements.insert(
        0, "settlement_key", settlements[KEY_COLUMNS].agg(":".join, axis=1)
    )
    settlements.insert(0, "settlement_id", range(1, len(settlements) + 1))
    settlements["is_selected"] = settlements.store_count >= MIN_STORES
    return settlements.reset_index(drop=True)


def aggregate_months(store_months: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Считает выручку и показатели чеков населенного пункта по месяцам."""
    keyed = store_months.merge(stores[["store_id", "settlement_key"]], on="store_id")
    grouped = keyed.groupby(["settlement_key", "year", "month"])
    return grouped.agg(
        stores_in_month=("store_id", "size"),
        revenue_total_rub=("revenue_rub", "sum"),
        revenue_per_store_rub=("revenue_rub", "mean"),
        revenue_median_store_rub=("revenue_rub", "median"),
        items_per_check_mean=("items_per_check", "mean"),
        cancellations_mean=("cancellations", "mean"),
        promo_items_per_check_mean=("promo_items_per_check", "mean"),
    ).reset_index()


def main() -> None:
    stores = pd.read_parquet(paths.STORES)
    stores = add_share_indicators(
        stores[~stores.settlement_type_class.isin(NOT_A_SETTLEMENT)]
    )
    settlements = assign_ids(aggregate_settlements(stores))
    stores = stores.merge(settlements[[*KEY_COLUMNS, "settlement_key"]], on=KEY_COLUMNS)

    months = aggregate_months(pd.read_parquet(paths.STORE_MONTHS), stores)
    counts = months.merge(
        settlements[["settlement_key", "store_count"]], on="settlement_key"
    )
    assert counts.stores_in_month.eq(counts.store_count).all(), (
        "состав магазинов меняется"
    )

    settlements.to_parquet(paths.SETTLEMENTS, index=False)
    months.to_parquet(paths.SETTLEMENT_MONTHS, index=False)
    merged = settlements[settlements.source_settlement_names.str.contains(";")]
    print(
        f"{paths.SETTLEMENTS.name}: {len(settlements)} населенных пунктов, "
        f"отобрано {settlements.is_selected.sum()} (не менее {MIN_STORES} магазинов), "
        f"склеено написаний {len(merged)}"
    )


if __name__ == "__main__":
    main()
