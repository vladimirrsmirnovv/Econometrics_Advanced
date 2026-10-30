"""Шаг 6. Очистка панели «магазин - месяц» сети «Пятерочка».

Результат - две таблицы: постоянные характеристики магазинов и помесячная
выручка. Исходный файл не меняется. Нули в зашумленных признаках
означают пропуск и заменяются на NA, рабочие часы ограничиваются сутками.
"""

import pandas as pd

from src import paths
from src.names import settlement_base_name, settlement_type_class, split_settlement

COLUMN_NAMES = {
    "new_id": "store_id",
    "Год": "year",
    "Месяц": "month",
    "РТО": "revenue_rub",
    "Среднее количество промо товаров в чеке": "promo_items_per_check",
    "Среднее количество товаров в чеке": "items_per_check",
    "Среднее количество отмен": "cancellations",
    "Рабочие часы в день": "working_hours_per_day",
    "Дата открытия, категориальный": "store_age_category",
    "Торговая площадь, категориальный": "sales_area_category",
    "Населенный пункт": "shop_settlement",
    "Регион": "shop_region",
    "Численность населения": "population_reported",
    "Количество домохозяйств": "households",
    "Трафик пеший, в час": "foot_traffic_per_hour",
    "Трафик авто, в час": "car_traffic_per_hour",
    "Маркетплейсы, доставки, постаматы (100 м)": "pickup_points_100m",
    "Медицинские уч. и аптеки (300 м)": "medical_and_pharmacies_300m",
    "Школы (300 м)": "schools_300m",
    "Остановки (300 м)": "public_transport_stops_300m",
    "Продуктовые магазины (500 м)": "grocery_competitors_500m",
    "Пятерочки (500 м)": "pyaterochka_stores_500m",
    "Количество касс": "checkouts",
    "Флаг алкогольной лицензии": "has_alcohol_license",
}

MONTHLY_COLUMNS = [
    "revenue_rub",
    "items_per_check",
    "cancellations",
    "promo_items_per_check",
]

# В этих признаках ноль означает пропуск: у магазинов Москвы и Сочи
# встречается нулевая численность населения, а у работающего магазина
# не бывает нуля касс.
ZERO_MEANS_MISSING = [
    "population_reported",
    "households",
    "foot_traffic_per_hour",
    "car_traffic_per_hour",
    "checkouts",
]

HOURS_IN_DAY = 24
REVENUE_JUMP_RATIO = 3


def read_shops() -> pd.DataFrame:
    """Читает исходную панель и переименовывает колонки."""
    shops = pd.read_csv(paths.SHOPS_RAW).rename(columns=COLUMN_NAMES)
    assert set(shops.columns) == set(COLUMN_NAMES.values()), "изменился состав колонок"
    return shops


def check_panel(shops: pd.DataFrame) -> None:
    """Проверяет ключ, сбалансированность и положительность выручки."""
    assert not shops.duplicated(["store_id", "year", "month"]).any(), (
        "дубли магазин-месяц"
    )
    months = shops[["year", "month"]].drop_duplicates()
    assert len(months) == 26, "ожидалось 26 месяцев"
    assert shops.groupby("store_id").size().eq(len(months)).all(), (
        "панель не сбалансирована"
    )
    assert shops.revenue_rub.gt(0).all(), "неположительная выручка"
    assert shops.notna().all().all(), "пропуски в исходных данных"


def build_stores(shops: pd.DataFrame) -> pd.DataFrame:
    """Строит таблицу постоянных характеристик: одна строка на магазин."""
    static_columns = [
        column
        for column in COLUMN_NAMES.values()
        if column not in MONTHLY_COLUMNS + ["year", "month"]
    ]
    changes = shops.groupby("store_id")[static_columns].nunique().gt(1).sum()
    assert changes.eq(0).all(), f"меняются во времени: {changes[changes > 0].to_dict()}"

    stores = shops.drop_duplicates("store_id")[static_columns].copy()
    stores[ZERO_MEANS_MISSING] = (
        stores[ZERO_MEANS_MISSING]
        .where(stores[ZERO_MEANS_MISSING].gt(0))
        .astype("Float64")
    )
    stores["working_hours_per_day"] = stores.working_hours_per_day.clip(
        upper=HOURS_IN_DAY
    )
    return stores


def add_settlement_keys(stores: pd.DataFrame) -> pd.DataFrame:
    """Разбирает населенный пункт на название и тип, добавляет код субъекта."""
    parsed = stores.shop_settlement.map(split_settlement)
    stores["settlement_type"] = parsed.str[1]
    stores["settlement_type_class"] = stores.settlement_type.map(settlement_type_class)
    stores["settlement_base_name"] = parsed.str[0].map(settlement_base_name)

    regions = pd.read_csv(paths.REGIONS, dtype={"okato_code": str})
    stores["region_okato_code"] = stores.shop_region.map(
        regions.set_index("shop_region").okato_code
    )
    assert stores.region_okato_code.notna().all()
    return stores


def flag_revenue_jumps(store_months: pd.DataFrame) -> pd.Series:
    """Отмечает магазины со скачком выручки более чем втрое между месяцами."""
    ordered = store_months.sort_values(["store_id", "year", "month"])
    ratio = ordered.revenue_rub / ordered.groupby("store_id").revenue_rub.shift()
    is_jump = ratio.gt(REVENUE_JUMP_RATIO) | ratio.lt(1 / REVENUE_JUMP_RATIO)
    return is_jump.groupby(ordered.store_id).any().rename("has_revenue_jump")


def main() -> None:
    shops = read_shops()
    check_panel(shops)

    store_months = shops[["store_id", "year", "month", *MONTHLY_COLUMNS]]
    store_months = store_months.sort_values(["store_id", "year", "month"])
    stores = add_settlement_keys(build_stores(shops))
    stores = stores.merge(flag_revenue_jumps(store_months), on="store_id")

    stores.to_parquet(paths.STORES, index=False)
    store_months.to_parquet(paths.STORE_MONTHS, index=False)
    print(
        f"{paths.STORES.name}: {len(stores)} магазинов, "
        f"со скачком выручки {stores.has_revenue_jump.sum()}; "
        f"{paths.STORE_MONTHS.name}: {len(store_months)} строк"
    )


if __name__ == "__main__":
    main()
