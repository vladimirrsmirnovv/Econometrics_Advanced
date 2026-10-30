"""Шаг 9. Численность населения Росстата для населенных пунктов сети.

Бюллетень содержит только городские населенные пункты, поэтому город
сопоставляется с городом, а поселок или пгт - с поселком или пгт того же
субъекта. Села, станицы, деревни и хутора в бюллетене отсутствуют, для них
остается медиана численности из данных сети с пометкой источника.
Спорные пары фиксируются вручную в `Data/Manual/population_links.csv`.
"""

import pandas as pd

from src import paths

URBAN_SETTLEMENT_CLASSES = {"поселок городского типа", "поселок", "населенный пункт"}
RURAL_SETTLEMENT_CLASSES = {"село", "деревня", "станица", "хутор", "аул", "слобода"}


def comparable_class(type_class: str) -> str:
    """Объединяет классы, которые в бюллетене и в сети записываются по-разному."""
    return "поселок" if type_class in URBAN_SETTLEMENT_CLASSES else type_class


def load_bulletin() -> pd.DataFrame:
    """Читает численность городских населенных пунктов из шага 4."""
    codes = {"oktmo_code": str, "region_okato_code": str}
    bulletin = pd.read_csv(paths.POPULATION_CSV, dtype=codes)
    bulletin["comparable_class"] = bulletin.settlement_type_class.map(comparable_class)
    return bulletin


def match_automatically(
    settlements: pd.DataFrame, bulletin: pd.DataFrame
) -> pd.DataFrame:
    """Сопоставляет по субъекту, названию и совместимому типу."""
    keys = ["region_okato_code", "settlement_base_name", "comparable_class"]
    settlements = settlements.assign(
        comparable_class=settlements.settlement_type_class.map(comparable_class)
    )
    federal = bulletin[bulletin.is_federal_city].set_index("region_okato_code")
    regular = bulletin[~bulletin.is_federal_city]
    assert not regular.duplicated(keys).any(), "неоднозначные строки бюллетеня"

    matched = settlements.merge(
        regular[keys + ["oktmo_code"]], on=keys, how="left", validate="many_to_one"
    )
    is_federal_city = matched.region_okato_code.isin(federal.index)
    matched.loc[is_federal_city, "oktmo_code"] = matched.region_okato_code.map(
        federal.oktmo_code
    )
    matched["population_match"] = matched.oktmo_code.notna().map(
        {True: "automatic", False: None}
    )
    return matched


def apply_manual_links(matched: pd.DataFrame) -> pd.DataFrame:
    """Подставляет пары, проверенные вручную; `none` означает отсутствие пары."""
    if not paths.MANUAL_POPULATION_LINKS.exists():
        return matched
    manual = pd.read_csv(paths.MANUAL_POPULATION_LINKS, dtype=str).set_index(
        "settlement_key"
    )
    unknown = set(manual.index) - set(matched.settlement_key)
    assert not unknown, f"в ручных связях неизвестные ключи: {unknown}"

    manual_code = matched.settlement_key.map(manual.oktmo_code)
    has_manual = manual_code.notna()
    matched.loc[has_manual, "oktmo_code"] = manual_code[has_manual].replace(
        "none", None
    )
    matched.loc[has_manual, "population_match"] = "manual"
    return matched


def cross_class_candidates(
    unmatched: pd.DataFrame, bulletin: pd.DataFrame
) -> pd.DataFrame:
    """Ищет для несопоставленных пунктов тезок того же субъекта с другим типом."""
    keys = ["region_okato_code", "settlement_base_name"]
    candidates = unmatched.merge(bulletin, on=keys, suffixes=("", "_bulletin"))
    return candidates[
        [
            "settlement_key",
            "settlement_name",
            "store_count",
            "population_reported_median",
            "oktmo_code_bulletin",
            "source_name",
            "population_total",
        ]
    ]


def main() -> None:
    settlements = pd.read_parquet(paths.SETTLEMENTS)
    bulletin = load_bulletin()
    matched = apply_manual_links(match_automatically(settlements, bulletin))

    is_open = (
        matched.is_selected
        & matched.oktmo_code.isna()
        & matched.population_match.isna()
        & ~matched.settlement_type_class.isin(RURAL_SETTLEMENT_CLASSES)
    )
    review = cross_class_candidates(matched[is_open], bulletin)
    review_path = paths.INTERIM / "population_to_review.csv"
    matched[is_open][
        [
            "settlement_key",
            "shop_region",
            "settlement_name",
            "store_count",
            "population_reported_median",
        ]
    ].merge(review, how="left").to_csv(review_path, index=False)
    assert not is_open.any(), (
        f"{is_open.sum()} отобранных городских пунктов без пары в бюллетене, "
        f"список в {review_path}"
    )

    population = bulletin.set_index("oktmo_code").population_total
    matched["population_rosstat_2024"] = matched.oktmo_code.map(population).astype(
        "Int64"
    )
    columns = [
        "settlement_key",
        "oktmo_code",
        "population_rosstat_2024",
        "population_match",
    ]
    matched[columns].to_parquet(paths.SETTLEMENT_POPULATION, index=False)

    selected = matched[matched.is_selected]
    print(
        f"{paths.SETTLEMENT_POPULATION.name}: численность Росстата у "
        f"{selected.population_rosstat_2024.notna().sum()} из {len(selected)} отобранных"
    )


if __name__ == "__main__":
    main()
