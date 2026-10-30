"""Шаг 8. Координаты населенных пунктов и расстояния до центров.

Координаты берутся из собственного парсинга GeoNames, где поиск ограничен
регионом. Внутри региона возможна подмена тезкой, поэтому для отобранных
населенных пунктов конвейер проверяет качество сопоставления и
останавливается, если сомнительная точка не подтверждена вручную в
`Data/Manual/coordinates.csv`.
"""

import numpy as np
import pandas as pd

from src import paths
from src.names import settlement_base_name, settlement_type_class, split_settlement

EARTH_RADIUS_KM = 6371.0
MOSCOW = (55.75204, 37.61781)
SAINT_PETERSBURG = (59.93863, 30.31413)

# Центр Московской и Ленинградской областей - ядро агломерации, где
# находится экономический центр и правительство (для Ленинградской
# области). Парсинг GeoNames выбирает для них Химки и Гатчину.
REGION_CENTER_OVERRIDES = {
    "Московская обл": ("Moscow", *MOSCOW),
    "Ленинградская обл": ("Saint Petersburg", *SAINT_PETERSBURG),
}

# Меньше этого расстояния две разные точки считаются совпадающими.
SAME_POINT_KM = 1.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Расстояние по большому кругу между точками, километры."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    half_chord = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(half_chord))


def parse_or_none(raw_name: str) -> tuple[str, str] | None:
    """Разбирает название или возвращает None для нераспознанного типа."""
    try:
        return split_settlement(raw_name)
    except ValueError:
        return None


def load_geonames_matches() -> pd.DataFrame:
    """Читает результаты парсинга GeoNames и строит ключ населенного пункта."""
    columns = {
        "Регион": "shop_region",
        "Населенный пункт": "shop_settlement",
        "matched_geonameid": "geonames_id",
        "matched_geoname_name": "geonames_name",
        "feature_code": "geonames_feature_code",
        "population": "geonames_population",
        "lon": "longitude",
        "lat": "latitude",
        "settlement_match_method": "geonames_match_method",
        "settlement_match_score": "geonames_match_score",
    }
    matches = pd.read_csv(paths.GEONAMES_DIAGNOSTICS, usecols=list(columns))
    matches = matches.rename(columns=columns).drop_duplicates(
        ["shop_region", "shop_settlement"]
    )

    regions = pd.read_csv(paths.REGIONS, dtype={"okato_code": str}).set_index(
        "shop_region"
    )
    matches["region_okato_code"] = matches.shop_region.map(regions.okato_code)
    parsed = matches.shop_settlement.map(parse_or_none)
    matches = matches[parsed.notna() & matches.region_okato_code.notna()].copy()
    matches["settlement_base_name"] = parsed.dropna().str[0].map(settlement_base_name)
    matches["settlement_type_class"] = parsed.dropna().str[1].map(settlement_type_class)
    return matches


def choose_point(candidates: pd.DataFrame) -> pd.Series:
    """Выбирает одну точку из вариантов написания и оценивает их разброс."""
    best = (
        candidates.sort_values("geonames_match_score", ascending=False).iloc[0].copy()
    )
    spread = haversine_km(
        candidates.latitude, candidates.longitude, best.latitude, best.longitude
    )
    best["geonames_variants"] = len(candidates)
    best["geonames_variants_spread_km"] = float(np.max(spread))
    return best


def attach_points(settlements: pd.DataFrame) -> pd.DataFrame:
    """Присоединяет точку GeoNames, ручные правки имеют приоритет."""
    keys = ["region_okato_code", "settlement_base_name", "settlement_type_class"]
    matches = load_geonames_matches()
    points = (
        matches.groupby(keys).apply(choose_point, include_groups=False).reset_index()
    )
    point_columns = [
        "latitude",
        "longitude",
        "geonames_id",
        "geonames_name",
        "geonames_feature_code",
        "geonames_population",
        "geonames_match_method",
        "geonames_match_score",
        "geonames_variants",
        "geonames_variants_spread_km",
    ]
    result = settlements.merge(points[keys + point_columns], on=keys, how="left")
    result["coordinate_source"] = np.where(result.latitude.notna(), "geonames", None)
    return apply_manual_coordinates(result)


def apply_manual_coordinates(settlements: pd.DataFrame) -> pd.DataFrame:
    """Подставляет координаты, проверенные вручную."""
    if not paths.MANUAL_COORDINATES.exists():
        settlements["manual_decision"] = None
        return settlements
    manual = pd.read_csv(paths.MANUAL_COORDINATES).set_index("settlement_key")
    unknown = set(manual.index) - set(settlements.settlement_key)
    assert not unknown, f"в ручных координатах неизвестные ключи: {unknown}"

    settlements["manual_decision"] = settlements.settlement_key.map(manual.decision)
    is_replaced = settlements.manual_decision.eq("replace")
    for column in ["latitude", "longitude"]:
        replacement = settlements.settlement_key.map(manual[column])
        settlements.loc[is_replaced, column] = replacement[is_replaced]
    settlements.loc[is_replaced, "coordinate_source"] = "manual"
    return settlements


def add_region_centers(settlements: pd.DataFrame) -> pd.DataFrame:
    """Добавляет центр субъекта и расстояния до него и до Москвы."""
    features = (
        pd.read_csv(
            paths.GEONAMES_FEATURES,
            usecols=[
                "source_region",
                "region_center_name",
                "region_center_lat",
                "region_center_lon",
            ],
        )
        .drop_duplicates("source_region")
        .set_index("source_region")
    )
    for region, (name, latitude, longitude) in REGION_CENTER_OVERRIDES.items():
        features.loc[region, "region_center_name"] = name
        features.loc[region, "region_center_lat"] = latitude
        features.loc[region, "region_center_lon"] = longitude

    centers = settlements.shop_region.map
    settlements["region_center_name"] = centers(features.region_center_name)
    settlements["region_center_latitude"] = centers(features.region_center_lat)
    settlements["region_center_longitude"] = centers(features.region_center_lon)
    settlements["distance_to_moscow_km"] = haversine_km(
        settlements.latitude, settlements.longitude, *MOSCOW
    )
    settlements["distance_to_region_center_km"] = haversine_km(
        settlements.latitude,
        settlements.longitude,
        settlements.region_center_latitude,
        settlements.region_center_longitude,
    )
    return settlements


def point_flags(settlements: pd.DataFrame) -> pd.Series:
    """Возвращает причины сомнений в точке для каждого населенного пункта."""
    reasons = pd.Series("", index=settlements.index)
    reasons[settlements.latitude.isna()] += "нет координат; "
    reasons[settlements.geonames_match_method.eq("fuzzy")] += (
        "нечеткое совпадение названия; "
    )
    is_spread = settlements.geonames_variants_spread_km.gt(SAME_POINT_KM)
    reasons[is_spread] += "варианты написания в разных точках; "
    is_not_place = ~settlements.geonames_feature_code.fillna("").str.startswith("PPL")
    reasons[is_not_place] += "объект GeoNames не населенный пункт; "

    selected = settlements[settlements.is_selected & settlements.latitude.notna()]
    for index, row in selected.iterrows():
        distances = haversine_km(
            selected.latitude, selected.longitude, row.latitude, row.longitude
        )
        if (distances < SAME_POINT_KM).sum() > 1:
            reasons[index] += "совпадает с точкой другого населенного пункта; "
    return reasons.str.rstrip("; ")


def main() -> None:
    settlements = pd.read_parquet(paths.SETTLEMENTS)
    settlements = add_region_centers(attach_points(settlements))
    settlements["point_flags"] = point_flags(settlements)

    is_confirmed = settlements.manual_decision.isin(["confirm", "replace"])
    is_doubtful = settlements.is_selected & settlements.point_flags.ne("")
    to_review = settlements[is_doubtful & ~is_confirmed]
    review_path = paths.INTERIM / "coordinates_to_review.csv"
    review_columns = [
        "settlement_key",
        "shop_region",
        "settlement_name",
        "store_count",
        "point_flags",
        "geonames_name",
        "geonames_feature_code",
        "geonames_population",
        "population_reported_median",
        "latitude",
        "longitude",
        "distance_to_region_center_km",
    ]
    to_review[review_columns].to_csv(review_path, index=False)
    assert to_review.empty, (
        f"{len(to_review)} отобранных точек требуют проверки, список в {review_path}"
    )

    settlements.to_parquet(paths.SETTLEMENT_GEOGRAPHY, index=False)
    print(
        f"{paths.SETTLEMENT_GEOGRAPHY.name}: координаты у "
        f"{settlements[settlements.is_selected].latitude.notna().sum()} "
        f"из {settlements.is_selected.sum()} отобранных"
    )


if __name__ == "__main__":
    main()
