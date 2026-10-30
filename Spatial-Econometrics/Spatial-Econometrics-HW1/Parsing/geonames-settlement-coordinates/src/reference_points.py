from __future__ import annotations

import pandas as pd

from src.constants import FEATURE_RANK


def build_region_center_frame(geonames_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Строит таблицу репрезентативных центров регионов на основе GeoNames.

    Parameters
    ----------
    geonames_frame : pd.DataFrame
        Полный GeoNames-датасет по России.

    Returns
    -------
    pd.DataFrame
        Таблица с `admin1_code`, названием центра региона и его координатами.
    """

    populated_rows = geonames_frame[geonames_frame["feature_class"] == "P"].copy()
    populated_rows["feature_rank"] = (
        populated_rows["feature_code"].map(FEATURE_RANK).fillna(100)
    )
    populated_rows["population_value"] = pd.to_numeric(
        populated_rows["population"],
        errors="coerce",
    ).fillna(0)

    sorted_rows = populated_rows.sort_values(
        by=["admin1_code", "feature_rank", "population_value", "geonameid"],
        ascending=[True, True, False, True],
    )
    best_center_rows = sorted_rows.groupby("admin1_code", as_index=False).first()

    return best_center_rows.rename(
        columns={
            "name": "region_center_name",
            "latitude": "region_center_lat",
            "longitude": "region_center_lon",
            "geonameid": "region_center_geonameid",
            "feature_code": "region_center_feature_code",
        }
    )[
        [
            "admin1_code",
            "region_center_geonameid",
            "region_center_name",
            "region_center_lat",
            "region_center_lon",
            "region_center_feature_code",
        ]
    ].copy()


def build_moscow_reference(region_center_frame: pd.DataFrame) -> dict[str, object]:
    """
    Выделяет координаты Москвы как отдельной опорной точки для всех наблюдений.

    Parameters
    ----------
    region_center_frame : pd.DataFrame
        Таблица центров регионов.

    Returns
    -------
    dict[str, object]
        Словарь с названием и координатами Москвы.

    Raises
    ------
    ValueError
        Если в таблице центров регионов не найден `admin1_code=48`.
    """

    moscow_rows = region_center_frame[region_center_frame["admin1_code"] == "48"].copy()
    if moscow_rows.empty:
        raise ValueError("Не удалось выделить координаты Москвы из GeoNames.")

    moscow_row = moscow_rows.iloc[0]
    return {
        "moscow_name": moscow_row["region_center_name"],
        "moscow_lat": float(moscow_row["region_center_lat"]),
        "moscow_lon": float(moscow_row["region_center_lon"]),
    }
