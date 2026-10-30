"""Сборка location-справочника географических фичей без `new_id`."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.preprocessing.geographic_features import (  # noqa: E402
    DENIS_SELECTED_GEOGRAPHIC_FEATURE_COLUMNS,
    NORMALIZED_REGION_COLUMN,
    NORMALIZED_SETTLEMENT_COLUMN,
    add_normalized_location_columns,
)

SOURCE_FEATURE_TABLE_PATH = (
    PROJECT_ROOT
    / "Parsing"
    / "geonames_settlement_coordinates"
    / "result"
    / "settlement_coordinates.csv"
)
RESULT_FEATURE_TABLE_PATH = (
    PROJECT_ROOT
    / "Parsing"
    / "geonames_settlement_coordinates"
    / "result"
    / "settlement_geographic_features.csv"
)


def build_settlement_geographic_feature_table(
    source_feature_table_path: str | Path = SOURCE_FEATURE_TABLE_PATH,
) -> pd.DataFrame:
    """
    Строит справочник географических фичей на уровне нормализованной локации.

    Parameters
    ----------
    source_feature_table_path : str | Path, default=SOURCE_FEATURE_TABLE_PATH
        Путь до старого справочника `settlement_coordinates.csv`, построенного на уровне
        `new_id` отборочного этапа.

    Returns
    -------
    pd.DataFrame
        Таблица без `new_id` и магазинных данных: нормализованный регион, нормализованный
        населённый пункт, сырые source-названия и выбранные географические признаки Дениса.
    """

    source_feature_table = pd.read_csv(source_feature_table_path)
    source_feature_table = add_normalized_location_columns(
        source_feature_table,
        region_column="region",
        settlement_column="settlement",
    )
    coordinate_consistency_issues = build_coordinate_consistency_issues(
        source_feature_table
    )
    if not coordinate_consistency_issues.empty:
        raise ValueError(
            "В исходном справочнике найдены разные координаты для одной normalized-локации. "
            "Проверьте `coordinate_consistency_issues` перед сборкой."
        )

    location_feature_table = (
        source_feature_table.sort_values(
            [NORMALIZED_REGION_COLUMN, NORMALIZED_SETTLEMENT_COLUMN]
        )
        .drop_duplicates(
            subset=[NORMALIZED_REGION_COLUMN, NORMALIZED_SETTLEMENT_COLUMN]
        )
        .rename(
            columns={
                "region": "source_region",
                "settlement": "source_settlement",
            }
        )
        .reset_index(drop=True)
    )
    location_feature_table["distance_to_moscow_bin_q10"] = build_quantile_distance_bin(
        location_feature_table["distance_to_moscow_m"],
        bin_count=10,
    )

    ordered_columns = [
        NORMALIZED_REGION_COLUMN,
        NORMALIZED_SETTLEMENT_COLUMN,
        "source_region",
        "source_settlement",
        "settlement_lon",
        "settlement_lat",
        "region_center_name",
        "region_center_lon",
        "region_center_lat",
        *DENIS_SELECTED_GEOGRAPHIC_FEATURE_COLUMNS,
    ]
    return location_feature_table[ordered_columns].copy()


def build_quantile_distance_bin(
    distance_values: pd.Series, bin_count: int = 10
) -> pd.Series:
    """
    Строит категориальные квантили расстояния до Москвы.

    Parameters
    ----------
    distance_values : pd.Series
        Расстояние до Москвы в метрах.
    bin_count : int, default=10
        Количество квантильных корзин.

    Returns
    -------
    pd.Series
        Строковые метки бинов вида `bin_00`, `bin_01`, ...
    """

    numeric_distance = pd.to_numeric(distance_values, errors="coerce").fillna(0.0)
    quantile_bins = pd.qcut(
        numeric_distance.rank(method="first"),
        q=bin_count,
        labels=[f"bin_{index:02d}" for index in range(bin_count)],
    )
    return quantile_bins.astype(str)


def build_coordinate_consistency_issues(
    source_feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Проверяет, нет ли разных координат внутри одной normalized-локации.

    Parameters
    ----------
    source_feature_table : pd.DataFrame
        Исходный справочник с нормализованными колонками и координатами.

    Returns
    -------
    pd.DataFrame
        Таблица конфликтных normalized-локаций; пустой результат означает, что конфликтов нет.
    """

    coordinate_columns = [
        "settlement_lon",
        "settlement_lat",
        "region_center_name",
        "region_center_lon",
        "region_center_lat",
    ]
    unique_value_count_frame = (
        source_feature_table.groupby(
            [NORMALIZED_REGION_COLUMN, NORMALIZED_SETTLEMENT_COLUMN],
            as_index=False,
        )[coordinate_columns]
        .nunique(dropna=False)
        .sort_values([NORMALIZED_REGION_COLUMN, NORMALIZED_SETTLEMENT_COLUMN])
    )
    inconsistent_mask = unique_value_count_frame[coordinate_columns].gt(1).any(axis=1)
    return unique_value_count_frame[inconsistent_mask].copy()


def main() -> None:
    """
    Создаёт CSV-справочник `settlement_geographic_features.csv` в директории `result`.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Итоговый CSV сохраняется на диск.
    """

    result_feature_table = build_settlement_geographic_feature_table()
    RESULT_FEATURE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_feature_table.to_csv(RESULT_FEATURE_TABLE_PATH, index=False)
    print(f"Rows: {len(result_feature_table)}")
    print(f"Columns: {result_feature_table.columns.tolist()}")
    print(f"Result path: {RESULT_FEATURE_TABLE_PATH}")


if __name__ == "__main__":
    main()
