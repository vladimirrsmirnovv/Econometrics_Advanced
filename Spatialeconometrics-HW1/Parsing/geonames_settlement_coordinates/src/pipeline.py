from __future__ import annotations

import pandas as pd

from Utils.features import add_geospatial_features
from src.constants import GEONAMES_COLUMNS
from src.download import ensure_geonames_files
from src.matching import (
    build_place_index,
    build_region_resolution_table,
    resolve_settlement_record,
)
from src.paths import (
    DIAGNOSTICS_PATH,
    GEONAMES_TEXT_PATH,
    REGION_RESOLUTION_PATH,
    RESULT_PATH,
    SERVICE_OUTPUT_DIR,
    TRAIN_PATH,
    ensure_runtime_directories,
)
from src.reference_points import build_moscow_reference, build_region_center_frame


def run_pipeline() -> None:
    """
    Запускает полный пайплайн подготовки координат населённых пунктов.

    Returns
    -------
    None
        Создаёт итоговый CSV в `result` и служебные диагностические файлы в `tmp`.
    """

    ensure_runtime_directories()
    ensure_geonames_files()

    geonames_frame = load_geonames_dataframe()
    region_resolution_frame, region_code_mapping = build_region_resolution_table(
        geonames_frame
    )
    region_center_frame = build_region_center_frame(geonames_frame)
    moscow_reference = build_moscow_reference(region_center_frame)
    place_index, region_name_lists = build_place_index(geonames_frame)

    result_frame, diagnostics_frame = build_result_tables(
        region_code_mapping=region_code_mapping,
        region_center_frame=region_center_frame,
        moscow_reference=moscow_reference,
        place_index=place_index,
        region_name_lists=region_name_lists,
    )
    save_tables(
        result_frame=result_frame,
        diagnostics_frame=diagnostics_frame,
        region_resolution_frame=region_resolution_frame,
    )
    print_summary(
        result_frame=result_frame,
        diagnostics_frame=diagnostics_frame,
        region_resolution_frame=region_resolution_frame,
    )


def load_geonames_dataframe() -> pd.DataFrame:
    """
    Загружает текстовый GeoNames-дамп России в `DataFrame`.

    Returns
    -------
    pd.DataFrame
        Таблица GeoNames с исходными колонками.
    """

    return pd.read_csv(
        GEONAMES_TEXT_PATH,
        sep="\t",
        names=GEONAMES_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )


def build_result_tables(
    region_code_mapping: dict[str, str],
    region_center_frame: pd.DataFrame,
    moscow_reference: dict[str, object],
    place_index: dict[tuple[str, str], list[dict[str, object]]],
    region_name_lists: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строит итоговую таблицу и диагностику по населённым пунктам.

    Parameters
    ----------
    region_code_mapping : dict[str, str]
        Сопоставление `normalized_region -> admin1_code`.
    region_center_frame : pd.DataFrame
        Таблица центров регионов и их координат.
    moscow_reference : dict[str, object]
        Координаты Москвы как общей опорной точки.
    place_index : dict[tuple[str, str], list[dict[str, object]]]
        Exact-индекс объектов GeoNames.
    region_name_lists : dict[str, list[str]]
        Каталог имён GeoNames по регионам.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Итоговая таблица для `result` и диагностическая таблица для `tmp`.
    """

    train_frame = load_unique_train_rows()
    unique_pairs = load_unique_region_settlement_pairs(train_frame)
    diagnostics_frame = build_diagnostics_table(
        unique_pairs=unique_pairs,
        region_code_mapping=region_code_mapping,
        place_index=place_index,
        region_name_lists=region_name_lists,
    )
    expanded_diagnostics = expand_diagnostics_to_new_id(train_frame, diagnostics_frame)
    feature_base_frame = build_feature_base_table(
        expanded_diagnostics=expanded_diagnostics,
        region_center_frame=region_center_frame,
        moscow_reference=moscow_reference,
    )
    result_frame = build_result_feature_table(feature_base_frame=feature_base_frame)
    return result_frame, expanded_diagnostics


def load_unique_train_rows() -> pd.DataFrame:
    """
    Загружает уникальные строки train на уровне `new_id`.

    Returns
    -------
    pd.DataFrame
        Таблица с уникальными магазинами и их географическими признаками.
    """

    return (
        pd.read_csv(TRAIN_PATH, usecols=["new_id", "Регион", "Населенный пункт"])
        .drop_duplicates(subset=["new_id"])
        .sort_values("new_id")
        .reset_index(drop=True)
    )


def load_unique_region_settlement_pairs(train_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Выделяет уникальные пары `Регион + Населенный пункт`.

    Parameters
    ----------
    train_frame : pd.DataFrame
        Таблица уникальных `new_id`.

    Returns
    -------
    pd.DataFrame
        Таблица уникальных пар для сокращения числа матчей.
    """

    return (
        train_frame[["Регион", "Населенный пункт"]]
        .drop_duplicates()
        .sort_values(["Регион", "Населенный пункт"])
        .reset_index(drop=True)
    )


def build_diagnostics_table(
    unique_pairs: pd.DataFrame,
    region_code_mapping: dict[str, str],
    place_index: dict[tuple[str, str], list[dict[str, object]]],
    region_name_lists: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Строит диагностическую таблицу по уникальным парам населённых пунктов.

    Parameters
    ----------
    unique_pairs : pd.DataFrame
        Таблица уникальных пар `Регион + Населенный пункт`.
    region_code_mapping : dict[str, str]
        Сопоставление `normalized_region -> admin1_code`.
    place_index : dict[tuple[str, str], list[dict[str, object]]]
        Exact-индекс объектов GeoNames.
    region_name_lists : dict[str, list[str]]
        Каталог имён GeoNames по регионам.

    Returns
    -------
    pd.DataFrame
        Диагностическая таблица по уникальным парам.
    """

    diagnostics_rows = []
    for _, row in unique_pairs.iterrows():
        resolved_row = resolve_settlement_record(
            region_name=row["Регион"],
            settlement_name=row["Населенный пункт"],
            region_code_mapping=region_code_mapping,
            place_index=place_index,
            region_name_lists=region_name_lists,
        )
        resolved_row["url"] = build_yandex_maps_url(
            longitude=resolved_row["lon"],
            latitude=resolved_row["lat"],
        )
        diagnostics_rows.append(resolved_row)
    return pd.DataFrame(diagnostics_rows)


def build_feature_base_table(
    expanded_diagnostics: pd.DataFrame,
    region_center_frame: pd.DataFrame,
    moscow_reference: dict[str, object],
) -> pd.DataFrame:
    """
    Собирает базовую таблицу координат перед расчётом производных гео-фичей.

    Parameters
    ----------
    expanded_diagnostics : pd.DataFrame
        Диагностика на уровне всех `new_id`.
    region_center_frame : pd.DataFrame
        Таблица центров регионов.
    moscow_reference : dict[str, object]
        Координаты Москвы.

    Returns
    -------
    pd.DataFrame
        Базовая таблица с English-колонками и координатами опорных точек.
    """

    result = expanded_diagnostics.rename(
        columns={
            "Регион": "region",
            "Населенный пункт": "settlement",
            "lat": "settlement_lat",
            "lon": "settlement_lon",
            "url": "yandex_maps_url",
        }
    ).copy()
    result = result.merge(
        region_center_frame, on="admin1_code", how="left", validate="many_to_one"
    )
    result["moscow_name"] = moscow_reference["moscow_name"]
    result["moscow_lat"] = moscow_reference["moscow_lat"]
    result["moscow_lon"] = moscow_reference["moscow_lon"]
    result["region_center_lat"] = result["region_center_lat"].astype(float)
    result["region_center_lon"] = result["region_center_lon"].astype(float)
    result["settlement_lat"] = result["settlement_lat"].astype(float)
    result["settlement_lon"] = result["settlement_lon"].astype(float)
    return result


def build_result_feature_table(feature_base_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Строит финальную таблицу результата с координатами и производными фичами.

    Parameters
    ----------
    feature_base_frame : pd.DataFrame
        Базовая таблица координат населённого пункта, Москвы и центра региона.

    Returns
    -------
    pd.DataFrame
        Финальный result-датасет с English-колонками.
    """

    enriched_frame = add_geospatial_features(feature_base_frame, copy=True)
    ordered_columns = [
        "new_id",
        "region",
        "settlement",
        "settlement_lon",
        "settlement_lat",
        "region_center_name",
        "region_center_lon",
        "region_center_lat",
        "moscow_name",
        "moscow_lon",
        "moscow_lat",
        "distance_to_moscow_m",
        "bearing_to_moscow_deg",
        "bearing_to_moscow_sin",
        "bearing_to_moscow_cos",
        "distance_to_region_center_m",
        "bearing_to_region_center_deg",
        "bearing_to_region_center_sin",
        "bearing_to_region_center_cos",
        "lat_delta_to_moscow_deg",
        "lon_delta_to_moscow_deg",
        "lat_delta_to_region_center_deg",
        "lon_delta_to_region_center_deg",
        "distance_to_moscow_minus_region_center_m",
        "distance_to_moscow_to_region_center_ratio",
        "is_moscow",
        "is_region_center_settlement",
        "yandex_maps_url",
    ]
    return enriched_frame[ordered_columns].copy()


def expand_diagnostics_to_new_id(
    train_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Размножает диагностику по уникальным парам обратно на все `new_id`.

    Parameters
    ----------
    train_frame : pd.DataFrame
        Таблица с уникальными магазинами.
    diagnostics_frame : pd.DataFrame
        Диагностика по уникальным парам `Регион + Населенный пункт`.

    Returns
    -------
    pd.DataFrame
        Диагностика на уровне всех `new_id`.
    """

    return (
        train_frame.merge(
            diagnostics_frame,
            on=["Регион", "Населенный пункт"],
            how="left",
            validate="many_to_one",
        )
        .sort_values("new_id")
        .reset_index(drop=True)
    )


def build_yandex_maps_url(longitude: float, latitude: float) -> str:
    """
    Формирует ссылку на Yandex Maps для ручной проверки координат.

    Parameters
    ----------
    longitude : float
        Долгота точки.
    latitude : float
        Широта точки.

    Returns
    -------
    str
        URL на карту с центром в найденных координатах.
    """

    return f"https://yandex.ru/maps/?ll={longitude:.6f}%2C{latitude:.6f}&z=11"


def save_tables(
    result_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    region_resolution_frame: pd.DataFrame,
) -> None:
    """
    Сохраняет итоговую таблицу и служебные диагностические артефакты.

    Parameters
    ----------
    result_frame : pd.DataFrame
        Итоговая таблица для пользователя.
    diagnostics_frame : pd.DataFrame
        Диагностика по населённым пунктам.
    region_resolution_frame : pd.DataFrame
        Диагностика по регионам.

    Returns
    -------
    None
        Итоговый CSV кладётся в `result`, диагностические файлы — в `tmp`.
    """

    SERVICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(RESULT_PATH, index=False)
    diagnostics_frame.to_csv(DIAGNOSTICS_PATH, index=False)
    region_resolution_frame.to_csv(REGION_RESOLUTION_PATH, index=False)


def print_summary(
    result_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    region_resolution_frame: pd.DataFrame,
) -> None:
    """
    Печатает краткую сводку по собранным артефактам.

    Parameters
    ----------
    result_frame : pd.DataFrame
        Итоговая пользовательская таблица.
    diagnostics_frame : pd.DataFrame
        Диагностика по населённым пунктам.
    region_resolution_frame : pd.DataFrame
        Диагностика по регионам.

    Returns
    -------
    None
        Сводка выводится в стандартный поток.
    """

    settlement_distribution = (
        diagnostics_frame["settlement_match_method"].value_counts().to_dict()
    )
    low_confidence_count = int(
        (diagnostics_frame["settlement_match_score"] < 0.9).sum()
    )
    region_non_exact_count = int(
        (region_resolution_frame["region_match_method"] != "exact").sum()
    )

    print(f"Rows in result: {len(result_frame)}")
    print(f"Settlement match distribution: {settlement_distribution}")
    print(f"Low confidence settlement matches (<0.9): {low_confidence_count}")
    print(f"Non-exact region matches: {region_non_exact_count}")
    print(f"Result path: {RESULT_PATH}")
    print(f"Diagnostics path: {DIAGNOSTICS_PATH}")
    print(f"Region resolution path: {REGION_RESOLUTION_PATH}")
