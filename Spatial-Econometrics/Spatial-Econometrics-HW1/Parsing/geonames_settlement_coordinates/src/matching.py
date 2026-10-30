from __future__ import annotations

from difflib import SequenceMatcher, get_close_matches

import pandas as pd

from src.constants import (
    EXTRA_INDEX_FEATURE_CODES,
    FEATURE_RANK,
    MANUAL_REGION_CODE_OVERRIDES,
)
from src.normalize import (
    build_region_variants,
    build_settlement_variants,
    iterate_normalized_names,
    normalize_region_name,
    normalize_settlement_name,
)
from src.paths import TRAIN_PATH


def compare_strings(left_value: str, right_value: str) -> float:
    """
    Рассчитывает похожесть двух строк для fuzzy-сопоставления.

    Parameters
    ----------
    left_value : str
        Первая строка.
    right_value : str
        Вторая строка.

    Returns
    -------
    float
        Значение похожести от 0 до 1.
    """

    if left_value == right_value:
        return 1.0
    return max(
        _direct_ratio(left_value, right_value),
        _token_ratio(left_value, right_value),
        _token_overlap_ratio(left_value, right_value) * 0.97,
        _containment_ratio(left_value, right_value) * 0.96,
    )


def _direct_ratio(left_value: str, right_value: str) -> float:
    """
    Считает прямое сходство двух строк.

    Parameters
    ----------
    left_value : str
        Первая строка.
    right_value : str
        Вторая строка.

    Returns
    -------
    float
        Значение `SequenceMatcher.ratio()`.
    """

    return SequenceMatcher(None, left_value, right_value).ratio()


def _token_ratio(left_value: str, right_value: str) -> float:
    """
    Считает сходство строк после сортировки токенов.

    Parameters
    ----------
    left_value : str
        Первая строка.
    right_value : str
        Вторая строка.

    Returns
    -------
    float
        Сходство строк с отсортированными токенами.
    """

    left_sorted = " ".join(sorted(left_value.split()))
    right_sorted = " ".join(sorted(right_value.split()))
    return SequenceMatcher(None, left_sorted, right_sorted).ratio()


def _token_overlap_ratio(left_value: str, right_value: str) -> float:
    """
    Считает долю пересечения множеств токенов.

    Parameters
    ----------
    left_value : str
        Первая строка.
    right_value : str
        Вторая строка.

    Returns
    -------
    float
        Отношение пересечения токенов к их объединению.
    """

    left_tokens = set(left_value.split())
    right_tokens = set(right_value.split())
    union_size = len(left_tokens | right_tokens)
    if union_size == 0:
        return 0.0
    return len(left_tokens & right_tokens) / union_size


def _containment_ratio(left_value: str, right_value: str) -> float:
    """
    Считает степень вложенности одной строки в другую.

    Parameters
    ----------
    left_value : str
        Первая строка.
    right_value : str
        Вторая строка.

    Returns
    -------
    float
        Доля длины короткой строки к длинной, если одна строка входит в другую.
    """

    if left_value not in right_value and right_value not in left_value:
        return 0.0
    shorter_length = min(len(left_value), len(right_value))
    longer_length = max(len(left_value), len(right_value))
    return shorter_length / longer_length


def choose_best_geonames_record(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    """
    Выбирает лучшую запись GeoNames из списка кандидатов.

    Parameters
    ----------
    candidates : list[dict[str, object]]
        Кандидатные записи GeoNames.

    Returns
    -------
    dict[str, object]
        Лучшая запись по приоритету типа объекта, населению и `geonameid`.
    """

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["feature_rank"],
            -candidate["population_value"],
            int(candidate["geonameid"]),
        ),
    )[0]


def build_region_resolution_table(
    geonames_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Сопоставляет регионы из train с кодами `admin1` GeoNames.

    Parameters
    ----------
    geonames_frame : pd.DataFrame
        Полный GeoNames-датасет по России.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        Диагностика по регионам и словарь `normalized_region -> admin1_code`.
    """

    admin1_rows = geonames_frame[
        (geonames_frame["feature_class"] == "A")
        & (geonames_frame["feature_code"] == "ADM1")
    ].copy()
    region_alias_frame = _build_region_alias_frame(admin1_rows)
    region_exact_index = _build_region_exact_index(region_alias_frame)
    unique_regions = _load_unique_regions()

    resolutions = []
    region_code_mapping: dict[str, str] = {}
    for region_name in unique_regions["Регион"]:
        resolution = _resolve_region_name(
            region_name, region_alias_frame, region_exact_index
        )
        region_code_mapping[resolution["normalized_region"]] = resolution["admin1_code"]
        resolutions.append(resolution)

    region_resolution_frame = (
        pd.DataFrame(resolutions).sort_values("Регион").reset_index(drop=True)
    )
    return region_resolution_frame, region_code_mapping


def _build_region_alias_frame(admin1_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Собирает таблицу алиасов регионов из записей `ADM1`.

    Parameters
    ----------
    admin1_rows : pd.DataFrame
        Строки GeoNames только для административных единиц первого уровня.

    Returns
    -------
    pd.DataFrame
        Таблица с полями `admin1_code`, `region_display_name`, `region_alias`.
    """

    region_alias_rows: list[dict[str, object]] = []
    for admin1_row in admin1_rows.itertuples(index=False):
        aliases = set(
            iterate_normalized_names(
                primary_name=admin1_row.name,
                ascii_name=admin1_row.asciiname,
                alternate_names=admin1_row.alternatenames,
                is_region=True,
            )
        )
        for alias in aliases:
            region_alias_rows.append(
                {
                    "admin1_code": admin1_row.admin1_code,
                    "region_display_name": admin1_row.name,
                    "region_alias": alias,
                }
            )
    return pd.DataFrame(region_alias_rows)


def _build_region_exact_index(region_alias_frame: pd.DataFrame) -> dict[str, set[str]]:
    """
    Строит exact-индекс региональных алиасов.

    Parameters
    ----------
    region_alias_frame : pd.DataFrame
        Таблица алиасов регионов.

    Returns
    -------
    dict[str, set[str]]
        Индекс `alias -> множество admin1_code`.
    """

    region_exact_index: dict[str, set[str]] = {}
    for alias_row in region_alias_frame.itertuples(index=False):
        region_exact_index.setdefault(alias_row.region_alias, set()).add(
            alias_row.admin1_code
        )
    return region_exact_index


def _load_unique_regions() -> pd.DataFrame:
    """
    Загружает уникальные значения региона из train.

    Returns
    -------
    pd.DataFrame
        Таблица с уникальными регионами.
    """

    return (
        pd.read_csv(TRAIN_PATH, usecols=["Регион"])
        .drop_duplicates()
        .sort_values("Регион")
        .reset_index(drop=True)
    )


def _resolve_region_name(
    region_name: str,
    region_alias_frame: pd.DataFrame,
    region_exact_index: dict[str, set[str]],
) -> dict[str, object]:
    """
    Сопоставляет одно название региона с `admin1_code`.

    Parameters
    ----------
    region_name : str
        Название региона из train.
    region_alias_frame : pd.DataFrame
        Таблица региональных алиасов.
    region_exact_index : dict[str, set[str]]
        Exact-индекс `alias -> admin1_code`.

    Returns
    -------
    dict[str, object]
        Диагностическая запись по сопоставлению региона.
    """

    normalized_region = normalize_region_name(region_name)
    if normalized_region in MANUAL_REGION_CODE_OVERRIDES:
        return _build_manual_region_resolution(
            region_name=region_name,
            normalized_region=normalized_region,
            region_alias_frame=region_alias_frame,
        )

    exact_codes = _collect_exact_region_codes(region_name, region_exact_index)
    if len(exact_codes) == 1:
        return _build_exact_region_resolution(
            region_name=region_name,
            normalized_region=normalized_region,
            admin1_code=next(iter(exact_codes)),
            region_alias_frame=region_alias_frame,
        )
    return _build_fuzzy_region_resolution(
        region_name=region_name,
        normalized_region=normalized_region,
        region_alias_frame=region_alias_frame,
    )


def _build_manual_region_resolution(
    region_name: str,
    normalized_region: str,
    region_alias_frame: pd.DataFrame,
) -> dict[str, object]:
    """
    Возвращает ручное сопоставление региона.

    Parameters
    ----------
    region_name : str
        Исходное название региона.
    normalized_region : str
        Нормализованное название региона.
    region_alias_frame : pd.DataFrame
        Таблица региональных алиасов.

    Returns
    -------
    dict[str, object]
        Диагностическая запись с ручным сопоставлением.
    """

    admin1_code = MANUAL_REGION_CODE_OVERRIDES[normalized_region]
    matched_rows = region_alias_frame[
        region_alias_frame["admin1_code"] == admin1_code
    ].copy()
    return {
        "Регион": region_name,
        "normalized_region": normalized_region,
        "admin1_code": admin1_code,
        "matched_region_name": matched_rows["region_display_name"].iloc[0],
        "matched_region_alias": sorted(matched_rows["region_alias"].tolist())[0],
        "region_match_method": "manual",
        "region_match_score": 1.0,
    }


def _collect_exact_region_codes(
    region_name: str,
    region_exact_index: dict[str, set[str]],
) -> set[str]:
    """
    Собирает множество exact-кандидатов `admin1_code` для региона.

    Parameters
    ----------
    region_name : str
        Название региона из train.
    region_exact_index : dict[str, set[str]]
        Exact-индекс `alias -> admin1_code`.

    Returns
    -------
    set[str]
        Множество кодов `admin1`, найденных по региональным вариантам.
    """

    region_variants = build_region_variants(region_name)
    return {
        code
        for variant in region_variants
        for code in region_exact_index.get(variant, set())
    }


def _build_exact_region_resolution(
    region_name: str,
    normalized_region: str,
    admin1_code: str,
    region_alias_frame: pd.DataFrame,
) -> dict[str, object]:
    """
    Возвращает exact-сопоставление региона.

    Parameters
    ----------
    region_name : str
        Исходное название региона.
    normalized_region : str
        Нормализованное название региона.
    admin1_code : str
        Найденный код `admin1`.
    region_alias_frame : pd.DataFrame
        Таблица региональных алиасов.

    Returns
    -------
    dict[str, object]
        Диагностическая запись exact-матчинга.
    """

    matched_rows = region_alias_frame[
        region_alias_frame["admin1_code"] == admin1_code
    ].copy()
    return {
        "Регион": region_name,
        "normalized_region": normalized_region,
        "admin1_code": admin1_code,
        "matched_region_name": matched_rows["region_display_name"].iloc[0],
        "matched_region_alias": sorted(matched_rows["region_alias"].tolist())[0],
        "region_match_method": "exact",
        "region_match_score": 1.0,
    }


def _build_fuzzy_region_resolution(
    region_name: str,
    normalized_region: str,
    region_alias_frame: pd.DataFrame,
) -> dict[str, object]:
    """
    Возвращает fuzzy-сопоставление региона.

    Parameters
    ----------
    region_name : str
        Исходное название региона.
    normalized_region : str
        Нормализованное название региона.
    region_alias_frame : pd.DataFrame
        Таблица региональных алиасов.

    Returns
    -------
    dict[str, object]
        Диагностическая запись fuzzy-матчинга.
    """

    alias_candidates = get_close_matches(
        normalized_region,
        region_alias_frame["region_alias"].tolist(),
        n=10,
        cutoff=0.0,
    )
    scored_candidates = []
    for alias_candidate in alias_candidates:
        matching_row = region_alias_frame[
            region_alias_frame["region_alias"] == alias_candidate
        ].iloc[0]
        scored_candidates.append(
            {
                "admin1_code": matching_row["admin1_code"],
                "matched_region_name": matching_row["region_display_name"],
                "matched_region_alias": alias_candidate,
                "region_match_score": compare_strings(
                    normalized_region, alias_candidate
                ),
            }
        )

    best_candidate = sorted(
        scored_candidates,
        key=lambda candidate: (
            -candidate["region_match_score"],
            candidate["matched_region_name"],
            candidate["admin1_code"],
        ),
    )[0]
    return {
        "Регион": region_name,
        "normalized_region": normalized_region,
        "admin1_code": best_candidate["admin1_code"],
        "matched_region_name": best_candidate["matched_region_name"],
        "matched_region_alias": best_candidate["matched_region_alias"],
        "region_match_method": "fuzzy",
        "region_match_score": best_candidate["region_match_score"],
    }


def build_place_index(
    geonames_frame: pd.DataFrame,
) -> tuple[dict[tuple[str, str], list[dict[str, object]]], dict[str, list[str]]]:
    """
    Строит индекс объектов GeoNames по региону и нормализованному имени.

    Parameters
    ----------
    geonames_frame : pd.DataFrame
        Полный GeoNames-датасет по России.

    Returns
    -------
    tuple[dict[tuple[str, str], list[dict[str, object]]], dict[str, list[str]]]
        Exact-индекс и каталог доступных имён по каждому региону.
    """

    indexed_rows = geonames_frame[
        (geonames_frame["feature_class"] == "P")
        | (geonames_frame["feature_code"].isin(EXTRA_INDEX_FEATURE_CODES))
    ].copy()

    place_index: dict[tuple[str, str], list[dict[str, object]]] = {}
    region_name_catalog: dict[str, set[str]] = {}
    for indexed_row in indexed_rows.itertuples(index=False):
        record = _build_geonames_record(indexed_row)
        normalized_names = set(
            iterate_normalized_names(
                primary_name=indexed_row.name,
                ascii_name=indexed_row.asciiname,
                alternate_names=indexed_row.alternatenames,
                is_region=False,
            )
        )
        for normalized_name in normalized_names:
            key = (indexed_row.admin1_code, normalized_name)
            place_index.setdefault(key, []).append(record)
            region_name_catalog.setdefault(indexed_row.admin1_code, set()).add(
                normalized_name
            )

    region_name_lists = {
        admin1_code: sorted(region_names)
        for admin1_code, region_names in region_name_catalog.items()
    }
    return place_index, region_name_lists


def _build_geonames_record(indexed_row: pd.Series) -> dict[str, object]:
    """
    Преобразует строку GeoNames в компактную словарную запись.

    Parameters
    ----------
    indexed_row : pd.Series
        Строка GeoNames.

    Returns
    -------
    dict[str, object]
        Словарь с полями, нужными для выбора лучшего кандидата.
    """

    return {
        "geonameid": indexed_row.geonameid,
        "name": indexed_row.name,
        "latitude": float(indexed_row.latitude),
        "longitude": float(indexed_row.longitude),
        "feature_code": indexed_row.feature_code,
        "feature_rank": FEATURE_RANK.get(indexed_row.feature_code, 100),
        "population_value": int(indexed_row.population or 0),
        "admin1_code": indexed_row.admin1_code,
    }


def resolve_settlement_record(
    region_name: str,
    settlement_name: str,
    region_code_mapping: dict[str, str],
    place_index: dict[tuple[str, str], list[dict[str, object]]],
    region_name_lists: dict[str, list[str]],
) -> dict[str, object]:
    """
    Подбирает координаты GeoNames для одной пары `регион + населённый пункт`.

    Parameters
    ----------
    region_name : str
        Регион из train.
    settlement_name : str
        Населённый пункт из train.
    region_code_mapping : dict[str, str]
        Сопоставление `normalized_region -> admin1_code`.
    place_index : dict[tuple[str, str], list[dict[str, object]]]
        Exact-индекс объектов GeoNames.
    region_name_lists : dict[str, list[str]]
        Каталог доступных имён по каждому региону.

    Returns
    -------
    dict[str, object]
        Диагностическая запись с выбранными координатами и метаданными матчинга.
    """

    normalized_region = normalize_region_name(region_name)
    normalized_settlement = normalize_settlement_name(settlement_name)
    admin1_code = region_code_mapping[normalized_region]
    exact_variants = build_settlement_variants(
        settlement_name,
        normalized_region_name=normalized_region,
    )

    exact_match = _find_exact_settlement_match(admin1_code, exact_variants, place_index)
    if exact_match is not None:
        matched_name, best_record = exact_match
        return _build_settlement_result(
            region_name=region_name,
            settlement_name=settlement_name,
            normalized_region=normalized_region,
            normalized_settlement=normalized_settlement,
            admin1_code=admin1_code,
            matched_name=matched_name,
            best_record=best_record,
            match_method="exact",
            match_score=1.0,
        )

    fuzzy_match = _find_fuzzy_settlement_match(
        admin1_code=admin1_code,
        normalized_settlement=normalized_settlement,
        exact_variants=exact_variants,
        place_index=place_index,
        region_name_lists=region_name_lists,
    )
    return _build_settlement_result(
        region_name=region_name,
        settlement_name=settlement_name,
        normalized_region=normalized_region,
        normalized_settlement=normalized_settlement,
        admin1_code=admin1_code,
        matched_name=fuzzy_match["matched_name"],
        best_record=fuzzy_match["record"],
        match_method="fuzzy",
        match_score=fuzzy_match["match_score"],
    )


def _find_exact_settlement_match(
    admin1_code: str,
    exact_variants: set[str],
    place_index: dict[tuple[str, str], list[dict[str, object]]],
) -> tuple[str, dict[str, object]] | None:
    """
    Ищет exact-кандидата среди всех вариантов названия населённого пункта.

    Parameters
    ----------
    admin1_code : str
        Код региона `admin1`.
    exact_variants : set[str]
        Варианты названия населённого пункта.
    place_index : dict[tuple[str, str], list[dict[str, object]]]
        Exact-индекс GeoNames.

    Returns
    -------
    tuple[str, dict[str, object]] | None
        Найденный алиас и лучшая запись GeoNames, либо `None`.
    """

    candidate_records: dict[str, dict[str, object]] = {}
    matched_name = None
    for exact_variant in sorted(exact_variants):
        for candidate_record in place_index.get((admin1_code, exact_variant), []):
            candidate_records[candidate_record["geonameid"]] = candidate_record
            matched_name = exact_variant
    if not candidate_records or matched_name is None:
        return None
    return matched_name, choose_best_geonames_record(list(candidate_records.values()))


def _find_fuzzy_settlement_match(
    admin1_code: str,
    normalized_settlement: str,
    exact_variants: set[str],
    place_index: dict[tuple[str, str], list[dict[str, object]]],
    region_name_lists: dict[str, list[str]],
) -> dict[str, object]:
    """
    Ищет fuzzy-кандидата среди доступных имён внутри региона.

    Parameters
    ----------
    admin1_code : str
        Код региона `admin1`.
    normalized_settlement : str
        Основной нормализованный вариант названия населённого пункта.
    exact_variants : set[str]
        Все варианты названия для сравнения.
    place_index : dict[tuple[str, str], list[dict[str, object]]]
        Exact-индекс GeoNames.
    region_name_lists : dict[str, list[str]]
        Каталог имён по регионам.

    Returns
    -------
    dict[str, object]
        Лучший fuzzy-кандидат с оценкой похожести и выбранной записью GeoNames.
    """

    candidate_names = region_name_lists.get(admin1_code, [])
    fuzzy_candidates = get_close_matches(
        normalized_settlement,
        candidate_names,
        n=10,
        cutoff=0.0,
    )
    if not fuzzy_candidates and candidate_names:
        fuzzy_candidates = candidate_names[:10]

    scored_candidates = []
    for fuzzy_candidate in fuzzy_candidates:
        best_candidate_record = choose_best_geonames_record(
            place_index[(admin1_code, fuzzy_candidate)]
        )
        match_score = max(
            compare_strings(exact_variant, fuzzy_candidate)
            for exact_variant in exact_variants
        )
        scored_candidates.append(
            {
                "matched_name": fuzzy_candidate,
                "match_score": match_score,
                "record": best_candidate_record,
            }
        )

    return sorted(
        scored_candidates,
        key=lambda candidate: (
            -candidate["match_score"],
            candidate["record"]["feature_rank"],
            -candidate["record"]["population_value"],
            candidate["record"]["geonameid"],
        ),
    )[0]


def _build_settlement_result(
    region_name: str,
    settlement_name: str,
    normalized_region: str,
    normalized_settlement: str,
    admin1_code: str,
    matched_name: str,
    best_record: dict[str, object],
    match_method: str,
    match_score: float,
) -> dict[str, object]:
    """
    Формирует итоговую диагностическую запись для населённого пункта.

    Parameters
    ----------
    region_name : str
        Исходное название региона.
    settlement_name : str
        Исходное название населённого пункта.
    normalized_region : str
        Нормализованное название региона.
    normalized_settlement : str
        Нормализованное название населённого пункта.
    admin1_code : str
        Код региона `admin1`.
    matched_name : str
        Нормализованный алиас GeoNames, по которому найден кандидат.
    best_record : dict[str, object]
        Выбранная запись GeoNames.
    match_method : str
        Метод сопоставления: `exact` или `fuzzy`.
    match_score : float
        Оценка качества матчинга.

    Returns
    -------
    dict[str, object]
        Полная диагностическая запись по населённому пункту.
    """

    return {
        "Регион": region_name,
        "Населенный пункт": settlement_name,
        "admin1_code": admin1_code,
        "normalized_region": normalized_region,
        "normalized_settlement": normalized_settlement,
        "matched_geonameid": best_record["geonameid"],
        "matched_geoname_name": best_record["name"],
        "matched_geoname_alias": matched_name,
        "feature_code": best_record["feature_code"],
        "population": best_record["population_value"],
        "lon": best_record["longitude"],
        "lat": best_record["latitude"],
        "settlement_match_method": match_method,
        "settlement_match_score": match_score,
    }
