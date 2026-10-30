from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from src.constants import (
    MANUAL_REGION_ALIASES,
    MANUAL_SETTLEMENT_ALIASES,
    REGION_ABBREVIATIONS,
    REGION_GENERIC_TOKENS,
    SETTLEMENT_EDGE_TOKENS,
    SETTLEMENT_SPECIAL_PATTERNS,
)


def normalize_base_text(value: str) -> str:
    """
    Приводит строку к базовой очищенной форме для последующего сопоставления.

    Parameters
    ----------
    value : str
        Исходный текст.

    Returns
    -------
    str
        Нормализованная строка в нижнем регистре без лишней пунктуации и скобок.
    """

    normalized_value = unicodedata.normalize("NFKC", str(value)).lower().strip()
    normalized_value = normalized_value.replace("ё", "е").replace("’", "'")
    normalized_value = re.sub(r"\([^)]*\)", " ", normalized_value)
    normalized_value = normalized_value.replace("-", " ")
    normalized_value = normalized_value.replace(".", " ")
    normalized_value = normalized_value.replace("'", " ")
    normalized_value = re.sub(r"[^0-9a-zа-я ]+", " ", normalized_value)
    normalized_value = re.sub(r"\s+", " ", normalized_value).strip()
    return normalized_value


def normalize_region_name(value: str) -> str:
    """
    Нормализует название региона и разворачивает типовые сокращения.

    Parameters
    ----------
    value : str
        Исходное название региона.

    Returns
    -------
    str
        Нормализованное название региона.
    """

    normalized_value = f" {normalize_base_text(value)} "
    for source, target in REGION_ABBREVIATIONS.items():
        normalized_value = normalized_value.replace(source, target)
    normalized_value = re.sub(r"\s+", " ", normalized_value).strip()
    return MANUAL_REGION_ALIASES.get(normalized_value, normalized_value)


def build_region_variants(value: str) -> set[str]:
    """
    Строит варианты нормализованного названия региона.

    Parameters
    ----------
    value : str
        Исходное название региона.

    Returns
    -------
    set[str]
        Набор вариантов названия региона для exact- и fuzzy-матчинга.
    """

    normalized_value = normalize_region_name(value)
    compact_tokens = [
        token
        for token in normalized_value.split()
        if token not in REGION_GENERIC_TOKENS
    ]
    variants = {normalized_value}
    if compact_tokens:
        variants.add(" ".join(compact_tokens))
    return {variant for variant in variants if variant}


def normalize_settlement_name(value: str, strip_edge_tokens: bool = True) -> str:
    """
    Нормализует название населённого пункта.

    Parameters
    ----------
    value : str
        Исходное название населённого пункта.
    strip_edge_tokens : bool, default=True
        Нужно ли убирать служебные типы населённых пунктов по краям строки.

    Returns
    -------
    str
        Нормализованное название населённого пункта.
    """

    normalized_value = normalize_base_text(value)
    for pattern, replacement in SETTLEMENT_SPECIAL_PATTERNS:
        normalized_value = pattern.sub(replacement, normalized_value)
    normalized_value = re.sub(r"\bим\b", " имени ", normalized_value)
    normalized_value = re.sub(r"\s+", " ", normalized_value).strip()

    tokens = normalized_value.split()
    if strip_edge_tokens:
        tokens = _strip_edge_tokens(tokens)
    return " ".join(tokens)


def _strip_edge_tokens(tokens: list[str]) -> list[str]:
    """
    Удаляет служебные токены типа населённого пункта по краям строки.

    Parameters
    ----------
    tokens : list[str]
        Список токенов исходного названия.

    Returns
    -------
    list[str]
        Список токенов без краевых служебных обозначений.
    """

    normalized_tokens = tokens[:]
    while len(normalized_tokens) > 1 and normalized_tokens[0] in SETTLEMENT_EDGE_TOKENS:
        normalized_tokens = normalized_tokens[1:]
    while (
        len(normalized_tokens) > 1 and normalized_tokens[-1] in SETTLEMENT_EDGE_TOKENS
    ):
        normalized_tokens = normalized_tokens[:-1]
    return normalized_tokens


def build_settlement_variants(
    value: str,
    normalized_region_name: str | None = None,
) -> set[str]:
    """
    Строит варианты нормализованного названия населённого пункта.

    Parameters
    ----------
    value : str
        Исходное название населённого пункта.
    normalized_region_name : str | None, default=None
        Нормализованное название региона для применения точечных алиасов.

    Returns
    -------
    set[str]
        Набор вариантов названия для поиска в GeoNames.
    """

    variants = {
        normalize_settlement_name(value, strip_edge_tokens=True),
        normalize_settlement_name(value, strip_edge_tokens=False),
    }
    variants.update(_build_service_stripped_variants(variants))
    variants.update(_build_named_object_variants(variants))

    if normalized_region_name is not None:
        normalized_value = normalize_settlement_name(value, strip_edge_tokens=True)
        variants.update(
            MANUAL_SETTLEMENT_ALIASES.get(
                (normalized_region_name, normalized_value), []
            )
        )

    return {variant for variant in variants if variant}


def _build_service_stripped_variants(variants: Iterable[str]) -> set[str]:
    """
    Строит дополнительные варианты названия без служебных токенов внутри строки.

    Parameters
    ----------
    variants : Iterable[str]
        Уже подготовленные варианты названия.

    Returns
    -------
    set[str]
        Дополнительные варианты после удаления служебных слов.
    """

    service_tokens = SETTLEMENT_EDGE_TOKENS | {"железнодорожная", "имени", "район"}
    stripped_variants = set()
    for variant in variants:
        tokens = [token for token in variant.split() if token not in service_tokens]
        if tokens:
            stripped_variants.add(" ".join(tokens))
    return stripped_variants


def _build_named_object_variants(variants: Iterable[str]) -> set[str]:
    """
    Строит сокращённые варианты для названий с конструкцией `имени`.

    Parameters
    ----------
    variants : Iterable[str]
        Текущие варианты названия.

    Returns
    -------
    set[str]
        Варианты, обрезанные до части перед словом `имени`.
    """

    named_variants = set()
    for variant in variants:
        tokens = variant.split()
        if "имени" not in tokens:
            continue
        marker_index = tokens.index("имени")
        if marker_index > 0:
            named_variants.add(" ".join(tokens[:marker_index]))
    return named_variants


def split_alternate_names(alternate_names: str) -> list[str]:
    """
    Разбивает строку `alternatenames` из GeoNames на отдельные значения.

    Parameters
    ----------
    alternate_names : str
        Содержимое поля `alternatenames`.

    Returns
    -------
    list[str]
        Список непустых альтернативных названий.
    """

    if not alternate_names:
        return []
    return [part for part in alternate_names.split(",") if part]


def iterate_normalized_names(
    primary_name: str,
    ascii_name: str,
    alternate_names: str,
    is_region: bool,
) -> Iterable[str]:
    """
    Возвращает поток нормализованных вариантов названий записи GeoNames.

    Parameters
    ----------
    primary_name : str
        Основное имя записи.
    ascii_name : str
        ASCII-вариант имени.
    alternate_names : str
        Поле `alternatenames` из GeoNames.
    is_region : bool
        Флаг выбора логики нормализации для региона или населённого пункта.

    Returns
    -------
    Iterable[str]
        Последовательность нормализованных вариантов названия.
    """

    raw_parts = [primary_name, ascii_name, *split_alternate_names(alternate_names)]
    for part in raw_parts:
        if not part:
            continue
        if is_region:
            yield from build_region_variants(part)
        else:
            yield from build_settlement_variants(part)
