"""Справочник субъектов и сопоставление регионов разных источников.

Опорный справочник - субъекты из таблицы ИПЦ: у них есть код ОКАТО,
а остальные таблицы Росстата и данные сети содержат только названия.
Шаг 5 конвейера сохраняет соответствие регионов сети кодам ОКАТО.
"""

import pandas as pd

from src import paths
from src.names import is_federal_district, region_key

# В данных сети ХМАО и ЯНАО выделены отдельными регионами, поэтому
# «Тюменская обл» означает юг области без автономных округов. Для
# Архангельской области поступаем так же: Ненецкий АО в сети не представлен.
SHOP_REGION_ALIASES = {
    "Тюменская обл": "Тюменская область без авт. округов",
    "Архангельская обл": "Архангельская область без авт. округа",
}

# ОКТМО автономных округов отличается от ОКАТО, остальные субъекты
# совпадают по первым двум цифрам.
OKTMO_PREFIX_OVERRIDES = {
    "71100000": "718",
    "71140000": "719",
    "11100000": "118",
}


def load_rosstat_regions() -> pd.DataFrame:
    """Возвращает справочник субъектов с кодом ОКАТО и ключом названия."""
    prices = pd.read_csv(paths.INFLATION_CSV, dtype={"okato_code": str})
    regions = prices.drop_duplicates("okato_code")[["okato_code", "region_name"]]
    regions = regions.assign(region_key=regions.region_name.map(region_key))
    assert regions.region_key.is_unique, "ключи субъектов неоднозначны"
    return regions.reset_index(drop=True)


def okato_code_by_name(region_names: pd.Series) -> pd.Series:
    """Находит код ОКАТО для каждого названия субъекта по ключу названия."""
    regions = load_rosstat_regions().set_index("region_key").okato_code
    codes = region_names.map(region_key).map(regions)
    unknown = sorted(set(region_names[codes.isna()]))
    assert not unknown, f"субъекты не найдены в справочнике: {unknown}"
    return codes


def rosstat_region_name(okato_codes: pd.Series) -> pd.Series:
    """Возвращает единое название субъекта из справочника по коду ОКАТО."""
    names = load_rosstat_regions().set_index("okato_code").region_name
    return okato_codes.map(names)


def keep_subjects(table: pd.DataFrame, name_column: str) -> pd.DataFrame:
    """Убирает строки России в целом и федеральных округов."""
    return table[~table[name_column].map(is_federal_district)]


def oktmo_prefix(okato_code: str) -> str:
    """Возвращает префикс ОКТМО, по которому в бюллетене узнается субъект."""
    return OKTMO_PREFIX_OVERRIDES.get(okato_code, okato_code[:2])


def build_shop_regions() -> pd.DataFrame:
    """Сопоставляет каждый регион сети субъекту Росстата."""
    shops = pd.read_csv(paths.SHOPS_RAW, usecols=["Регион"])
    shop_regions = pd.DataFrame({"shop_region": sorted(shops["Регион"].unique())})
    keys = shop_regions.shop_region.map(
        lambda name: region_key(SHOP_REGION_ALIASES.get(name, name))
    )
    rosstat = load_rosstat_regions().set_index("region_key")
    shop_regions["okato_code"] = keys.map(rosstat.okato_code)
    shop_regions["rosstat_region_name"] = keys.map(rosstat.region_name)

    unmatched = shop_regions[shop_regions.okato_code.isna()]
    assert unmatched.empty, f"регионы сети без пары: {unmatched.shop_region.tolist()}"
    assert shop_regions.okato_code.is_unique, "два региона сети на один субъект"
    return shop_regions


def main() -> None:
    shop_regions = build_shop_regions()
    shop_regions.to_csv(paths.REGIONS, index=False)
    print(f"{paths.REGIONS.name}: {len(shop_regions)} регионов сети сопоставлены")


if __name__ == "__main__":
    main()
