"""Шаг 4. Численность городских населенных пунктов на 1 января 2024 года.

Лист `Численность_по_МО` содержит вперемешку субъекты, муниципальные
образования, населенные пункты и внутригородские районы. Населенный пункт
узнается по коду ТЕРСОН-МО длиннее десяти знаков и типу в названии
(«г Майкоп», «пгт Энем»). Субъект определяется по блоку листа, в котором
стоит строка, а префикс кода служит перекрестной проверкой. Города
федерального значения берутся из строк субъектов целиком. Покинутые
поселки с нулевой численностью (Кадыкчан и подобные) исключаются.
"""

import pandas as pd

from src import paths
from src.names import (
    region_key,
    settlement_base_name,
    settlement_type_class,
    split_settlement,
)
from src.regions import load_rosstat_regions

FEDERAL_CITY_CODES = {
    "4500000000": "Москва",
    "4000000000": "Санкт-Петербург",
    "6700000000": "Севастополь",
}

# Префикс ОКТМО -> код ОКАТО субъекта. Юг Тюменской и Архангельская область
# без Ненецкого АО получают коды территорий без автономных округов.
SPECIAL_PREFIXES = {
    "718": "71100000",
    "719": "71140000",
    "118": "11100000",
    "71": "71001000",
    "11": "11001000",
}


# Строка субъекта имеет код вида `XX000000YZ`, где YZ - служебные цифры.
# Автономные округа открываются собственными строками с особыми кодами.
REGION_HEADER_PATTERN = r"\d{2}0{6}\d{2}"
AUTONOMOUS_OKRUG_HEADERS = {"7180000000", "7190000010", "1180010"}

# Опечатки в кодах бюллетеня: префикс кода указывает на чужой субъект.
# пгт Таежный стоит в блоке ХМАО, в коде 728 вместо 718.
KNOWN_CODE_TYPOS = {"728241590511200"}


def region_okato_code(oktmo_code: str) -> str:
    """Определяет код ОКАТО субъекта по коду ТЕРСОН-МО строки."""
    return SPECIAL_PREFIXES.get(
        oktmo_code[:3], SPECIAL_PREFIXES.get(oktmo_code[:2], oktmo_code[:2] + "000000")
    )


def read_bulletin() -> pd.DataFrame:
    """Читает лист бюллетеня и очищает коды от пробелов."""
    sheet = pd.read_excel(
        paths.POPULATION_RAW, "Численность_по_МО", header=None, dtype=str
    )
    sheet.columns = [
        "oktmo_code",
        "name",
        "population_total",
        "population_urban",
        "population_rural",
    ]
    sheet["oktmo_code"] = sheet.oktmo_code.fillna("").str.replace(r"\s", "", regex=True)
    sheet["name"] = sheet.name.fillna("").str.strip()
    sheet = sheet[sheet.oktmo_code.str.fullmatch(r"\d+")].copy()
    sheet["is_region_header"] = sheet.oktmo_code.str.fullmatch(
        REGION_HEADER_PATTERN
    ) | sheet.oktmo_code.isin(AUTONOMOUS_OKRUG_HEADERS)
    is_header = sheet.is_region_header
    sheet["block_okato_code"] = (
        sheet.oktmo_code.where(is_header).ffill().map(region_okato_code)
    )
    return sheet


def parse_type(name: str) -> tuple[str, str] | None:
    """Возвращает название и тип или None, если строка не населенный пункт."""
    try:
        return split_settlement(name)
    except ValueError:
        return None


def check_region_headers(sheet: pd.DataFrame) -> None:
    """Сверяет названия субъектов бюллетеня со справочником по префиксу кода."""
    headers = sheet[sheet.is_region_header]
    headers = headers[~headers.name.str.contains("включая|федерального значения")]
    assert len(headers) == len(set(headers.block_okato_code)), (
        "субъект встречается дважды"
    )
    directory = load_rosstat_regions().set_index("okato_code").region_key
    for code, name in zip(headers.oktmo_code, headers.name):
        okato_code = region_okato_code(code)
        assert directory[okato_code] == region_key(name), f"{code}: {name}"


def settlement_rows(sheet: pd.DataFrame) -> pd.DataFrame:
    """Оставляет строки городских населенных пунктов с кодом длиннее 10 знаков."""
    candidates = sheet[sheet.oktmo_code.str.len() > 10]
    parsed = candidates.name.map(parse_type)
    settlements = candidates[parsed.notna()].copy()
    settlements["settlement_name"] = parsed.dropna().str[0]
    settlements["settlement_type"] = parsed.dropna().str[1]
    settlements["is_federal_city"] = False
    return settlements


def federal_city_rows(sheet: pd.DataFrame) -> pd.DataFrame:
    """Превращает строки городов федерального значения в населенные пункты."""
    cities = sheet[sheet.oktmo_code.isin(FEDERAL_CITY_CODES)].copy()
    cities["settlement_name"] = cities.oktmo_code.map(FEDERAL_CITY_CODES).str.lower()
    cities["settlement_type"] = "г"
    cities["is_federal_city"] = True
    return cities


def build_population() -> pd.DataFrame:
    """Собирает таблицу численности городских населенных пунктов."""
    sheet = read_bulletin()
    check_region_headers(sheet)
    table = pd.concat([settlement_rows(sheet), federal_city_rows(sheet)])

    table["region_okato_code"] = table.block_okato_code
    code_region = table.oktmo_code.map(region_okato_code)
    conflicts = table[code_region.ne(table.block_okato_code)].oktmo_code
    assert set(conflicts) == KNOWN_CODE_TYPOS, (
        f"новые расхождения кода и блока: {set(conflicts)}"
    )
    names = load_rosstat_regions().set_index("okato_code").region_name
    table["region_name"] = table.region_okato_code.map(names)
    table["settlement_base_name"] = table.settlement_name.map(settlement_base_name)
    table["settlement_type_class"] = table.settlement_type.map(settlement_type_class)
    for column in ["population_total", "population_urban", "population_rural"]:
        table[column] = pd.to_numeric(table[column]).astype("Int64")

    columns = [
        "oktmo_code",
        "region_okato_code",
        "region_name",
        "name",
        "settlement_name",
        "settlement_type",
        "settlement_type_class",
        "settlement_base_name",
        "is_federal_city",
        "population_total",
        "population_urban",
        "population_rural",
    ]
    table = table[table.population_total.gt(0)]
    return table[columns].rename(columns={"name": "source_name"}).reset_index(drop=True)


def check_population(table: pd.DataFrame) -> None:
    """Проверяет коды, субъекты и численность."""
    assert table.region_name.notna().all(), "строки без субъекта"
    assert not table.oktmo_code.duplicated().any()
    assert table.population_total.gt(0).all()
    assert table.settlement_type_class.isin(
        ["город", "поселок городского типа", "поселок", "населенный пункт"]
    ).all()


def main() -> None:
    table = build_population()
    check_population(table)
    table.to_csv(paths.POPULATION_CSV, index=False)
    print(
        f"{paths.POPULATION_CSV.name}: {len(table)} населенных пунктов, "
        f"{table.region_okato_code.nunique()} субъектов"
    )


if __name__ == "__main__":
    main()
