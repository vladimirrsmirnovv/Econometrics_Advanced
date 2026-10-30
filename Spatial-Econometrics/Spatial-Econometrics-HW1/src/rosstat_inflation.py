"""Шаг 1. Индексы потребительских цен по субъектам: из xlsx Росстата в плоский csv.

Каждый лист книги соответствует одному месяцу и называется `MM(YYYY)`.
Из листа берутся строки субъектов с восьмизначным кодом ОКАТО.
"""

import re

import pandas as pd

from src import paths
from src.names import LATIN_LOOKALIKES

PRICE_COLUMNS = {
    2: "cpi_all_items",
    3: "cpi_food",
    4: "cpi_non_food",
    5: "cpi_services",
}


def read_month_sheet(workbook: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """Читает лист одного месяца и оставляет строки субъектов."""
    sheet = pd.read_excel(workbook, sheet_name, header=None, dtype=str)
    assert sheet.iloc[3, 3].strip() == "продовольственные товары", sheet_name

    territory_code = sheet[0].str.strip()
    is_region = territory_code.str.fullmatch(r"\d{7,8}", na=False)
    regions = sheet[is_region]

    month, year = re.fullmatch(r"(\d{2})\((\d{4})\)", sheet_name).groups()
    table = pd.DataFrame(
        {
            "okato_code": territory_code[is_region].str.zfill(8),
            "region_name": (
                regions[1]
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .str.translate(LATIN_LOOKALIKES)
            ),
            "year": int(year),
            "month": int(month),
        }
    )
    for column_number, column_name in PRICE_COLUMNS.items():
        table[column_name] = pd.to_numeric(regions[column_number])
    return table


def build_consumer_price_index() -> pd.DataFrame:
    """Собирает все месячные листы в одну длинную таблицу."""
    workbook = pd.ExcelFile(paths.INFLATION_RAW)
    month_sheets = [
        name for name in workbook.sheet_names if re.fullmatch(r"\d{2}\(\d{4}\)", name)
    ]
    table = pd.concat([read_month_sheet(workbook, name) for name in month_sheets])
    # В листах разных лет название одного субъекта пишется по-разному
    # («г. Москва» и «г.Москва»), поэтому берется первое написание.
    table["region_name"] = table.groupby("okato_code").region_name.transform("first")
    return table.sort_values(["okato_code", "year", "month"]).reset_index(drop=True)


def check_consumer_price_index(table: pd.DataFrame) -> None:
    """Проверяет полноту и правдоподобие индексов."""
    assert not table.duplicated(["okato_code", "year", "month"]).any()
    months_per_region = table.groupby("okato_code").size()
    assert months_per_region.nunique() == 1, "у субъектов разное число месяцев"
    price_values = table[list(PRICE_COLUMNS.values())]
    assert price_values.notna().all().all(), "есть пропуски в индексах"
    assert price_values.gt(80).all().all() and price_values.lt(130).all().all()


def main() -> None:
    table = build_consumer_price_index()
    check_consumer_price_index(table)
    table.to_csv(paths.INFLATION_CSV, index=False)
    print(
        f"{paths.INFLATION_CSV.name}: {table.okato_code.nunique()} субъектов, "
        f"{len(table)} строк"
    )


if __name__ == "__main__":
    main()
