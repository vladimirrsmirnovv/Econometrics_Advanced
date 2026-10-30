"""Шаг 2. Номинальная зарплата по субъектам по месяцам: из xlsx в плоский csv.

На листе `с 2019` строка 2 содержит годы (только в первой колонке года),
строка 3 - названия месяцев, данные начинаются с 4-й строки.
"""

import re

import pandas as pd

from src import paths
from src.names import clean_text
from src.regions import keep_subjects, okato_code_by_name, rosstat_region_name

MONTH_NUMBERS = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}


def month_number(header: str) -> int:
    """Достает номер месяца из заголовка вида `декабрь1)`."""
    return MONTH_NUMBERS[re.match(r"[а-я]+", clean_text(header)).group()]


def read_wage_sheet() -> pd.DataFrame:
    """Читает лист и превращает широкую таблицу в длинную «субъект - месяц»."""
    sheet = pd.read_excel(paths.WAGES_RAW, "с 2019", header=None, dtype=str)
    years = sheet.iloc[1, 1:].str.extract(r"(\d{4})")[0].ffill().astype(int)
    months = sheet.iloc[2, 1:].map(month_number)

    body = sheet.iloc[3:].dropna(subset=[1])
    body = body.assign(
        region_name=body[0].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    body = keep_subjects(body, "region_name")
    body = body[~body.region_name.str.fullmatch(r"(?i)в том числе:?")]

    long_table = body.melt(
        id_vars="region_name",
        value_vars=list(sheet.columns[1:]),
        var_name="column",
        value_name="nominal_wage_rub",
    )
    long_table["year"] = long_table.column.map(years)
    long_table["month"] = long_table.column.map(months)
    long_table["nominal_wage_rub"] = pd.to_numeric(long_table.nominal_wage_rub)
    return long_table.drop(columns="column").dropna(subset=["nominal_wage_rub"])


def build_wages() -> pd.DataFrame:
    """Добавляет код ОКАТО и упорядочивает таблицу."""
    table = read_wage_sheet()
    table.insert(0, "okato_code", okato_code_by_name(table.region_name))
    table["region_name"] = rosstat_region_name(table.okato_code)
    columns = ["okato_code", "region_name", "year", "month", "nominal_wage_rub"]
    return (
        table[columns]
        .sort_values(["okato_code", "year", "month"])
        .reset_index(drop=True)
    )


def check_wages(table: pd.DataFrame) -> None:
    """Проверяет единственность строк и правдоподобие уровня зарплат."""
    assert not table.duplicated(["okato_code", "year", "month"]).any()
    assert table.nominal_wage_rub.between(10_000, 500_000).all()
    months_per_year = table.groupby(["okato_code", "year"]).size()
    assert months_per_year.le(12).all()


def main() -> None:
    table = build_wages()
    check_wages(table)
    table.to_csv(paths.WAGES_CSV, index=False)
    print(
        f"{paths.WAGES_CSV.name}: {table.okato_code.nunique()} субъектов, "
        f"{table.year.min()}-{table.year.max()}, {len(table)} строк"
    )


if __name__ == "__main__":
    main()
