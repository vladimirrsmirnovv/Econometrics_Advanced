"""Шаг 3. Уровень безработицы по субъектам: из xlsx в два плоских csv.

Лист `4` - скользящие трехмесячные периоды («январь - март 2023»), что
дает помесячный ряд. Лист `2` - годовые значения. Оба показателя - уровень
безработицы населения в возрасте 15 лет и старше, проценты.
"""

import re

import pandas as pd

from src import paths
from src.names import clean_text
from src.regions import keep_subjects, okato_code_by_name, rosstat_region_name
from src.rosstat_wages import MONTH_NUMBERS

HEADER_ROW = 4


def read_subject_rows(sheet_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Возвращает строки субъектов листа и строку заголовков периодов."""
    sheet = pd.read_excel(paths.UNEMPLOYMENT_RAW, sheet_name, header=None, dtype=str)
    headers = sheet.iloc[HEADER_ROW, 1:]
    body = sheet.iloc[HEADER_ROW + 1 :].dropna(subset=[1])
    body = body.assign(
        region_name=body[0].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return keep_subjects(body, "region_name"), headers


def window_end(header: str) -> tuple[int, int]:
    """Возвращает год и месяц последнего месяца трехмесячного окна."""
    text = clean_text(header)
    last_month = [
        word for word in re.findall(r"[а-я]+", text) if word in MONTH_NUMBERS
    ][-1]
    last_year = re.findall(r"\d{4}", text)[-1]
    return int(last_year), MONTH_NUMBERS[last_month]


def shift_month(year: int, month: int, months: int) -> tuple[int, int]:
    """Сдвигает пару «год - месяц» на заданное число месяцев."""
    index = year * 12 + month - 1 + months
    return index // 12, index % 12 + 1


def to_long(body: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Переводит широкую таблицу в длинную с кодом ОКАТО субъекта."""
    long_table = body.melt(
        id_vars="region_name",
        value_vars=list(body.columns[1:-1]),
        var_name="column",
        value_name=value_name,
    )
    long_table[value_name] = pd.to_numeric(long_table[value_name])
    long_table.insert(0, "okato_code", okato_code_by_name(long_table.region_name))
    long_table["region_name"] = rosstat_region_name(long_table.okato_code)
    return long_table


def build_monthly_unemployment() -> pd.DataFrame:
    """Строит помесячный ряд: окну присваивается его центральный месяц."""
    body, headers = read_subject_rows("4")
    table = to_long(body, "unemployment_rate_percent")
    ends = table.column.map(headers).map(window_end)
    centers = ends.map(lambda end: shift_month(*end, -1))
    table["window_end_year"] = ends.str[0]
    table["window_end_month"] = ends.str[1]
    table["year"] = centers.str[0]
    table["month"] = centers.str[1]
    columns = [
        "okato_code",
        "region_name",
        "year",
        "month",
        "window_end_year",
        "window_end_month",
        "unemployment_rate_percent",
    ]
    return (
        table[columns]
        .sort_values(["okato_code", "year", "month"])
        .reset_index(drop=True)
    )


def build_annual_unemployment() -> pd.DataFrame:
    """Строит годовой ряд с листа `2`."""
    body, headers = read_subject_rows("2")
    table = to_long(body, "unemployment_rate_percent")
    table["year"] = table.column.map(headers).str.extract(r"(\d{4})")[0].astype(int)
    columns = ["okato_code", "region_name", "year", "unemployment_rate_percent"]
    return table[columns].sort_values(["okato_code", "year"]).reset_index(drop=True)


def check_unemployment(table: pd.DataFrame, key: list[str]) -> None:
    """Проверяет единственность строк и допустимый диапазон процентов."""
    assert not table.duplicated(key).any()
    assert table.unemployment_rate_percent.notna().all()
    assert table.unemployment_rate_percent.between(0, 50).all()


def main() -> None:
    monthly = build_monthly_unemployment()
    check_unemployment(monthly, ["okato_code", "year", "month"])
    monthly.to_csv(paths.UNEMPLOYMENT_CSV, index=False)

    annual = build_annual_unemployment()
    check_unemployment(annual, ["okato_code", "year"])
    annual.to_csv(paths.UNEMPLOYMENT_ANNUAL_CSV, index=False)
    print(
        f"{paths.UNEMPLOYMENT_CSV.name}: {monthly.okato_code.nunique()} субъектов, "
        f"{len(monthly)} строк; {paths.UNEMPLOYMENT_ANNUAL_CSV.name}: {len(annual)} строк"
    )


if __name__ == "__main__":
    main()
