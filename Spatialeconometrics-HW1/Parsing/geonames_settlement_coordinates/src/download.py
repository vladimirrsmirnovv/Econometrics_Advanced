from __future__ import annotations

import shutil
import zipfile

import requests

from src.constants import GEONAMES_URL
from src.paths import (
    GEONAMES_ARCHIVE_PATH,
    GEONAMES_README_PATH,
    GEONAMES_TEXT_PATH,
    GEONAMES_WORKDIR,
    LOCAL_SAMPLE_DIR,
)


def download_geonames_archive() -> None:
    """
    Скачивает архив GeoNames по России во временную директорию проекта.

    Returns
    -------
    None
        Архив `RU.zip` сохраняется в рабочую папку `tmp`.
    """

    GEONAMES_WORKDIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(GEONAMES_URL, timeout=120, stream=True)
    response.raise_for_status()

    with GEONAMES_ARCHIVE_PATH.open("wb") as archive_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                archive_file.write(chunk)


def extract_geonames_archive() -> None:
    """
    Распаковывает архив GeoNames в рабочую временную директорию.

    Returns
    -------
    None
        Из архива извлекаются `RU.txt` и сопутствующий `readme.txt`.
    """

    GEONAMES_WORKDIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(GEONAMES_ARCHIVE_PATH) as archive:
        archive.extractall(GEONAMES_WORKDIR)


def seed_geonames_from_local_sample() -> bool:
    """
    Копирует локальный пример GeoNames из репозитория во временную директорию.

    Returns
    -------
    bool
        `True`, если пример найден и скопирован, иначе `False`.
    """

    sample_text_path = LOCAL_SAMPLE_DIR / "RU.txt"
    sample_readme_path = LOCAL_SAMPLE_DIR / "readme.txt"
    if not sample_text_path.exists() or not sample_readme_path.exists():
        return False

    GEONAMES_WORKDIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_text_path, GEONAMES_TEXT_PATH)
    shutil.copy2(sample_readme_path, GEONAMES_README_PATH)
    return True


def ensure_geonames_files() -> None:
    """
    Гарантирует наличие локальных файлов GeoNames для парсинга.

    Returns
    -------
    None
        При отсутствии файлов используется локальный пример или выполняется скачивание.
    """

    if GEONAMES_TEXT_PATH.exists() and GEONAMES_README_PATH.exists():
        return

    if seed_geonames_from_local_sample():
        return

    download_geonames_archive()
    extract_geonames_archive()
