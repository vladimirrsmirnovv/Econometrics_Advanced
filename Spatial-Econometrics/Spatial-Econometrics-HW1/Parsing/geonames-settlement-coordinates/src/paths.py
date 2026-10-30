from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
PARSING_ROOT = ROOT / "Parsing" / 'geonames-settlement-coordinates'
SRC_DIR = PARSING_ROOT / "src"
TMP_DIR = PARSING_ROOT / "tmp"
RESULT_DIR = PARSING_ROOT / "result"

TRAIN_PATH = ROOT / "Source" / "Qualifying" / "train.csv"
LOCAL_SAMPLE_DIR = ROOT / "tmp" / "RU"

GEONAMES_WORKDIR = TMP_DIR / "geonames_ru"
GEONAMES_ARCHIVE_PATH = GEONAMES_WORKDIR / "RU.zip"
GEONAMES_TEXT_PATH = GEONAMES_WORKDIR / "RU.txt"
GEONAMES_README_PATH = GEONAMES_WORKDIR / "readme.txt"

SERVICE_OUTPUT_DIR = GEONAMES_WORKDIR / "service_outputs"
DIAGNOSTICS_PATH = SERVICE_OUTPUT_DIR / "settlement_coordinates_diagnostics.csv"
REGION_RESOLUTION_PATH = SERVICE_OUTPUT_DIR / "region_resolution.csv"

RESULT_PATH = RESULT_DIR / "settlement_coordinates.csv"
RESULT_README_PATH = RESULT_DIR / "README.md"


def ensure_runtime_directories() -> None:
    """
    Создаёт директории, необходимые для выполнения пайплайна.

    Returns
    -------
    None
        На диске гарантированно существуют папки `tmp`, `result` и служебный каталог.
    """

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SERVICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
