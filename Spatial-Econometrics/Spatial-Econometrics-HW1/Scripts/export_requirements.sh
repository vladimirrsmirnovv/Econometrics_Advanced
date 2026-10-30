#!/usr/bin/env bash
# Выгружает зафиксированные версии зависимостей из uv.lock в requirements.txt.
# Нужен участникам, которые ставят пакеты через pip, а не через uv.
# Запускать после каждого изменения зависимостей в pyproject.toml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
uv export \
  --no-hashes \
  --no-annotate \
  --no-emit-project \
  --format requirements.txt \
  --output-file requirements.txt
echo "requirements.txt обновлен"
