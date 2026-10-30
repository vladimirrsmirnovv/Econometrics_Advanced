#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/tmp/paper-build"
SOURCE_FILE="$SCRIPT_DIR/main.tex"
OUTPUT_FILE="$SCRIPT_DIR/Пространственная-эконометрика-ДЗ1.pdf"

cd "$PROJECT_ROOT"
mkdir -p "$BUILD_DIR"
rm -f "$OUTPUT_FILE"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" "$SOURCE_FILE" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" "$SOURCE_FILE" >/dev/null
cp "$BUILD_DIR/main.pdf" "$OUTPUT_FILE"
if command -v exiftool >/dev/null 2>&1; then
  exiftool -all= -overwrite_original "$OUTPUT_FILE" >/dev/null
fi
rm -rf "$BUILD_DIR"
