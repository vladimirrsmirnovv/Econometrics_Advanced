#!/usr/bin/env bash
set -euo pipefail
PRESENTATION_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIRECTORY="$(cd "$PRESENTATION_DIRECTORY/.." && pwd)"
PRESENTATION_BUILD_DIRECTORY="$PROJECT_DIRECTORY/tmp/presentation/build"
mkdir -p "$PRESENTATION_BUILD_DIRECTORY"
cd "$PRESENTATION_DIRECTORY"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir="$PRESENTATION_BUILD_DIRECTORY" main.tex > "$PRESENTATION_BUILD_DIRECTORY/slides-console.log" 2>&1
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 -outdir="$PRESENTATION_BUILD_DIRECTORY" script.tex > "$PRESENTATION_BUILD_DIRECTORY/script-console.log" 2>&1
cp "$PRESENTATION_BUILD_DIRECTORY/main.pdf" "$PRESENTATION_DIRECTORY/Презентация.pdf"
cp "$PRESENTATION_BUILD_DIRECTORY/script.pdf" "$PRESENTATION_DIRECTORY/Речь.pdf"
