#!/usr/bin/env bash
# Build Topic manuscript (MDPI LaTeX).
# Run from repo root: ./article/build.sh
# Or from article/: ./build.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

build_one() {
  local series="$1"
  echo "Building $series manuscript..."
  cd "$SCRIPT_DIR/$series"
  pdflatex -interaction=nonstopmode main_mdpi.tex
  bibtex main_mdpi || true   # allow no citations during draft
  pdflatex -interaction=nonstopmode main_mdpi.tex
  pdflatex -interaction=nonstopmode main_mdpi.tex
  echo "  -> $series/main_mdpi.pdf"
  cd "$SCRIPT_DIR"
}

build_one topic
echo "Done. Output: article/topic/main_mdpi.pdf"
