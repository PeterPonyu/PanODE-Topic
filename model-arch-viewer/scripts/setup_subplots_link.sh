#!/bin/bash
# setup_subplots_link.sh — Create symlinks so Next.js can serve figure assets
#
# This links:
#   model-arch-viewer/public/subplots/{series}/fig{N}/
#   → benchmarks/paper_figures/{series}/subplots/fig{N}/
# and
#   model-arch-viewer/public/statistical/
#   → benchmarks/paper_figures/statistical/
#
# Run from the PanODE-LAB root directory.

set -euo pipefail

VIEWER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$VIEWER_DIR/.." && pwd)"
PUBLIC_DIR="$VIEWER_DIR/public"
SUBPLOTS_BASE="$ROOT/benchmarks/paper_figures"

echo "Setting up subplot symlinks..."
echo "  Viewer dir: $VIEWER_DIR"
echo "  Public dir: $PUBLIC_DIR"
echo "  Source base: $SUBPLOTS_BASE"

mkdir -p "$PUBLIC_DIR/subplots"

for series in dpmm topic; do
  SERIES_SRC="$SUBPLOTS_BASE/$series/subplots"
  SERIES_DST="$PUBLIC_DIR/subplots/$series"

  if [ -d "$SERIES_SRC" ]; then
    # Remove existing symlink or directory
    rm -rf "$SERIES_DST"
    ln -s "$SERIES_SRC" "$SERIES_DST"
    echo "  ✓ $series → $SERIES_SRC"
  else
    echo "  ⚠ $SERIES_SRC not found (run generate_subplots.py first)"
  fi
done

# Statistical figure assets (shared across series)
STAT_SRC="$SUBPLOTS_BASE/statistical"
STAT_DST="$PUBLIC_DIR/statistical"
if [ -d "$STAT_SRC" ]; then
  rm -rf "$STAT_DST"
  ln -s "$STAT_SRC" "$STAT_DST"
  echo "  ✓ statistical → $STAT_SRC"
else
  echo "  ⚠ $STAT_SRC not found (run generate_statistical_figures.py first)"
fi

echo "Done."
