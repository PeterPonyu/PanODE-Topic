#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# refresh_figures.sh — End-to-end figure regeneration pipeline (Python-only)
#
# Regenerates subplot PNGs, composes experiment figures, and validates outputs.
# Next.js + Playwright screenshots are available as an optional debug step
# via --with-frontend.
#
# Usage:
#   ./refresh_figures.sh                           # all figures, both series
#   ./refresh_figures.sh --figures 7 8             # specific figures only
#   ./refresh_figures.sh --series dpmm             # single series
#   ./refresh_figures.sh --series topic --figures 7 # combine flags
#   ./refresh_figures.sh --with-frontend           # include Next.js screenshots
#
# Prerequisites:
#   - conda environment activated (with matplotlib, numpy, pandas, etc.)
#   - For --with-frontend: Node.js + npm + Playwright browsers
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIEWER_DIR="$PROJ_ROOT/model-arch-viewer"
PORT=3099
BLUEPRINTS=(
  "$PROJ_ROOT/docs/PAPER_BLUEPRINT_DPMM.md"
  "$PROJ_ROOT/docs/PAPER_BLUEPRINT_TOPIC.md"
)

# ── Parse arguments ──────────────────────────────────────────────────────────
FIGURES=()
SERIES=()
WITH_FRONTEND=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --figures)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        FIGURES+=("$1"); shift
      done
      ;;
    --series)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SERIES+=("$1"); shift
      done
      ;;
    --with-frontend)
      WITH_FRONTEND=true; shift
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Defaults
[[ ${#SERIES[@]} -eq 0 ]]  && SERIES=(topic)
[[ ${#FIGURES[@]} -eq 0 ]] && FIGURES=(1 2 3 4 6 7 8 9 10 11 12)

FIG_ARGS="${FIGURES[*]}"
SERIES_LIST="${SERIES[*]}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           PanODE-Topic  Figure Refresh Pipeline                  ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Series : $SERIES_LIST"
echo "║  Figures: $FIG_ARGS"
echo "║  Mode   : $(if $WITH_FRONTEND; then echo 'Python + Frontend'; else echo 'Python-only (release)'; fi)"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate subplot PNGs (matplotlib) for benchmark-pipeline figures
# ═══════════════════════════════════════════════════════════════════════════════
# Only generate subplots for figures that use the benchmark pipeline (3-4, 6-9 + 10).
# Figures 2 and 10-12 use the experiment pipeline. Figure 5 removed.
SUBPLOT_FIGS=()
for f in "${FIGURES[@]}"; do
  case "$f" in
    1|2|5|11|12) ;; # skip: Fig1 manual, Fig2/11/12 experiment pipeline, Fig5 removed
    *) SUBPLOT_FIGS+=("$f") ;;
  esac
done

if [[ ${#SUBPLOT_FIGS[@]} -gt 0 ]]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "STEP 1/5  Generating subplot PNGs (Figs ${SUBPLOT_FIGS[*]}) …"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$PROJ_ROOT"
  for s in "${SERIES[@]}"; do
    echo ""
    echo "  ▸ Series: $s | Figures: ${SUBPLOT_FIGS[*]}"
    python -m benchmarks.figure_generators.generate_subplots \
      --series "$s" --figures ${SUBPLOT_FIGS[@]}
  done

  echo ""
  echo "  ✓ Subplot PNGs generated."
  echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1b — Generate statistical/significance figure assets
# ═══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1b/5  Generating statistical/significance figures …"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJ_ROOT"
python scripts/generate_statistical_figures.py
python scripts/generate_statistical_figures_2seed.py

echo ""
echo "  ✓ Statistical/significance figures generated (statistical/dpmm/ and statistical/topic/)."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compose experiment-pipeline figures (Figs 2, 10–12) via Python
# ═══════════════════════════════════════════════════════════════════════════════
# Check if any experiment-pipeline figures are requested
NEEDS_EXP_COMPOSE=false
for f in "${FIGURES[@]}"; do
  case "$f" in
    2|10|11|12) NEEDS_EXP_COMPOSE=true ;;
  esac
done

if $NEEDS_EXP_COMPOSE; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "STEP 2/5  Composing experiment figures (Figs 2, 10–12) via Python …"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cd "$PROJ_ROOT"
  python scripts/compose_experiment_figures.py --all --layout auto

  echo ""
  echo "  ✓ Experiment figures composed."
  echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 (optional) — Frontend screenshots (debug only)
# ═══════════════════════════════════════════════════════════════════════════════
SERVER_STARTED_HERE=false

if $WITH_FRONTEND; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "STEP 3/5  [FRONTEND] Next.js + Playwright screenshots (debug) …"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Ensure symlinks
  bash "$VIEWER_DIR/scripts/setup_subplots_link.sh"

  # Start or detect server
  if curl -s --max-time 2 "http://localhost:$PORT" >/dev/null 2>&1; then
    echo "  ✓ Server already running on port $PORT."
  else
    echo "  ▸ Starting Next.js dev server …"
    pkill -f "next dev" 2>/dev/null || true
    rm -f "$VIEWER_DIR/.next/dev/lock" 2>/dev/null || true
    sleep 1

    cd "$VIEWER_DIR"
    PORT=$PORT npm run dev > /tmp/panode_nextjs.log 2>&1 &
    SERVER_PID=$!
    SERVER_STARTED_HERE=true

    echo -n "  ▸ Waiting for server"
    for i in $(seq 1 30); do
      if curl -s --max-time 1 "http://localhost:$PORT" >/dev/null 2>&1; then
        echo " ready! (${i}s)"
        break
      fi
      echo -n "."
      sleep 1
      if [[ $i -eq 30 ]]; then
        echo ""
        echo "  ✗ Server failed to start after 30s. Check /tmp/panode_nextjs.log"
        tail -20 /tmp/panode_nextjs.log
        exit 1
      fi
    done
  fi

  # Take screenshots
  cd "$VIEWER_DIR"
  for s in "${SERIES[@]}"; do
    echo ""
    echo "  ▸ Series: $s"
    node screenshot_figures.mjs --series "$s" --figures $FIG_ARGS
  done

  echo ""
  echo "  ✓ Frontend screenshots saved."
  echo ""
else
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "STEP 3/5  Frontend screenshots — SKIPPED (use --with-frontend to enable)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Verify output files & print sizes
# ═══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4/5  Verifying output …"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FIGURES_DIR="$PROJ_ROOT/benchmarks/paper_figures"
EXP_DIR="$PROJ_ROOT/experiments/results"
MISSING=0

# Experiment-pipeline figure paths (relative to EXP_DIR/{series}/)
declare -A EXP_FIG_PATHS
EXP_FIG_PATHS[2]="ablation/figures/composed_metrics.png"
EXP_FIG_PATHS[10]="vs_external/figures/proposed/composed_metrics.png"
EXP_FIG_PATHS[11]="vs_external/figures/classical/composed_metrics.png"
EXP_FIG_PATHS[12]="vs_external/figures/deep/composed_metrics.png"

for s in "${SERIES[@]}"; do
  for f in "${FIGURES[@]}"; do
    if [[ -n "${EXP_FIG_PATHS[$f]+x}" ]]; then
      # Experiment-pipeline figure
      exp_path="$EXP_DIR/$s/${EXP_FIG_PATHS[$f]}"
      if [[ -f "$exp_path" ]]; then
        sz=$(du -h "$exp_path" | cut -f1)
        echo "  ✓ $exp_path  ($sz)"
      else
        echo "  ✗ MISSING: Fig${f} for series $s ($exp_path)"
        MISSING=$((MISSING + 1))
      fi
    else
      # Benchmark-pipeline figure
      found=$(find "$FIGURES_DIR/$s" -maxdepth 1 -name "Fig${f}_*_${s}.png" -print -quit 2>/dev/null)
      if [[ -n "$found" ]]; then
        sz=$(du -h "$found" | cut -f1)
        echo "  ✓ $found  ($sz)"
      else
        echo "  ✗ MISSING: Fig${f} for series $s"
        MISSING=$((MISSING + 1))
      fi
    fi
  done
done

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Composed-figure-level conflict validation
# ═══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5/5  Composed-figure validation (size ≤ 17×21cm content-fitted, content, conflicts) …"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJ_ROOT"
COMPOSED_FAIL=0
for s in "${SERIES[@]}"; do
  python -m benchmarks.figure_generators.composed_figure_validator \
    --series "$s" --figures ${FIGURES[@]} || COMPOSED_FAIL=1
done

if [[ $COMPOSED_FAIL -ne 0 ]]; then
  echo ""
  echo "  ⚠  Composed-figure validation found issues — review warnings above."
  echo ""
fi

# ── Clean up server if we started it ─────────────────────────────────────────
if [[ "$SERVER_STARTED_HERE" = true ]]; then
  echo "  ▸ Stopping Next.js server (PID $SERVER_PID) …"
  kill "$SERVER_PID" 2>/dev/null || true
  echo "  ✓ Server stopped."
  echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# REMINDER — Update paper blueprint text descriptions
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║   ⚠  ACTION REQUIRED: UPDATE PAPER BLUEPRINT TEXT  ⚠          ║"
echo "║                                                                ║"
echo "║   Figures have been regenerated. You MUST now review and       ║"
echo "║   update the figure descriptions and results narrative in      ║"
echo "║   the paper blueprint files to reflect the new figures:        ║"
echo "║                                                                ║"
echo "║     1. docs/PAPER_BLUEPRINT_DPMM.md                          ║"
echo "║     2. docs/PAPER_BLUEPRINT_TOPIC.md                         ║"
echo "║                                                                ║"
echo "║   Note: Former Figure 10 is now split into:                    ║"
echo "║     • Figure 10 (Proposed external benchmark)                  ║"
echo "║     • Figure 11 (Classical external benchmark)                 ║"
echo "║     • Figure 12 (Deep external benchmark)                      ║"
echo "║                                                                ║"
echo "║   Updated figures:                                             ║"
for s in "${SERIES[@]}"; do
  for f in "${FIGURES[@]}"; do
    printf "║     • Figure %-2s  (%s series)                               ║\n" "$f" "$s"
  done
done
echo "║                                                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [[ $MISSING -gt 0 ]]; then
  echo "⚠  $MISSING figure(s) are missing — check errors above."
  exit 1
fi

echo "Done. $(date '+%Y-%m-%d %H:%M:%S')"
