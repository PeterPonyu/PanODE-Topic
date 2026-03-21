#!/bin/bash
# Run biological validation pipeline for ALL 12 model variants.
# Step 1: latent_component_umap.py → extract latent + save .npz
# Step 2: perturbation_analysis.py  → gene importance + GO enrichment
# Step 3: compose_figures.py        → multi-panel composite

set -e

PYTHON="/home/zeyufu/miniconda3/envs/dl/bin/python"
ROOT="/home/zeyufu/Desktop/PanODE-LAB"
MODELS_DIR="$ROOT/benchmarks/benchmark_results/models"
RESULTS_DIR="$ROOT/benchmarks/biological_validation/results"
DATASET="setty"

mkdir -p "$RESULTS_DIR"

# Model → series mapping
declare -A SERIES_MAP=(
  ["Pure-AE"]="dpmm"
  ["Pure-Transformer-AE"]="dpmm"
  ["Pure-Contrastive-AE"]="dpmm"
  ["DPMM-Base"]="dpmm"
  ["DPMM-Transformer"]="dpmm"
  ["DPMM-Contrastive"]="dpmm"
  ["Pure-VAE"]="topic"
  ["Pure-Transformer-VAE"]="topic"
  ["Pure-Contrastive-VAE"]="topic"
  ["Topic-Base"]="topic"
  ["Topic-Transformer"]="topic"
  ["Topic-Contrastive"]="topic"
)

cd "$ROOT"

for MODEL_NAME in "${!SERIES_MAP[@]}"; do
  SERIES="${SERIES_MAP[$MODEL_NAME]}"

  # Find the .pt file
  PT_FILE=$(ls "$MODELS_DIR/${MODEL_NAME}_${DATASET}"*.pt 2>/dev/null | head -1)
  if [ -z "$PT_FILE" ]; then
    echo "WARNING: No .pt file found for $MODEL_NAME — skipping"
    continue
  fi

  # Per-model output sub-directory
  MODEL_OUT="$RESULTS_DIR/$MODEL_NAME"
  mkdir -p "$MODEL_OUT"

  echo ""
  echo "============================================================"
  echo "  BIO VALIDATION: $MODEL_NAME (series=$SERIES)"
  echo "  Model: $PT_FILE"
  echo "  Output: $MODEL_OUT"
  echo "============================================================"

  # Step 1: Extract latent components
  TAG="${MODEL_NAME}_${DATASET}"
  if [ -f "$MODEL_OUT/${TAG}_latent_data.npz" ]; then
    echo "  [skip] Latent data already exists"
  else
    echo "  [1/3] Extracting latent components..."
    $PYTHON -m benchmarks.biological_validation.latent_component_umap \
      --model-path "$PT_FILE" \
      --dataset "$DATASET" \
      --series "$SERIES" \
      --no-title
  fi

  # Step 2: Perturbation + enrichment
  if [ -f "$MODEL_OUT/${TAG}_importance.npz" ]; then
    echo "  [skip] Importance data already exists"
  else
    echo "  [2/3] Perturbation analysis + GO enrichment..."
    $PYTHON -m benchmarks.biological_validation.perturbation_analysis \
      --model-path "$PT_FILE" \
      --dataset "$DATASET" \
      --series "$SERIES" \
      --organism human \
      --no-title
  fi

  # Step 3: Composite figure
  echo "  [3/3] Composing multi-panel figure..."
  $PYTHON -m benchmarks.biological_validation.compose_figures \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --series "$SERIES" \
    --no-title

  echo "  ✓ Done: $MODEL_NAME"
done

echo ""
echo "============================================================"
echo "  ALL BIOLOGICAL VALIDATION COMPLETE"
echo "  Results: $RESULTS_DIR"
echo "============================================================"
