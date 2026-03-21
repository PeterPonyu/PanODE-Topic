#!/bin/bash
# Run biological validation for 3 representative datasets × prior-based models.
# Trains a fresh model for each (model, dataset) pair that lacks a .pt,
# then runs the full pipeline: latent extraction → perturbation → enrichment.
#
# Usage:  bash benchmarks/biological_validation/run_bio_multi_dataset.sh
#
set -e

PYTHON="/home/zeyufu/Desktop/.conda/bin/python"
ROOT="/home/zeyufu/Desktop/PanODE-LAB"
MODELS_DIR="$ROOT/benchmarks/benchmark_results/models"
RESULTS_DIR="$ROOT/benchmarks/biological_validation/results"

mkdir -p "$RESULTS_DIR" "$MODELS_DIR"

# Representative datasets for Figure 6
DATASETS=("setty" "endo" "dentate")

# Prior-based models (6 models total — both series)
declare -A SERIES_MAP=(
  ["DPMM-Base"]="dpmm"
  ["DPMM-Transformer"]="dpmm"
  ["DPMM-Contrastive"]="dpmm"
  ["Topic-Base"]="topic"
  ["Topic-Transformer"]="topic"
  ["Topic-Contrastive"]="topic"
)

cd "$ROOT"

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "################################################################"
  echo "  DATASET: $DATASET"
  echo "################################################################"

  for MODEL_NAME in "${!SERIES_MAP[@]}"; do
    SERIES="${SERIES_MAP[$MODEL_NAME]}"
    TAG="${MODEL_NAME}_${DATASET}"

    # Skip if bio validation already completed
    if [ -f "$RESULTS_DIR/${TAG}_importance.npz" ] && \
       [ -f "$RESULTS_DIR/${TAG}_latent_data.npz" ]; then
      echo "  [skip] ${TAG} — bio validation already exists"
      continue
    fi

    # Find or train a .pt model
    PT_FILE=$(ls "$MODELS_DIR/${MODEL_NAME}_${DATASET}"*.pt 2>/dev/null | head -1)
    if [ -z "$PT_FILE" ]; then
      echo "  [train] Training $MODEL_NAME on $DATASET..."
      $PYTHON -c "
import sys, torch
sys.path.insert(0, '$ROOT')
from benchmarks.model_registry import MODELS
from benchmarks.data_utils import load_data
from benchmarks.dataset_registry import DATASET_REGISTRY
from utils.data import DataSplitter

ds_info = DATASET_REGISTRY['$DATASET']
adata = load_data(ds_info['path'], max_cells=3000, hvg_top_genes=3000, seed=42)
splitter = DataSplitter(adata=adata, batch_size=128, random_seed=42, verbose=False)
model_info = MODELS['$MODEL_NAME']
params = dict(model_info['params'])
# Remove fit params
fit_lr = params.pop('fit_lr', 1e-3)
fit_wd = params.pop('fit_weight_decay', 0)
fit_ep = params.pop('fit_epochs', 600)
model = model_info['class'](input_dim=splitter.n_var, **params)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.fit(splitter.train_loader, epochs=fit_ep, lr=fit_lr,
          weight_decay=fit_wd, device=device, verbose=False)
torch.save({
    'state_dict': model.state_dict(),
    'config': {
        'model_name': '$MODEL_NAME',
        'input_dim': splitter.n_var,
        'params': dict(model_info['params']),
    },
}, '$MODELS_DIR/${TAG}_bio.pt')
print('  Saved model: $MODELS_DIR/${TAG}_bio.pt')
"
      PT_FILE="$MODELS_DIR/${TAG}_bio.pt"
    fi

    if [ ! -f "$PT_FILE" ]; then
      echo "  WARNING: Could not train $MODEL_NAME on $DATASET — skipping"
      continue
    fi

    echo ""
    echo "============================================================"
    echo "  BIO VALIDATION: $TAG (series=$SERIES)"
    echo "============================================================"

    # Determine organism (dentate is mouse, others are human)
    ORGANISM="human"
    if [ "$DATASET" = "dentate" ]; then
      ORGANISM="mouse"
    fi

    # Step 1: Extract latent components
    if [ ! -f "$RESULTS_DIR/${TAG}_latent_data.npz" ]; then
      echo "  [1/2] Extracting latent components..."
      $PYTHON -m benchmarks.biological_validation.latent_component_umap \
        --model-path "$PT_FILE" \
        --dataset "$DATASET" \
        --series "$SERIES" \
        --no-title 2>&1 || echo "  WARNING: latent extraction failed"
    fi

    # Step 2: Perturbation + enrichment
    if [ ! -f "$RESULTS_DIR/${TAG}_importance.npz" ]; then
      echo "  [2/2] Perturbation analysis + GO enrichment..."
      $PYTHON -m benchmarks.biological_validation.perturbation_analysis \
        --model-path "$PT_FILE" \
        --dataset "$DATASET" \
        --series "$SERIES" \
        --organism "$ORGANISM" \
        --no-title 2>&1 || echo "  WARNING: perturbation analysis failed"
    fi

    echo "  Done: $TAG"
  done
done

echo ""
echo "========================================"
echo "  Bio validation complete for all datasets."
echo "  Results in: $RESULTS_DIR"
echo "========================================"
