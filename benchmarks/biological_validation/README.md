# Biological Validation Scripts for PanODE

This directory contains scripts for the biological validation section of both the DPMM paper and the Topic paper.

## Pipeline Overview

The biological validation proceeds in three stages:

### Stage 1: Train model & save state
```bash
# Train with full dynamics logging (requires GPU)
python benchmarks/training_dynamics.py \
    --model DPMM-Base --dataset setty --epochs 600 --snapshot-every 10

python benchmarks/training_dynamics.py \
    --model Topic-Base --dataset setty --epochs 800 --snapshot-every 10
```

Output: `benchmarks/training_dynamics_results/{model}_{dataset}_model.pt` + `_history.json`

### Stage 2: Latent component UMAP visualization
```bash
# Project each latent dim / topic proportion onto UMAP space
python benchmarks/biological_validation/latent_component_umap.py \
    --model-path benchmarks/training_dynamics_results/DPMM-Base_setty_model.pt \
    --dataset setty --series dpmm

python benchmarks/biological_validation/latent_component_umap.py \
    --model-path benchmarks/training_dynamics_results/Topic-Base_setty_model.pt \
    --dataset setty --series topic
```

Output: `benchmarks/biological_validation/results/{model}_{dataset}_components.png`

### Stage 3: Perturbation analysis + gene enrichment
```bash
# Perturb each component, find responsive genes, run GO enrichment
python benchmarks/biological_validation/perturbation_analysis.py \
    --model-path benchmarks/training_dynamics_results/DPMM-Base_setty_model.pt \
    --dataset setty --series dpmm \
    --gene-sets GO_Biological_Process_2021 --organism human

python benchmarks/biological_validation/perturbation_analysis.py \
    --model-path benchmarks/training_dynamics_results/Topic-Base_setty_model.pt \
    --dataset setty --series topic \
    --gene-sets GO_Biological_Process_2021 --organism human
```

Output:
- `*_importance_heatmap.png` — Gene importance per component
- `*_enrichment_{k}.png` — Per-component enrichment dot plot
- `*_enrichment_summary.png` — Combined enrichment overview
- `*_top_genes.json` — Top responsive genes per component
- `*_enrichment_comp{k}.csv` — Enrichment tables per component

## Plot Style

All scripts use `utils/paper_style.py` for consistent:
- Font sizes (14pt base, 300 DPI)
- Model colors (warm for DPMM/Topic, cool for AE/VAE baselines)
- Model ordering (ablation-aware)

Override via CLI: `--dpi 600`, `--font-scale 1.2`, `--fig-format pdf`, `--no-title`

## Dependencies

- `gseapy` (for Enrichr enrichment)
- `scanpy` (for UMAP computation)
- `torch` (for model inference)
- Standard: numpy, pandas, matplotlib
