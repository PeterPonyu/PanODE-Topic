# PanODE-Topic

Single-cell representation learning with logistic-normal topic models, featuring Transformer / contrastive / standalone ablation variants.

## Project Structure

```
PanODE-LAB/
├── models/                       # Core model implementations
│   ├── encoders.py               # Shared encoder architectures
│   ├── shared_modules.py         # Reusable layers & blocks
│   ├── dpmm_base.py              # DPMM base model (AE + DPMM clustering)
│   ├── dpmm_transformer.py       # DPMM + Transformer encoder
│   ├── dpmm_contrastive.py       # DPMM + MoCo contrastive learning
│   ├── topic_base.py             # Logistic-normal topic model
│   ├── topic_transformer.py      # Topic + Transformer encoder
│   └── topic_contrastive.py      # Topic + MoCo contrastive learning
│
├── metrics/                      # Evaluation metrics
│   ├── dre.py                    # Dimensionality Reduction Evaluator
│   ├── lse.py                    # Latent Space Evaluator
│   └── rhe.py                    # Representation Health Evaluator
│
├── utils/                        # Shared utilities
│   ├── base_model.py             # Unified model interface (fit / extract_latent)
│   ├── mixins.py                 # Shared model components & mixins
│   ├── data.py                   # Data loading and preprocessing
│   └── viz.py                    # Publication-quality visualisation tools
│
└── benchmarks/                   # Benchmarking
    ├── benchmark_base.py         # Consolidated benchmark (12 model variants)
    └── config.py                 # Centralised benchmark configuration
```

## Quick Start

### Installation

```bash
pip install torch numpy scipy scikit-learn scanpy torchdiffeq pandas matplotlib seaborn
```

### Run Benchmarks

```bash
# Full benchmark — all 12 model variants, 200 epochs, no early stopping
python benchmarks/benchmark_base.py --epochs 200 --no-early-stopping --series all

# DPMM series only
python benchmarks/benchmark_base.py --epochs 200 --series dpmm

# Topic series only
python benchmarks/benchmark_base.py --epochs 200 --series topic

# Quick smoke test (5 epochs)
python benchmarks/benchmark_base.py --epochs 5 --no-early-stopping --series all
```

## Model Overview

### 12 Benchmark Variants

| # | Model | Family | Description |
|---|-------|--------|-------------|
| 1 | Pure-AE | Baseline | Vanilla autoencoder (reconstruction only) |
| 2 | DPMM-Base | DPMM | AE + DPMM clustering with two-phase warmup |
| 3 | DPMM-Transformer | DPMM | DPMM + gene-as-token self-attention |
| 4 | DPMM-Contrastive | DPMM | DPMM + MoCo-v2 contrastive learning |
| 5 | Pure-VAE | Baseline | Vanilla VAE (reconstruction + KL) |
| 6 | Topic-Base | Topic | Logistic-normal topic model |
| 7 | Topic-Transformer | Topic | Topic + gene-as-token self-attention |
| 8 | Topic-Contrastive | Topic | Topic + MoCo-v2 contrastive learning |
| 9 | Pure-Transformer-AE | Ablation | Transformer encoder only (no DPMM/Topic) |
| 10 | Pure-Contrastive-AE | Ablation | Contrastive encoder only (no DPMM/Topic) |
| 11 | Pure-Transformer-VAE | Ablation | Transformer + KL (no Topic) |
| 12 | Pure-Contrastive-VAE | Ablation | Contrastive + KL (no Topic) |

### Unified Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-3 | All models |
| Latent dim | 10 | All models |
| Batch size | 128 | All models |
| Gradient clipping | 10.0 | All models |
| DPMM warmup ratio | 0.9 | DPMM series only |
| KL weight | 0.1 | Topic & VAE series |

## Evaluation Metrics

### Clustering
- **NMI** — Normalised Mutual Information
- **ARI** — Adjusted Rand Index
- **ASW** — Average Silhouette Width
- **DAV** — Davies–Bouldin Index (lower is better)
- **CAL** — Calinski–Harabasz Index

### DRE (Dimensionality Reduction Evaluation)
Evaluates structure preservation (embedding, UMAP, t-SNE) using co-ranking matrices.

### LSE (Latent Space Evaluation)
Evaluates intrinsic latent-space properties (structure, trajectory).

### Visualisation Output
- **UMAP grids** — coloured by true labels & model
- **Horizontal bar charts** — grouped by metric family (6 panels)
- **Heatmap** — compact normalised overview of all models × metrics

## Data Format

Input: AnnData (`.h5ad`) with:
- Raw counts in `adata.X` or `adata.layers['counts']`
- Optional labels in `adata.obs['cell_type']` or `adata.obs['clusters']`

## Configuration

Benchmark defaults are defined in [benchmarks/config.py](benchmarks/config.py).

## Development Workflow

- Keep the main repository on `main` for stable, release-ready code.
- Use `develop` as the integration branch for ongoing work.
- Create short-lived `feature/*` branches from `develop` and merge them back there.
- Attach every git worktree to a named branch. Because git does not allow the same branch to be checked out in multiple worktrees at once, extra worktrees should use distinct `feature/*` branches rather than staying detached.
- Generated experiment outputs and benchmark exports under `experiments/results/`, `lightning_logs/`, and selected `benchmarks/benchmark_results/` paths are ignored and should be regenerated, not committed.

See [docs/BRANCHING.md](docs/BRANCHING.md) for the full workflow.

---

## Changelog

### v2.0 (2025-02-08)
- **Consolidated benchmark**: merged 3 separate scripts (`benchmark_base.py`, `benchmark_base_contrastive.py`, `benchmark_attention.py`) into a single `benchmark_base.py` handling all 12 variants.
- **Added 4 standalone ablation variants**: Pure-Transformer-AE, Pure-Contrastive-AE, Pure-Transformer-VAE, Pure-Contrastive-VAE — isolate Transformer / contrastive contributions without DPMM or Topic heads.
- **Hyperparameter audit & unification**: fixed `kl_weight` (0.01→0.1 for Topic series), added `dpmm_warmup_ratio=0.9` to DPMM-Transformer & DPMM-Contrastive, unified gradient clipping to 10.0 across all models.
- **Visualisation overhaul** (`utils/viz.py`):
  - Removed redundant `plot_metrics_comparison`; consolidated 9→6 metric panels.
  - Switched to horizontal bar charts for readability with 12 models.
  - Added `plot_metrics_heatmap` for compact cross-model comparison.
  - Publication-quality fonts (DejaVu Sans, 12 pt) and 12-colour palette.
- **Removed GAT models** (`dpmm_gat.py`, `topic_gat.py`) — retired from the benchmark.
- **Cleaned up**: removed archive notebooks, stale cache files, backup files, and orphaned `__pycache__` entries.

### v1.0 (2025-01-29)
- Initial 8-model benchmark (DPMM × {Base, Transformer, GAT, Contrastive} + Topic × {Base, Transformer, GAT, Contrastive}).
- Separate benchmark scripts per model group.
- DRE + LSE evaluation metrics.
