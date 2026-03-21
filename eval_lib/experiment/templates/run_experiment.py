#!/usr/bin/env python
"""
Experiment Runner Template
============================

Project-agnostic experiment runner using eval_lib for metric computation.
Copy into your project's ``experiments/`` directory, then customise only
the ``# ═══ PROJECT-SPECIFIC`` sections.

Portable sections (do NOT modify):
  - train_single_model() core logic
  - run_experiment() main loop (dataset × model iteration, CSV output, resume)
  - CLI argument parser

Project-specific sections (MUST be customised):
  - Data loading function
  - Label standardisation
  - DataSplitter / data-loader creation
  - Model hyperparameter CLI overrides map
  - PRESETS import

Usage
-----
    python -m experiments.run_experiment --preset my_ablation
    python -m experiments.run_experiment --preset my_ablation --epochs 50
    python -m experiments.run_experiment --preset my_ablation --datasets ds_a ds_b
"""

import argparse
import gc
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

warnings.filterwarnings("ignore")

# ── eval_lib imports (portable) ───────────────────────────────────────────────
# All metric helpers live in eval_lib.metrics.battery — single source of truth.
from eval_lib.metrics.battery import (
    compute_metrics,
    compute_latent_diagnostics,
    convergence_diagnostics,
    METRIC_COLUMNS)
from eval_lib.experiment.config import ExperimentConfig


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PROJECT-SPECIFIC: Project root & config imports
# ═══════════════════════════════════════════════════════════════════════════════
# Make sure your project root is on sys.path and import your config.

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# TODO: Import your PRESETS dict from your project's experiment_config.
# from experiments.experiment_config import PRESETS
PRESETS = {}  # placeholder — replace with project PRESETS


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PROJECT-SPECIFIC: Data loading
# ═══════════════════════════════════════════════════════════════════════════════
# Replace this with your project's data loading / preprocessing pipeline.
# The function must return an AnnData object with:
#   - .X or .layers["counts"] containing expression data
#   - .obs containing at least a label column
#
# Example: For scRNA-seq projects, this typically involves:
#   - Reading an .h5ad file
#   - Subsampling to max_cells
#   - Selecting top n_hvg highly variable genes
#   - Log-normalisation
#   - Caching preprocessed data for faster re-runs

def load_data(path: str, max_cells: int, n_hvg: int, seed: int,
              cache_dir: Path = None) -> "sc.AnnData":
    """Load and preprocess a single dataset.

    TODO: Replace with your data loading pipeline.
    """
    adata = sc.read_h5ad(path)

    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)

    if adata.n_vars > n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg,
                                     flavor="seurat_v3", span=0.3)
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PROJECT-SPECIFIC: Label standardisation
# ═══════════════════════════════════════════════════════════════════════════════

def standardize_labels(adata, label_key: str):
    """Copy *label_key* → ``adata.obs['cell_type']`` for DataSplitter.

    TODO: Extend CANDIDATE_KEYS with your project's label column names.
    """
    CANDIDATE_KEYS = [
        label_key,
        "cell_type", "celltype", "clusters", "Clusters",
        "Cell type", "Main_cell_type", "cell.type", "CellType",
    ]
    for key in CANDIDATE_KEYS:
        if key in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[key].astype(str).copy()
            print(f"  Labels: using '{key}' ({adata.obs['cell_type'].nunique()} types)")
            return adata

    print(f"  WARNING: No label column found — will generate KMeans pseudo-labels")
    return adata


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PROJECT-SPECIFIC: DataSplitter / data-loader creation
# ═══════════════════════════════════════════════════════════════════════════════
# Your DataSplitter must produce:
#   .train_loader   (PyTorch DataLoader)
#   .val_loader     (PyTorch DataLoader)
#   .test_loader    (PyTorch DataLoader)
#   .labels_test    (np.ndarray of string/int labels for test set)
#   .n_var          (int — input dimensionality)

def create_data_splitter(adata, batch_size: int, seed: int, latent_dim: int):
    """Create train/val/test data loaders from an AnnData object.

    TODO: Replace with your project's DataSplitter.
    """
    raise NotImplementedError(
        "Replace create_data_splitter() with your project's data splitting logic."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def is_cuda_oom(exc: Exception) -> bool:
    """Return True if *exc* is a CUDA out-of-memory error."""
    msg = str(exc).lower()
    return ("out of memory" in msg or "cuda" in msg and "alloc" in msg)


def _ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: Single model training + evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_single_model(
    model_name: str,
    model_spec: dict,
    splitter,
    cfg: ExperimentConfig,
    device: torch.device,
    data_type: str = "cluster") -> dict:
    """Train one model variant, extract latent, compute full metrics.

    The model class must conform to the interface:
      __init__(input_dim=..., **params)
      fit(train_loader, val_loader, epochs, lr, device, patience,
          verbose, verbose_every, weight_decay) -> dict
      extract_latent(loader, device) -> {"latent": ndarray}

    Returns
    -------
    dict
        {"Model": name, metric columns..., "latent": ndarray, "history": dict}
    """
    gc.collect()
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_cls = model_spec["class"]
    params = dict(model_spec["params"])

    fit_lr = params.pop("fit_lr", 1e-3)
    fit_wd = params.pop("fit_weight_decay", 1e-5)
    params.pop("fit_epochs", None)

    start = time.time()
    print(f"\n{'─'*60}\n  Training: {model_name}  ({cfg.epochs} epochs)\n{'─'*60}")

    try:
        model = model_cls(input_dim=splitter.n_var, **params)
        model = model.to(device)

        history = model.fit(
            train_loader=splitter.train_loader,
            val_loader=splitter.val_loader,
            epochs=cfg.epochs,
            lr=fit_lr,
            device=str(device),
            patience=cfg.patience,
            verbose=1,
            verbose_every=cfg.verbose_every,
            weight_decay=fit_wd)

        elapsed = time.time() - start
        epochs_trained = len(history.get("train_loss", [])) or cfg.epochs
        sec_per_epoch = elapsed / max(epochs_trained, 1)

        peak_gpu_mb = 0.0
        if device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        latent_dict = model.extract_latent(
            splitter.test_loader, device=str(device))
        latent = latent_dict["latent"]

        # Full metric battery from eval_lib
        metrics = compute_metrics(
            latent, splitter.labels_test,
            data_type=data_type, dre_k=cfg.dre_k)
        diagnostics = compute_latent_diagnostics(latent)
        n_params = sum(p.numel() for p in model.parameters())

        print(
            f"  NMI={metrics['NMI']:.4f}  ARI={metrics['ARI']:.4f}  "
            f"ASW={metrics.get('ASW', float('nan')):.4f}  |  "
            f"{elapsed:.1f}s ({sec_per_epoch:.2f}s/ep, GPU {peak_gpu_mb:.0f}MB, "
            f"params {n_params:,})"
        )

        result = {
            "Model": model_name,
            "Time_s": elapsed,
            "SecPerEpoch": sec_per_epoch,
            "PeakGPU_MB": peak_gpu_mb,
            "NumParams": n_params,
            "Epochs": cfg.epochs,
            "EpochsTrained": epochs_trained,
            "latent": latent,
            "history": history,
        }
        result.update(metrics)
        result.update(diagnostics)
        return result

    except Exception as exc:
        if device.type == "cuda" and is_cuda_oom(exc):
            print(f"  CUDA OOM → retrying on CPU ...")
            torch.cuda.empty_cache()
            gc.collect()
            return train_single_model(
                model_name, model_spec, splitter, cfg,
                torch.device("cpu"), data_type)
        elapsed = time.time() - start
        print(f"  ERROR: {str(exc)[:150]}")
        traceback.print_exc()
        return {
            "Model": model_name,
            "Time_s": elapsed,
            "Error": str(exc)[:200],
            "latent": None,
            "history": {},
            "NMI": 0, "ARI": 0,
            "Epochs": cfg.epochs,
            "EpochsTrained": 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: Main experiment loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg: ExperimentConfig, device: torch.device = None):
    """Execute full experiment: iterate datasets × models → CSV tables.

    Supports resume: already-completed datasets (CSV exists) are skipped.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print(cfg.summary())
    print("=" * 70 + "\n")

    _ensure_dirs(cfg.tables_dir, cfg.series_dir)

    # Resume support
    existing_csvs = list(cfg.tables_dir.glob("*.csv"))
    done_datasets = set()
    for csv_path in existing_csvs:
        stem = csv_path.stem
        if stem.endswith("_df"):
            done_datasets.add(stem[:-3])
    if done_datasets:
        print(f"Resume: {len(done_datasets)} datasets already completed → "
              f"{sorted(done_datasets)}")

    cache_dir = cfg.output_root / cfg.name / "cache"
    _ensure_dirs(cache_dir)

    total_datasets = len(cfg.datasets)
    completed = 0

    for ds_idx, (ds_key, ds_info) in enumerate(cfg.datasets.items(), 1):
        if ds_key in done_datasets:
            print(f"\n[{ds_idx}/{total_datasets}] {ds_key}: already done, skipping")
            completed += 1
            continue

        ds_path = ds_info["path"]
        label_key = ds_info.get("label_key", "cell_type")
        data_type = ds_info.get("data_type", "cluster")

        print(f"\n{'='*70}")
        print(f"[{ds_idx}/{total_datasets}] Dataset: {ds_key}")
        print(f"  Path: {ds_path}")
        print(f"{'='*70}")

        if not Path(ds_path).exists():
            print(f"  WARNING: File not found → skipping")
            continue

        try:
            adata = load_data(ds_path, cfg.max_cells, cfg.n_hvg, cfg.seed,
                              cache_dir=cache_dir)
            adata = standardize_labels(adata, label_key)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue

        try:
            splitter = create_data_splitter(
                adata, cfg.batch_size, cfg.seed, cfg.latent_dim)
        except Exception as e:
            print(f"  ERROR creating DataSplitter: {e}")
            continue

        results = []
        for model_name, model_spec in cfg.models.items():
            res = train_single_model(
                model_name, model_spec, splitter, cfg, device, data_type)
            results.append(res)
            gc.collect()
            torch.cuda.empty_cache()

        # Assemble per-dataset CSV
        rows = []
        for res in results:
            row = {"method": res["Model"]}
            for col in METRIC_COLUMNS:
                row[col] = res.get(col, np.nan)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = cfg.tables_dir / f"{ds_key}_df.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}  "
              f"({len(df)} methods × {len(METRIC_COLUMNS)} metrics)")

        # Save training series CSV
        series_rows = []
        for res in results:
            history = res.get("history", {})
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            recon_loss = history.get("train_recon_loss", train_loss)
            val_recon = history.get("val_recon_loss", [])
            for ep_idx in range(len(train_loss)):
                series_rows.append({
                    "epoch": ep_idx + 1,
                    "hue": res["Model"],
                    "train_loss": train_loss[ep_idx],
                    "val_loss": (val_loss[ep_idx]
                                 if ep_idx < len(val_loss) else np.nan),
                    "recon_loss": (recon_loss[ep_idx]
                                  if ep_idx < len(recon_loss) else np.nan),
                    "val_recon_loss": (val_recon[ep_idx]
                                      if ep_idx < len(val_recon) else np.nan),
                })
        if series_rows:
            dfs = pd.DataFrame(series_rows)
            series_path = cfg.series_dir / f"{ds_key}_dfs.csv"
            dfs.to_csv(series_path, index=False)

        completed += 1
        print(f"\n  Done: {ds_key}  ({completed}/{total_datasets})")

    print(f"\n{'='*70}")
    print(f"Experiment '{cfg.name}' finished: {completed}/{total_datasets} datasets")
    print(f"Results: {cfg.tables_dir}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.run_experiment --preset my_ablation
  python -m experiments.run_experiment --preset my_ablation --epochs 50
  python -m experiments.run_experiment --preset my_ablation --datasets ds_a ds_b
        """)
    parser.add_argument(
        "--preset", type=str, required=True,
        choices=list(PRESETS.keys()),
        help="Experiment preset name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--n-hvg", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dre-k", type=int, default=None,
                        help="k for DRE k-nearest-neighbour quality")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Run only these dataset keys")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--verbose-every", type=int, default=None)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution")

    # ── Model hyperparameter overrides ────────────────────────────────────
    # ██  PROJECT-SPECIFIC: Add/remove CLI flags matching your models' params
    model_grp = parser.add_argument_group(
        "Model hyperparameters",
        "Override model params across ALL model variants in the preset.")
    model_grp.add_argument("--model-lr", type=float, default=None,
                           help="Learning rate override")
    model_grp.add_argument("--model-wd", type=float, default=None,
                           help="Weight decay override")
    model_grp.add_argument("--model-dropout", type=float, default=None,
                           help="Dropout rate override")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PROJECT-SPECIFIC: Model HP override mapping
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_HP_MAP = {
    'model_lr': [('fit_lr', None)],
    'model_wd': [('fit_weight_decay', None)],
    'model_dropout': [('dropout_rate', None), ('dropout', None),
                      ('encoder_drop', None)],
}


def main():
    args = parse_args()

    cfg = PRESETS[args.preset]

    overrides = {}
    for field_name in ('epochs', 'max_cells', 'n_hvg', 'latent_dim',
                       'batch_size', 'seed', 'verbose_every', 'dre_k'):
        val = getattr(args, field_name.replace('-', '_'), None)
        if val is not None:
            overrides[field_name] = val
    if args.output_root is not None:
        overrides["output_root"] = Path(args.output_root)
    if args.datasets is not None:
        overrides["datasets"] = {
            k: v for k, v in cfg.datasets.items() if k in args.datasets
        }
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    # Apply model-level HP overrides
    hp_applied = []
    for cli_key, param_variants in _MODEL_HP_MAP.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            for _, model_spec in cfg.models.items():
                for param_key, _ in param_variants:
                    if param_key in model_spec["params"]:
                        model_spec["params"][param_key] = val
            hp_applied.append(f"{cli_key}={val}")
    if hp_applied:
        print(f"Model HP overrides: {', '.join(hp_applied)}")

    _set_global_seed(cfg.seed)

    device = (torch.device("cpu") if args.cpu else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    run_experiment(cfg, device=device)


if __name__ == "__main__":
    main()
