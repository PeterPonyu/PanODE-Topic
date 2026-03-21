#!/usr/bin/env python
"""
External Baseline Benchmark Runner
====================================

Trains external baseline models on the same 16 scRNA-seq datasets used by
``run_experiment.py``, using **identical preprocessing, data splits, and
metric computation**.

Model groups (for figure integration):
  - **Generative**: CellBLAST, SCALEX, scDiffusion, siVAE, scDAC,
    scDeepCluster, scDHMap, scSMD
  - **Gaussian geometric (GM-VAE series)**: GMVAE, GMVAE-Poincare,
    GMVAE-PGM, GMVAE-LearnablePGM, GMVAE-HW
  - **Disentanglement**: VAE-DIP, VAE-TC, InfoVAE, BetaVAE
  - **Graph & contrastive**: CLEAR, scGCC, scGNN
  - **scVI family** (requires scvi-tools): scVI, PeakVI, PoissonVI

Architecture
------------
- Uses ``eval_lib.baselines.registry.EXTERNAL_MODELS`` for model catalogue
- Uses ``eval_lib.metrics.battery.METRIC_COLUMNS`` for canonical column order
- Uses ``eval_lib.metrics.battery.compute_metrics()`` for unified evaluation
- Uses ``eval_lib.metrics.battery.compute_latent_diagnostics()`` for latent QC
- Uses ``eval_lib.metrics.battery.convergence_diagnostics()`` for convergence
- Identical preprocessing via ``benchmarks.data_utils.load_or_preprocess_adata``
- Identical data splits via ``utils.data.DataSplitter``

Output Layout
-------------
::

    experiments/results/external/
        tables/      <- one CSV per dataset ({ds}_df.csv)
        series/      <- per-epoch loss series
        latents/     <- .npz per model per dataset

    # When using --group:
    experiments/results/external/{group_name}/
        tables/  series/  latents/

Usage
-----
::

    # All external models on all 16 datasets
    python -m experiments.run_external_benchmark

    # Specific models only
    python -m experiments.run_external_benchmark \\
        --models CellBLAST GMVAE scVI --datasets setty endo

    # Run a single model group
    python -m experiments.run_external_benchmark --group generative

    # Run all groups into separate subdirectories
    python -m experiments.run_external_benchmark --all-groups

    # Skip models with special dependencies
    python -m experiments.run_external_benchmark --skip scGCC scGNN

    # Override training budget for quick test
    python -m experiments.run_external_benchmark --epochs 50 --datasets setty

    # Force CPU
    python -m experiments.run_external_benchmark --cpu
"""

from __future__ import annotations

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
import torch

warnings.filterwarnings("ignore")

# -- Project imports -----------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# eval_lib -- portable evaluation toolkit
from eval_lib.baselines.registry import (
    EXTERNAL_MODELS,
    MODEL_GROUPS,
    list_external_models,
    list_model_groups)
from eval_lib.metrics.battery import (
    METRIC_COLUMNS,
    compute_metrics,
    compute_latent_diagnostics,
    convergence_diagnostics)

# Project infrastructure (data loading, splitting, seeding)
from benchmarks.config import set_global_seed, ensure_dirs
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.model_registry import is_cuda_oom
from utils.data import DataSplitter

# Dataset catalogue (16 scRNA-seq datasets + full catalogue)
from experiments.experiment_config import SCRNA_16_DATASETS, SCRNA_ALL_DATASETS

# Label standardisation (shared with run_experiment.py)
from experiments.run_experiment import standardize_labels


# ==============================================================================
# Defaults -- match internal experiments exactly for fair comparison
# ==============================================================================

DEFAULT_EPOCHS     = 1000   # production; use --epochs 100 for quick runs
DEFAULT_LR         = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_LATENT_DIM = 10
DEFAULT_MAX_CELLS  = 3000
DEFAULT_N_HVG      = 2000
DEFAULT_SEED       = 42
DEFAULT_PATIENCE   = 100      # external models use early stopping
DEFAULT_VERBOSE    = 25
DEFAULT_DRE_K      = 15       # k for DRE k-nearest-neighbour quality

OUTPUT_ROOT = Path("experiments/results")
EXPERIMENT_NAME = "external"


# ==============================================================================
# Single model training + metric computation
# ==============================================================================

def train_external_model(
    model_name: str,
    model_cfg: dict,
    splitter: DataSplitter,
    device: torch.device,
    data_type: str,
    epochs: int,
    lr: float,
    patience: int,
    verbose_every: int,
    dre_k: int = DEFAULT_DRE_K) -> dict:
    """Train one external baseline, extract latent, compute full metrics.

    Uses ``eval_lib.metrics.battery.compute_metrics()`` -- identical to the
    internal runner, producing columns matching ``METRIC_COLUMNS``.

    Parameters
    ----------
    model_name : str
        Display name for this model (e.g. ``'CellBLAST'``).
    model_cfg : dict
        Registry entry with ``factory``, ``params``, ``fit_params``, ``notes``.
    splitter : DataSplitter
        Project DataSplitter with ``.train_loader``, ``.test_loader``,
        ``.labels_test``, ``.n_var``.
    device : torch.device
        Compute device.
    data_type : str
        ``'cluster'`` or ``'trajectory'`` -- affects LSE scoring.
    epochs : int
        Training epochs (may differ from registry ``fit_params``).
    lr : float
        Learning rate.
    patience : int
        Early stopping patience.
    verbose_every : int
        Print progress interval.
    dre_k : int
        k for DRE k-nearest-neighbour quality.

    Returns
    -------
    dict
        ``{"Model": name, <metric columns>, "latent": ndarray, "history": dict}``
    """
    gc.collect()
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    print(f"\n{'─'*60}\n  Training: {model_name}  ({epochs} ep, lr={lr})\n{'─'*60}")

    try:
        factory = model_cfg["factory"]
        params = dict(model_cfg["params"])
        fit_params = model_cfg.get("fit_params") or {}
        # Per-model training overrides from registry (e.g. scVI 400 ep, patience 45)
        run_epochs = fit_params.get("epochs", epochs)
        run_lr = fit_params.get("lr", lr)
        run_patience = fit_params.get("patience", patience)

        # Create model via registry factory
        model = factory(input_dim=splitter.n_var, **params)
        model = model.to(device)

        # Train using the unified BaseModel.fit() interface
        history = model.fit(
            train_loader=splitter.train_loader,
            val_loader=splitter.val_loader,
            epochs=run_epochs,
            lr=run_lr,
            device=str(device),
            patience=run_patience,
            verbose=1,
            verbose_every=verbose_every)

        elapsed = time.time() - start
        epochs_trained = len(history.get("train_loss", [])) or run_epochs
        sec_per_epoch = elapsed / max(epochs_trained, 1)

        peak_gpu_mb = 0.0
        if device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Extract latent representations (test split for evaluation)
        latent_dict = model.extract_latent(
            splitter.test_loader, device=str(device))
        latent = latent_dict["latent"]

        # Full metric battery from eval_lib (all 6 suites)
        metrics = compute_metrics(
            latent, splitter.labels_test,
            data_type=data_type, dre_k=dre_k)
        diagnostics = compute_latent_diagnostics(latent)
        n_params = sum(p.numel() for p in model.parameters())

        # Convergence diagnostics
        conv_diag = convergence_diagnostics(history, window=50)

        print(
            f"  NMI={metrics['NMI']:.4f}  ARI={metrics['ARI']:.4f}  "
            f"ASW={metrics.get('ASW', float('nan')):.4f}  |  "
            f"{elapsed:.1f}s ({sec_per_epoch:.2f}s/ep, GPU {peak_gpu_mb:.0f}MB, "
            f"params {n_params:,})"
        )
        if conv_diag.get("converged") is not None:
            print(f"  Converged: {conv_diag['converged']}  "
                  f"(recon delta: {conv_diag.get('recon_rel_change_pct', '?')}%)")

        result = {
            "Model": model_name,
            "Time_s": elapsed,
            "SecPerEpoch": sec_per_epoch,
            "PeakGPU_MB": peak_gpu_mb,
            "NumParams": n_params,
            "Epochs": run_epochs,
            "EpochsTrained": epochs_trained,
            "latent": latent,
            "history": history,
        }
        result.update(metrics)
        result.update(diagnostics)
        return result

    except Exception as exc:
        if device.type == "cuda" and is_cuda_oom(exc):
            print(f"  CUDA OOM -> retrying on CPU ...")
            torch.cuda.empty_cache()
            gc.collect()
            return train_external_model(
                model_name, model_cfg, splitter,
                torch.device("cpu"), data_type,
                epochs, lr, patience, verbose_every, dre_k)
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
            "Epochs": epochs,
            "EpochsTrained": 0,
        }


# ==============================================================================
# Main benchmark loop
# ==============================================================================

def run_external_benchmark(
    selected_models: dict,
    datasets: dict,
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    latent_dim: int = DEFAULT_LATENT_DIM,
    max_cells: int = DEFAULT_MAX_CELLS,
    n_hvg: int = DEFAULT_N_HVG,
    seed: int = DEFAULT_SEED,
    patience: int = DEFAULT_PATIENCE,
    verbose_every: int = DEFAULT_VERBOSE,
    dre_k: int = DEFAULT_DRE_K,
    output_root: Path = OUTPUT_ROOT,
    experiment_name: str = EXPERIMENT_NAME,
    device: torch.device = None):
    """Execute external benchmark: iterate datasets x models -> CSV tables.

    Supports resume: already-completed datasets (CSV exists) are skipped.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tables_dir = output_root / experiment_name / "tables"
    series_dir = output_root / experiment_name / "series"
    latents_dir = output_root / experiment_name / "latents"
    cache_dir = output_root / experiment_name / "cache"
    ensure_dirs(tables_dir, series_dir, latents_dir, cache_dir)

    method_names = list(selected_models.keys())

    # -- Print experiment summary ----------------------------------------------
    print("\n" + "=" * 70)
    print("EXTERNAL BASELINE BENCHMARK")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Models ({len(method_names)}): {method_names}")
    print(f"Datasets   : {len(datasets)} datasets")
    print(f"Epochs     : {epochs}")
    print(f"LR         : {lr}")
    print(f"Batch size : {batch_size}")
    print(f"Latent dim : {latent_dim}")
    print(f"Max cells  : {max_cells}")
    print(f"HVGs       : {n_hvg}")
    print(f"Seed       : {seed}")
    print(f"Patience   : {patience}")
    print(f"DRE k      : {dre_k}")
    print(f"Output     : {output_root / experiment_name}")
    print("=" * 70 + "\n")

    # -- Resume support --------------------------------------------------------
    existing_csvs = list(tables_dir.glob("*.csv"))
    done_datasets = set()
    for csv_path in existing_csvs:
        stem = csv_path.stem
        if stem.endswith("_df"):
            done_datasets.add(stem[:-3])
    if done_datasets:
        print(f"Resume: {len(done_datasets)} datasets already completed -> "
              f"{sorted(done_datasets)}")

    total = len(datasets)
    completed = 0

    for ds_idx, (ds_key, ds_info) in enumerate(datasets.items(), 1):
        if ds_key in done_datasets:
            print(f"\n[{ds_idx}/{total}] {ds_key}: already done, skipping")
            completed += 1
            continue

        ds_path = ds_info["path"]
        label_key = ds_info.get("label_key", "cell_type")
        data_type = ds_info.get("data_type", "cluster")

        print(f"\n{'='*70}")
        print(f"[{ds_idx}/{total}] Dataset: {ds_key}")
        print(f"  Path: {ds_path}")
        print(f"{'='*70}")

        if not Path(ds_path).exists():
            print(f"  WARNING: File not found -> skipping")
            continue

        # Load and preprocess (identical to run_experiment.py)
        try:
            adata = load_or_preprocess_adata(
                ds_path, max_cells, n_hvg, seed,
                cache_dir=cache_dir, use_cache=True)
            adata = standardize_labels(adata, label_key)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue

        # Create DataSplitter (same 70/15/15 split)
        try:
            splitter = DataSplitter(
                adata,
                layer="counts",
                batch_size=batch_size,
                random_seed=seed,
                latent_dim=latent_dim,
                verbose=True)
        except Exception as e:
            print(f"  ERROR creating DataSplitter: {e}")
            continue

        # Train all selected external models
        results = []
        for model_name, model_cfg in selected_models.items():
            res = train_external_model(
                model_name, model_cfg, splitter, device,
                data_type, epochs, lr, patience, verbose_every, dre_k)
            results.append(res)
            gc.collect()
            torch.cuda.empty_cache()

        # Assemble per-dataset CSV table
        rows = []
        for res in results:
            row = {"method": res["Model"]}
            for col in METRIC_COLUMNS:
                row[col] = res.get(col, np.nan)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = tables_dir / f"{ds_key}_df.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}  "
              f"({len(df)} methods x {len(METRIC_COLUMNS)} metrics)")

        # Save training series (per-epoch loss curves)
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
            series_path = series_dir / f"{ds_key}_dfs.csv"
            dfs.to_csv(series_path, index=False)

        # Save latent arrays (.npz for downstream analysis)
        for res in results:
            lat = res.get("latent")
            if lat is not None:
                safe_name = res["Model"].replace("/", "_").replace(" ", "_")
                lat_path = latents_dir / f"{ds_key}_{safe_name}.npz"
                np.savez_compressed(lat_path, latent=lat,
                                    labels=splitter.labels_test)

        completed += 1
        print(f"\n  Done: {ds_key}  ({completed}/{total})")

    print(f"\n{'='*70}")
    print(f"External benchmark finished: {completed}/{total} datasets")
    print(f"Results: {tables_dir}")
    print(f"{'='*70}\n")


# ==============================================================================
# CLI entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run external baseline models on scRNA-seq datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All external models on all 16 datasets
  python -m experiments.run_external_benchmark

  # Quick test: 2 models, 1 dataset (short epochs)
  python -m experiments.run_external_benchmark \\
      --models CellBLAST GMVAE --datasets setty --epochs 100

  # Run a single model group into external/{group}/
  python -m experiments.run_external_benchmark --group generative
  python -m experiments.run_external_benchmark --group disentanglement

  # Run ALL groups, each into its own subdirectory
  python -m experiments.run_external_benchmark --all-groups

  # Skip models requiring torch_geometric
  python -m experiments.run_external_benchmark --skip scGCC scGNN

  # Include scVI-family models
  python -m experiments.run_external_benchmark --models scVI PeakVI PoissonVI

  # Force CPU
  python -m experiments.run_external_benchmark --cpu
        """)

    parser.add_argument("--models", nargs="+", default=None,
                        help="External models to run (default: all). "
                             f"Available: {list(EXTERNAL_MODELS.keys())}")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Models to skip (e.g. --skip scGCC scGNN)")
    parser.add_argument("--group", type=str, default=None,
                        choices=list(MODEL_GROUPS.keys()),
                        help="Run only models in this group. Output goes to "
                             "external/{group}/. "
                             f"Groups: {list(MODEL_GROUPS.keys())}")
    parser.add_argument("--all-groups", action="store_true",
                        help="Run each group sequentially into separate "
                             "subdirectories (external/{group}/)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to run (default: all 16). "
                             f"Available: {list(SCRNA_16_DATASETS.keys())}")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Use the full dataset catalogue (56 datasets) instead of the core 16")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM,
                        help=f"Latent dimension (default: {DEFAULT_LATENT_DIM})")
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS,
                        help=f"Max cells (default: {DEFAULT_MAX_CELLS})")
    parser.add_argument("--n-hvg", type=int, default=DEFAULT_N_HVG,
                        help=f"HVG count (default: {DEFAULT_N_HVG})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})")
    parser.add_argument("--dre-k", type=int, default=DEFAULT_DRE_K,
                        help=f"k for DRE quality (default: {DEFAULT_DRE_K})")
    parser.add_argument("--verbose-every", type=int, default=DEFAULT_VERBOSE,
                        help=f"Print every N epochs (default: {DEFAULT_VERBOSE})")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT),
                        help="Output root directory")
    parser.add_argument("--name", type=str, default=EXPERIMENT_NAME,
                        help=f"Experiment name (default: {EXPERIMENT_NAME})")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution")

    args = parser.parse_args()

    # -- Select models ---------------------------------------------------------
    if args.models:
        selected = {}
        for m in args.models:
            if m in EXTERNAL_MODELS:
                selected[m] = EXTERNAL_MODELS[m]
            else:
                print(f"WARNING: Unknown model '{m}' -- skipping. "
                      f"Available: {list(EXTERNAL_MODELS.keys())}")
        if not selected:
            print("ERROR: No valid models selected.")
            sys.exit(1)
    else:
        selected = dict(EXTERNAL_MODELS)

    if args.skip:
        for s in args.skip:
            removed = selected.pop(s, None)
            if removed:
                print(f"  Skipping: {s}")

    if not selected:
        print("ERROR: No models remaining after --skip.")
        sys.exit(1)

    # -- Select datasets -------------------------------------------------------
    base_datasets = SCRNA_ALL_DATASETS if args.all_datasets else SCRNA_16_DATASETS
    if args.datasets:
        datasets = {}
        for d in args.datasets:
            if d in base_datasets:
                datasets[d] = base_datasets[d]
            else:
                print(f"WARNING: Unknown dataset '{d}' -- skipping. "
                      f"Available: {list(base_datasets.keys())}")
    else:
        datasets = dict(base_datasets)

    if not datasets:
        print("ERROR: No valid datasets selected.")
        sys.exit(1)

    # -- Print model registry and group summary --------------------------------
    print("\nAvailable external models:")
    list_external_models()
    print()
    list_model_groups()
    scvi_models = [m for m in ["scVI", "PeakVI", "PoissonVI"] if m in selected]
    if scvi_models:
        print(f"\n  scVI family included: {scvi_models} (will use registry epochs/patience)")
    else:
        from eval_lib.baselines.registry import _SCVI_AVAILABLE
        if _SCVI_AVAILABLE:
            print("\n  scVI family available but not in --models; add e.g. --models scVI PeakVI PoissonVI to run them.")
        else:
            print("\n  scVI family not installed (pip install scvi-tools for scVI, PeakVI, PoissonVI).")

    # -- Seed and device -------------------------------------------------------
    set_global_seed(args.seed)
    device = (torch.device("cpu") if args.cpu else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # -- Common run kwargs -----------------------------------------------------
    run_kwargs = dict(
        datasets=datasets,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        max_cells=args.max_cells,
        n_hvg=args.n_hvg,
        seed=args.seed,
        patience=args.patience,
        verbose_every=args.verbose_every,
        dre_k=args.dre_k,
        output_root=Path(args.output_root),
        device=device)

    # -- Group filter ----------------------------------------------------------
    if args.group:
        # Run a single group into external/{group}/
        group_model_names = set(MODEL_GROUPS.get(args.group, []))
        group_selected = {k: v for k, v in selected.items()
                          if k in group_model_names}
        if not group_selected:
            print(f"ERROR: No available models for group '{args.group}'.")
            sys.exit(1)
        experiment_name = (f"external/{args.group}"
                           if args.name == EXPERIMENT_NAME else args.name)
        print(f"\nGroup '{args.group}': {list(group_selected.keys())}")
        print(f"Output: {Path(args.output_root) / experiment_name}\n")
        run_external_benchmark(
            selected_models=group_selected,
            experiment_name=experiment_name,
            **run_kwargs)

    elif args.all_groups:
        # Run every group sequentially, each into external/{group}/
        for group_name, group_model_names in MODEL_GROUPS.items():
            group_selected = {k: v for k, v in selected.items()
                              if k in group_model_names}
            if not group_selected:
                print(f"\nGroup '{group_name}': no available models, skipping")
                continue
            experiment_name = f"external/{group_name}"
            print(f"\n{'#'*70}")
            print(f"# GROUP: {group_name} ({len(group_selected)} models)")
            print(f"# Output: {Path(args.output_root) / experiment_name}")
            print(f"{'#'*70}")
            run_external_benchmark(
                selected_models=group_selected,
                experiment_name=experiment_name,
                **run_kwargs)

    else:
        # Default: all selected models into external/
        print(f"\nSelected ({len(selected)}): {list(selected.keys())}\n")
        run_external_benchmark(
            selected_models=selected,
            experiment_name=args.name,
            **run_kwargs)


if __name__ == "__main__":
    main()
