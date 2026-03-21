#!/usr/bin/env python
"""
External Model Benchmark — Cross-Dataset Evaluation

.. deprecated::
    This runner is **superseded** by ``experiments/run_external_benchmark.py``
    which uses the complete ``eval_lib.baselines.registry`` (21+ models,
    5 groups) and writes to the canonical ``experiments/results/external/``
    directory.  Use::

        python -m experiments.run_external_benchmark          # all models
        python -m experiments.run_external_benchmark --group generative

Evaluates 12 external baseline models (from Liora unified_models) on the
same 12 datasets used for internal DPMM/Topic variant evaluation.  Results
are saved alongside internal results for direct comparison.

External models:
  CellBLAST, GMVAE, SCALEX, scDiffusion, siVAE, CLEAR,
  scDAC, scDeepCluster, scDHMap, scGNN, scGCC, scSMD

Outputs (under benchmark_results/external/):
  csv/        — per-dataset + combined CSV results
  plots/      — per-dataset metric barplots
  meta/       — JSON run metadata
  latents/    — per-model latent .npz files

Usage:
  # All models, all datasets
  python benchmarks/benchmark_external.py

  # Specific models
  python benchmarks/benchmark_external.py --models CellBLAST GMVAE SCALEX

  # Specific datasets
  python benchmarks/benchmark_external.py --datasets setty lung endo

  # Skip problematic models (e.g. those needing torch_geometric)
  python benchmarks/benchmark_external.py --skip scGCC

  # Compare with internal best
  python benchmarks/benchmark_external.py --compare
"""

import sys
import os
import argparse
import json
import gc
import time
import warnings
import traceback
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from benchmarks.config import BASE_CONFIG, DEFAULT_OUTPUT_DIR, ensure_dirs, set_global_seed
from benchmarks.dataset_registry import DATASET_REGISTRY
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.metrics_utils import (
    compute_metrics,
    compute_latent_diagnostics,
    convergence_diagnostics)
from benchmarks.model_registry import is_cuda_oom

from utils.data import DataSplitter

# ── Defaults ──────────────────────────────────────────────────────────────────
LATENT_DIM    = BASE_CONFIG.latent_dim
LR            = BASE_CONFIG.lr
BATCH_SIZE    = BASE_CONFIG.batch_size
DEVICE        = BASE_CONFIG.device
SEED          = BASE_CONFIG.seed
VERBOSE_EVERY = BASE_CONFIG.verbose_every
HVG_TOP       = BASE_CONFIG.hvg_top_genes
MAX_CELLS     = BASE_CONFIG.max_cells
DRE_K         = BASE_CONFIG.dre_k

# Output dirs
EXT_DIR   = DEFAULT_OUTPUT_DIR / "external"
CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
OUT_DIRS  = {
    "csv":     EXT_DIR / "csv",
    "plots":   EXT_DIR / "plots",
    "meta":    EXT_DIR / "meta",
    "latents": EXT_DIR / "latents",
}


def standardize_labels(adata, label_key):
    """Copy dataset-specific label column to 'cell_type'."""
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()
    elif "cell_type" not in adata.obs.columns:
        print(f"  WARNING: '{label_key}' not found; KMeans pseudo-labels.")
    return adata


# ═══════════════════════════════════════════════════════════════════════════════
# Training wrapper for external models
# ═══════════════════════════════════════════════════════════════════════════════

def train_external_model(model_name, model_cfg, splitter, device, data_type,
                         verbose_every=50, dre_k=15):
    """Train a single external model, compute metrics, return result dict.

    Mirrors the internal ``train_and_evaluate`` but works with the
    Liora BaseModel interface (factory + .fit + .extract_latent).
    """
    gc.collect()
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    print(f"\n{'='*60}\nTraining External: {model_name}\n{'='*60}")

    try:
        factory = model_cfg["factory"]
        params  = dict(model_cfg["params"])
        fp      = dict(model_cfg.get("fit_params", {}))

        epochs       = fp.get("epochs", 1000)
        lr           = fp.get("lr", LR)
        patience     = fp.get("patience", 100)
        vb_every     = fp.get("verbose_every", verbose_every)

        # Create model
        model = factory(input_dim=splitter.n_var, **params)
        model = model.to(device)

        # Train
        history = model.fit(
            train_loader=splitter.train_loader,
            val_loader=splitter.val_loader,
            epochs=epochs,
            lr=lr,
            device=str(device),
            patience=patience,
            verbose=1,
            verbose_every=vb_every)

        epochs_trained = len(history.get("train_loss", [])) or epochs
        elapsed = time.time() - start
        sec_per_epoch = elapsed / max(epochs_trained, 1)

        peak_gpu_mb = 0.0
        if device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Convergence diagnostics
        conv_diag = convergence_diagnostics(history, window=50)

        # Extract latent
        latent_dict = model.extract_latent(
            splitter.test_loader, device=str(device))
        latent = latent_dict["latent"]

        # Compute metrics
        metrics = compute_metrics(
            latent, splitter.labels_test,
            data_type=data_type, dre_k=dre_k)
        diagnostics = compute_latent_diagnostics(latent)
        n_params = sum(p.numel() for p in model.parameters())

        print(
            f"  NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f} "
            f"ASW={metrics['ASW']:.4f} | {elapsed:.1f}s "
            f"({sec_per_epoch:.2f}s/ep, GPU {peak_gpu_mb:.0f}MB, "
            f"params {n_params:,})"
        )

        result = {
            "Model": model_name,
            "Series": "external",
            "Time_s": elapsed,
            "SecPerEpoch": sec_per_epoch,
            "PeakGPU_MB": peak_gpu_mb,
            "NumParams": n_params,
            "LR": lr,
            "Epochs": epochs,
            "EpochsTrained": epochs_trained,
            "latent": latent,
        }
        result.update(metrics)
        result.update(diagnostics)
        if conv_diag:
            result.update(conv_diag)
        return result

    except Exception as exc:
        if device.type == "cuda" and is_cuda_oom(exc):
            print("  CUDA OOM → retry on CPU…")
            torch.cuda.empty_cache()
            gc.collect()
            return train_external_model(
                model_name, model_cfg, splitter,
                torch.device("cpu"), data_type,
                verbose_every, dre_k)
        elapsed = time.time() - start
        print(f"  ERROR ({model_name}): {str(exc)[:200]}")
        traceback.print_exc()
        return {
            "Model": model_name,
            "Series": "external",
            "Time_s": elapsed,
            "Error": str(exc)[:300],
            "latent": None,
            "NMI": np.nan, "ARI": np.nan, "ASW": np.nan,
            "Epochs": model_cfg.get("fit_params", {}).get("epochs", 0),
            "EpochsTrained": 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_dataset(ds_key, ds_info, selected_models, device, seed, no_plots):
    """Run external models on one dataset."""
    print(f"\n{'#'*70}")
    print(f"# DATASET: {ds_key} — {ds_info['desc']}")
    print(f"# Path: {ds_info['path']}")
    print(f"# Data type: {ds_info['data_type']}")
    print(f"{'#'*70}")

    adata = load_or_preprocess_adata(
        ds_info["path"], max_cells=MAX_CELLS, hvg_top_genes=HVG_TOP,
        seed=seed, cache_dir=str(CACHE_DIR), use_cache=True)
    adata = standardize_labels(adata, ds_info["label_key"])

    splitter = DataSplitter(
        adata=adata, layer="counts",
        train_size=0.7, val_size=0.15, test_size=0.15,
        batch_size=BATCH_SIZE, latent_dim=LATENT_DIM,
        random_seed=seed, verbose=True)

    n_cells  = int(adata.n_obs)
    n_genes  = int(adata.n_vars)
    n_labels = len(np.unique(splitter.labels)) if splitter.labels is not None else 0
    print(f"  Shape: {n_cells} cells × {n_genes} genes, {n_labels} labels")

    results, latents = [], {}
    for mname, mcfg in selected_models.items():
        r = train_external_model(
            mname, mcfg, splitter, device,
            ds_info["data_type"], VERBOSE_EVERY, DRE_K)
        if r.get("latent") is not None:
            latents[mname] = r.pop("latent")
        else:
            r.pop("latent", None)
        r["Dataset"]  = ds_key
        r["DataType"] = ds_info["data_type"]
        r["Cells"]    = n_cells
        r["Genes"]    = n_genes
        r["Labels"]   = n_labels
        results.append(r)
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ds_key}_{timestamp}"

    # Save CSV
    csv_p = OUT_DIRS["csv"] / f"results_{tag}.csv"
    df.to_csv(csv_p, index=False)
    print(f"\n  CSV: {csv_p}")

    # Save latents
    try:
        lat_dir = OUT_DIRS["latents"] / ds_key
        ensure_dirs(lat_dir)
        labels_test = splitter.labels_test
        for model_name, lat_arr in latents.items():
            safe_name = model_name.replace("/", "_").replace(" ", "_")
            lp = lat_dir / f"{safe_name}_{tag}.npz"
            if labels_test is not None:
                np.savez(lp, latent=lat_arr, labels=labels_test)
            else:
                np.savez(lp, latent=lat_arr)
        print(f"  Latents: {lat_dir}")
    except Exception as e:
        print(f"  Latent save err: {e}")

    # Save metadata
    meta = {
        "timestamp": timestamp,
        "script": "benchmark_external.py",
        "dataset": ds_key,
        "data_path": ds_info["path"],
        "data_type": ds_info["data_type"],
        "label_key": ds_info["label_key"],
        "n_cells": n_cells, "n_genes": n_genes, "n_labels": n_labels,
        "lr": LR, "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "hvg": HVG_TOP, "max_cells": MAX_CELLS,
        "seed": seed,
        "models": df["Model"].tolist(),
    }
    meta_p = OUT_DIRS["meta"] / f"run_{tag}.json"
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison with internal best
# ═══════════════════════════════════════════════════════════════════════════════

def load_internal_best():
    """Load best internal model results per dataset.

    Returns DataFrame with columns: Dataset, Model, Series, NMI, ARI, ASW, DAV
    for the best DPMM variant and best Topic variant per dataset.
    """
    cross_dir = DEFAULT_OUTPUT_DIR / "crossdata" / "csv"

    # DPMM series results (latest complete run)
    dpmm_file = None
    topic_file = None
    pure_file = None

    for f in sorted(cross_dir.glob("results_combined_*.csv"), reverse=True):
        try:
            df = pd.read_csv(f, usecols=["Model", "Series"])
            models = df["Model"].unique().tolist()
            series_vals = set(df["Series"].unique()) if "Series" in df.columns else set()
            if any("DPMM" in m for m in models) or "dpmm" in series_vals:
                if dpmm_file is None:
                    dpmm_file = f
            if any("Topic" in m for m in models) or "topic" in series_vals:
                if topic_file is None:
                    topic_file = f
            if any("Pure-AE" in m or "Pure-VAE" in m for m in models) \
               or "pure-ae" in series_vals or "pure-vae" in series_vals:
                if pure_file is None:
                    pure_file = f
        except Exception:
            continue

    dfs = []
    metric_cols = ["NMI", "ARI", "ASW"]

    file_series_pairs = [
        (dpmm_file, "dpmm"),
        (topic_file, "topic"),
    ]
    if pure_file is not None and pure_file not in (dpmm_file, topic_file):
        file_series_pairs.append((pure_file, "pure"))

    for fpath, series_name in file_series_pairs:
        if fpath is None:
            print(f"  WARNING: No crossdata results found for {series_name} series")
            continue
        df = pd.read_csv(fpath)
        # Ensure we have the needed columns
        avail = [c for c in metric_cols if c in df.columns]
        if not avail:
            continue
        # Compute composite score (mean of available metrics)
        df["_score"] = df[avail].mean(axis=1)
        # Best model per dataset
        best_idx = df.groupby("Dataset")["_score"].idxmax()
        best = df.loc[best_idx, ["Dataset", "Model"] + avail].copy()
        best["Series"] = series_name
        dfs.append(best)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def print_comparison(external_df, internal_best_df):
    """Print side-by-side comparison table."""
    if internal_best_df.empty:
        print("  No internal results available for comparison.")
        return

    metric_cols = ["NMI", "ARI", "ASW"]
    datasets = sorted(external_df["Dataset"].unique())

    print("\n" + "=" * 120)
    print("COMPARISON: External Models vs Best Internal Variants")
    print("=" * 120)

    for ds in datasets:
        ext_ds = external_df[external_df["Dataset"] == ds].copy()
        int_ds = internal_best_df[internal_best_df["Dataset"] == ds]

        if ext_ds.empty:
            continue

        print(f"\n── {ds} ──")
        print(f"  {'Model':<22} {'Type':<10} {'NMI':>7} {'ARI':>7} {'ASW':>7}")
        print("  " + "-" * 58)

        # Internal best
        for _, row in int_ds.iterrows():
            print(f"  {row['Model']:<22} {row['Series']:<10} "
                  f"{row.get('NMI', 0):>7.4f} {row.get('ARI', 0):>7.4f} "
                  f"{row.get('ASW', 0):>7.4f}  ★")

        # External models (sorted by NMI descending)
        ext_sorted = ext_ds.sort_values("NMI", ascending=False)
        for _, row in ext_sorted.iterrows():
            err_mark = " ✗" if pd.notna(row.get("Error")) else ""
            print(f"  {row['Model']:<22} {'external':<10} "
                  f"{row.get('NMI', 0):>7.4f} {row.get('ARI', 0):>7.4f} "
                  f"{row.get('ASW', 0):>7.4f}{err_mark}")

    # Aggregate ranking
    print(f"\n── Aggregate: Mean Metrics Across Datasets ──")
    avail = [c for c in metric_cols if c in external_df.columns]
    ext_mean = external_df.groupby("Model")[avail].mean()

    # Add internal bests
    all_rows = []
    for _, row in internal_best_df.iterrows():
        all_rows.append({
            "Model": f"{row['Model']} (best-{row['Series']})",
            **{c: row.get(c, np.nan) for c in avail},
        })
    for mname, mrow in ext_mean.iterrows():
        all_rows.append({"Model": mname, **{c: mrow[c] for c in avail}})

    agg = pd.DataFrame(all_rows)
    if avail:
        agg["Score"] = agg[avail].mean(axis=1)
        agg = agg.sort_values("Score", ascending=False)
        print(f"  {'Model':<35} " + " ".join(f"{c:>7}" for c in avail) + f" {'Score':>7}")
        print("  " + "-" * (35 + 8 * (len(avail) + 1)))
        for _, row in agg.iterrows():
            vals = " ".join(f"{row[c]:>7.4f}" if pd.notna(row[c]) else f"{'N/A':>7}"
                           for c in avail)
            score = f"{row['Score']:>7.4f}" if pd.notna(row.get("Score")) else f"{'N/A':>7}"
            print(f"  {row['Model']:<35} {vals} {score}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="External model benchmark (cross-dataset)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None,
                    choices=list(DATASET_REGISTRY.keys()),
                    help="Which datasets to run (default: all).")
    ap.add_argument("--models", nargs="+", default=None,
                    help="External models to run (default: all).")
    ap.add_argument("--skip", nargs="+", default=None,
                    help="External models to skip.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="Compare with best internal DPMM/Topic variants.")
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(DEVICE)

    # Import external model registry
    from benchmarks.external_model_registry import EXTERNAL_MODELS

    # Select models
    if args.models:
        selected_models = {k: v for k, v in EXTERNAL_MODELS.items()
                          if k in args.models}
    else:
        selected_models = dict(EXTERNAL_MODELS)

    if args.skip:
        for s in args.skip:
            selected_models.pop(s, None)

    if not selected_models:
        print("ERROR: No models selected.")
        sys.exit(1)

    ds_keys = args.datasets or list(DATASET_REGISTRY.keys())

    ensure_dirs(*OUT_DIRS.values(), CACHE_DIR)

    total_runs = len(ds_keys) * len(selected_models)
    print("=" * 70)
    print("EXTERNAL MODEL BENCHMARK")
    print("=" * 70)
    print(f"Device     : {DEVICE}")
    print(f"Datasets   : {ds_keys}")
    print(f"Models     : {list(selected_models.keys())}")
    print(f"Total runs : {total_runs}")
    print(f"HVG/Cells  : {HVG_TOP} / {MAX_CELLS}")
    print("=" * 70)

    all_dfs = []
    for ds_key in ds_keys:
        ds_info = DATASET_REGISTRY[ds_key]
        df = run_dataset(ds_key, ds_info, selected_models, device,
                        args.seed, args.no_plots)
        all_dfs.append(df)

    # Combined results
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_csv = OUT_DIRS["csv"] / f"results_combined_{timestamp}.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nCombined CSV: {combined_csv}")
    else:
        combined = pd.DataFrame()

    # Print summary
    print("\n" + "=" * 110)
    print("EXTERNAL MODEL SUMMARY")
    print("=" * 110)

    metric_cols = ["NMI", "ARI", "ASW", "DAV"]
    for ds_key in ds_keys:
        sdf = combined[combined["Dataset"] == ds_key] if "Dataset" in combined.columns else combined
        if sdf.empty:
            continue
        dt = DATASET_REGISTRY[ds_key]["data_type"]
        print(f"\n── {ds_key} ({dt}) ──")
        print(f"  {'Model':<18} {'NMI':>7} {'ARI':>7} {'ASW':>7} {'DAV':>7} "
              f"{'s/ep':>6} {'GPU':>6} {'Params':>10} {'Error'}")
        print("  " + "-" * 95)
        for _, r in sdf.iterrows():
            err = str(r.get("Error", ""))[:30] if pd.notna(r.get("Error")) else ""
            print(f"  {r.get('Model','?'):<18} "
                  f"{r.get('NMI',0):>7.4f} {r.get('ARI',0):>7.4f} "
                  f"{r.get('ASW',0):>7.4f} {r.get('DAV',0):>7.4f} "
                  f"{r.get('SecPerEpoch',0):>6.2f} "
                  f"{r.get('PeakGPU_MB',0):>6.0f} "
                  f"{r.get('NumParams',0):>10,} {err}")

    # Cross-dataset ranking
    if len(ds_keys) > 1 and "Dataset" in combined.columns:
        print(f"\n── Avg Rank Across Datasets ──")
        rank_parts = []
        for ds_key in ds_keys:
            sdf = combined[combined["Dataset"] == ds_key].copy()
            for mc in metric_cols:
                if mc in sdf.columns:
                    asc = (mc == "DAV")
                    sdf[f"{mc}_rank"] = sdf[mc].rank(ascending=asc)
            rank_parts.append(sdf)
        rank_df = pd.concat(rank_parts)
        rank_cols = [f"{mc}_rank" for mc in metric_cols if f"{mc}_rank" in rank_df.columns]
        if rank_cols:
            avg_ranks = rank_df.groupby("Model")[rank_cols].mean()
            avg_ranks["AvgRank"] = avg_ranks.mean(axis=1)
            avg_ranks = avg_ranks.sort_values("AvgRank")
            print(f"  {'Model':<18} " + " ".join(f"{c:>10}" for c in rank_cols) + f" {'AvgRank':>10}")
            print("  " + "-" * (18 + 11 * (len(rank_cols) + 1)))
            for mname, row in avg_ranks.iterrows():
                vals = " ".join(f"{row[c]:>10.2f}" for c in rank_cols)
                print(f"  {mname:<18} {vals} {row['AvgRank']:>10.2f}")

    # Comparison with internal best
    if args.compare and not combined.empty:
        internal_best = load_internal_best()
        print_comparison(combined, internal_best)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
