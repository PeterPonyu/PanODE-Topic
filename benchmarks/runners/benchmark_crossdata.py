#!/usr/bin/env python
"""
Cross-Dataset Generalization Benchmark

Evaluates all 12 model variants on multiple datasets spanning three
data types: trajectory, cluster, and mixed. All model/training params
are held at optimal defaults from single-dataset experiments.

Datasets:
  - setty.h5ad       (trajectory, 5780 cells, hematopoiesis)
  - GSE130148_LungHmDev.h5ad (cluster, 10360 cells, 13 types, lung dev)
  - endo.h5ad         (mixed, 2531 cells, 7 types + pseudotime, pancreas)

Outputs (under benchmark_results/crossdata/):
  csv/        — per-dataset CSV results
  plots/      — per-dataset UMAP + metrics barplots
  meta/       — JSON configs for reproducibility

Usage:
  python benchmarks/benchmark_crossdata.py
  python benchmarks/benchmark_crossdata.py --datasets setty endo
"""

import sys, os, argparse, json, gc, warnings
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
from benchmarks.dataset_registry import DATASET_REGISTRY, ALL_DATASET_REGISTRY
from benchmarks.model_registry import MODELS, SERIES_GROUPS
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.train_utils import (
    train_and_evaluate, add_common_cli_args,
    select_models, apply_model_overrides)
from utils.data import DataSplitter
from utils.viz import plot_umap_grid, plot_all_metrics_barplot
from utils.paper_style import MODEL_SHORT_NAMES, MODEL_ORDER_DPMM, MODEL_ORDER_TOPIC

# ── defaults ──────────────────────────────────────────────────────────────────
LATENT_DIM    = BASE_CONFIG.latent_dim
LR            = BASE_CONFIG.lr
BATCH_SIZE    = BASE_CONFIG.batch_size
EPOCHS        = BASE_CONFIG.epochs
DEVICE        = BASE_CONFIG.device
SEED          = BASE_CONFIG.seed
VERBOSE_EVERY = BASE_CONFIG.verbose_every
HVG_TOP       = BASE_CONFIG.hvg_top_genes
MAX_CELLS     = BASE_CONFIG.max_cells

def standardize_labels(adata, label_key):
    """Copy dataset-specific label column to 'cell_type' for DataSplitter compatibility."""
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()
    elif "cell_type" not in adata.obs.columns:
        print(f"  WARNING: Neither '{label_key}' nor 'cell_type' found in obs. "
              "Will use KMeans pseudo-labels.")
    return adata


def train_one(model_name, model_info, splitter, data_type, device, verbose_every):
    """Train a single model variant via the shared train_and_evaluate loop."""
    return train_and_evaluate(
        name=model_name,
        model_cls=model_info["class"],
        params=model_info["params"],
        splitter=splitter,
        device=device,
        lr=LR,
        verbose_every=verbose_every,
        data_type=data_type,
        extra_fields={
            "Series": model_info.get("series", ""),
        })


def run_dataset(ds_key, ds_info, device, verbose_every, seed, cache_dir, no_plots, out_dirs,
                selected_models=None):
    """Run selected models on one dataset."""
    if selected_models is None:
        selected_models = MODELS
    print(f"\n{'#'*70}")
    print(f"# DATASET: {ds_key} — {ds_info['desc']}")
    print(f"# Path: {ds_info['path']}")
    print(f"# Data type: {ds_info['data_type']}")
    print(f"{'#'*70}")

    # Load & preprocess
    adata = load_or_preprocess_adata(
        ds_info["path"], max_cells=MAX_CELLS, hvg_top_genes=HVG_TOP,
        seed=seed, cache_dir=str(cache_dir), use_cache=True)

    # Standardize label column
    adata = standardize_labels(adata, ds_info["label_key"])

    splitter = DataSplitter(
        adata=adata, layer="counts",
        train_size=0.7, val_size=0.15, test_size=0.15,
        batch_size=BATCH_SIZE, latent_dim=LATENT_DIM,
        random_seed=seed, verbose=True)

    n_cells = int(adata.n_obs)
    n_genes = int(adata.n_vars)
    n_labels = len(np.unique(splitter.labels)) if splitter.labels is not None else 0
    print(f"  Shape after preprocess: {n_cells} cells × {n_genes} genes, {n_labels} labels")

    results, latents = [], {}
    for mname, minfo in selected_models.items():
        r = train_one(mname, minfo, splitter, ds_info["data_type"],
                       device, verbose_every)
        if r.get("latent") is not None:
            latents[mname] = r.pop("latent")
        else:
            r.pop("latent", None)
        r["Dataset"] = ds_key
        r["DataType"] = ds_info["data_type"]
        r["Cells"] = n_cells
        r["Genes"] = n_genes
        r["Labels"] = n_labels
        results.append(r)
        gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ds_key}_{timestamp}"

    # Save CSV
    csv_p = out_dirs["csv"] / f"results_{tag}.csv"
    df.to_csv(csv_p, index=False)
    print(f"\n  CSV: {csv_p}")

    # Save per-model latent files for composite Figure 5 UMAP panels
    try:
        lat_dir = out_dirs.get("latents", out_dirs["csv"].parent / "latents") / ds_key
        ensure_dirs(lat_dir)
        labels_test = splitter.labels_test if splitter.labels_test is not None else None
        for model_name, latent_arr in latents.items():
            safe_name = str(model_name).replace("/", "_").replace(" ", "_")
            lp = lat_dir / f"{safe_name}_{tag}.npz"
            if labels_test is None:
                np.savez(lp, latent=latent_arr)
            else:
                np.savez(lp, latent=latent_arr, labels=labels_test)
        print(f"  Latents: {lat_dir}")
    except Exception as e:
        print(f"  Latent save err: {e}")

    # Save metadata
    meta = {
        "timestamp": timestamp,
        "script": "benchmark_crossdata.py",
        "dataset": ds_key,
        "data_path": ds_info["path"],
        "data_type": ds_info["data_type"],
        "label_key": ds_info["label_key"],
        "n_cells": n_cells, "n_genes": n_genes, "n_labels": n_labels,
        "epochs_per_model": {m: info["params"].get("fit_epochs", EPOCHS)
                             for m, info in selected_models.items()},
        "lr": LR, "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "hvg": HVG_TOP, "max_cells": MAX_CELLS,
        "seed": seed,
        "models": df["Model"].tolist(),
    }
    meta_p = out_dirs["meta"] / f"run_{tag}.json"
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Plots
    if not no_plots and latents:
        try:
            labels = splitter.labels_test
            up = out_dirs["plots"] / f"umap_{tag}.png"
            ordered_models = [m for m in df["Model"].tolist() if m in latents]
            ordered_latents = {m: latents[m] for m in ordered_models}
            plot_umap_grid(ordered_latents, labels, f"Cross-Dataset: {ds_key}", str(up))
            print(f"  UMAP: {up}")
        except Exception as e:
            print(f"  UMAP err: {e}")

        try:
            bp = out_dirs["plots"] / f"metrics_{tag}.png"
            plot_all_metrics_barplot(df, str(bp),
                                     title=f"Cross-Dataset: {ds_key} ({ds_info['data_type']})")
            print(f"  Barplot: {bp}")
        except Exception as e:
            print(f"  Barplot err: {e}")

        # Also produce per-series separated barplots with improved style
        try:
            from utils.viz import plot_core_metrics_barplot
            for series_tag, order in [("dpmm", MODEL_ORDER_DPMM), ("topic", MODEL_ORDER_TOPIC)]:
                s_models = [m for m in order if m in df["Model"].values]
                if len(s_models) < 2:
                    continue
                sdf = df[df["Model"].isin(s_models)].copy()
                sp = out_dirs["plots"] / f"core_metrics_{tag}_{series_tag}.png"
                plot_core_metrics_barplot(sdf, str(sp),
                                          title=f"{ds_key} ({series_tag.upper()} series)",
                                          series=series_tag)
                print(f"  Core barplot ({series_tag}): {sp}")
        except Exception as e:
            print(f"  Per-series barplot err: {e}")

    return df


def main():
    ap = argparse.ArgumentParser(
        description="Cross-dataset generalization benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # All models, all datasets
  python benchmarks/benchmark_crossdata.py

  # Only DPMM series on setty
  python benchmarks/benchmark_crossdata.py --series dpmm --datasets setty

  # Only Pure baselines, override epochs to 500
  python benchmarks/benchmark_crossdata.py --series pure --override-epochs 500

  # Specific models
  python benchmarks/benchmark_crossdata.py --models Pure-Transformer-VAE Topic-Transformer

  # DPMM series with custom dropout
  python benchmarks/benchmark_crossdata.py --series dpmm --override-dropout 0.15
""")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Which datasets to run (core + extra). Default: all core.")
    ap.add_argument("--series", type=str, default="all",
                    help="Model series to run: all, dpmm, topic, pure, pure-ae, "
                         "pure-vae, or comma-separated combination. Default: all.")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Explicit model names to run (overrides --series).")
    ap.add_argument("--override-epochs", type=int, default=None,
                    help="Override fit_epochs for all selected models.")
    ap.add_argument("--override-wd", type=float, default=None,
                    help="Override fit_weight_decay for all selected models.")
    ap.add_argument("--override-dropout", type=float, default=None,
                    help="Override dropout for all selected models.")
    ap.add_argument("--override-kl-weight", type=float, default=None,
                    help="Override kl_weight for all selected VAE models.")
    ap.add_argument("--verbose-every", type=int, default=VERBOSE_EVERY)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(DEVICE)

    ds_keys = args.datasets or list(DATASET_REGISTRY.keys())
    # Resolve dataset info from ALL registries (core + extra)
    _ds_lookup = ALL_DATASET_REGISTRY

    # Select & override models
    selected_models = select_models(args)
    if not selected_models:
        print("ERROR: No models selected. Check --series / --models.")
        sys.exit(1)
    apply_model_overrides(selected_models, args)

    CROSS_DIR = DEFAULT_OUTPUT_DIR / "crossdata"
    CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
    out_dirs = {
        "csv": CROSS_DIR / "csv",
        "plots": CROSS_DIR / "plots",
        "meta": CROSS_DIR / "meta",
        "latents": CROSS_DIR / "latents",
    }
    ensure_dirs(*out_dirs.values(), CACHE_DIR)

    # Build epoch summary from selected models
    ep_summary = {}
    for mname, minfo in selected_models.items():
        ep = minfo["params"].get("fit_epochs", EPOCHS)
        ep_summary.setdefault(ep, []).append(mname)
    ep_display = ", ".join(f"{ep}ep:[{','.join(ms)}]" for ep, ms in sorted(ep_summary.items()))

    print("=" * 70)
    print("CROSS-DATASET GENERALIZATION BENCHMARK")
    print("=" * 70)
    print(f"Device     : {DEVICE}")
    print(f"Datasets   : {ds_keys}")
    print(f"Models     : {len(selected_models)} / {len(MODELS)} variants")
    print(f"Selected   : {list(selected_models.keys())}")
    print(f"Epochs     : {ep_display}")
    print(f"HVG/Cells  : {HVG_TOP} / {MAX_CELLS}")
    total_runs = len(ds_keys) * len(selected_models)
    print(f"Total runs : {total_runs}")

    all_dfs = []
    for ds_key in ds_keys:
        ds_info = _ds_lookup[ds_key]
        df = run_dataset(ds_key, ds_info, device, args.verbose_every,
                         args.seed, CACHE_DIR, args.no_plots, out_dirs,
                         selected_models=selected_models)
        all_dfs.append(df)

    # Combined summary
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_csv = out_dirs["csv"] / f"results_combined_{timestamp}.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nCombined CSV: {combined_csv}")
    else:
        combined = all_dfs[0] if all_dfs else pd.DataFrame()

    # Print final summary table
    print("\n" + "=" * 100)
    print("CROSS-DATASET SUMMARY")
    print("=" * 100)

    metric_cols = ["NMI", "ARI", "ASW", "DAV"]
    for ds_key in ds_keys:
        sdf = combined[combined["Dataset"] == ds_key] if "Dataset" in combined.columns else combined
        if sdf.empty:
            continue
        dt = _ds_lookup.get(ds_key, {}).get("data_type", "unknown")
        print(f"\n── {ds_key} ({dt}) ──")
        print(f"{'Model':<25} {'NMI':>7} {'ARI':>7} {'ASW':>7} {'DAV':>7} "
              f"{'DRE':>7} {'LSE':>7} {'s/ep':>6} {'GPU':>6} {'Params':>10}")
        print("-" * 105)
        for _, r in sdf.iterrows():
            print(f"{r.get('Model','?'):<25} "
                  f"{r.get('NMI',0):>7.4f} {r.get('ARI',0):>7.4f} "
                  f"{r.get('ASW',0):>7.4f} {r.get('DAV',0):>7.4f} "
                  f"{r.get('DRE_umap_overall_quality',0):>7.4f} "
                  f"{r.get('LSE_overall_quality',0):>7.4f} "
                  f"{r.get('SecPerEpoch',0):>6.2f} "
                  f"{r.get('PeakGPU_MB',0):>6.0f} "
                  f"{r.get('NumParams',0):>10,}")

    # Cross-dataset ranking
    if len(ds_keys) > 1 and "Dataset" in combined.columns:
        print(f"\n── Avg Rank Across Datasets ──")
        rank_df_parts = []
        for ds_key in ds_keys:
            sdf = combined[combined["Dataset"] == ds_key].copy()
            for mc in metric_cols:
                if mc in sdf.columns:
                    asc = (mc == "DAV")  # DAV lower=better
                    sdf[f"{mc}_rank"] = sdf[mc].rank(ascending=asc)
            rank_df_parts.append(sdf)
        rank_df = pd.concat(rank_df_parts)

        rank_cols = [f"{mc}_rank" for mc in metric_cols if f"{mc}_rank" in rank_df.columns]
        if rank_cols:
            avg_ranks = rank_df.groupby("Model")[rank_cols].mean()
            avg_ranks["AvgRank"] = avg_ranks.mean(axis=1)
            avg_ranks = avg_ranks.sort_values("AvgRank")
            print(f"{'Model':<25} " + " ".join(f"{c:>10}" for c in rank_cols) + f" {'AvgRank':>10}")
            print("-" * (25 + 11 * (len(rank_cols) + 1)))
            for mname, row in avg_ranks.iterrows():
                vals = " ".join(f"{row[c]:>10.2f}" for c in rank_cols)
                print(f"{mname:<25} {vals} {row['AvgRank']:>10.2f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
