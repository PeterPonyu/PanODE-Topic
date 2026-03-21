#!/usr/bin/env python
"""
Data Preprocessing Sensitivity Analysis

Evaluates the impact of preprocessing choices on model performance:
  1) HVG top-genes  ∈ {1000, 2000, 3000, 5000}

Max-cells sweep removed — use fixed 3000 cells (default).

All model/training params held at optimal defaults.
Representative models: Topic-Base (kl=0.01) and DPMM-Base (wr=0.6).

Outputs (under benchmark_results/preprocessing/):
  csv/{topic,dpmm}/   — per-series CSV
  plots/{topic,dpmm}/ — per-sweep UMAP + metrics
  meta/{topic,dpmm}/  — JSON configs

Usage:
  python benchmarks/benchmark_preprocessing.py
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
from benchmarks.dataset_registry import DATASET_REGISTRY, resolve_datasets
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.train_utils import (
    make_topic_params, make_dpmm_params,
    setup_series_dirs, add_common_cli_args,
    train_and_evaluate)
from utils.data import DataSplitter
from utils.viz import plot_umap_grid, plot_all_metrics_barplot

from models.dpmm_base import DPMMODEModel
from models.topic_base import TopicODEModel

# ── defaults ──────────────────────────────────────────────────────────────────
DATA_PATH     = str(BASE_CONFIG.data_path)
LATENT_DIM    = BASE_CONFIG.latent_dim
LR            = BASE_CONFIG.lr
BATCH_SIZE    = BASE_CONFIG.batch_size
EPOCHS        = BASE_CONFIG.epochs
DEVICE        = BASE_CONFIG.device
SEED          = BASE_CONFIG.seed
DATA_TYPE     = BASE_CONFIG.data_type
VERBOSE_EVERY = BASE_CONFIG.verbose_every

# ── sweep grids ───────────────────────────────────────────────────────────────
HVG_GRID   = [1000, 2000, 3000, 5000]
CELLS_GRID = [500, 1000, 3000, 5000]

# Default baseline
HVG_DEFAULT   = BASE_CONFIG.hvg_top_genes   # 3000
CELLS_DEFAULT = BASE_CONFIG.max_cells       # 3000


def _topic_params():
    return make_topic_params(latent_dim=LATENT_DIM)


def _dpmm_params():
    return make_dpmm_params(latent_dim=LATENT_DIM)


MODELS = {
    "topic": {"class": TopicODEModel, "params_fn": _topic_params, "label": "Topic-Base"},
    "dpmm":  {"class": DPMMODEModel,  "params_fn": _dpmm_params,  "label": "DPMM-Base"},
}


def build_sweep():
    """Build sweep configs for HVG and cell-count variations."""
    runs = []
    for sk, mi in MODELS.items():
        lb = mi["label"]

        # 1. HVG sweep (cells=3000 fixed)
        for hvg in HVG_GRID:
            runs.append({
                "series": sk, "name": f"{lb}(hvg={hvg})", "sweep": "hvg_top_genes",
                "sweep_val": hvg, "hvg": hvg, "max_cells": CELLS_DEFAULT,
            })

        # 2. Cell count sweep (hvg=3000 fixed)
        for nc in CELLS_GRID:
            if nc == CELLS_DEFAULT:
                continue  # covered by hvg baseline
            runs.append({
                "series": sk, "name": f"{lb}(cells={nc})", "sweep": "max_cells",
                "sweep_val": nc, "hvg": HVG_DEFAULT, "max_cells": nc,
            })

    return runs

def train_variant(run_cfg, device, verbose_every, seed, cache_dir, data_path):
    """Train one preprocessing variant, delegating core training to shared loop."""
    sk   = run_cfg["series"]
    name = run_cfg["name"]
    mi   = MODELS[sk]
    hvg  = run_cfg["hvg"]
    nc   = run_cfg["max_cells"]

    print(f"\n{'='*60}\nPreprocessing: {name}  (hvg={hvg}, cells={nc})\n{'='*60}")

    # Re-preprocess with different params
    adata = load_or_preprocess_adata(
        data_path, max_cells=nc, hvg_top_genes=hvg,
        seed=seed, cache_dir=str(cache_dir), use_cache=True)

    splitter = DataSplitter(
        adata=adata, layer="counts",
        train_size=0.7, val_size=0.15, test_size=0.15,
        batch_size=BATCH_SIZE, latent_dim=LATENT_DIM,
        random_seed=seed, verbose=False)

    params = mi["params_fn"]()

    # Delegate core training to the shared loop
    result = train_and_evaluate(
        name=name,
        model_cls=mi["class"],
        params=params,
        splitter=splitter,
        device=device,
        lr=LR,
        epochs=EPOCHS,
        verbose_every=verbose_every,
        data_type=DATA_TYPE)

    # Extract full-dataset latent for UMAP (test-only is too sparse
    # for small-cells sweeps like cells=500)
    model = result.pop("model_obj", None)
    if model is not None:
        full_latent_parts = []
        for loader in (splitter.train_loader, splitter.val_loader, splitter.test_loader):
            ld = model.extract_latent(loader, device=str(device))
            full_latent_parts.append(ld["latent"])
        result["latent"] = np.concatenate(full_latent_parts, axis=0)

    # Add preprocessing-specific fields
    n_actual_cells = int(adata.n_obs)
    n_actual_genes = int(adata.n_vars)
    result.update({
        "Series": sk,
        "Sweep": run_cfg["sweep"], "SweepVal": run_cfg["sweep_val"],
        "HVG": hvg, "MaxCells": nc,
        "ActualCells": n_actual_cells, "ActualGenes": n_actual_genes,
    })
    result.pop("history", None)

    return result


def main():
    ap = argparse.ArgumentParser(description="Preprocessing sensitivity")
    add_common_cli_args(ap)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(DEVICE)

    PRE_DIR  = DEFAULT_OUTPUT_DIR / "preprocessing"
    CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
    SDIRS = setup_series_dirs(PRE_DIR, include_latents=False)
    ensure_dirs(CACHE_DIR)

    runs = build_sweep()
    n_t = sum(1 for r in runs if r["series"] == "topic")
    n_d = sum(1 for r in runs if r["series"] == "dpmm")

    print("=" * 60)
    print("PREPROCESSING SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    ds_keys = resolve_datasets(args.datasets)
    print(f"Data   : {ds_keys}")
    print(f"HVG sweep   : {HVG_GRID}  (default={HVG_DEFAULT})")
    print(f"Cells sweep : {CELLS_GRID}  (default={CELLS_DEFAULT})")
    print(f"Total runs  : {len(runs)} (Topic: {n_t}, DPMM: {n_d})")

    results, latents = [], {}
    for ds_key in ds_keys:
        ds_info = DATASET_REGISTRY[ds_key]
        print(f"\n--- Dataset: {ds_key} ({ds_info['data_type']}) ---")
        for run_cfg in runs:
            r = train_variant(run_cfg, device, args.verbose_every,
                              args.seed, CACHE_DIR, ds_info["path"])
            if r.get("latent") is not None:
                latents[f"{ds_key}::{r['Model']}"] = r.pop("latent")
            else:
                r.pop("latent", None)
            r["Dataset"] = ds_key
            r["DataType"] = ds_info["data_type"]
            results.append(r)
            gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    # ── save ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"preprocessing_{timestamp}"

    # Save latent representations for downstream UMAP in composite figures
    PREPROC_DIR = DEFAULT_OUTPUT_DIR / "preprocessing"
    for sk in ("topic", "dpmm"):
        lat_dir = PREPROC_DIR / "latents" / sk
        ensure_dirs(lat_dir)
        for lat_key, lat_arr in latents.items():
            ds_key, model_key = lat_key.split("::", 1)
            # Match model to series
            is_topic = any(t in model_key for t in ("Topic", "topic"))
            is_dpmm = any(t in model_key for t in ("DPMM", "dpmm"))
            if (sk == "topic" and is_topic) or (sk == "dpmm" and is_dpmm):
                safe_name = model_key.replace("/", "_").replace(" ", "_")
                np.savez(lat_dir / f"{safe_name}_{ds_key}_{tag}.npz", latent=lat_arr)
                if len(ds_keys) == 1:
                    np.savez(lat_dir / f"{safe_name}_{tag}.npz", latent=lat_arr)
        print(f"  Saved latents for {sk}: {lat_dir}")

    for sk in ("topic", "dpmm"):
        sdf = df[df["Series"] == sk]
        if sdf.empty:
            continue
        csv_p = SDIRS[sk]["csv"] / f"results_{tag}.csv"
        sdf.to_csv(csv_p, index=False)
        print(f"CSV: {csv_p}")

        meta = {
            "timestamp": timestamp,
            "script": "benchmark_preprocessing.py",
            "series": sk,
            "datasets": ds_keys,
            "hvg_grid": HVG_GRID, "cells_grid": CELLS_GRID,
            "defaults": {"hvg": HVG_DEFAULT, "cells": CELLS_DEFAULT},
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "seed": args.seed,
            "models": sdf["Model"].tolist(),
        }
        meta_p = SDIRS[sk]["meta"] / f"run_{tag}.json"
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PREPROCESSING SENSITIVITY SUMMARY")
    print("=" * 80)

    for sk, sl in [("topic", "Topic"), ("dpmm", "DPMM")]:
        sdf = df[df["Series"] == sk]
        if sdf.empty:
            continue
        for sw in sdf["Sweep"].unique():
            swdf = sdf[sdf["Sweep"] == sw]
            print(f"\n── {sl} — {sw} sweep ──")
            print(f"{'Model':<28} {'NMI':>7} {'ARI':>7} {'ASW':>7} {'DAV':>7} "
                  f"{'DRE':>7} {'LSE':>7} {'Cells':>6} {'Genes':>6} "
                  f"{'s/ep':>6} {'GPU':>6}")
            print("-" * 110)
            for _, r in swdf.iterrows():
                print(f"{r['Model']:<28} "
                      f"{r.get('NMI',0):>7.4f} {r.get('ARI',0):>7.4f} "
                      f"{r.get('ASW',0):>7.4f} {r.get('DAV',0):>7.4f} "
                      f"{r.get('DRE_umap_overall_quality',0):>7.4f} "
                      f"{r.get('LSE_overall_quality',0):>7.4f} "
                      f"{r.get('ActualCells',0):>6} "
                      f"{r.get('ActualGenes',0):>6} "
                      f"{r.get('SecPerEpoch',0):>6.2f} "
                      f"{r.get('PeakGPU_MB',0):>6.0f}")

    # ── plots ─────────────────────────────────────────────────────────────
    if args.no_plots or len(ds_keys) > 1:
        if len(ds_keys) > 1:
            print("\nMulti-dataset run: skipped per-run UMAP/bar plots; CSV supports dataset-wise boxplot aggregation.")
        print("\nPlots disabled."); return

    generated = []
    for sk, sl in [("topic", "Topic"), ("dpmm", "DPMM")]:
        sdf = df[df["Series"] == sk]
        if sdf.empty:
            continue

        for sw in sdf["Sweep"].unique():
            swdf = sdf[sdf["Sweep"] == sw]
            # Note: latent dims differ across HVG but UMAPs still meaningful
            ordered_models = [m for m in swdf["Model"].tolist() if m in latents]
            sw_latents = {m: latents[m] for m in ordered_models}
            stag = f"{sw}_{tag}"

            if sw_latents:
                up = SDIRS[sk]["plots"] / f"umap_{stag}.png"
                try:
                    # Use labels from the largest test set; for umap just visualize
                    # the latent space shapes (labels may have different sizes)
                    first_key = list(sw_latents.keys())[0]
                    n_lat = sw_latents[first_key].shape[0]
                    # We need consistent labels — use dummy if sizes differ
                    compatible = all(v.shape[0] == n_lat for v in sw_latents.values())
                    if compatible:
                        plot_umap_grid(sw_latents, None,
                                       f"{sl} — {sw}", str(up))
                        generated.append(str(up))
                except Exception as e:
                    print(f"UMAP ({sk}/{sw}) err: {e}")

            bp = SDIRS[sk]["plots"] / f"metrics_{stag}.png"
            try:
                plot_all_metrics_barplot(swdf, str(bp),
                                         title=f"{sl} — {sw}")
                generated.append(str(bp))
            except Exception as e:
                print(f"Barplot ({sk}/{sw}) err: {e}")

    print(f"\nGenerated {len(generated)} plots.")
    for p in generated:
        print(f"  ✓ {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
