#!/usr/bin/env python
"""
Training / Optimisation Hyperparameter Sweep

Sweeps training-related hyperparameters (one-at-a-time, control-variable
principle: all others held at default).

Sweep dimensions (on Topic-Base and DPMM-Base):
  1) Learning rate   ∈ {1e-4, 1e-3, 5e-3, 1e-2}
  2) Epochs          ∈ {200, 600, 1200, 1600}
  3) Batch size      ∈ {32, 64, 256, 512}
  4) Weight decay    ∈ {0, 1e-5, 1e-4, 1e-3}

Defaults (held constant unless being swept):
  lr=1e-3, epochs=600, batch_size=128, weight_decay=1e-5,
  kl_weight=0.01, warmup_ratio=0.9, latent_dim=10

Efficiency is recorded per-run: sec/epoch, peak GPU MB, param count.

Outputs (under benchmark_results/training/):
  csv/{topic,dpmm}/   — per-series CSV
  plots/{topic,dpmm}/ — UMAP + metrics bar charts per sweep
  meta/{topic,dpmm}/  — JSON configs

Usage:
  python benchmarks/benchmark_training.py
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
    make_topic_params, make_dpmm_params, train_and_evaluate,
    setup_series_dirs, save_latents, add_common_cli_args)
from utils.data import DataSplitter
from utils.viz import plot_umap_grid, plot_all_metrics_barplot

from models.dpmm_base import DPMMODEModel
from models.topic_base import TopicODEModel

# ── defaults ──────────────────────────────────────────────────────────────────
DATA_PATH     = str(BASE_CONFIG.data_path)
LATENT_DIM    = BASE_CONFIG.latent_dim     # 10
LR            = BASE_CONFIG.lr             # 1e-3
BATCH_SIZE    = BASE_CONFIG.batch_size     # 128
DEVICE        = BASE_CONFIG.device
HVG_TOP_GENES = BASE_CONFIG.hvg_top_genes  # 3000
MAX_CELLS     = BASE_CONFIG.max_cells      # 3000
SEED          = BASE_CONFIG.seed           # 42
DATA_TYPE     = BASE_CONFIG.data_type
VERBOSE_EVERY = BASE_CONFIG.verbose_every
DEFAULT_EPOCHS = BASE_CONFIG.epochs        # 600
WEIGHT_DECAY  = 1e-5  # default weight decay (hardcoded in all model .fit())

# ── sweep grids (4 values each for uniform Fig 3/4 layout) ─────────────────────
LR_GRID      = [1e-4, 1e-3, 5e-3, 1e-2]
EPOCHS_GRID  = [200, 600, 1200, 1600]
BATCH_GRID   = [32, 64, 256, 512]
WD_GRID      = [0.0, 1e-5, 1e-4, 1e-3]

# ── model configs (architecture fixed at sensitivity-optimal) ─────────────────
def _topic_params():
    return make_topic_params(latent_dim=LATENT_DIM)


def _dpmm_params():
    return make_dpmm_params(latent_dim=LATENT_DIM)


MODELS = {
    "topic": {"class": TopicODEModel, "params_fn": _topic_params, "label": "Topic-Base"},
    "dpmm":  {"class": DPMMODEModel,  "params_fn": _dpmm_params,  "label": "DPMM-Base"},
}


def build_sweep():
    """Build all run configs. Each changes ONE training param, rest at default."""
    runs = []

    for sk, mi in MODELS.items():
        lb = mi["label"]

        # 1. Learning rate sweep
        for lr in LR_GRID:
            runs.append({
                "series": sk, "name": f"{lb}(lr={lr:.0e})", "sweep": "lr",
                "sweep_val": lr, "lr": lr, "epochs": DEFAULT_EPOCHS,
                "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
            })

        # 2. Epochs sweep
        for ep in EPOCHS_GRID:
            if ep == DEFAULT_EPOCHS:
                continue  # covered by lr baseline
            runs.append({
                "series": sk, "name": f"{lb}(ep={ep})", "sweep": "epochs",
                "sweep_val": ep, "lr": LR, "epochs": ep,
                "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
            })

        # 3. Batch size sweep
        for bs in BATCH_GRID:
            if bs == BATCH_SIZE:
                continue  # covered by lr baseline
            runs.append({
                "series": sk, "name": f"{lb}(bs={bs})", "sweep": "batch_size",
                "sweep_val": bs, "lr": LR, "epochs": DEFAULT_EPOCHS,
                "batch_size": bs, "weight_decay": WEIGHT_DECAY,
            })

        # 4. Weight decay sweep
        for wd in WD_GRID:
            if wd == WEIGHT_DECAY:
                continue  # covered by lr baseline
            runs.append({
                "series": sk, "name": f"{lb}(wd={wd:.0e})", "sweep": "weight_decay",
                "sweep_val": wd, "lr": LR, "epochs": DEFAULT_EPOCHS,
                "batch_size": BATCH_SIZE, "weight_decay": wd,
            })

    return runs


def train_variant(run_cfg, adata, device, verbose_every, seed):
    """Train one variant — rebuilds DataSplitter when batch_size varies."""
    sk    = run_cfg["series"]
    name  = run_cfg["name"]
    mi    = MODELS[sk]
    lr    = run_cfg["lr"]
    epochs = run_cfg["epochs"]
    bs    = run_cfg["batch_size"]
    wd    = run_cfg["weight_decay"]

    splitter = DataSplitter(
        adata=adata, layer="counts",
        train_size=0.7, val_size=0.15, test_size=0.15,
        batch_size=bs, latent_dim=LATENT_DIM,
        random_seed=seed, verbose=False)

    return train_and_evaluate(
        name=name,
        model_cls=mi["class"],
        params=mi["params_fn"](),
        splitter=splitter,
        device=device,
        lr=lr, epochs=epochs, weight_decay=wd,
        verbose_every=verbose_every,
        data_type=DATA_TYPE,
        extra_fields={
            "Series": sk,
            "Sweep": run_cfg["sweep"],
            "SweepVal": str(run_cfg["sweep_val"]),
            "BatchSize": bs,
            "WeightDecay": wd,
        })


def main():
    ap = argparse.ArgumentParser(description="Training hyperparameter sweep")
    add_common_cli_args(ap)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(DEVICE)

    # ── dirs ──────────────────────────────────────────────────────────────
    TRAIN_DIR = DEFAULT_OUTPUT_DIR / "training"
    CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
    SDIRS = setup_series_dirs(TRAIN_DIR, include_latents=False)
    ensure_dirs(CACHE_DIR)

    # ── header ────────────────────────────────────────────────────────────
    runs = build_sweep()
    n_t = sum(1 for r in runs if r["series"] == "topic")
    n_d = sum(1 for r in runs if r["series"] == "dpmm")

    print("=" * 60)
    print("TRAINING HYPERPARAMETER SWEEP")
    print("=" * 60)
    print(f"Device     : {DEVICE}")
    ds_keys = resolve_datasets(args.datasets)
    print(f"Data       : {ds_keys}")
    print(f"LR sweep   : {LR_GRID}")
    print(f"Epochs     : {EPOCHS_GRID}")
    print(f"Batch      : {BATCH_GRID}")
    print(f"WeightDecay: {WD_GRID}")
    print(f"Baseline   : lr={LR:.0e}, ep={DEFAULT_EPOCHS}, bs={BATCH_SIZE}, wd={WEIGHT_DECAY:.0e}")
    print(f"Total runs : {len(runs)} (Topic: {n_t}, DPMM: {n_d})")

    # ── data + train (possibly multi-dataset) ─────────────────────────────
    results, latents = [], {}
    last_splitter = None
    for ds_key in ds_keys:
        ds_info = DATASET_REGISTRY[ds_key]
        print(f"\n--- Dataset: {ds_key} ({ds_info['data_type']}) ---")
        adata = load_or_preprocess_adata(
            ds_info["path"], max_cells=MAX_CELLS, hvg_top_genes=HVG_TOP_GENES,
            seed=args.seed, cache_dir=str(CACHE_DIR), use_cache=True)
        label_key = ds_info.get("label_key", "cell_type")
        if label_key in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[label_key].copy()
            print(f"  Labels: {len(np.unique(adata.obs[label_key].values))} types ('{label_key}')")

        for run_cfg in runs:
            r = train_variant(run_cfg, adata, device, args.verbose_every, args.seed)
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
    tag = f"training_{timestamp}"

    # Save latent representations for downstream UMAP in composite figures
    for sk in ("topic", "dpmm"):
        lat_dir = TRAIN_DIR / "latents" / sk
        ensure_dirs(lat_dir)
        for lat_key, lat_arr in latents.items():
            ds_key, model_key = lat_key.split("::", 1)
            # model_key is like "Topic-Base(ep=800)", sk check via run list
            matching = [r for r in runs if r["name"] == model_key and r["series"] == sk]
            if matching:
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
            "script": "benchmark_training.py",
            "series": sk,
            "datasets": ds_keys,
            "baseline": {
                "lr": LR, "epochs": DEFAULT_EPOCHS,
                "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
                "latent_dim": LATENT_DIM,
            },
            "sweeps": {
                "lr": LR_GRID, "epochs": EPOCHS_GRID,
                "batch_size": BATCH_GRID, "weight_decay": WD_GRID,
            },
            "seed": args.seed,
            "models": sdf["Model"].tolist(),
        }
        meta_p = SDIRS[sk]["meta"] / f"run_{tag}.json"
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING SWEEP SUMMARY")
    print("=" * 80)

    for sk, sl in [("topic", "Topic-Base"), ("dpmm", "DPMM-Base")]:
        sdf = df[df["Series"] == sk]
        if sdf.empty:
            continue
        for sw in sdf["Sweep"].unique():
            swdf = sdf[sdf["Sweep"] == sw]
            print(f"\n── {sl} — {sw} sweep ──")
            print(f"{'Model':<28} {'NMI':>7} {'ARI':>7} {'ASW':>7} {'DAV':>7} "
                  f"{'DRE':>7} {'LSE':>7} {'s/ep':>6} {'GPU':>6} "
                  f"{'Δ%':>7} {'Conv':>5}")
            print("-" * 110)
            for _, r in swdf.iterrows():
                rpct = f"{r['recon_rel_change_pct']:+.1f}" if pd.notna(r.get('recon_rel_change_pct')) else "N/A"
                conv = "✓" if r.get("converged", False) else "✗"
                print(f"{r['Model']:<28} "
                      f"{r.get('NMI',0):>7.4f} {r.get('ARI',0):>7.4f} "
                      f"{r.get('ASW',0):>7.4f} {r.get('DAV',0):>7.4f} "
                      f"{r.get('DRE_umap_overall_quality',0):>7.4f} "
                      f"{r.get('LSE_overall_quality',0):>7.4f} "
                      f"{r.get('SecPerEpoch',0):>6.2f} "
                      f"{r.get('PeakGPU_MB',0):>6.0f} "
                      f"{rpct:>7} {conv:>5}")

    # ── plots ─────────────────────────────────────────────────────────────
    if args.no_plots or len(ds_keys) > 1:
        if len(ds_keys) > 1:
            print("\nMulti-dataset run: skipped per-run UMAP/bar plots; CSV supports dataset-wise boxplot aggregation.")
        print("\nPlots disabled."); return

    generated = []
    for sk, sl in [("topic", "Topic-Base"), ("dpmm", "DPMM-Base")]:
        sdf = df[df["Series"] == sk]
        if sdf.empty:
            continue
        slatents = {k: v for k, v in latents.items()
                    if any(r["name"] == k and r["series"] == sk for r in runs)}

        for sw in sdf["Sweep"].unique():
            swdf = sdf[sdf["Sweep"] == sw]
            sw_latents = {k: v for k, v in slatents.items()
                          if k in swdf["Model"].values}
            stag = f"{sw}_{tag}"

            if sw_latents:
                up = SDIRS[sk]["plots"] / f"umap_{stag}.png"
                try:
                    # legacy plotting path only for single dataset mode
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
