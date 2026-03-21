#!/usr/bin/env python
"""
Targeted runner for missing sensitivity sweep variants.

Only trains the NEW encoder_size tier (Topic hidden=512, DPMM enc=1024x512)
across all 12 datasets, then merges with existing CSVs while filtering out
the removed warmup_ratio steps (0.5, 0.7).

This avoids re-running the full benchmark_sensitivity.py (~3 hours).
"""

import sys, os, gc, json, shutil, warnings
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
    setup_series_dirs, add_common_cli_args)
from utils.data import DataSplitter

from models.dpmm_base import DPMMODEModel
from models.topic_base import TopicODEModel

# ── defaults ──────────────────────────────────────────────────────────────────
LATENT_DIM    = BASE_CONFIG.latent_dim
LR            = BASE_CONFIG.lr
BATCH_SIZE    = BASE_CONFIG.batch_size
DEVICE        = BASE_CONFIG.device
HVG_TOP_GENES = BASE_CONFIG.hvg_top_genes
MAX_CELLS     = BASE_CONFIG.max_cells
DATA_TYPE     = BASE_CONFIG.data_type
VERBOSE_EVERY = BASE_CONFIG.verbose_every
DEFAULT_EPOCHS = BASE_CONFIG.epochs

# Defaults for other params (held constant)
TOPIC_KL_DEFAULT     = 0.01
TOPIC_HIDDEN_DEFAULT = 128
TOPIC_DROP_DEFAULT   = 0.1

DPMM_WR_DEFAULT      = 0.9
DPMM_EDIMS_DEFAULT   = [256, 128]
DPMM_DDIMS_DEFAULT   = [128, 256]
DPMM_DROP_DEFAULT    = 0.1


def main():
    set_global_seed(42)
    device = torch.device(DEVICE)
    epochs = DEFAULT_EPOCHS
    lr = LR

    SENS_DIR = DEFAULT_OUTPUT_DIR / "sensitivity"
    CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
    ensure_dirs(CACHE_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"sensitivity_{timestamp}"

    # ── new variants to train ─────────────────────────────────────────────
    new_variants = [
        {
            "name": "Topic(hidden=512)", "series": "topic",
            "sweep": "encoder_size", "sweep_val": 512,
            "cls": TopicODEModel,
            "params": make_topic_params(
                latent_dim=LATENT_DIM, kl_weight=TOPIC_KL_DEFAULT,
                encoder_hidden=512, dropout=TOPIC_DROP_DEFAULT),
        },
        {
            "name": "DPMM(enc=1024x512)", "series": "dpmm",
            "sweep": "encoder_size", "sweep_val": "1024x512",
            "cls": DPMMODEModel,
            "params": make_dpmm_params(
                latent_dim=LATENT_DIM, warmup_ratio=DPMM_WR_DEFAULT,
                encoder_dims=[1024, 512], decoder_dims=[512, 1024],
                dropout=DPMM_DROP_DEFAULT),
        },
    ]

    ds_keys = resolve_datasets()
    print("=" * 60)
    print("TARGETED SENSITIVITY RUN — NEW ENCODER_SIZE TIER")
    print("=" * 60)
    print(f"Device  : {DEVICE}")
    print(f"Datasets: {ds_keys}")
    print(f"Epochs  : {epochs},  LR: {lr:.0e},  Batch: {BATCH_SIZE}")
    print(f"Variants: {[v['name'] for v in new_variants]}")
    print(f"Total runs: {len(new_variants) * len(ds_keys)}")

    # ── train new variants ────────────────────────────────────────────────
    results, latents = [], {}

    for ds_key in ds_keys:
        ds_info = DATASET_REGISTRY[ds_key]
        print(f"\n--- Dataset: {ds_key} ({ds_info['data_type']}) ---")
        adata = load_or_preprocess_adata(
            ds_info["path"], max_cells=MAX_CELLS, hvg_top_genes=HVG_TOP_GENES,
            seed=42, cache_dir=str(CACHE_DIR), use_cache=True)

        label_key = ds_info.get("label_key", "cell_type")
        if label_key in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[label_key].copy()

        splitter = DataSplitter(
            adata=adata, layer="counts",
            train_size=0.7, val_size=0.15, test_size=0.15,
            batch_size=BATCH_SIZE, latent_dim=LATENT_DIM,
            random_seed=42, verbose=True)

        for v in new_variants:
            print(f"  Training {v['name']} ...")
            r = train_and_evaluate(
                name=v["name"],
                model_cls=v["cls"],
                params=v["params"],
                splitter=splitter,
                device=device,
                lr=lr,
                epochs=epochs,
                verbose_every=VERBOSE_EVERY,
                data_type=DATA_TYPE,
                extra_fields={
                    "Series": v["series"],
                    "Sweep": v["sweep"],
                    "SweepVal": str(v["sweep_val"]),
                })
            if r.get("latent") is not None:
                latents[(ds_key, v["name"], v["series"])] = r.pop("latent")
            else:
                r.pop("latent", None)
            r["Dataset"] = ds_key
            r["DataType"] = ds_info["data_type"]
            results.append(r)
            gc.collect(); torch.cuda.empty_cache()

    new_df = pd.DataFrame(results)

    # ── save latent .npz for new variants ─────────────────────────────────
    lat_sdirs = setup_series_dirs(SENS_DIR, include_latents=True)
    for (ds_key, model_name, series), lat in latents.items():
        lat_dir = lat_sdirs[series]["latents"]
        npz_path = lat_dir / f"{model_name}_{ds_key}_{tag}.npz"
        np.savez_compressed(str(npz_path), latent=lat)
        print(f"  Saved latent: {npz_path.name}")

    # ── merge with existing CSVs ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MERGING WITH EXISTING CSVs")
    print("=" * 60)

    for sk in ("topic", "dpmm"):
        csv_dir = SENS_DIR / "csv" / sk
        existing_csvs = sorted(csv_dir.glob("results_sensitivity_*.csv"),
                               key=lambda x: x.name, reverse=True)
        if not existing_csvs:
            print(f"  [{sk}] No existing CSV found — saving new results only.")
            out_p = csv_dir / f"results_{tag}.csv"
            new_df[new_df["Series"] == sk].to_csv(out_p, index=False)
            continue

        existing_df = pd.read_csv(existing_csvs[0])
        print(f"  [{sk}] Loaded {existing_csvs[0].name}: {len(existing_df)} rows")

        # Remove warmup_ratio 0.5 and 0.7 (DPMM only)
        if sk == "dpmm":
            before = len(existing_df)
            existing_df = existing_df[
                ~((existing_df["Sweep"] == "warmup_ratio") &
                  (existing_df["SweepVal"].astype(str).isin(["0.5", "0.7"])))
            ]
            removed = before - len(existing_df)
            print(f"  [{sk}] Removed {removed} rows (warmup_ratio 0.5, 0.7)")

        # Append new encoder_size variant rows
        new_series = new_df[new_df["Series"] == sk]
        if not new_series.empty:
            # Align columns
            for col in existing_df.columns:
                if col not in new_series.columns:
                    new_series[col] = np.nan
            new_series = new_series[existing_df.columns]
            merged = pd.concat([existing_df, new_series], ignore_index=True)
            print(f"  [{sk}] Added {len(new_series)} new rows → {len(merged)} total")
        else:
            merged = existing_df

        out_p = csv_dir / f"results_{tag}.csv"
        merged.to_csv(out_p, index=False)
        print(f"  [{sk}] Saved merged CSV: {out_p.name}")

    # ── clean up removed warmup_ratio latent files ────────────────────────
    print("\n--- Cleaning warmup_ratio 0.5/0.7 latent files ---")
    dpmm_lat_dir = SENS_DIR / "latents" / "dpmm"
    removed_count = 0
    for wr_val in ["0.5", "0.7"]:
        for f in dpmm_lat_dir.glob(f"DPMM(wr={wr_val})_*.npz"):
            f.unlink()
            removed_count += 1
    print(f"  Removed {removed_count} latent files for warmup_ratio 0.5/0.7")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NEW VARIANT RESULTS")
    print("=" * 60)
    for _, r in new_df.iterrows():
        print(f"  {r['Model']:<28} {r['Dataset']:<16} "
              f"NMI={r.get('NMI',0):.4f}  ARI={r.get('ARI',0):.4f}  "
              f"ASW={r.get('ASW',0):.4f}  DAV={r.get('DAV',0):.4f}")

    print("\n✓ Done. Merged CSVs are ready for figure regeneration.")


if __name__ == "__main__":
    main()
