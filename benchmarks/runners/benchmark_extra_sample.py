"""
benchmark_extra_sample.py
─────────────────────────
Sample benchmark runner for extra / cancer / disease datasets.

Mirrors the interface of benchmark_crossdata.py but targets the
EXTRA_DATASET_REGISTRY (or a user-specified subset), writing
results to benchmark_results/extra_sample/.

Usage
─────
# Check which extra datasets are ready (preprocessed files exist):
python benchmarks/benchmark_extra_sample.py --check-only

# Run ALL model variants on ALL available extra datasets (seed 42):
python benchmarks/benchmark_extra_sample.py --all --seed 42

# Run a specific subset of datasets + models:
python benchmarks/benchmark_extra_sample.py \
    --datasets melanoma tnbc_brain lbm_brain bc_ec \
    --models DPMM-Base DPMM-Contrastive Topic-Base Topic-Transformer Topic-Contrastive \
    --seed 42

# Run using a dataset group defined in dataset_registry.py:
python benchmarks/benchmark_extra_sample.py --group sample_cancer --seed 42

# Append results to existing CSV (for multi-seed runs):
python benchmarks/benchmark_extra_sample.py --all --seed 1 --append
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Working directory / path setup ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config import DEFAULT_OUTPUT_DIR
from benchmarks.dataset_registry import (
    EXTRA_DATASET_REGISTRY,
    ALL_DATASET_REGISTRY,
    DATASET_GROUPS,
    resolve_datasets)

# ── Output root ──────────────────────────────────────────────────────────────
RESULTS_ROOT = DEFAULT_OUTPUT_DIR / "_legacy" / "extra_sample"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ── Model / variant catalogue (mirrors benchmark_crossdata.py) ──────────────
ALL_MODELS = [
    "DPMM-Base",
    "DPMM-Transformer",
    "DPMM-Contrastive",
    "Pure-AE",
    "Pure-Transformer-AE",
    "Pure-Contrastive-AE",
    "Pure-VAE",
    "Pure-Transformer-VAE",
    "Pure-Contrastive-VAE",
    "Topic-Base",
    "Topic-Transformer",
    "Topic-Contrastive",
]

# Compact five-model set useful for quick sanity runs
QUICK_MODELS = [
    "DPMM-Base",
    "DPMM-Contrastive",
    "Topic-Base",
    "Topic-Transformer",
    "Topic-Contrastive",
]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def check_ready(dataset_keys: list[str]) -> tuple[list[str], list[str]]:
    """Return (ready, missing) lists."""
    ready, missing = [], []
    for key in dataset_keys:
        meta = EXTRA_DATASET_REGISTRY.get(key)
        if meta and Path(meta["path"]).exists():
            ready.append(key)
        else:
            missing.append(key)
    return ready, missing


def build_model_instance(model_name: str, n_genes: int, n_clusters: int, seed: int, device: str):
    """Instantiate and return a model variant given its string name.

    Uses the canonical MODELS registry from benchmarks.model_registry.
    """
    import torch
    from benchmarks.model_registry import MODELS

    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_name not in MODELS:
        raise ValueError(f"Unknown model name: {model_name}. "
                         f"Available: {list(MODELS.keys())}")

    entry = MODELS[model_name]
    model_cls = entry['class']
    params = {k: v for k, v in entry['params'].items()
              if not k.startswith('fit_')}
    params['input_dim'] = n_genes

    model = model_cls(**params)
    return model.to(device)


def evaluate_dataset(
    dataset_key: str,
    model_names: list[str],
    seed: int,
    device: str,
    train_kwargs: dict | None = None) -> list[dict]:
    """Run all requested models on one dataset. Return list of result dicts."""
    import torch
    import anndata as ad
    from sklearn.preprocessing import LabelEncoder
    from benchmarks.utils import (
        build_data_loaders,
        compute_cluster_metrics,
        compute_trajectory_metrics)

    train_kwargs = train_kwargs or {}
    meta = (EXTRA_DATASET_REGISTRY.get(dataset_key)
            or ALL_DATASET_REGISTRY.get(dataset_key))
    if meta is None:
        raise KeyError(f"Dataset '{dataset_key}' not found in registry.")

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_key}")
    print(f"  Path    : {meta['path']}")
    print(f"  Domain  : {meta.get('domain', 'N/A')}")
    print(f"  Seed    : {seed}")
    print(f"{'='*60}")

    adata = ad.read_h5ad(meta["path"])
    label_key = meta["label_key"]

    if label_key not in adata.obs.columns:
        print(f"  WARNING: label_key '{label_key}' missing — skipping dataset.")
        return []

    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[label_key].astype(str).values)
    n_clusters = len(le.classes_)
    n_genes    = adata.n_vars

    train_loader, val_loader, test_loader, test_idx = build_data_loaders(
        adata, label_key=label_key, seed=seed
    )

    results = []
    for mname in model_names:
        t0 = time.time()
        print(f"  → {mname} ...", end="", flush=True)
        try:
            model = build_model_instance(mname, n_genes, n_clusters, seed, device)
            model.fit(train_loader, val_loader, **train_kwargs)

            # Cluster metrics on test split
            cluster_metrics = compute_cluster_metrics(
                model, test_loader, labels[test_idx], n_clusters
            )
            # Trajectory metrics (if applicable)
            traj_metrics = {}
            try:
                traj_metrics = compute_trajectory_metrics(model, test_loader)
            except Exception:
                pass

            elapsed = time.time() - t0
            row = {
                "dataset":   dataset_key,
                "domain":    meta.get("domain", "N/A"),
                "model":     mname,
                "seed":      seed,
                "n_cells":   adata.n_obs,
                "n_genes":   n_genes,
                "n_clusters": n_clusters,
                "runtime_s": round(elapsed, 1),
                **cluster_metrics,
                **traj_metrics,
            }
            # Composite score
            nmi = row.get("NMI", 0.0)
            ari = row.get("ARI", 0.0)
            asw = row.get("ASW", 0.0)
            row["Score"] = round((nmi + ari + asw) / 3, 6)
            results.append(row)
            print(f" Score={row['Score']:.4f}  [{elapsed:.1f}s]")

        except Exception as exc:
            elapsed = time.time() - t0
            print(f" FAILED ({exc})")
            results.append({
                "dataset": dataset_key, "model": mname, "seed": seed,
                "error": str(exc), "runtime_s": round(elapsed, 1),
            })

    return results


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample benchmark runner for extra/cancer/disease datasets."
    )

    # Dataset selection
    ds_group = parser.add_mutually_exclusive_group()
    ds_group.add_argument("--all",         action="store_true",
                          help="Run on all available (preprocessed) extra datasets.")
    ds_group.add_argument("--datasets",    nargs="+", metavar="KEY",
                          help="Explicit dataset keys from EXTRA_DATASET_REGISTRY.")
    ds_group.add_argument("--group",       metavar="GROUP",
                          help=f"Dataset group: {list(DATASET_GROUPS.keys())}")
    ds_group.add_argument("--check-only",  action="store_true",
                          help="Check which extra datasets are ready; do not run.")

    # Model selection
    parser.add_argument("--models",  nargs="+", metavar="MODEL", default=None,
                        help="Model variants to include (default: all 12).")
    parser.add_argument("--quick",   action="store_true",
                        help="Use QUICK_MODELS (5 variants) instead of all 12.")

    # Run settings
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  default="auto",
                        help="'cuda', 'cpu', or 'auto' (default: auto).")
    parser.add_argument("--append",  action="store_true",
                        help="Append results to existing CSV rather than overwrite.")
    parser.add_argument("--output",  default=None,
                        help="Custom output CSV path (default: auto-named in RESULTS_ROOT).")
    parser.add_argument("--no-plots", action="store_true", help="Skip figures.")

    args = parser.parse_args()

    # ── Check-only mode ──────────────────────────────────────────────────
    if args.check_only:
        all_keys = list(EXTRA_DATASET_REGISTRY.keys())
        ready, missing = check_ready(all_keys)
        print(f"\nExtra datasets — status ({len(all_keys)} total):")
        print(f"  READY   ({len(ready)}): {', '.join(ready) if ready else 'none'}")
        print(f"  MISSING ({len(missing)}): {', '.join(missing) if missing else 'none'}")
        if missing:
            print("\n  Run prep_extra_datasets.py to preprocess missing datasets:")
            print(f"    python scripts/prep_extra_datasets.py --datasets {' '.join(missing)}")
        return

    # ── Resolve datasets ─────────────────────────────────────────────────
    if args.group:
        keys = DATASET_GROUPS.get(args.group, [])
        if not keys:
            parser.error(f"Unknown group '{args.group}'. Available: {list(DATASET_GROUPS.keys())}")
        keys = [k for k in keys if k in EXTRA_DATASET_REGISTRY]
    elif args.datasets:
        keys = args.datasets
    else:  # --all (default)
        keys = list(EXTRA_DATASET_REGISTRY.keys())

    ready, missing = check_ready(keys)
    if missing:
        print(f"\nWARNING: these datasets are not yet preprocessed and will be skipped:")
        print(f"  {', '.join(missing)}")
        print(f"  Run: python scripts/prep_extra_datasets.py --datasets {' '.join(missing)}")
    if not ready:
        print("No datasets ready. Exiting.")
        return
    print(f"\nWill benchmark {len(ready)} dataset(s): {', '.join(ready)}")

    # ── Resolve models ────────────────────────────────────────────────────
    if args.quick:
        model_names = QUICK_MODELS
    elif args.models:
        model_names = [m for m in args.models if m in ALL_MODELS]
        unknown = [m for m in args.models if m not in ALL_MODELS]
        if unknown:
            print(f"WARNING: unknown models ignored: {unknown}")
    else:
        model_names = ALL_MODELS
    print(f"Models ({len(model_names)}): {', '.join(model_names)}")

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ── Run ───────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv   = Path(args.output) if args.output else (
        RESULTS_ROOT / "csv" / f"extra_results_{timestamp}_seed{args.seed}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    t_start = time.time()

    for i, dkey in enumerate(ready, 1):
        print(f"\n[{i}/{len(ready)}] {dkey}")
        rows = evaluate_dataset(dkey, model_names, seed=args.seed, device=device)
        all_rows.extend(rows)
        # Intermediate save
        df_partial = pd.DataFrame(all_rows)
        df_partial.to_csv(out_csv, index=False)

    # ── Final summary ─────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    if args.append and out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done!  Total time: {elapsed_total/60:.1f} min")
    print(f"  Results: {out_csv}")
    print(f"  Rows   : {len(df)}")
    print(f"{'='*60}")

    # ── Per-model ranking ─────────────────────────────────────────────────
    if "Score" in df.columns:
        rank_df = (
            df.groupby("model")["Score"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        rank_df.columns = ["Model", "Mean_Score"]
        rank_df["Rank"] = range(1, len(rank_df) + 1)
        print("\nRanking (by mean Score across extra datasets):")
        print(rank_df.to_string(index=False))

        rank_csv = RESULTS_ROOT / "csv" / f"extra_ranking_seed{args.seed}.csv"
        rank_df.to_csv(rank_csv, index=False)
        print(f"\nRanking saved: {rank_csv}")


if __name__ == "__main__":
    main()
