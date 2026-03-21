#!/usr/bin/env python3
"""
Offline GSE (DRE + DREX + LSEX) evaluation on cached latent spaces.

Reads biological_validation latent_data.npz for all 12 models,
computes UMAP/t-SNE 2-D projections from each latent space, and
evaluates dimensionality-reduction quality (DRE, DREX) by comparing
the **latent space** against its own 2-D projections — not against
the original expression space.

No model training is performed — this is pure metric computation.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_lib.metrics import (
    evaluate_dimensionality_reduction,
    evaluate_extended_dimensionality_reduction,
    evaluate_single_cell_latent_space,
    evaluate_extended_latent_space)
from benchmarks.config import (
    DEFAULT_OUTPUT_DIR, CACHE_DIR as _CACHE_DIR, BIO_RESULTS_DIR)


# ── Paths ────────────────────────────────────────────────────────────────────
BIO_DIR = BIO_RESULTS_DIR
CACHE_DIR = _CACHE_DIR
BASE_CSV_DIR = DEFAULT_OUTPUT_DIR / "base" / "csv"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "base" / "gse_results"

# The standard setty 3000c × 3000g cache (used for base benchmark)
CACHE_HASH = "c4e1e1aa80205650ccd822f1a9e9c8cc"


MODEL_SERIES = {
    "dpmm": [
        "Pure-AE", "Pure-Transformer-AE", "Pure-Contrastive-AE",
        "DPMM-Base", "DPMM-Contrastive", "DPMM-Transformer",
    ],
    "topic": [
        "Pure-VAE", "Pure-Transformer-VAE", "Pure-Contrastive-VAE",
        "Topic-Base", "Topic-Contrastive", "Topic-Transformer",
    ],
}


def _compute_2d_projections(latent: np.ndarray):
    """Compute UMAP and t-SNE 2-D projections from the latent space."""
    adata = sc.AnnData(latent.astype(np.float32))
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    sc.tl.tsne(adata, use_rep='X')
    return adata.obsm['X_umap'], adata.obsm['X_tsne']


def load_latent(model_name: str) -> np.ndarray | None:
    """Load cached latent from biological_validation results."""
    path = BIO_DIR / f"{model_name}_setty_latent_data.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return data["latent"]


def to_jsonable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_jsonable(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def evaluate_model(
    model_name: str,
    latent: np.ndarray,
    k: int = 15,
    data_type: str = "trajectory") -> dict:
    """Compute DRE (latent→2-D) + LSE + DREX (latent→2-D) + LSEX for one model.

    All DR quality metrics compare the latent space against its own
    UMAP / t-SNE 2-D projections, not the original expression space.
    """
    _DRE_KEYS = ["distance_correlation", "Q_local", "Q_global", "K_max", "overall_quality"]
    _DREX_KEYS = ["trustworthiness", "continuity", "distance_spearman", "distance_pearson",
                  "local_scale_quality", "neighborhood_symmetry", "overall_quality"]

    result = {"Model": model_name}

    # ── 2-D projections from latent ──────────────────────────────────────
    umap_2d = tsne_2d = None
    try:
        umap_2d, tsne_2d = _compute_2d_projections(latent)
    except Exception as e:
        print(f"  [WARN] UMAP/tSNE failed for {model_name}: {e}")

    # DRE — latent → UMAP
    if umap_2d is not None:
        try:
            dre = evaluate_dimensionality_reduction(latent, umap_2d, k=k, verbose=False)
            for key in _DRE_KEYS:
                result[f"DRE_umap_{key}"] = dre.get(key, np.nan)
        except Exception as e:
            print(f"  [WARN] DRE-umap failed for {model_name}: {e}")
            for key in _DRE_KEYS:
                result[f"DRE_umap_{key}"] = np.nan
    else:
        for key in _DRE_KEYS:
            result[f"DRE_umap_{key}"] = np.nan

    # DRE — latent → t-SNE
    if tsne_2d is not None:
        try:
            dre = evaluate_dimensionality_reduction(latent, tsne_2d, k=k, verbose=False)
            for key in _DRE_KEYS:
                result[f"DRE_tsne_{key}"] = dre.get(key, np.nan)
        except Exception as e:
            print(f"  [WARN] DRE-tsne failed for {model_name}: {e}")
            for key in _DRE_KEYS:
                result[f"DRE_tsne_{key}"] = np.nan
    else:
        for key in _DRE_KEYS:
            result[f"DRE_tsne_{key}"] = np.nan

    # LSE (intrinsic latent structure)
    try:
        lse = evaluate_single_cell_latent_space(latent, data_type=data_type, verbose=False)
        for key in ["manifold_dimensionality", "spectral_decay_rate", "participation_ratio",
                     "anisotropy_score", "trajectory_directionality", "noise_resilience",
                     "core_quality", "overall_quality"]:
            result[f"LSE_{key}"] = lse.get(key, np.nan)
    except Exception as e:
        print(f"  [WARN] LSE failed for {model_name}: {e}")
        for key in ["manifold_dimensionality", "spectral_decay_rate", "participation_ratio",
                     "anisotropy_score", "trajectory_directionality", "noise_resilience",
                     "core_quality", "overall_quality"]:
            result[f"LSE_{key}"] = np.nan

    # DREX — extended DR fidelity (latent → UMAP 2-D)
    if umap_2d is not None:
        try:
            drex = evaluate_extended_dimensionality_reduction(latent, umap_2d, n_neighbors=k)
            for key in _DREX_KEYS:
                result[f"DREX_{key}"] = drex.get(key, np.nan)
        except Exception as e:
            print(f"  [WARN] DREX failed for {model_name}: {e}")
            for key in _DREX_KEYS:
                result[f"DREX_{key}"] = np.nan
    else:
        for key in _DREX_KEYS:
            result[f"DREX_{key}"] = np.nan

    # LSEX (extended latent geometry)
    try:
        lsex = evaluate_extended_latent_space(latent, n_neighbors=k)
        for key in ["two_hop_connectivity", "radial_concentration_quality",
                     "local_curvature_linearity", "neighbor_entropy_stability", "overall_quality"]:
            result[f"LSEX_{key}"] = lsex.get(key, np.nan)
    except Exception as e:
        print(f"  [WARN] LSEX failed for {model_name}: {e}")
        for key in ["two_hop_connectivity", "radial_concentration_quality",
                     "local_curvature_linearity", "neighbor_entropy_stability", "overall_quality"]:
            result[f"LSEX_{key}"] = np.nan

    return result


def main():
    parser = argparse.ArgumentParser(description="Offline GSE evaluation on cached latent spaces")
    parser.add_argument("--k", type=int, default=15, help="Neighborhood size for metrics")
    parser.add_argument("--series", choices=["dpmm", "topic", "all"], default="all")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 80)
    print("  Offline GSE (DRE + DREX + LSEX) Evaluation on Cached Latent Spaces")
    print("=" * 80)

    # Evaluate each model series (no expression-space data needed)
    series_to_run = list(MODEL_SERIES.keys()) if args.series == "all" else [args.series]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for series_name in series_to_run:
        model_names = MODEL_SERIES[series_name]
        print(f"\n{'='*60}")
        print(f"  Series: {series_name.upper()} ({len(model_names)} models)")
        print(f"{'='*60}")

        rows = []
        for mname in model_names:
            latent = load_latent(mname)
            if latent is None:
                print(f"  [SKIP] {mname}: no cached latent found")
                continue

            print(f"  Evaluating {mname} (latent {latent.shape}) ...", end=" ", flush=True)
            t0 = time.time()
            result = evaluate_model(mname, latent, k=args.k)
            elapsed = time.time() - t0
            result["eval_time_sec"] = round(elapsed, 2)
            rows.append(result)

            dre_umap_ov = result.get("DRE_umap_overall_quality", np.nan)
            drex_ov = result.get("DREX_overall_quality", np.nan)
            lsex_ov = result.get("LSEX_overall_quality", np.nan)
            print(f"DRE_umap={dre_umap_ov:.4f}  DREX={drex_ov:.4f}  LSEX={lsex_ov:.4f}  ({elapsed:.1f}s)")

        if not rows:
            print(f"  No models evaluated for {series_name}")
            continue

        # Save CSV
        df = pd.DataFrame(rows)
        csv_path = OUTPUT_DIR / f"gse_{series_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"\n  CSV saved: {csv_path}")

        # Save JSON
        json_path = OUTPUT_DIR / f"gse_{series_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(rows), f, ensure_ascii=False, indent=2)
        print(f"  JSON saved: {json_path}")

        # Also try to merge into existing base CSV
        _try_merge_into_base_csv(series_name, df)

        # Print summary table
        print(f"\n  {'Model':<25} {'DRE_umap':>8} {'LSE':>6} {'DREX':>6} {'LSEX':>6}")
        print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
        for _, row in df.iterrows():
            print(f"  {row['Model']:<25} "
                  f"{row.get('DRE_umap_overall_quality', np.nan):>8.3f} "
                  f"{row.get('LSE_overall_quality', np.nan):>6.3f} "
                  f"{row.get('DREX_overall_quality', np.nan):>6.3f} "
                  f"{row.get('LSEX_overall_quality', np.nan):>6.3f}")

    print(f"\n{'='*80}")
    print("  Done!")
    print(f"{'='*80}")


def _try_merge_into_base_csv(series_name: str, gse_df: pd.DataFrame):
    """Try to merge DREX/LSEX columns into the existing base benchmark CSV."""
    csv_dir = BASE_CSV_DIR / series_name
    if not csv_dir.exists():
        return

    csvs = sorted(csv_dir.glob("results_*.csv"))
    if not csvs:
        return

    latest_csv = csvs[-1]
    try:
        base_df = pd.read_csv(latest_csv)

        # Extract DRE_umap / DRE_tsne / DREX / LSEX columns
        merge_prefixes = ("DRE_umap_", "DRE_tsne_", "DREX_", "LSEX_")
        merge_cols = ["Model"] + [c for c in gse_df.columns
                                   if any(c.startswith(p) for p in merge_prefixes)]
        gse_subset = gse_df[merge_cols]

        # Drop existing columns with these prefixes (avoid duplicates)
        drop_cols = [c for c in base_df.columns
                     if any(c.startswith(p) for p in merge_prefixes)]
        if drop_cols:
            base_df = base_df.drop(columns=drop_cols)

        merged = base_df.merge(gse_subset, on="Model", how="left")

        # Save merged version
        merged_path = latest_csv.parent / latest_csv.name.replace(".csv", "_with_gse.csv")
        merged.to_csv(merged_path, index=False, float_format="%.6f")
        print(f"  Merged CSV: {merged_path}")
    except Exception as e:
        print(f"  [WARN] Could not merge into base CSV: {e}")


if __name__ == "__main__":
    main()
