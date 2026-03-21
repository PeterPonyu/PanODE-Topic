#!/usr/bin/env python3
"""Repair corrupted NMI/ASW/DAV values in crossdata and joint CSVs.

The crossdata and joint CSVs written on Feb 11–15, 2026 contain fabricated
NMI/ASW/DAV for prior-augmented models (DPMM-*, Topic-*).  ARI and all
DRE/LSE/DREX/LSEX metrics are correct.

This script:
  1. Loads each saved latent NPZ for every model × dataset pair
  2. Recomputes NMI, ASW, DAV using the same KMeans + sklearn pipeline
  3. Overwrites the corrupted columns in-place, preserving all other data
  4. Writes repaired CSVs alongside the originals (timestamped)

Usage:
    python repair_crossdata_csv.py [--dry-run]
"""

import argparse
import glob
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.config import DEFAULT_OUTPUT_DIR

RESULTS_DIR = DEFAULT_OUTPUT_DIR

PRIOR_MODELS = {
    "DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
    "Topic-Base", "Topic-Transformer", "Topic-Contrastive",
}

ALL_DATASETS = [
    "setty", "lung", "hesc", "retina", "pituitary", "endo",
    "dentate", "hemato", "blood_aged", "teeth",
    "pansci_muscle", "pansci_tcell",
]


def recompute_metrics(latent, labels):
    """Recompute NMI, ARI, ASW, DAV from latent and ground-truth labels."""
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(latent)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    try:
        asw = silhouette_score(latent, pred) if len(np.unique(pred)) > 1 else np.nan
    except Exception:
        asw = np.nan
    try:
        dav = davies_bouldin_score(latent, pred)
    except Exception:
        dav = np.nan
    return {"NMI": nmi, "ARI": ari, "ASW": asw, "DAV": dav}


def find_latest_latent(lat_dir, model_name, dataset):
    """Find the latest latent NPZ for a model on a dataset."""
    safe = model_name.replace("/", "_").replace(" ", "_")
    pattern = f"{safe}_{dataset}_*.npz"
    files = sorted(glob.glob(str(lat_dir / pattern)), reverse=True)
    return files[0] if files else None


def repair_crossdata(dry_run=False):
    """Repair all crossdata combined CSVs."""
    csv_dir = RESULTS_DIR / "crossdata" / "csv"
    lat_dir = RESULTS_DIR / "crossdata" / "latents"

    combined_files = sorted(csv_dir.glob("results_combined_*.csv"))
    if not combined_files:
        print("  No crossdata combined CSVs found.")
        return

    # Build recomputed metrics cache from latent NPZ files
    cache = {}
    for dataset in ALL_DATASETS:
        ds_lat_dir = lat_dir / dataset
        if not ds_lat_dir.exists():
            print(f"  WARNING: No latent dir for {dataset}")
            continue
        for model in PRIOR_MODELS:
            npz_path = find_latest_latent(ds_lat_dir, model, dataset)
            if npz_path is None:
                print(f"  WARNING: No latent for {model} on {dataset}")
                continue
            data = np.load(npz_path)
            latent = data["latent"]
            labels = data["labels"]
            metrics = recompute_metrics(latent, labels)
            cache[(model, dataset)] = metrics
            print(f"  Recomputed {model} @ {dataset}: "
                  f"NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f} "
                  f"ASW={metrics['ASW']:.4f} DAV={metrics['DAV']:.4f}")

    # Repair each combined CSV
    for csv_path in combined_files:
        print(f"\n  Repairing {csv_path.name}...")
        df = pd.read_csv(csv_path)
        n_fixed = 0
        for idx, row in df.iterrows():
            model, dataset = row.get("Model"), row.get("Dataset")
            if model in PRIOR_MODELS and (model, dataset) in cache:
                new = cache[(model, dataset)]
                for col in ["NMI", "ASW", "DAV"]:
                    old_val = df.at[idx, col]
                    df.at[idx, col] = new[col]
                    if abs(old_val - new[col]) > 1e-6:
                        n_fixed += 1
        if dry_run:
            print(f"  DRY RUN: Would fix {n_fixed} values in {csv_path.name}")
        else:
            # Backup original
            backup = csv_path.with_suffix(".csv.bak")
            if not backup.exists():
                shutil.copy2(csv_path, backup)
            df.to_csv(csv_path, index=False)
            print(f"  Fixed {n_fixed} values in {csv_path.name}")

    # Also write a fresh combined CSV with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dpmm_files = sorted(csv_dir.glob("results_combined_*.csv"), reverse=True)
    if dpmm_files and not dry_run:
        frames = [pd.read_csv(f) for f in dpmm_files]
        merged = pd.concat(frames, ignore_index=True)
        if "Model" in merged.columns and "Dataset" in merged.columns:
            merged = merged.drop_duplicates(subset=["Model", "Dataset"], keep="first")
        out_path = csv_dir / f"results_combined_{ts}.csv"
        merged.to_csv(out_path, index=False)
        print(f"  Wrote fresh combined: {out_path.name}")


def repair_joint(dry_run=False):
    """Repair joint training CSVs."""
    csv_dir = RESULTS_DIR / "joint" / "csv"
    lat_dir = RESULTS_DIR / "joint" / "latents"

    if not lat_dir.exists():
        print("  No joint latent dir found.")
        return

    for csv_path in sorted(csv_dir.glob("results_joint_*.csv")):
        print(f"\n  Repairing {csv_path.name}...")
        df = pd.read_csv(csv_path)
        n_fixed = 0
        for idx, row in df.iterrows():
            model = row.get("Model")
            if model not in PRIOR_MODELS:
                continue
            safe = model.replace("/", "_").replace(" ", "_")
            npz_files = sorted(lat_dir.glob(f"{safe}_joint_*.npz"), reverse=True)
            if not npz_files:
                print(f"  WARNING: No joint latent for {model}")
                continue
            data = np.load(npz_files[0], allow_pickle=True)
            latent = data["latent"]
            labels_str = data.get("dataset_labels")
            if labels_str is None:
                labels_str = data.get("labels")
            if labels_str is None:
                print(f"  WARNING: No labels in NPZ for {model}")
                continue
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(np.array(labels_str, dtype=str))
            metrics = recompute_metrics(latent, labels)
            print(f"  Recomputed {model} (joint): "
                  f"NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f} "
                  f"ASW={metrics['ASW']:.4f} DAV={metrics['DAV']:.4f}")
            for col in ["NMI", "ASW", "DAV"]:
                old_val = df.at[idx, col]
                df.at[idx, col] = metrics[col]
                if abs(old_val - metrics[col]) > 1e-6:
                    n_fixed += 1
        if dry_run:
            print(f"  DRY RUN: Would fix {n_fixed} values in {csv_path.name}")
        else:
            backup = csv_path.with_suffix(".csv.bak")
            if not backup.exists():
                shutil.copy2(csv_path, backup)
            df.to_csv(csv_path, index=False)
            print(f"  Fixed {n_fixed} values in {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Repair corrupted crossdata/joint CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fixed without writing")
    args = parser.parse_args()

    print("=" * 60)
    print("CROSSDATA CSV REPAIR")
    print("=" * 60)
    repair_crossdata(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("JOINT CSV REPAIR")
    print("=" * 60)
    repair_joint(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
