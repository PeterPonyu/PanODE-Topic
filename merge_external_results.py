#!/usr/bin/env python3
"""
Merge 5 external baseline group result directories into one unified external_full directory.

Groups:
  - gaussian_geometric  (56/56 datasets)
  - disentanglement     (56/56 datasets)
  - graph_contrastive   (56/56 datasets)
  - scvi_family         (56/56 datasets)
  - generative          (34/56 datasets)

For each dataset, concatenate rows from all groups that have it,
producing a single CSV per dataset in the output directory.
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/home/zeyufu/Desktop/PanODE-LAB/experiments/results/external")
GROUPS = ["gaussian_geometric", "disentanglement", "graph_contrastive", "scvi_family", "generative"]

OUT_BASE = Path("/home/zeyufu/Desktop/PanODE-LAB/experiments/results/external_full")
OUT_TABLES = OUT_BASE / "tables"
OUT_SERIES = OUT_BASE / "series"

# ── Create output dirs ─────────────────────────────────────────────────────────
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_SERIES.mkdir(parents=True, exist_ok=True)

# ── Helper: discover and merge ─────────────────────────────────────────────────
def merge_subdirectory(subdir_name: str, out_dir: Path) -> dict:
    """
    For a given sub-directory name (e.g. 'tables' or 'series'),
    discover all CSVs across groups, concat per dataset, and write out.
    Returns stats dict.
    """
    # Map dataset filename -> list of (group, full_path)
    dataset_sources = defaultdict(list)

    for group in GROUPS:
        src_dir = BASE / group / subdir_name
        if not src_dir.is_dir():
            print(f"  [skip] {group}/{subdir_name}/ does not exist")
            continue
        for csv_file in sorted(src_dir.glob("*.csv")):
            dataset_sources[csv_file.name].append((group, csv_file))

    total_datasets = 0
    all_methods = set()
    datasets_written = []

    for filename in sorted(dataset_sources.keys()):
        sources = dataset_sources[filename]
        frames = []
        for group, fpath in sources:
            try:
                df = pd.read_csv(fpath)
                frames.append(df)
            except Exception as e:
                print(f"  [warn] Could not read {fpath}: {e}")

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)

        # Deduplicate: if a method appears in multiple groups (shouldn't, but safety)
        if "method" in merged.columns:
            before = len(merged)
            merged = merged.drop_duplicates(subset=["method"], keep="first")
            if len(merged) < before:
                print(f"  [dedup] {filename}: {before} -> {len(merged)} rows (removed duplicate methods)")
            all_methods.update(merged["method"].tolist())
        elif "hue" in merged.columns:
            # Series files use 'hue' as the method identifier; duplicates across groups
            # are expected per (epoch, hue) but we just concat since each group has different hues
            before = len(merged)
            merged = merged.drop_duplicates(subset=["epoch", "hue"], keep="first")
            if len(merged) < before:
                print(f"  [dedup] {filename}: {before} -> {len(merged)} rows (removed duplicate epoch/hue)")
            all_methods.update(merged["hue"].tolist())

        out_path = out_dir / filename
        merged.to_csv(out_path, index=False)
        total_datasets += 1
        datasets_written.append((filename, len(sources), len(merged)))

    return {
        "total_datasets": total_datasets,
        "all_methods": all_methods,
        "details": datasets_written,
    }


# ── Merge tables ───────────────────────────────────────────────────────────────
print("=" * 70)
print("MERGING tables/")
print("=" * 70)
tables_stats = merge_subdirectory("tables", OUT_TABLES)

print(f"\nDatasets written: {tables_stats['total_datasets']}")
print(f"Unique methods:  {len(tables_stats['all_methods'])}")
print(f"\nMethods list ({len(tables_stats['all_methods'])}):")
for m in sorted(tables_stats["all_methods"]):
    print(f"  - {m}")

# Show per-dataset detail
print(f"\nPer-dataset breakdown (tables):")
print(f"  {'Dataset':<40s} {'Groups':>6s} {'Rows':>6s}")
print(f"  {'-'*40} {'-'*6} {'-'*6}")
for fname, ngroups, nrows in tables_stats["details"]:
    print(f"  {fname:<40s} {ngroups:>6d} {nrows:>6d}")


# ── Merge series ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MERGING series/")
print("=" * 70)
series_stats = merge_subdirectory("series", OUT_SERIES)

print(f"\nDatasets written: {series_stats['total_datasets']}")
print(f"Unique methods (hue): {len(series_stats['all_methods'])}")

# Show per-dataset detail
print(f"\nPer-dataset breakdown (series):")
print(f"  {'Dataset':<40s} {'Groups':>6s} {'Rows':>6s}")
print(f"  {'-'*40} {'-'*6} {'-'*6}")
for fname, ngroups, nrows in series_stats["details"]:
    print(f"  {fname:<40s} {ngroups:>6d} {nrows:>6d}")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  tables/  -> {tables_stats['total_datasets']} datasets, {len(tables_stats['all_methods'])} unique methods")
print(f"  series/  -> {series_stats['total_datasets']} datasets, {len(series_stats['all_methods'])} unique methods (hue)")
print(f"  Output:  {OUT_BASE}")
print("Done.")
