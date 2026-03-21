"""Lightweight run manifest for benchmark result tracking.

Every benchmark script appends a JSON-lines entry to
``<results_dir>/run_manifest.jsonl`` on completion so users can:

* List all previous runs at a glance (``list_runs``)
* Compare two runs side-by-side (``compare_runs``)
* Filter runs by script / dataset / date

File format
-----------
Each line is one self-contained JSON object with keys:

    timestamp, script, tag, series, dataset, n_cells, epochs, lr, seed,
    csv_path, models

Usage from CLI
--------------
::

    python -m benchmarks.run_manifest                     # list all
    python -m benchmarks.run_manifest --last 5            # last 5 runs
    python -m benchmarks.run_manifest --compare TAG1 TAG2 # compare two
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.config import DEFAULT_OUTPUT_DIR

MANIFEST_NAME = "run_manifest.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# Write API
# ═══════════════════════════════════════════════════════════════════════════════

def append_run(
    results_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    script: str,
    tag: str,
    series: str = "",
    dataset: str = "",
    n_cells: int = 0,
    epochs: int = 0,
    lr: float = 0.0,
    seed: int = 42,
    csv_path: str = "",
    models: list[str] | None = None,
    extra: dict[str, Any] | None = None) -> Path:
    """Append one line to the run manifest.

    Parameters
    ----------
    results_dir : Path
        Top-level results directory (contains ``run_manifest.jsonl``).
    script : str
        Name of the entry-point script (e.g. ``"benchmark_base"``).
    tag : str
        Human-readable run tag (usually contains timestamp).
    series, dataset, n_cells, epochs, lr, seed
        Key experimental parameters recorded for quick filtering.
    csv_path : str
        Path to the primary CSV output of this run.
    models : list[str]
        Model names included in this run.
    extra : dict
        Any additional metadata to attach.

    Returns
    -------
    Path
        Path to the manifest file.
    """
    manifest = Path(results_dir) / MANIFEST_NAME
    manifest.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": script,
        "tag": tag,
        "series": series,
        "dataset": dataset,
        "n_cells": n_cells,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "csv_path": csv_path,
        "models": models or [],
    }
    if extra:
        entry.update(extra)

    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Read API
# ═══════════════════════════════════════════════════════════════════════════════

def list_runs(
    results_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    last: int | None = None,
    script: str | None = None,
    dataset: str | None = None) -> pd.DataFrame:
    """Return a DataFrame of all recorded runs.

    Parameters
    ----------
    results_dir : Path
        Top-level results directory.
    last : int, optional
        Return only the *last* N entries.
    script : str, optional
        Filter to runs from this script.
    dataset : str, optional
        Filter to runs on this dataset.

    Returns
    -------
    pd.DataFrame
    """
    manifest = Path(results_dir) / MANIFEST_NAME
    if not manifest.exists():
        return pd.DataFrame()

    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if script:
        df = df[df["script"] == script]
    if dataset:
        df = df[df["dataset"] == dataset]
    if last:
        df = df.tail(last)

    return df.reset_index(drop=True)


def compare_runs(
    tag_a: str,
    tag_b: str,
    results_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Load CSVs for two runs and return a merged comparison DataFrame.

    Merges on ``Model`` column with suffixes ``_a`` / ``_b`` and computes
    delta columns for the core metrics (NMI, ARI, ASW).

    Returns
    -------
    pd.DataFrame
    """
    runs = list_runs(results_dir)
    if runs.empty:
        raise FileNotFoundError("No runs recorded in manifest.")

    def _load(tag):
        matches = runs[runs["tag"] == tag]
        if matches.empty:
            raise KeyError(f"Tag '{tag}' not found in manifest.")
        csv = matches.iloc[0]["csv_path"]
        return pd.read_csv(csv)

    df_a = _load(tag_a)
    df_b = _load(tag_b)

    merged = df_a.merge(df_b, on="Model", suffixes=("_a", "_b"), how="outer")

    for m in ("NMI", "ARI", "ASW"):
        ca, cb = f"{m}_a", f"{m}_b"
        if ca in merged.columns and cb in merged.columns:
            merged[f"Δ{m}"] = merged[cb] - merged[ca]

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Browse / compare benchmark runs")
    parser.add_argument("--results-dir", type=str,
                        default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--last", type=int, default=None,
                        help="Show only the last N runs")
    parser.add_argument("--script", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--compare", nargs=2, metavar="TAG",
                        help="Compare two run tags")
    args = parser.parse_args()

    rdir = Path(args.results_dir)

    if args.compare:
        df = compare_runs(args.compare[0], args.compare[1], rdir)
        print(df.to_string(index=False))
        return

    df = list_runs(rdir, last=args.last, script=args.script,
                   dataset=args.dataset)
    if df.empty:
        print("No runs found.")
        return

    # Compact display
    show_cols = ["timestamp", "script", "tag", "dataset", "epochs", "lr",
                 "seed", "csv_path"]
    show = [c for c in show_cols if c in df.columns]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    _cli()
