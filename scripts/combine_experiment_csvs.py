#!/usr/bin/env python3
"""
Combine Experiment-Style Per-Dataset CSVs into a Combined CrossData CSV.

Reads all {dataset}_df.csv files from an experiment directory and produces a
single combined CSV compatible with statistical_analysis.py.

Usage:
    python scripts/combine_experiment_csvs.py \
        --input experiments/results/ablation_dpmm_full/tables \
        --output benchmarks/benchmark_results/crossdata/csv/results_combined_full.csv

    # Or for merged (internal + external):
    python scripts/combine_experiment_csvs.py \
        --input experiments/results/full_vs_external_all/tables \
        --output benchmarks/benchmark_results/crossdata/csv/results_combined_full.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def combine_tables(input_dir: Path, output_path: Path):
    """Read per-dataset CSVs and combine into one DataFrame."""
    csv_files = sorted(input_dir.glob("*_df.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    frames = []
    for csv_path in csv_files:
        dataset_key = csv_path.stem.replace("_df", "")
        df = pd.read_csv(csv_path)
        df.insert(0, "Dataset", dataset_key)
        # Rename 'method' -> 'Model' for consistency with statistical_analysis.py
        if "method" in df.columns and "Model" not in df.columns:
            df.rename(columns={"method": "Model"}, inplace=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Combined {len(csv_files)} datasets -> {output_path}")
    print(f"  Shape: {combined.shape}")
    print(f"  Models: {sorted(combined['Model'].unique())}")
    print(f"  Datasets: {sorted(combined['Dataset'].unique())}")


def main():
    parser = argparse.ArgumentParser(description="Combine experiment CSVs")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing per-dataset CSVs")
    parser.add_argument("--output", type=str, required=True,
                        help="Output combined CSV path")
    args = parser.parse_args()

    combine_tables(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
