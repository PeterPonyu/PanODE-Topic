#!/usr/bin/env python3
"""Update hardcoded Table 1b rows in PAPER_BLUEPRINT_*.md files.

Reads the multi-seed crossdata CSV, computes seed-averaged per-model
means, and replaces the Table 1b body rows in all 4 blueprint files.

Table 1 (single-dataset) and Table 1c (joint-training) are NOT updated
automatically because they come from different data sources.

Usage:
    python benchmarks/scripts/update_blueprint_tables.py          # dry-run
    python benchmarks/scripts/update_blueprint_tables.py --write  # apply
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.config import DEFAULT_OUTPUT_DIR

CSV_DIR = DEFAULT_OUTPUT_DIR / "crossdata" / "csv"

# ── Blueprint files ──────────────────────────────────────────────────────────

DOCS = ROOT / "docs"
BLUEPRINT_FILES = [
    DOCS / "PAPER_BLUEPRINT_DPMM.md",
    DOCS / "PAPER_BLUEPRINT_DPMM_IEEE.md",
    DOCS / "PAPER_BLUEPRINT_TOPIC.md",
    DOCS / "PAPER_BLUEPRINT_TOPIC_IEEE.md",
]

# ── Model groups ─────────────────────────────────────────────────────────────

DPMM_MODELS = [
    "DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
    "Pure-AE", "Pure-Transformer-AE", "Pure-Contrastive-AE",
]
TOPIC_MODELS = [
    "Topic-Base", "Topic-Transformer", "Topic-Contrastive",
    "Pure-VAE", "Pure-Transformer-VAE", "Pure-Contrastive-VAE",
]

# Metric columns in CSV → display names for Table 1b
TABLE1B_METRICS = [
    ("NMI", "NMI(avg)"),
    ("ARI", "ARI(avg)"),
    ("ASW", "ASW(avg)"),
    ("DAV", "DAV(avg)"),
    ("DRE_umap_overall_quality", "DRE UMAP(avg)"),
    ("LSE_overall_quality", "LSE(avg)"),
    ("DREX_overall_quality", "DREX(avg)"),
    ("LSEX_overall_quality", "LSEX(avg)"),
]


def load_seed_averaged_means(csv_path: Path) -> pd.DataFrame:
    """Load CSV and compute seed-averaged model means."""
    df = pd.read_csv(csv_path)
    metric_cols = [m for m, _ in TABLE1B_METRICS]

    # Average across seeds first (if Seed/seed column exists)
    seed_col = "Seed" if "Seed" in df.columns else "seed" if "seed" in df.columns else None
    if seed_col:
        avg = df.groupby(["Model", "Dataset"])[metric_cols].mean().reset_index()
    else:
        avg = df.copy()

    return avg.groupby("Model")[metric_cols].mean()


def format_table1b_rows(model_means: pd.DataFrame, models: list[str]) -> str:
    """Generate table body rows sorted by descending NMI."""
    sub = model_means.loc[model_means.index.isin(models)].copy()
    sub = sub.sort_values("NMI", ascending=False)

    lines = []
    for model_name in sub.index:
        vals = []
        for col, _ in TABLE1B_METRICS:
            v = sub.loc[model_name, col]
            vals.append(f"{v:.4f}")
        lines.append(f"| {model_name} | {' | '.join(vals)} |")
    return "\n".join(lines)


def update_table1b_in_file(filepath: Path, new_rows: str, dry_run: bool) -> bool:
    """Replace Table 1b body rows in a blueprint file.

    Looks for the Table 1b header row (starting with |:--) followed by
    data rows, and replaces them.
    """
    text = filepath.read_text()

    # Pattern: find "Table 1b" section, then the separator row, then data rows
    # The header line starts with "| Model | NMI(avg)" and separator is "|:--|"
    # Data rows start with "| " and contain digits — stop at a blank line or
    # a line that doesn't start with "|"
    pattern = re.compile(
        r"(\*\*Table 1b\..*?\n"            # Table 1b title line
        r"\n"                               # blank line after title
        r"\| Model \| NMI\(avg\).*?\n"      # header row
        r"\|:--\|.*?\n)"                    # separator row
        r"((?:\| [^\n]+\n)+)",              # data rows (greedy within pipe-lines)
        re.DOTALL)

    m = pattern.search(text)
    if not m:
        print(f"  ⚠ Table 1b not found in {filepath.name}")
        return False

    old_rows = m.group(2).rstrip("\n")
    if old_rows == new_rows:
        print(f"  ✓ {filepath.name}: Table 1b already up-to-date")
        return False

    if dry_run:
        print(f"  📝 {filepath.name}: Table 1b would be updated")
        print(f"     Old rows:\n{old_rows}")
        print(f"     New rows:\n{new_rows}")
        return True

    new_text = text[:m.start(2)] + new_rows + "\n" + text[m.end(2):]
    filepath.write_text(new_text)
    print(f"  ✓ {filepath.name}: Table 1b updated")
    return True


def main():
    parser = argparse.ArgumentParser(description="Update Table 1b in blueprint files")
    parser.add_argument("--write", action="store_true", help="Actually write changes (default: dry-run)")
    args = parser.parse_args()

    # Load CSV
    csvs = sorted(CSV_DIR.glob("results_combined_*.csv"))
    if not csvs:
        sys.exit(f"No CSV found in {CSV_DIR}")
    csv_path = csvs[-1]
    print(f"Using CSV: {csv_path.name}")

    model_means = load_seed_averaged_means(csv_path)
    print(f"Computed seed-averaged means for {len(model_means)} models")

    dpmm_rows = format_table1b_rows(model_means, DPMM_MODELS)
    topic_rows = format_table1b_rows(model_means, TOPIC_MODELS)

    if not args.write:
        print("\n[DRY RUN — pass --write to apply changes]\n")

    n_updated = 0
    for bp in BLUEPRINT_FILES:
        if not bp.exists():
            print(f"  ⚠ Not found: {bp.name}")
            continue
        # Pick the right model set based on filename
        if "DPMM" in bp.name:
            rows = dpmm_rows
        else:
            rows = topic_rows
        if update_table1b_in_file(bp, rows, dry_run=not args.write):
            n_updated += 1

    action = "would update" if not args.write else "updated"
    print(f"\n{n_updated} files {action}.")


if __name__ == "__main__":
    main()
