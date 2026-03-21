"""Refined Figure 9 — GO Enrichment Dot Plots (composed figure).

Gene Ontology enrichment results for perturbation and beta-decoder gene sets,
across representative datasets × prior models.

Data source: experiments/results/ (bio-validation enrichment data)

Usage:
    python -m refined_figures.fig09_enrichment --series dpmm
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion)
from benchmarks.figure_generators.common import (
    MODEL_SHORT_NAMES, REPRESENTATIVE_DATASETS,
    PRIOR_MODELS_DPMM, PRIOR_MODELS_TOPIC, BIO_RESULTS)

DPI = 300


def _load_enrichment(ds_name, model_name, method="perturbation"):
    """Load GO enrichment results.

    Lookup order
    ------------
    1. Original flat paths: ``BIO_RESULTS / ds_name / model / <filename>``
    2. Per-component CSVs (10 components) located either in a model
       sub-directory or directly under *BIO_RESULTS*:
       - perturbation:  ``{model}_{ds}_enrichment_comp*.csv``
       - beta:          ``{model}_{ds}_beta_enrich_comp*.csv``

    When loading per-component files the individual DataFrames are
    concatenated with an extra ``Component`` column and trimmed to the top
    20 terms per component (by ascending *P-value*) to keep memory
    reasonable.
    """
    safe_model = model_name.replace("/", "_").replace(" ", "_")

    # --- Attempt 1: original flat-directory paths -------------------------
    bio_dir = BIO_RESULTS / ds_name / safe_model
    for fname in [f"go_enrichment_{method}.csv",
                  f"enrichment_{method}.csv",
                  "go_enrichment.csv",
                  "enrichment_results.csv"]:
        p = bio_dir / fname
        if p.exists():
            return pd.read_csv(p)
    # Try JSON format
    for fname in [f"go_enrichment_{method}.json",
                  "go_enrichment.json"]:
        p = bio_dir / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and "results" in data:
                return pd.DataFrame(data["results"])

    # --- Attempt 2: per-component CSVs ------------------------------------
    if method == "beta":
        pattern = f"{safe_model}_{ds_name}_beta_enrich_comp*.csv"
    else:                       # "perturbation" or default
        pattern = f"{safe_model}_{ds_name}_enrichment_comp*.csv"

    # Look in model sub-directory first, then top-level BIO_RESULTS
    comp_files = []
    for search_dir in [BIO_RESULTS / safe_model, BIO_RESULTS]:
        comp_files = sorted(search_dir.glob(pattern))
        if comp_files:
            break

    if not comp_files:
        return None

    frames = []
    for csv_path in comp_files:
        # Extract component number from filename  (e.g. "…_comp3.csv")
        comp_str = csv_path.stem.rsplit("comp", 1)[-1]
        try:
            comp_num = int(comp_str)
        except ValueError:
            comp_num = comp_str
        df = pd.read_csv(csv_path)
        df["Component"] = comp_num
        frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Trim to top terms per component so the DataFrame stays manageable
    pval_col = None
    for c in ["P-value", "Adjusted P-value", "p_value", "pvalue",
              "p.adjust", "padj", "FDR", "qvalue"]:
        if c in combined.columns:
            pval_col = c
            break
    if pval_col is not None:
        combined = (combined
                    .sort_values(pval_col)
                    .groupby("Component", sort=False)
                    .head(20)
                    .reset_index(drop=True))

    return combined


def _draw_enrichment_dotplot(ax, enrich_df, title, top_n=15):
    """Draw a GO enrichment dot plot."""
    if enrich_df is None or len(enrich_df) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
        return

    # Find p-value and term columns
    pval_col = None
    for c in ["P-value", "p_value", "pvalue", "p.adjust",
              "Adjusted P-value", "padj", "FDR", "qvalue"]:
        if c in enrich_df.columns:
            pval_col = c
            break
    term_col = None
    for c in ["Term", "name", "Description", "GO_term", "term_name"]:
        if c in enrich_df.columns:
            term_col = c
            break
    count_col = None
    for c in ["Count", "gene_count", "Overlap", "n_genes", "size"]:
        if c in enrich_df.columns:
            count_col = c
            break

    if pval_col is None or term_col is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing columns", ha="center", va="center",
                fontsize=8)
        return

    df = enrich_df.nsmallest(top_n, pval_col).copy()
    df["_neg_log_p"] = -np.log10(df[pval_col].clip(lower=1e-300))
    terms = df[term_col].values
    # Truncate long term names
    terms = [t[:45] + "…" if len(str(t)) > 45 else str(t) for t in terms]
    y_pos = np.arange(len(terms))

    sizes = 40
    if count_col and count_col in df.columns:
        counts = pd.to_numeric(df[count_col], errors="coerce").fillna(1)
        sizes = 20 + counts / counts.max() * 80

    sc = ax.scatter(df["_neg_log_p"].values, y_pos, s=sizes,
                    c=df["_neg_log_p"].values, cmap="YlOrRd",
                    edgecolors="black", linewidths=0.3, zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=7)
    ax.set_xlabel("-log₁₀(p)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=3, loc="left", fontweight="normal")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2, lw=0.4)


def generate(out_dir):
    """Generate refined Figure 9."""
    series = "topic"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    prior_models = (PRIOR_MODELS_DPMM if series == "dpmm"
                    else PRIOR_MODELS_TOPIC)
    datasets = REPRESENTATIVE_DATASETS
    n_rows = len(datasets)
    n_cols = len(prior_models)

    fig = plt.figure(figsize=(17.0, 5.5 * n_rows + 1.0))
    root = bind_figure_region(fig, (0.12, 0.04, 0.96, 0.95))
    grid = root.grid(n_rows, n_cols, wgap=0.06, hgap=0.06)

    for r_idx, ds in enumerate(datasets):
        for c_idx, model in enumerate(prior_models):
            ax = grid[r_idx][c_idx].add_axes(fig)
            style_axes(ax, kind="default")
            enrich_df = _load_enrichment(ds, model)
            short = MODEL_SHORT_NAMES.get(model, model)
            title = f"{short} — {ds}"
            _draw_enrichment_dotplot(ax, enrich_df, title)
            if r_idx == 0 and c_idx == 0:
                add_panel_label(ax, "a")

    out_path = out_dir / "Fig9_enrichment_topic.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "refined_figures" / "output" / series)
    generate(out)
