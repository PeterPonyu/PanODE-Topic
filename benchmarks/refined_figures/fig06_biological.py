"""Refined Figure 6 — Biological Validation Heatmaps (composed figure).

Produces gene importance heatmaps across representative datasets and models.

Data source: benchmarks/biological_validation/results/

Usage:
    python -m benchmarks.refined_figures.fig06_biological --series dpmm
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion, add_colorbar_safe)
from benchmarks.figure_generators.common import (
    MODEL_SHORT_NAMES, REPRESENTATIVE_DATASETS,
    PRIOR_MODELS_DPMM, PRIOR_MODELS_TOPIC, BIO_RESULTS)

DPI = 300

_COMP_PALETTE = [
    "#4E79A7", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1",
    "#9C755F", "#BAB0AC", "#86BCB6", "#A0CBE8", "#CFCFCF",
]


def _load_gene_importance(series, ds_name, model_name):
    """Load gene importance scores from biological validation results."""
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    bio_dir = BIO_RESULTS / ds_name / safe_model
    # Try common file patterns
    for pattern in ["gene_importance.csv", "feature_importance.csv",
                    "gene_weights.csv", "beta_decoder_weights.csv"]:
        p = bio_dir / pattern
        if p.exists():
            return pd.read_csv(p)
    # Try NPZ
    for pattern in ["gene_importance.npz", "feature_importance.npz"]:
        p = bio_dir / pattern
        if p.exists():
            data = np.load(p, allow_pickle=True)
            if "importance" in data:
                return pd.DataFrame({
                    "gene": data.get("genes", np.arange(len(data["importance"]))),
                    "importance": data["importance"],
                })
    return None


def _draw_gene_heatmap(ax, importance_df, title, top_n=30):
    """Draw a gene importance heatmap."""
    if importance_df is None or len(importance_df) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
        return

    # Sort by importance and take top N
    score_col = [c for c in importance_df.columns
                 if c in ("importance", "weight", "score", "mean_abs")]
    if not score_col:
        score_col = [c for c in importance_df.select_dtypes(include="number").columns
                     if c != "index"]
    if not score_col:
        ax.axis("off")
        return
    sc = score_col[0]
    df_sorted = importance_df.nlargest(top_n, sc)
    genes = df_sorted.iloc[:, 0].values if importance_df.columns[0] != sc else np.arange(len(df_sorted))
    vals = df_sorted[sc].values.reshape(1, -1)

    im = ax.imshow(vals, aspect="auto", cmap="YlOrRd")
    ax.set_yticks([])
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, fontsize=7, rotation=90, ha="center")
    ax.set_title(title, fontsize=10, pad=3, loc="left", fontweight="normal")
    add_colorbar_safe(im, ax=ax, shrink=0.5, pad=0.02, label="Score")


def generate(series, out_dir):
    """Generate refined Figure 6."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    prior_models = (PRIOR_MODELS_DPMM if series == "dpmm"
                    else PRIOR_MODELS_TOPIC)
    datasets = REPRESENTATIVE_DATASETS
    n_rows = len(datasets)
    n_cols = len(prior_models)

    fig = plt.figure(figsize=(17.0, 4.0 * n_rows + 1.0))
    root = bind_figure_region(fig, (0.05, 0.05, 0.96, 0.95))
    grid = root.grid(n_rows, n_cols, wgap=0.03, hgap=0.06)

    for r_idx, ds in enumerate(datasets):
        for c_idx, model in enumerate(prior_models):
            ax = grid[r_idx][c_idx].add_axes(fig)
            style_axes(ax, kind="heatmap")
            imp_df = _load_gene_importance(series, ds, model)
            short = MODEL_SHORT_NAMES.get(model, model)
            title = f"{short} — {ds}"
            _draw_gene_heatmap(ax, imp_df, title)
            if r_idx == 0 and c_idx == 0:
                add_panel_label(ax, "a")

    out_path = out_dir / f"Fig6_biological_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "refined_figures" / "output" / args.series)
    generate(args.series, out)
