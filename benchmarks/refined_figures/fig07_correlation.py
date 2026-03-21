"""Refined Figure 7 — Latent-Gene Correlation Heatmaps (composed figure).

Pearson correlation between latent dimensions and marker genes across
representative datasets × prior models.

Data source: benchmarks/biological_validation/results/

Usage:
    python -m benchmarks.refined_figures.fig07_correlation --series dpmm
"""

import argparse
import sys
import numpy as np
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


def _load_correlation(ds_name, model_name):
    """Load latent-gene correlation data."""
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    bio_dir = BIO_RESULTS / ds_name / safe_model
    for fname in ["latent_gene_correlation.npz", "correlation_matrix.npz",
                  "pearson_correlation.npz"]:
        p = bio_dir / fname
        if p.exists():
            data = np.load(p, allow_pickle=True)
            corr = data.get("correlation", data.get("corr", None))
            genes = data.get("genes", data.get("gene_names", None))
            if corr is not None:
                genes = genes if genes is not None else np.arange(corr.shape[1])
                return corr, genes
    return None, None


def _draw_corr_heatmap(ax, corr, genes, title, top_n=30):
    """Draw a latent-gene correlation heatmap showing top correlated genes."""
    if corr is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
        return

    # Select top genes by max absolute correlation across all components
    max_abs = np.abs(corr).max(axis=0)
    top_idx = np.argsort(max_abs)[-top_n:][::-1]
    corr_sub = corr[:, top_idx]
    gene_sub = np.array(genes)[top_idx] if hasattr(genes, '__len__') else top_idx

    n_comp = min(corr_sub.shape[0], 10)  # Show at most 10 components
    corr_show = corr_sub[:n_comp, :]

    im = ax.imshow(corr_show, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([f"z{i}" for i in range(n_comp)], fontsize=8)
    ax.set_xticks(range(len(gene_sub)))
    ax.set_xticklabels(gene_sub, fontsize=6, rotation=90, ha="center")
    ax.set_title(title, fontsize=10, pad=3, loc="left", fontweight="normal")
    add_colorbar_safe(im, ax=ax, shrink=0.6, pad=0.02, label="Pearson r")


def generate(series, out_dir):
    """Generate refined Figure 7."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    prior_models = (PRIOR_MODELS_DPMM if series == "dpmm"
                    else PRIOR_MODELS_TOPIC)
    datasets = REPRESENTATIVE_DATASETS
    n_rows = len(datasets)
    n_cols = len(prior_models)

    fig = plt.figure(figsize=(17.0, 5.0 * n_rows + 1.0))
    root = bind_figure_region(fig, (0.05, 0.05, 0.96, 0.95))
    grid = root.grid(n_rows, n_cols, wgap=0.04, hgap=0.06)

    for r_idx, ds in enumerate(datasets):
        for c_idx, model in enumerate(prior_models):
            ax = grid[r_idx][c_idx].add_axes(fig)
            style_axes(ax, kind="heatmap")
            corr, genes = _load_correlation(ds, model)
            short = MODEL_SHORT_NAMES.get(model, model)
            title = f"{short} — {ds}"
            _draw_corr_heatmap(ax, corr, genes, title)
            if r_idx == 0 and c_idx == 0:
                add_panel_label(ax, "a")

    out_path = out_dir / f"Fig7_correlation_{series}.png"
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
