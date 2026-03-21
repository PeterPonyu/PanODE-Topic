"""Refined Figure 5 — Cross-Dataset Metric Trade-Off Scatter (composed figure).

Produces scatter + convex hull plots showing how different model architectures
balance competing quality objectives across all 56 datasets.

Data source: benchmarks/benchmark_results/crossdata/csv/

Usage:
    python -m benchmarks.refined_figures.fig05_crossdataset --series dpmm
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion)
from benchmarks.figure_generators.common import (
    get_model_order, get_color, MODEL_SHORT_NAMES, METRIC_DIRECTION)
from benchmarks.figure_generators.data_loaders import load_crossdata_combined

DPI = 300

# Metric pairs for scatter panels (DPMM has 6 core; Topic has 4 core)
_METRIC_PAIRS_DPMM = [
    ("NMI", "ASW"),
    ("NMI", "DAV"),
    ("ARI", "ASW"),
    ("ASW", "DRE_umap_overall_quality"),
    ("ASW", "LSE_overall_quality"),
    ("DRE_umap_overall_quality", "LSE_overall_quality"),
]
_METRIC_PAIRS_TOPIC = [
    ("NMI", "ASW"),
    ("NMI", "DAV"),
    ("ARI", "ASW"),
    ("ARI", "DAV"),
]

_METRIC_LABELS = {
    "NMI": "NMI", "ARI": "ARI", "ASW": "ASW", "DAV": "DAV",
    "DRE_umap_overall_quality": "DRE UMAP",
    "LSE_overall_quality": "LSE Overall",
}


def _draw_scatter_hull(ax, df, order, x_col, y_col):
    """Scatter + convex hull for per-model means across datasets."""
    for model in order:
        sub = df[df["Model"] == model]
        if x_col not in sub.columns or y_col not in sub.columns:
            continue
        xv = sub.groupby("Dataset")[x_col].mean()
        yv = sub.groupby("Dataset")[y_col].mean()
        common = xv.index.intersection(yv.index)
        if len(common) < 2:
            continue
        xvals = xv[common].values
        yvals = yv[common].values
        color = get_color(model)
        ax.scatter(xvals, yvals, s=20, c=[color], alpha=0.7,
                   edgecolors="black", linewidths=0.2, zorder=5,
                   label=MODEL_SHORT_NAMES.get(model, model))
        # Convex hull
        if len(xvals) >= 3:
            pts = np.column_stack([xvals, yvals])
            try:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1],
                            color=color, alpha=0.25, lw=0.8)
                hull_pts = pts[hull.vertices]
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        color=color, alpha=0.08)
            except Exception:
                pass

    xl = _METRIC_LABELS.get(x_col, x_col)
    yl = _METRIC_LABELS.get(y_col, y_col)
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.2, lw=0.4)


def generate(series, out_dir):
    """Generate refined Figure 5."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    df = load_crossdata_combined(prefer_multiseed=True)
    order = get_model_order(series)
    valid = set(order)
    df = df[df["Model"].isin(valid)].copy()

    pairs = _METRIC_PAIRS_TOPIC if series == "topic" else _METRIC_PAIRS_DPMM
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    figw = 17.0
    figh = 4.5 * n_rows + 1.5
    fig = plt.figure(figsize=(figw, figh))
    root = bind_figure_region(fig, (0.06, 0.06, 0.96, 0.94))
    grid = root.grid(n_rows, n_cols, wgap=0.04, hgap=0.06)

    for idx, (x_col, y_col) in enumerate(pairs):
        r, c = divmod(idx, n_cols)
        ax = grid[r][c].add_axes(fig)
        style_axes(ax, kind="scatter")
        _draw_scatter_hull(ax, df, order, x_col, y_col)
        if idx == 0:
            add_panel_label(ax, "a")

    # Shared legend
    handles, labels = [], []
    for model in order:
        handles.append(plt.Line2D([0], [0], marker="o", ls="",
                                  color=get_color(model), markersize=5))
        labels.append(MODEL_SHORT_NAMES.get(model, model))
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=min(len(order), 5),
               fontsize=9, frameon=False, handletextpad=0.3,
               columnspacing=0.6)

    out_path = out_dir / f"Fig5_crossdataset_{series}.png"
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
