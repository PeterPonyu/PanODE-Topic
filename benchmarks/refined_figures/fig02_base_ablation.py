"""Refined Figure 2 — Base Ablation (multi-panel composed figure).

Loads cross-dataset benchmark results and produces a single composed figure:
  Panel (a) — Cross-dataset UMAP embeddings (1 per model, 3×3 grid)
  Panel (b) — Core metric boxplots (NMI, ARI, ASW, DAV + DRE/LSE for DPMM)
  Panel (c) — Efficiency boxplots (s/epoch, GPU MB, Params)

Data source: benchmarks/benchmark_results/ (crossdata CSV + latent NPZ)

Usage:
    python -m benchmarks.refined_figures.fig02_base_ablation --series dpmm
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
    bind_figure_region, LayoutRegion, MODEL_COLORS, VIS_STYLE)
from benchmarks.figure_generators.common import (
    get_model_order, get_color, compute_umap, clip_extreme_outliers,
    get_core_metrics, get_ext_metrics,
    MODEL_SHORT_NAMES, METRIC_DIRECTION)
from benchmarks.figure_generators.data_loaders import (
    load_crossdata_per_dataset, load_cross_latent)
from benchmarks.figure_generators.significance_brackets import (
    draw_significance_brackets)

# ── Constants ─────────────────────────────────────────────────────────────────

FIGSIZE = (17.0, 22.0)  # Full-page composed figure
DPI = 300
EFFICIENCY_METRICS = [
    ("SecPerEpoch", "s / epoch"),
    ("PeakGPU_MB",  "GPU (MB)"),
    ("NumParams",   "Parameters"),
]


def _draw_boxplot(ax, data_arrays, model_order, metric_col, metric_label,
                  series, higher_better=True):
    """Draw a styled boxplot on the given axes."""
    short = [MODEL_SHORT_NAMES.get(m, m) for m in model_order]
    n = len(model_order)
    data_arrays = clip_extreme_outliers(data_arrays)

    bp = ax.boxplot(data_arrays, vert=True, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(get_color(model_order[j]))
        patch.set_alpha(0.65)
        patch.set_edgecolor("gray")
        patch.set_linewidth(0.8)

    rng = np.random.RandomState(42)
    for j in range(n):
        vals = data_arrays[j]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(j + 1 + jitter, vals, s=14,
                   c=[get_color(model_order[j])],
                   edgecolors="black", linewidths=0.2, zorder=5, alpha=0.85)

    # Highlight best model
    medians = [np.nanmedian(d) if len(d) else np.nan for d in data_arrays]
    if not all(np.isnan(m) for m in medians):
        best_idx = int(np.nanargmax(medians) if higher_better
                       else np.nanargmin(medians))
        bp["boxes"][best_idx].set_edgecolor("red")
        bp["boxes"][best_idx].set_linewidth(1.3)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(short, fontsize=10, rotation=65, ha="right")
    ax.set_title(metric_label, fontsize=12, pad=3, loc="left",
                 fontweight="normal")
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.2, lw=0.4)

    # Significance brackets
    draw_significance_brackets(ax, model_order, metric_col, series,
                               data_per_model=data_arrays,
                               bracket_gap_frac=0.045, show_ns=False)

    # Y-axis padding
    ymin, ymax = ax.get_ylim()
    pad = abs(ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad * 0.3, ymax + pad)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))


def _draw_umap(ax, model_name, datasets, per_ds):
    """Draw a cross-dataset UMAP on the given axes."""
    blocks, ds_labels = [], []
    for ds in datasets:
        lat = load_cross_latent(model_name, ds)
        if lat is None or len(lat) == 0:
            continue
        blocks.append(lat)
        ds_labels.extend([ds] * len(lat))
    if not blocks:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        return
    X = np.vstack(blocks)
    emb = compute_umap(X)
    ds_labels = np.array(ds_labels)
    cmap = plt.colormaps["tab20"]
    color_map = {ds: cmap(i / max(len(datasets) - 1, 1))
                 for i, ds in enumerate(datasets)}
    for ds in datasets:
        mask = ds_labels == ds
        if not np.any(mask):
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.55,
                   color=[color_map[ds]], label=ds, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    short = MODEL_SHORT_NAMES.get(model_name, model_name)
    ax.set_title(short, fontsize=11, pad=2, loc="left", fontweight="normal")
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)


def generate(series, out_dir):
    """Generate refined Figure 2."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    # Load data
    per_ds = load_crossdata_per_dataset()
    order = get_model_order(series)
    valid = set(order)
    for ds in list(per_ds.keys()):
        per_ds[ds] = per_ds[ds][per_ds[ds]["Model"].isin(valid)].copy()
        if per_ds[ds].empty:
            per_ds.pop(ds, None)
    datasets = sorted(per_ds.keys())

    core_metrics = get_core_metrics(series)
    n_models = len(order)
    n_core = len(core_metrics)

    # ── Create composed figure ────────────────────────────────────────────
    fig = plt.figure(figsize=FIGSIZE)
    root = bind_figure_region(fig, (0.04, 0.03, 0.96, 0.97))

    # Split into 3 horizontal bands: UMAPs | Core boxplots | Efficiency
    rows = root.split_rows([3.0, 3.5, 2.0], gap=0.04)
    umap_region, core_region, eff_region = rows

    # Panel (a): UMAPs — 3 rows × 3 cols
    n_umap_rows = (n_models + 2) // 3
    umap_grid = umap_region.grid(n_umap_rows, 3, wgap=0.02, hgap=0.03)
    for i, model in enumerate(order):
        r, c = divmod(i, 3)
        if r < len(umap_grid) and c < len(umap_grid[r]):
            ax = umap_grid[r][c].add_axes(fig)
            style_axes(ax, kind="umap")
            _draw_umap(ax, model, datasets, per_ds)
            if i == 0:
                add_panel_label(ax, "a")

    # Add shared legend for UMAPs
    if datasets:
        cmap = plt.colormaps["tab20"]
        color_map = {ds: cmap(i / max(len(datasets) - 1, 1))
                     for i, ds in enumerate(datasets)}
        handles = [plt.Line2D([0], [0], marker="o", ls="",
                              color=color_map[ds], markersize=4, alpha=0.75)
                   for ds in datasets]
        fig.legend(handles, datasets, loc="lower center",
                   bbox_to_anchor=(0.5, umap_region.bottom - 0.015),
                   ncol=min(len(datasets), 8), fontsize=9,
                   frameon=False, handletextpad=0.3, columnspacing=0.6)

    # Panel (b): Core metric boxplots — 1 row × n_core cols
    core_cols = core_region.split_cols(n_core, gap=0.025)
    for idx, (col_name, label, higher) in enumerate(core_metrics):
        ax = core_cols[idx].add_axes(fig)
        style_axes(ax, kind="boxplot")
        # Build data arrays
        data_arr = []
        for m in order:
            vals = []
            for ds in datasets:
                rows_ds = per_ds[ds]
                sub = rows_ds[rows_ds["Model"] == m]
                if col_name in sub.columns:
                    vals.extend(sub[col_name].dropna().values.tolist())
            data_arr.append(np.array(vals))
        _draw_boxplot(ax, data_arr, order, col_name, label, series,
                      higher_better=higher)
        if idx == 0:
            add_panel_label(ax, "b")

    # Panel (c): Efficiency boxplots — 1 row × 3 cols
    eff_cols = eff_region.split_cols(len(EFFICIENCY_METRICS), gap=0.025)
    for idx, (col_name, label) in enumerate(EFFICIENCY_METRICS):
        ax = eff_cols[idx].add_axes(fig)
        style_axes(ax, kind="boxplot")
        data_arr = []
        for m in order:
            vals = []
            for ds in datasets:
                rows_ds = per_ds[ds]
                sub = rows_ds[rows_ds["Model"] == m]
                if col_name in sub.columns:
                    vals.extend(sub[col_name].dropna().values.tolist())
            data_arr.append(np.array(vals))
        _draw_boxplot(ax, data_arr, order, col_name, label, series,
                      higher_better=False)
        if idx == 0:
            add_panel_label(ax, "c")

    out_path = out_dir / f"Fig2_base_ablation_{series}.png"
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
