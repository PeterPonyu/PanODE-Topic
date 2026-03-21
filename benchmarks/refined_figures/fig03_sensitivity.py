"""Refined Figure 3 — Sensitivity Analysis (multi-panel composed figure).

Loads sensitivity / training / preprocessing sweep CSVs and produces a
composed grid of sweep-trend boxplots.  Each row = one hyperparameter sweep,
each column = one core metric.

Data source: benchmarks/benchmark_results/sensitivity/csv/{series}/
             benchmarks/benchmark_results/training/csv/{series}/
             benchmarks/benchmark_results/preprocessing/csv/{series}/

Usage:
    python -m benchmarks.refined_figures.fig03_sensitivity --series dpmm
"""

import argparse
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion)
from benchmarks.figure_generators.common import METRIC_DIRECTION
from benchmarks.figure_generators.data_loaders import (
    load_sensitivity_csv, load_training_csv, load_preprocessing_csv,
    parse_sweep_value)

DPI = 300

_CORE_METRICS_ALL = ["NMI", "ARI", "ASW", "DAV",
                     "DRE_umap_overall_quality", "LSE_overall_quality"]
_CORE_METRICS_TOPIC = ["NMI", "ARI", "ASW", "DAV"]
_DISPLAY = {
    "NMI": "NMI ↑", "ARI": "ARI ↑", "ASW": "ASW ↑", "DAV": "DAV ↓",
    "DRE_umap_overall_quality": "DRE UMAP ↑",
    "LSE_overall_quality": "LSE Overall ↑",
}
_EXCLUDED_SWEEPS = {"max_cells"}
_SWEEP_PALETTE = [
    "#4C78A8", "#6BAED6", "#9ECAE1", "#FDD49E", "#FDAE6B",
    "#FD8D3C", "#E6550D", "#A63603",
]


def _format_sweep_label(val_str):
    try:
        val = float(val_str)
        if val == 0:
            return "0"
        abs_val = abs(val)
        if abs_val >= 1000 or (0 < abs_val < 0.01):
            return f"{val:.0e}"
        if val == int(val):
            return str(int(val))
        return f"{val:g}"
    except (ValueError, TypeError):
        return str(val_str)


def _draw_sweep_boxplot(ax, df, param_col, metric):
    """Draw one sweep-trend boxplot."""
    if metric not in df.columns:
        ax.axis("off")
        return
    work = df.copy()
    if param_col not in work.columns:
        work[param_col] = work["Model"].map(parse_sweep_value)
    numeric_param = pd.to_numeric(work[param_col], errors="coerce")
    if numeric_param.notna().any():
        work["_sort"] = numeric_param
    else:
        work["_sort"] = work[param_col].astype(str)
    ordered = list(dict.fromkeys(
        work.sort_values("_sort")[param_col].astype(str).tolist()))
    n_sv = len(ordered)
    if n_sv <= len(_SWEEP_PALETTE):
        box_colors = _SWEEP_PALETTE[:n_sv]
    else:
        cmap = mpl.colormaps["RdYlBu_r"].resampled(n_sv)
        box_colors = [cmap(i / max(n_sv - 1, 1)) for i in range(n_sv)]

    grouped = []
    for sv in ordered:
        vals = pd.to_numeric(
            work.loc[work[param_col].astype(str) == sv, metric],
            errors="coerce").dropna().values
        grouped.append(vals)
    if not any(len(g) > 0 for g in grouped):
        ax.axis("off")
        return

    bp = ax.boxplot(grouped, positions=np.arange(len(ordered)),
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for j, patch in enumerate(bp["boxes"]):
        c = box_colors[j] if j < len(box_colors) else "#6BAED6"
        patch.set_facecolor(c)
        patch.set_alpha(0.60)
        patch.set_edgecolor("gray")
        patch.set_linewidth(0.8)

    rng = np.random.RandomState(42)
    for j, vals in enumerate(grouped):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        c = box_colors[j] if j < len(box_colors) else "#1F77B4"
        ax.scatter(np.full(len(vals), j) + jitter, vals, s=14,
                   color=c, edgecolors="black", linewidths=0.2,
                   alpha=0.80, zorder=5)

    title_text = _DISPLAY.get(metric, metric)
    ax.set_title(title_text, fontsize=11, pad=2, loc="left",
                 fontweight="normal")
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels([_format_sweep_label(v) for v in ordered],
                       fontsize=9, rotation=0)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.4)

    # Best marker
    medians = [np.nanmedian(g) if len(g) else np.nan for g in grouped]
    if any(pd.notna(m) for m in medians):
        hb = METRIC_DIRECTION.get(metric, True)
        best_i = int(np.nanargmax(medians) if hb else np.nanargmin(medians))
        bp["boxes"][best_i].set_edgecolor("red")
        bp["boxes"][best_i].set_linewidth(1.3)

    ymin, ymax = ax.get_ylim()
    pad = abs(ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad * 0.3, ymax + pad)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))


def generate(series, out_dir):
    """Generate refined Figure 3."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    core_metrics = _CORE_METRICS_TOPIC if series == "topic" else _CORE_METRICS_ALL
    n_cols = len(core_metrics)

    # Collect all sweeps
    all_sweeps = []
    for loader, source in [
        (load_sensitivity_csv, "sensitivity"),
        (load_training_csv, "training"),
        (load_preprocessing_csv, "preprocessing"),
    ]:
        try:
            df = loader(series)
            if "Series" in df.columns:
                df = df[df["Series"] == series].copy()
            if "Sweep" not in df.columns:
                continue
            for sweep_name in df["Sweep"].dropna().unique():
                if sweep_name in _EXCLUDED_SWEEPS:
                    continue
                sub = df[df["Sweep"] == sweep_name].copy()
                if "SweepVal" in sub.columns and len(sub) >= 2:
                    all_sweeps.append((sweep_name, sub, source))
        except Exception as e:
            print(f"    Warning: {source}: {e}")

    n_rows = len(all_sweeps)
    if n_rows == 0:
        print("    No sweep data found — skipping Fig 3")
        return

    figw = 17.0
    figh = 3.0 * n_rows + 1.0
    fig = plt.figure(figsize=(figw, figh))
    root = bind_figure_region(fig, (0.05, 0.03, 0.97, 0.96))
    row_regions = root.split_rows(n_rows, gap=0.035)

    for r_idx, (sweep_name, sub_df, source) in enumerate(all_sweeps):
        col_regions = row_regions[r_idx].split_cols(n_cols, gap=0.02)
        for c_idx, metric in enumerate(core_metrics):
            ax = col_regions[c_idx].add_axes(fig)
            style_axes(ax, kind="boxplot")
            _draw_sweep_boxplot(ax, sub_df, "SweepVal", metric)
            if c_idx == 0:
                # Add sweep name as ylabel
                ax.set_ylabel(sweep_name.replace("_", " ").title(),
                              fontsize=10, fontweight="bold")
            if r_idx == 0 and c_idx == 0:
                add_panel_label(ax, "a")

    out_path = out_dir / f"Fig3_sensitivity_{series}.png"
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
