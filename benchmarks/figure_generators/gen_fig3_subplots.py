"""Generate individual subplot PNGs for Figure 3 (Sensitivity Analysis).

Produces sweep boxplot trend subplots for each hyperparameter parameter:
  For each sweep parameter, 6 core metric boxplots.

Output: benchmarks/paper_figures/{series}/subplots/fig3/

Usage:
    python -m benchmarks.figure_generators.gen_fig3_subplots --series dpmm
"""

import argparse
import sys
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

from benchmarks.figure_generators.subplot_style import (
    apply_subplot_style, save_subplot, build_manifest,
    FIGSIZE_BOXPLOT, SUBPLOT_DPI,
    SCATTER_SIZE_SWEEP, LINE_WIDTH_MEDIAN, LINE_WIDTH_BOX,
    FONTSIZE_TITLE, FONTSIZE_TICK, FONTSIZE_LABEL)
from benchmarks.figure_generators.common import (
    METRIC_DIRECTION
)
from benchmarks.figure_generators.data_loaders import (
    load_sensitivity_csv, load_training_csv, load_preprocessing_csv,
    parse_sweep_value)


_CORE_METRICS_ALL = ["NMI", "ARI", "ASW", "DAV",
                     "DRE_umap_overall_quality", "LSE_overall_quality"]
# Topic models use a simplex latent space — LSE/DRE are not meaningful.
_CORE_METRICS_TOPIC = ["NMI", "ARI", "ASW", "DAV"]

_DISPLAY = {
    "NMI": "NMI \u2191", "ARI": "ARI \u2191", "ASW": "ASW \u2191",
    "DAV": "DAV \u2193", "DRE_umap_overall_quality": "DRE UMAP \u2191",
    "LSE_overall_quality": "LSE Overall \u2191",
}

# Sweep parameters to exclude from figures (no meaningful variation)
_EXCLUDED_SWEEPS = {"max_cells"}
_SWEEP_PALETTE = [
    "#4C78A8", "#6BAED6", "#9ECAE1", "#FDD49E", "#FDAE6B",
    "#FD8D3C", "#E6550D", "#A63603",
]


def _format_sweep_label(val_str):
    """Format a sweep value for x-axis display.

    Uses scientific notation (e.g., 1e-3, 1e3) when values are very
    small (< 0.01) or very large (>= 1000) for readability.  Mid-range
    values keep their compact decimal form.
    """
    try:
        val = float(val_str)
        if val == 0:
            return "0"
        abs_val = abs(val)
        # Use scientific notation for very small or very large values
        if abs_val >= 1000 or (0 < abs_val < 0.01):
            return f"{val:.0e}"
        if val == int(val):
            return str(int(val))
        # Strip trailing zeros for compact display
        s = f"{val:g}"
        return s
    except (ValueError, TypeError):
        return str(val_str)


def gen_sweep_boxplot(df, param_col, metric, out_path, dataset_col="Dataset"):
    """Generate one sweep-trend boxplot subplot PNG."""
    if metric not in df.columns:
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
        work.sort_values("_sort")[param_col].astype(str).tolist()
    ))
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
        return

    fig = plt.figure(figsize=FIGSIZE_BOXPLOT)

    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))

    ax = layout.add_axes(fig)

    style_axes(ax)
    bp = ax.boxplot(grouped, positions=np.arange(len(ordered)),
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=LINE_WIDTH_MEDIAN))
    for j, patch in enumerate(bp["boxes"]):
        c = box_colors[j] if j < len(box_colors) else "#6BAED6"
        patch.set_facecolor(c)
        patch.set_alpha(0.60)
        patch.set_edgecolor("gray")
        patch.set_linewidth(LINE_WIDTH_BOX)

    rng = np.random.RandomState(42)
    for j, vals in enumerate(grouped):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        c = box_colors[j] if j < len(box_colors) else "#1F77B4"
        ax.scatter(np.full(len(vals), j) + jitter, vals, s=SCATTER_SIZE_SWEEP,
                   color=c, edgecolors="black",
                   linewidths=0.2, alpha=0.80, zorder=5)

    title_text = _DISPLAY.get(metric, metric)
    ds_n = work[dataset_col].nunique() if dataset_col in work.columns else None
    if ds_n and ds_n > 1:
        title_text = f"{title_text} (n={ds_n})"
    ax.set_title(title_text, fontsize=FONTSIZE_TITLE, pad=2,
                 loc="left", fontweight="normal")
    ax.set_xticks(np.arange(len(ordered)))
    display_labels = [_format_sweep_label(v) for v in ordered]
    ax.set_xticklabels(display_labels, fontsize=FONTSIZE_TICK, rotation=0)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(axis="y", alpha=0.2, lw=0.4)

    medians = [np.nanmedian(g) if len(g) else np.nan for g in grouped]
    if any(pd.notna(m) for m in medians):
        hb = METRIC_DIRECTION.get(metric, True)
        best_i = int(np.nanargmax(medians) if hb else np.nanargmin(medians))
        bp["boxes"][best_i].set_edgecolor("red")
        bp["boxes"][best_i].set_linewidth(1.3)
        ax.scatter([best_i], [medians[best_i]], s=40, facecolors="none",
                   edgecolors="#D62728", linewidths=1.2, zorder=6)

    # y-axis padding + prune edge ticks to prevent VCD truncation warnings
    ymin, ymax = ax.get_ylim()
    y_range = abs(ymax - ymin)
    pad_y = y_range * 0.08
    if ymax > ymin:
        ax.set_ylim(ymin - pad_y * 0.3, ymax + pad_y)
    else:
        ax.set_ylim(ymin - pad_y, ymax + pad_y * 0.3)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))

    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def generate(out_dir):
    """Generate all subplot PNGs for Figure 3."""
    series = "topic"
    print(f"\n  Figure 3 subplots ({series})")
    sub_dir = out_dir / "fig3"
    sub_dir.mkdir(parents=True, exist_ok=True)
    apply_subplot_style()

    # Select metric set based on series
    core_metrics = _CORE_METRICS_TOPIC if series == "topic" else _CORE_METRICS_ALL

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
                    print(f"    Skipping excluded sweep: {sweep_name}")
                    continue
                sub = df[df["Sweep"] == sweep_name].copy()
                if "SweepVal" in sub.columns and len(sub) >= 2:
                    all_sweeps.append((sweep_name, sub, source))
        except Exception as e:
            print(f"    Warning: {source}: {e}")

    sweep_manifest = {}
    for sweep_name, sub_df, source in all_sweeps:
        safe_name = sweep_name.replace("/", "_").replace(" ", "_")
        sweep_plots = []
        for metric in core_metrics:
            fname = f"{safe_name}_{metric}.png"
            gen_sweep_boxplot(sub_df, "SweepVal", metric, sub_dir / fname)
            if (sub_dir / fname).exists():
                sweep_plots.append(fname)
        if sweep_plots:
            sweep_manifest[sweep_name] = {
                "source": source,
                "plots": sweep_plots,
            }

    manifest = build_manifest(sub_dir, {
        "sweeps": sweep_manifest,
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 3 subplots")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(out)
