"""Refined Figure 10 — External Model Benchmark (composed figure).

Compares internal best variant against 11 external baselines with:
  Panel (a) — Internal-only core metric boxplots (6 models)
  Panel (b) — External comparison core metric boxplots (best + 11 ext)
  Panel (c) — Efficiency comparison
  Panel (d) — Aggregate ranking bar chart

Data source: experiments/results/external/ + experiments/results/full_vs_external_all/
             benchmarks/benchmark_results/external/csv/
             benchmarks/benchmark_results/crossdata/csv/

Usage:
    python -m benchmarks.refined_figures.fig10_external --series dpmm
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
    bind_figure_region, LayoutRegion)
from benchmarks.figure_generators.common import (
    METRIC_DIRECTION, MODEL_SHORT_NAMES, clip_extreme_outliers,
    get_core_metrics, get_ext_metrics)
from benchmarks.figure_generators.significance_brackets import (
    draw_external_significance_stars)
from benchmarks.config import DEFAULT_OUTPUT_DIR as _RESULTS_DIR
from benchmarks.figure_generators.data_loaders import load_crossdata_combined

DPI = 300

EXTERNAL_MODEL_ORDER = [
    "scDHMap", "CLEAR", "scDAC", "scDeepCluster", "scGNN",
    "siVAE", "scSMD", "scDiffusion", "CellBLAST", "SCALEX", "GMVAE",
]

EXTERNAL_SHORT = {
    "CellBLAST": "CB", "GMVAE": "GM", "SCALEX": "SX",
    "scDiffusion": "sD", "siVAE": "si", "CLEAR": "CL",
    "scDAC": "DC", "scDeepCluster": "DP", "scDHMap": "DH",
    "scGNN": "GN", "scSMD": "SM", "Best-DPMM": "B-D",
    "Best-Topic": "B-T",
}

# Color palettes
EXT_COLORS_DPMM = {
    "CellBLAST": "#B2182B", "GMVAE": "#D6604D", "SCALEX": "#F4A582",
    "scDiffusion": "#FDDBC7", "siVAE": "#FEE08B", "CLEAR": "#E08214",
    "scDAC": "#D73027", "scDeepCluster": "#A50026", "scDHMap": "#8C510A",
    "scGNN": "#BF812D", "scSMD": "#DFC27D",
}
EXT_COLORS_TOPIC = {
    "CellBLAST": "#053061", "GMVAE": "#2166AC", "SCALEX": "#4393C3",
    "scDiffusion": "#92C5DE", "siVAE": "#D1E5F0", "CLEAR": "#67A9CF",
    "scDAC": "#3690C0", "scDeepCluster": "#02818A", "scDHMap": "#016C59",
    "scGNN": "#7B68AE", "scSMD": "#B2ABD2",
}
REF_COLORS = {"Best-DPMM": "#E6550D", "Best-Topic": "#756BB1"}


def _load_external():
    """Load external results CSV."""
    csv_dir = _RESULTS_DIR / "external" / "csv"
    cand = sorted(csv_dir.glob("results_combined_*.csv"),
                  key=lambda p: p.name, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No external CSV in {csv_dir}")
    expected = set(EXTERNAL_MODEL_ORDER)
    for c in cand:
        tmp = pd.read_csv(c, usecols=["Model"], nrows=200)
        if expected & set(tmp["Model"].unique()):
            return pd.read_csv(c)
    return pd.read_csv(cand[0])


def _load_internal_best(series):
    """Extract best internal variant for the series."""
    df = load_crossdata_combined(prefer_multiseed=True)
    if series == "topic":
        prior = {"Topic-Base", "Topic-Transformer", "Topic-Contrastive"}
        ref = "Best-Topic"
    else:
        prior = {"DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive"}
        ref = "Best-DPMM"

    rows = []
    for ds in df["Dataset"].unique():
        sub = df[(df["Dataset"] == ds) & (df["Model"].isin(prior))]
        if len(sub) == 0:
            continue
        mean_scores = sub.groupby("Model")[["NMI", "ARI", "ASW"]].mean()
        mean_scores["_s"] = mean_scores.mean(axis=1)
        best = mean_scores["_s"].idxmax()
        best_rows = sub[sub["Model"] == best].copy()
        num_cols = best_rows.select_dtypes(include="number").columns.tolist()
        avg = best_rows[num_cols].mean().to_frame().T
        avg["Dataset"] = ds
        avg["Model"] = ref
        rows.append(avg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _draw_ext_boxplot(ax, data_dict, model_order, metric, ext_colors,
                      ref_label):
    """Draw external comparison boxplot."""
    data_arrays = []
    labels = []
    colors = []
    for m in model_order:
        arr = data_dict.get(m, np.array([]))
        data_arrays.append(arr)
        labels.append(EXTERNAL_SHORT.get(m, m))
        if m == ref_label:
            colors.append(REF_COLORS.get(m, "#E6550D"))
        else:
            colors.append(ext_colors.get(m, "#999999"))

    data_arrays = clip_extreme_outliers(data_arrays)
    n = len(data_arrays)
    if n == 0:
        ax.axis("off")
        return

    bp = ax.boxplot(data_arrays, vert=True, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[j])
        patch.set_alpha(0.65)
        patch.set_edgecolor("gray")
        patch.set_linewidth(0.8)

    rng = np.random.RandomState(42)
    for j in range(n):
        vals = data_arrays[j]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(j + 1 + jitter, vals, s=12, c=[colors[j]],
                   edgecolors="black", linewidths=0.2, zorder=5, alpha=0.8)

    display = {"NMI": "NMI ↑", "ARI": "ARI ↑", "ASW": "ASW ↑",
               "DAV": "DAV ↓"}.get(metric, metric)
    ax.set_title(display, fontsize=11, pad=3, loc="left", fontweight="normal")
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels, fontsize=8, rotation=65, ha="right")
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.4)

    # Highlight best
    hb = METRIC_DIRECTION.get(metric, True)
    medians = [np.nanmedian(d) if len(d) else np.nan for d in data_arrays]
    if not all(np.isnan(m) for m in medians):
        best_i = int(np.nanargmax(medians) if hb else np.nanargmin(medians))
        bp["boxes"][best_i].set_edgecolor("red")
        bp["boxes"][best_i].set_linewidth(1.3)

    ymin, ymax = ax.get_ylim()
    pad = abs(ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad * 0.3, ymax + pad)


def _draw_ranking_bar(ax, ext_df, ref_df, ref_label, ext_colors, core_cols):
    """Draw aggregate ranking bar chart."""
    all_models = [ref_label] + EXTERNAL_MODEL_ORDER
    merged = pd.concat([ref_df, ext_df], ignore_index=True)
    merged = merged[merged["Model"].isin(all_models)]

    # Compute mean rank across core metrics and datasets
    mean_scores = {}
    for m in all_models:
        sub = merged[merged["Model"] == m]
        if len(sub) == 0:
            mean_scores[m] = np.nan
            continue
        vals = []
        for col in core_cols:
            if col in sub.columns:
                v = sub[col].mean()
                if pd.notna(v):
                    vals.append(v)
        mean_scores[m] = np.mean(vals) if vals else np.nan

    sorted_models = sorted(all_models,
                           key=lambda m: mean_scores.get(m, -1),
                           reverse=True)
    scores = [mean_scores.get(m, 0) for m in sorted_models]
    colors = []
    for m in sorted_models:
        if m == ref_label:
            colors.append(REF_COLORS.get(m, "#E6550D"))
        else:
            colors.append(ext_colors.get(m, "#999999"))
    labels = [EXTERNAL_SHORT.get(m, m) for m in sorted_models]

    bars = ax.barh(range(len(sorted_models)), scores, color=colors,
                   edgecolor="gray", linewidth=0.5, alpha=0.75)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Core Score", fontsize=10)
    ax.set_title("Aggregate Ranking", fontsize=11, pad=3, loc="left",
                 fontweight="normal")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2, lw=0.4)


def generate(series, out_dir):
    """Generate refined Figure 10."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    ext_df = _load_external()
    ref_df = _load_internal_best(series)
    ref_label = "Best-Topic" if series == "topic" else "Best-DPMM"
    ext_colors = EXT_COLORS_TOPIC if series == "topic" else EXT_COLORS_DPMM

    core_metrics = get_core_metrics(series)
    core_cols = [c for c, _, _ in core_metrics]
    model_order = [ref_label] + EXTERNAL_MODEL_ORDER
    n_metrics = len(core_metrics)

    # Build per-model data arrays
    merged = pd.concat([ref_df, ext_df], ignore_index=True)

    fig = plt.figure(figsize=(17.0, 14.0))
    root = bind_figure_region(fig, (0.05, 0.04, 0.96, 0.96))

    # 2 rows: top = metric boxplots, bottom = ranking bar
    top, bottom = root.split_rows([3.0, 1.5], gap=0.05)

    # Top: core metric boxplots
    metric_cols = top.split_cols(n_metrics, gap=0.02)
    for idx, (col, label, higher) in enumerate(core_metrics):
        ax = metric_cols[idx].add_axes(fig)
        style_axes(ax, kind="boxplot")
        data_dict = {}
        for m in model_order:
            sub = merged[merged["Model"] == m]
            if col in sub.columns:
                data_dict[m] = sub[col].dropna().values
            else:
                data_dict[m] = np.array([])
        _draw_ext_boxplot(ax, data_dict, model_order, col, ext_colors,
                          ref_label)
        if idx == 0:
            add_panel_label(ax, "a")

    # Bottom: ranking bar
    ax_rank = bottom.add_axes(fig)
    style_axes(ax_rank, kind="bar")
    _draw_ranking_bar(ax_rank, ext_df, ref_df, ref_label, ext_colors,
                      core_cols)
    add_panel_label(ax_rank, "b")

    out_path = out_dir / f"Fig10_external_{series}.png"
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
