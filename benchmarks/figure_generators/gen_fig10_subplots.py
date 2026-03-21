"""Generate individual subplot PNGs for Figure 10 (External Model Benchmark).

Compares 11 external baseline models against the best internal variant
for **one** series at a time. Internal vs external are fully separated:
no single overcrowded panel mixing all methods.

- ``--series dpmm``  →  internal reference = Best-DPMM  (warm palette)
- ``--series topic`` →  internal reference = Best-Topic (cool palette)

The two series are never mixed in the same figure.

Produces:
  Panel A — Workflow diagram (selection → benchmark → comparison)
  Panel B (internal) — Core metric boxplots, 6 internal models only (3 prior + 3 pure)
  Panel B (external) — Core metric boxplots: 1 best internal + 11 external
  Panel C — Extended metric boxplots (DPMM only; same suite as Figure 2 Panel D)
  Panel D (internal) — Efficiency boxplots, 6 internal models only
  Panel D (external) — Efficiency boxplots: 1 best internal + 11 external
  Panel E — Aggregate ranking bar chart (external comparison)

External models:
  CellBLAST, GMVAE, SCALEX, scDiffusion, siVAE, CLEAR,
  scDAC, scDeepCluster, scDHMap, scGNN, scSMD

Internal reference (exactly one per run):
  - Best DPMM prior model  (when --series dpmm)
  - Best Topic prior model (when --series topic)

Output: benchmarks/paper_figures/{series}/subplots/fig10/

Usage:
    python -m benchmarks.figure_generators.gen_fig10_subplots --series dpmm
    python -m benchmarks.figure_generators.gen_fig10_subplots --series topic
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

from benchmarks.config import DEFAULT_OUTPUT_DIR as _RESULTS_DIR
from benchmarks.figure_generators.subplot_style import (
    apply_subplot_style, save_subplot, build_manifest,
    SUBPLOT_DPI, FIGSIZE_BOXPLOT, FIGSIZE_4COL,
    SCATTER_SIZE_BOXPLOT, LINE_WIDTH_MEDIAN, LINE_WIDTH_BOX,
    FONTSIZE_TITLE, FONTSIZE_TICK, FONTSIZE_LABEL, FONTSIZE_LEGEND,
    CONTAINER_CSS_PX, DPR, _col_figsize)
from benchmarks.figure_generators.common import (
    METRIC_DIRECTION, MODEL_SHORT_NAMES,
    ALL_BOXPLOT_METRICS_CORE, ALL_BOXPLOT_METRICS_EXT,
    get_core_metrics, get_ext_metrics, clip_extreme_outliers)
from benchmarks.figure_generators.significance_brackets import (
    draw_external_significance_stars)
from benchmarks.figure_generators.gen_workflow import gen_workflow_png


# ═══════════════════════════════════════════════════════════════════════════════
# External model color palettes — separate schemes for DPMM vs Topic series.
# ═══════════════════════════════════════════════════════════════════════════════

# DPMM series: warm red / orange / earth tones
EXTERNAL_MODEL_COLORS_DPMM = {
    "CellBLAST":     "#B2182B",   # dark red
    "GMVAE":         "#D6604D",   # salmon red
    "SCALEX":        "#F4A582",   # peach
    "scDiffusion":   "#FDDBC7",   # light peach
    "siVAE":         "#FEE08B",   # light gold
    "CLEAR":         "#E08214",   # amber
    "scDAC":         "#D73027",   # bright red
    "scDeepCluster": "#A50026",   # deep red
    "scDHMap":       "#8C510A",   # brown
    "scGNN":         "#BF812D",   # tan
    "scSMD":         "#DFC27D",   # sand
}

# Topic series: cool blue / teal / purple tones
EXTERNAL_MODEL_COLORS_TOPIC = {
    "CellBLAST":     "#053061",   # navy
    "GMVAE":         "#2166AC",   # deep blue
    "SCALEX":        "#4393C3",   # medium blue
    "scDiffusion":   "#92C5DE",   # sky blue
    "siVAE":         "#D1E5F0",   # ice blue
    "CLEAR":         "#67A9CF",   # steel blue
    "scDAC":         "#3690C0",   # ocean blue
    "scDeepCluster": "#02818A",   # teal
    "scDHMap":       "#016C59",   # dark teal
    "scGNN":         "#7B68AE",   # muted purple
    "scSMD":         "#B2ABD2",   # light purple
}

# Reference model colors (internal best) — only the relevant one is used
REFERENCE_COLORS = {
    "Best-DPMM":  "#E6550D",     # warm orange (DPMM family)
    "Best-Topic": "#756BB1",     # purple (Topic family)
}

# Internal-only comparison: all 6 models per series (3 prior + 3 pure)
INTERNAL_MODEL_ORDER_DPMM = [
    "DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
    "Pure-AE", "Pure-Transformer-AE", "Pure-Contrastive-AE",
]
INTERNAL_MODEL_ORDER_TOPIC = [
    "Topic-Base", "Topic-Transformer", "Topic-Contrastive",
    "Pure-VAE", "Pure-Transformer-VAE", "Pure-Contrastive-VAE",
]
INTERNAL_SHORT_NAMES = {
    "DPMM-Base": "D-B", "DPMM-Transformer": "D-T", "DPMM-Contrastive": "D-C",
    "Pure-AE": "P-AE", "Pure-Transformer-AE": "P-TAE", "Pure-Contrastive-AE": "P-CAE",
    "Topic-Base": "T-B", "Topic-Transformer": "T-T", "Topic-Contrastive": "T-C",
    "Pure-VAE": "P-V", "Pure-Transformer-VAE": "P-TV", "Pure-Contrastive-VAE": "P-CV",
}
INTERNAL_COLORS = {
    "DPMM-Base": "#E6550D", "DPMM-Transformer": "#E6550D", "DPMM-Contrastive": "#E6550D",
    "Pure-AE": "#9ECAE1", "Pure-Transformer-AE": "#9ECAE1", "Pure-Contrastive-AE": "#9ECAE1",
    "Topic-Base": "#756BB1", "Topic-Transformer": "#756BB1", "Topic-Contrastive": "#756BB1",
    "Pure-VAE": "#C7E9C0", "Pure-Transformer-VAE": "#C7E9C0", "Pure-Contrastive-VAE": "#C7E9C0",
}

# Active palette holder — set by generate() based on --series
_active_ext_colors = EXTERNAL_MODEL_COLORS_DPMM  # default

EXTERNAL_MODEL_ORDER = [
    "scDHMap", "CLEAR", "scDAC", "scDeepCluster", "scGNN",
    "siVAE", "scSMD", "scDiffusion", "CellBLAST", "SCALEX", "GMVAE",
]

EXTERNAL_SHORT_NAMES = {
    "CellBLAST":     "CB",
    "GMVAE":         "GM",
    "SCALEX":        "SX",
    "scDiffusion":   "sD",
    "siVAE":         "si",
    "CLEAR":         "CL",
    "scDAC":         "DC",
    "scDeepCluster": "DP",
    "scDHMap":       "DH",
    "scGNN":         "GN",
    "scSMD":         "SM",
    "Best-DPMM":     "B-D",
    "Best-Topic":    "B-T",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_external_results():
    """Load the combined external model benchmark CSV.

    Picks the CSV that contains the expected 11 external models rather
    than blindly selecting the latest file by timestamp.
    """
    csv_dir = _RESULTS_DIR / "external" / "csv"
    cand = sorted(csv_dir.glob("results_combined_*.csv"),
                  key=lambda p: p.name, reverse=True)
    if not cand:
        raise FileNotFoundError(
            f"No external benchmark CSV in {csv_dir}. "
            f"Run benchmarks/runners/benchmark_external.py first.")
    # Prefer the CSV that has the 11 expected external models
    expected = set(EXTERNAL_MODEL_ORDER)
    for c in cand:
        tmp = pd.read_csv(c, usecols=["Model"], nrows=200)
        if expected & set(tmp["Model"].unique()):
            return pd.read_csv(c)
    # Fallback to latest
    return pd.read_csv(cand[0])


def _load_internal_best(series="dpmm"):
    """Load crossdata results and extract the best variant for *one* series.

    When the 5-seed CSV is available, returns one row per (dataset, seed)
    pair so that boxplots show seed-level variance (up to 60 points for
    12 datasets × 5 seeds).

    Parameters
    ----------
    series : {"dpmm", "topic"}
        Which prior family to extract.  ``"dpmm"`` returns rows labelled
        ``"Best-DPMM"``; ``"topic"`` returns ``"Best-Topic"``.

    Returns
    -------
    pd.DataFrame
        Rows with ``Model`` = ``"Best-DPMM"`` or ``"Best-Topic"``,
        preserving seed-level granularity when available.
    """
    from benchmarks.figure_generators.data_loaders import load_crossdata_combined

    df = load_crossdata_combined(prefer_multiseed=True)

    if series == "topic":
        prior_models = {"Topic-Base", "Topic-Transformer", "Topic-Contrastive"}
        ref_label = "Best-Topic"
    else:
        prior_models = {"DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive"}
        ref_label = "Best-DPMM"

    has_seed = "seed" in df.columns

    rows = []
    for ds in df["Dataset"].unique():
        ds_df = df[df["Dataset"] == ds].copy()

        if has_seed:
            # Multi-seed: pick best variant per dataset (by mean across seeds),
            # then average seed values to get ONE row per dataset
            sub = ds_df[ds_df["Model"].isin(prior_models)].copy()
            if len(sub) == 0:
                continue
            mean_scores = sub.groupby("Model")[["NMI", "ARI", "ASW"]].mean()
            mean_scores["_score"] = mean_scores.mean(axis=1)
            best_model = mean_scores["_score"].idxmax()
            best_sub = sub[sub["Model"] == best_model].copy()
            # Average across seeds to get one row per dataset
            num_cols = best_sub.select_dtypes(include="number").columns.tolist()
            avg_row = best_sub[num_cols].mean().to_frame().T
            avg_row["Dataset"] = ds
            avg_row["OrigModel"] = best_model
            avg_row["Model"] = ref_label
            rows.append(avg_row)
        else:
            # Single-seed fallback: 1 row per dataset
            ds_df["_score"] = ds_df[["NMI", "ARI", "ASW"]].mean(axis=1)
            sub = ds_df[ds_df["Model"].isin(prior_models)]
            if len(sub):
                best = sub.loc[sub["_score"].idxmax()]
                row = best.to_dict()
                row["OrigModel"] = row["Model"]
                row["Model"] = ref_label
                rows.append(pd.DataFrame([row]))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_internal_all(series="dpmm"):
    """Load crossdata results for all 6 internal models (3 prior + 3 pure) for one series.

    Used for internal-only comparison panels (no external models).
    Returns one row per (Dataset, Model) with seed-averaged values when multi-seed.
    """
    from benchmarks.figure_generators.data_loaders import load_crossdata_combined

    df = load_crossdata_combined(prefer_multiseed=True)
    if series == "topic":
        models = set(INTERNAL_MODEL_ORDER_TOPIC)
    else:
        models = set(INTERNAL_MODEL_ORDER_DPMM)

    sub = df[df["Model"].isin(models)].copy()
    if sub.empty:
        return pd.DataFrame()

    if "seed" in sub.columns:
        # Average across seeds per (Dataset, Model)
        num_cols = sub.select_dtypes(include="number").columns.difference(["seed"]).tolist()
        group_cols = ["Dataset", "Model"]
        avg = sub.groupby(group_cols)[num_cols].mean().reset_index()
        return avg
    return sub


def _build_internal_matrix(int_df, metric_col, model_order):
    """Build per-model data arrays for internal-only (6 models)."""
    data_per_model = []
    for model in model_order:
        sub = int_df[int_df["Model"] == model]
        if metric_col in sub.columns and "Dataset" in sub.columns:
            vals = sub.groupby("Dataset")[metric_col].mean().dropna().values
        elif metric_col in sub.columns:
            vals = sub[metric_col].dropna().values
        else:
            vals = np.array([])
        data_per_model.append(vals)
    return data_per_model, model_order


def _build_combined_matrix(ext_df, int_df, metric_col, ref_model="Best-DPMM"):
    """Build per-model data arrays combining external + internal best.

    Uses seed-averaged values: one data point per (Model, Dataset).
    Multi-seed variance is shown only in Figure 5.

    Returns
    -------
    data_per_model : list[np.ndarray]
        One array per model (one value per dataset).
    all_models : list[str]
        Model names in display order.
    datasets : list[str]
        Sorted dataset names.
    """
    all_models = [ref_model] + EXTERNAL_MODEL_ORDER
    combined = pd.concat([int_df, ext_df], ignore_index=True)
    datasets = sorted(combined["Dataset"].unique()) if "Dataset" in combined.columns else []

    data_per_model = []
    for model in all_models:
        sub = combined[combined["Model"] == model]
        if metric_col in sub.columns and "Dataset" in sub.columns:
            # Average across seeds per dataset
            avg = sub.groupby("Dataset")[metric_col].mean()
            vals = avg.dropna().values
        elif metric_col in sub.columns:
            vals = sub[metric_col].dropna().values
        else:
            vals = np.array([])
        data_per_model.append(vals)

    return data_per_model, all_models, datasets


# ═══════════════════════════════════════════════════════════════════════════════
# Subplot generators
# ═══════════════════════════════════════════════════════════════════════════════

def _get_color(model_name):
    """Get color for a model name (external or reference)."""
    global _active_ext_colors
    if model_name in REFERENCE_COLORS:
        return REFERENCE_COLORS[model_name]
    return _active_ext_colors.get(model_name, "#999999")


def gen_ext_boxplot(ext_df, int_df, metric_col, metric_label, out_path,
                    ref_model="Best-DPMM"):
    """Generate one grouped boxplot: external models + best DPMM or Topic.

    Internal reference model now shows seed-level variance (up to 60
    data points from 5 seeds × 12 datasets) when the 5-seed CSV is
    available.
    """
    data_per_model, all_models, datasets = _build_combined_matrix(
        ext_df, int_df, metric_col, ref_model=ref_model)
    # Clip extreme outliers (e.g. CAL artefacts) to prevent y-axis distortion
    data_per_model = clip_extreme_outliers(data_per_model)
    n_models = len(all_models)
    short = [EXTERNAL_SHORT_NAMES.get(m, m) for m in all_models]

    higher_better = METRIC_DIRECTION.get(metric_col, True)

    # Wider figure to fit 13 models; extra height for rotated labels
    fig_w = max(FIGSIZE_BOXPLOT[0], 4.5)
    fig_h = FIGSIZE_BOXPLOT[1] + 0.30
    fig = plt.figure(figsize=(fig_w, fig_h))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)

    bp = ax.boxplot(data_per_model, vert=True, patch_artist=True,
                    widths=0.50, showfliers=False,
                    medianprops=dict(color="black", lw=LINE_WIDTH_MEDIAN))

    for j, patch in enumerate(bp["boxes"]):
        c = _get_color(all_models[j])
        patch.set_facecolor(c)
        patch.set_alpha(0.70)
        patch.set_edgecolor("gray")
        patch.set_linewidth(LINE_WIDTH_BOX)

    rng = np.random.RandomState(42)
    for j in range(n_models):
        vals = data_per_model[j]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(j + 1 + jitter, vals, s=SCATTER_SIZE_BOXPLOT * 0.8,
                   c=[_get_color(all_models[j])],
                   edgecolors="black", linewidths=0.15, zorder=5, alpha=0.80)

    # Highlight best-performing model with red border
    medians = [np.nanmedian(d) if len(d) else np.nan for d in data_per_model]
    if any(pd.notna(m) for m in medians):
        best_idx = int(np.nanargmax(medians) if higher_better
                       else np.nanargmin(medians))
        bp["boxes"][best_idx].set_edgecolor("red")
        bp["boxes"][best_idx].set_linewidth(1.3)

    # Visual separator between internal ref and external models
    ax.axvline(x=1.5, color="#CCCCCC", ls="--", lw=0.6, zorder=0)

    ax.set_xticks(range(1, n_models + 1))
    tick_fs = FONTSIZE_TICK                    # compact tick labels
    ax.set_xticklabels(short, fontsize=tick_fs, rotation=90, ha="center")
    ax.set_title(metric_label, fontsize=FONTSIZE_TITLE, pad=3,
                 loc="left", fontweight="normal")
    ax.tick_params(labelsize=tick_fs)
    ax.grid(axis="y", alpha=0.2, lw=0.4)
    fig.subplots_adjust(bottom=0.40)          # room for rotated labels

    # Significance stars (external models vs internal reference)
    series = "dpmm" if ref_model == "Best-DPMM" else "topic"
    draw_external_significance_stars(ax, all_models, metric_col, series,
                                     data_per_model=data_per_model)

    # y-axis padding + prune upper tick to prevent VCD truncation warnings
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


def gen_internal_boxplot(int_df, metric_col, metric_label, out_path, series="dpmm"):
    """Generate one boxplot for internal comparison only (6 models: 3 prior + 3 pure).

    No external models — readable internal ablation view per series.
    """
    model_order = INTERNAL_MODEL_ORDER_TOPIC if series == "topic" else INTERNAL_MODEL_ORDER_DPMM
    data_per_model, all_models = _build_internal_matrix(int_df, metric_col, model_order)
    data_per_model = clip_extreme_outliers(data_per_model)
    n_models = len(all_models)
    short = [INTERNAL_SHORT_NAMES.get(m, m[:6]) for m in all_models]
    higher_better = METRIC_DIRECTION.get(metric_col, True)

    fig_w = max(FIGSIZE_BOXPLOT[0] * 0.85, 3.2)
    fig_h = FIGSIZE_BOXPLOT[1]
    fig = plt.figure(figsize=(fig_w, fig_h))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)

    bp = ax.boxplot(data_per_model, vert=True, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=LINE_WIDTH_MEDIAN))
    for j, patch in enumerate(bp["boxes"]):
        c = INTERNAL_COLORS.get(all_models[j], "#999999")
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("gray")
        patch.set_linewidth(LINE_WIDTH_BOX)

    rng = np.random.RandomState(42)
    for j in range(n_models):
        vals = data_per_model[j]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(j + 1 + jitter, vals, s=SCATTER_SIZE_BOXPLOT * 0.8,
                   c=[INTERNAL_COLORS.get(all_models[j], "#999999")],
                   edgecolors="black", linewidths=0.15, zorder=5, alpha=0.80)

    medians = [np.nanmedian(d) if len(d) else np.nan for d in data_per_model]
    if any(pd.notna(m) for m in medians):
        best_idx = int(np.nanargmax(medians) if higher_better else np.nanargmin(medians))
        bp["boxes"][best_idx].set_edgecolor("red")
        bp["boxes"][best_idx].set_linewidth(1.2)

    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels(short, fontsize=FONTSIZE_TICK, rotation=45, ha="right")
    ax.set_title(metric_label, fontsize=FONTSIZE_TITLE, pad=2, loc="left", fontweight="normal")
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(axis="y", alpha=0.2, lw=0.4)
    fig.subplots_adjust(bottom=0.22)

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


def gen_ranking_bar(ext_df, int_df, out_path, ref_model="Best-DPMM"):
    """Generate aggregate ranking bar chart (5-metric composite score).

    Uses min-max normalisation per metric across all models so that each
    metric contributes equally.  DAV is inverted (lower-is-better) before
    normalisation so that a higher normalised value is always better.
    """
    # 4-metric composite: NMI, ARI, ASW, DAV (min-max normalised)
    RANKING_METRICS = [
        ("NMI", True),   # higher-is-better
        ("ARI", True),
        ("ASW", True),
        ("DAV", False),  # Davies-Bouldin: lower-is-better → inverted
    ]
    all_models_order = [ref_model] + EXTERNAL_MODEL_ORDER

    combined = pd.concat([int_df, ext_df], ignore_index=True)

    # Per-model mean for each metric (handles multi-seed by averaging)
    avail = [m for m, _ in RANKING_METRICS if m in combined.columns]
    model_means = combined.groupby("Model")[avail].mean()

    # Min-max normalise; invert DAV so higher = better for all
    normed = pd.DataFrame(index=model_means.index)
    for col, hb in RANKING_METRICS:
        if col not in model_means.columns:
            continue
        vals = model_means[col].copy()
        if not hb:  # invert lower-is-better
            vals = -vals
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-12:
            normed[col] = (vals - vmin) / (vmax - vmin)
        else:
            normed[col] = 0.5

    normed["_composite"] = normed.mean(axis=1)

    scores = []
    for m in all_models_order:
        scores.append(normed.loc[m, "_composite"] if m in normed.index
                      else np.nan)

    short = [EXTERNAL_SHORT_NAMES.get(m, m) for m in all_models_order]
    colors = [_get_color(m) for m in all_models_order]

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    sorted_short = [short[i] for i in order]
    sorted_scores = [scores[i] for i in order]
    sorted_colors = [colors[i] for i in order]
    sorted_models = [all_models_order[i] for i in order]

    n_bars = len(sorted_short)
    fig_w = CONTAINER_CSS_PX * DPR / SUBPLOT_DPI
    fig_h = max(0.30 * n_bars, 2.5)          # taller bars for readability

    _TICK_SM = max(FONTSIZE_TICK, 7.5)        # slightly larger ticks
    _LABEL_SM = max(FONTSIZE_LABEL, 8.0)
    _TITLE_SM = max(FONTSIZE_TITLE, 8.5)

    fig = plt.figure(figsize=(fig_w, fig_h))

    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))

    ax = layout.add_axes(fig)

    style_axes(ax)

    bars = ax.barh(range(n_bars), sorted_scores,
                   color=sorted_colors, edgecolor="gray", linewidth=0.4,
                   alpha=0.80, height=0.55)

    # Highlight internal references
    for i, m in enumerate(sorted_models):
        if m in REFERENCE_COLORS:
            bars[i].set_edgecolor("#333333")
            bars[i].set_linewidth(1.0)
            ax.text(sorted_scores[i] + 0.005, i, "\u2605",
                    va="center", ha="left", fontsize=_TICK_SM,
                    color="#333333")

    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(sorted_short, fontsize=_TICK_SM)
    metric_label = "NMI+ARI+ASW+\u0394DAV"
    ax.set_xlabel(f"Composite Score ({metric_label})",
                  fontsize=_LABEL_SM)
    ax.set_title("Aggregate Ranking (4 metrics, norm.)",
                 fontsize=_TITLE_SM, pad=2,
                 loc="left", fontweight="normal")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    import matplotlib.ticker as mtk
    ax.xaxis.set_major_locator(mtk.MaxNLocator(nbins='auto', prune='both'))
    ax.tick_params(axis="x", labelsize=_TICK_SM)
    ax.grid(axis="x", alpha=0.2, lw=0.4)
    fig.subplots_adjust(left=0.16, bottom=0.14)

    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_legend_strip(out_path, ref_model="Best-DPMM"):
    """Generate a standalone legend strip for external model colors.

    Groups entries into 'Internal' (reference model) and 'External
    baselines' sections with enlarged markers for print readability.
    """
    global _active_ext_colors
    # Internal reference — only the relevant one
    ref_items = [(ref_model, REFERENCE_COLORS[ref_model])]
    # External models
    ext_items = [(m, _active_ext_colors[m]) for m in EXTERNAL_MODEL_ORDER]

    fig_w = CONTAINER_CSS_PX * DPR / SUBPLOT_DPI * 1.3  # wider for 14 entries
    fig = plt.figure(figsize=(fig_w, 0.85))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Build handles with group separators
    handles = []
    labels_list = []

    # Group label: Internal
    handles.append(plt.Line2D([0], [0], ls="", marker="", alpha=0))
    labels_list.append("Internal:")
    for name, c in ref_items:
        handles.append(
            plt.Line2D([0], [0], marker="s", ls="", color=c, markersize=7,
                       markeredgecolor="gray", markeredgewidth=0.3, alpha=0.80))
        labels_list.append(name)

    # Group label: External baselines
    handles.append(plt.Line2D([0], [0], ls="", marker="", alpha=0))
    labels_list.append("External:")
    for name, c in ext_items:
        handles.append(
            plt.Line2D([0], [0], marker="s", ls="", color=c, markersize=7,
                       markeredgecolor="gray", markeredgewidth=0.3, alpha=0.80))
        labels_list.append(name)

    _leg_fs = max(FONTSIZE_LEGEND, 12)
    ax.legend(handles, labels_list, loc="center",
              ncol=min(len(handles), 7),
              fontsize=_leg_fs, frameon=False,
              handletextpad=0.2, columnspacing=0.4, markerscale=1.0)

    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Workflow steps for Figure 10
# ═══════════════════════════════════════════════════════════════════════════════

def _workflow_steps(series):
    """Return series-specific workflow steps for Panel A."""
    if series == "topic":
        ref_label = "Best-Topic"
        ref_desc  = "Topic series"
    else:
        ref_label = "Best-DPMM"
        ref_desc  = "DPMM series"
    return [
        {"label": "11 External Models",
         "sub": "CellBLAST · GMVAE · …",
         "icon": "train"},
        {"label": "12 scRNA-seq",
         "sub": "datasets",
         "icon": "cells"},
        {"label": "Unified Eval",
         "sub": "NMI · ARI · ASW · DAV",
         "icon": "metrics"},
        {"label": f"vs {ref_label}",
         "sub": ref_desc,
         "icon": "compare"},
        {"label": "Benchmark Report",
         "sub": "ranking · boxplots",
         "icon": "evaluate"},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

CORE_METRICS_FIG10 = [
    ("NMI", "NMI ↑"),
    ("ARI", "ARI ↑"),
    ("ASW", "ASW ↑"),
    ("DAV", "DAV ↓"),
    ("DRE_umap_overall_quality", "DRE UMAP ↑"),
    ("LSE_overall_quality", "LSE Overall ↑"),
]

# Efficiency metrics (same as Figure 2 Panel E)
EFFICIENCY_METRICS_FIG10 = [
    ("SecPerEpoch", "Sec/Epoch ↓"),
    ("PeakGPU_MB", "Peak GPU (MB) ↓"),
    ("NumParams", "Num Params ↓"),
]


def gen_ext_boxplot_compact(ext_df, int_df, metric_col, metric_label,
                            out_path, higher_better=True, ref_model="Best-DPMM"):
    """Generate a compact boxplot for extended metrics (8-col grid style).

    Similar to gen_ext_boxplot but uses a smaller figsize matching the
    8-column grid layout used in Figure 2 Panel D.  Only includes the
    internal reference model matching the active series.
    """
    data_per_model, all_models, datasets = _build_combined_matrix(
        ext_df, int_df, metric_col, ref_model=ref_model)
    # Clip extreme outliers to prevent y-axis distortion
    data_per_model = clip_extreme_outliers(data_per_model)
    n_models = len(all_models)
    # Even shorter names for the compact 8-col grid
    _COMPACT_NAMES = {
        "CellBLAST": "CB", "GMVAE": "GM", "SCALEX": "SX",
        "scDiffusion": "sD", "siVAE": "si", "CLEAR": "CL",
        "scDAC": "DC", "scDeepCluster": "DP", "scDHMap": "DH",
        "scGNN": "GN", "scSMD": "SM",
        "Best-DPMM": "B-D", "Best-Topic": "B-T",
    }
    short = [_COMPACT_NAMES.get(m, m[:4]) for m in all_models]

    fig_w = max(FIGSIZE_4COL[0], 3.0)
    fig_h = FIGSIZE_4COL[1] + 0.10
    fig = plt.figure(figsize=(fig_w, fig_h))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)

    bp = ax.boxplot(data_per_model, vert=True, patch_artist=True,
                    widths=0.50, showfliers=False,
                    medianprops=dict(color="black", lw=LINE_WIDTH_MEDIAN))

    for j, patch in enumerate(bp["boxes"]):
        c = _get_color(all_models[j])
        patch.set_facecolor(c)
        patch.set_alpha(0.70)
        patch.set_edgecolor("gray")
        patch.set_linewidth(LINE_WIDTH_BOX)

    rng = np.random.RandomState(42)
    for j in range(n_models):
        vals = data_per_model[j]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(j + 1 + jitter, vals, s=SCATTER_SIZE_BOXPLOT * 0.5,
                   c=[_get_color(all_models[j])],
                   edgecolors="black", linewidths=0.10, zorder=5, alpha=0.75)

    # Highlight best
    medians = [np.nanmedian(d) if len(d) else np.nan for d in data_per_model]
    if any(pd.notna(m) for m in medians):
        best_idx = int(np.nanargmax(medians) if higher_better
                       else np.nanargmin(medians))
        bp["boxes"][best_idx].set_edgecolor("red")
        bp["boxes"][best_idx].set_linewidth(1.2)

    # Separator — 1 internal ref
    ax.axvline(x=1.5, color="#CCCCCC", ls="--", lw=0.5, zorder=0)

    ax.set_xticks(range(1, n_models + 1))
    tick_fs = FONTSIZE_TICK
    ax.set_xticklabels(short, fontsize=tick_fs, rotation=90, ha="center")
    ax.set_title(metric_label, fontsize=FONTSIZE_TITLE, pad=2,
                 loc="left", fontweight="normal")
    # Use scientific notation for y-axis to save space
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.tick_params(axis='x', labelsize=tick_fs)
    ax.grid(axis="y", alpha=0.2, lw=0.3)
    fig.subplots_adjust(bottom=0.30)

    # y-axis padding + prune upper tick to prevent VCD truncation warnings
    ymin, ymax = ax.get_ylim()
    y_range = abs(ymax - ymin)
    pad_y = y_range * 0.08
    if ymax > ymin:
        ax.set_ylim(ymin - pad_y * 0.3, ymax + pad_y)
    else:
        ax.set_ylim(ymin - pad_y, ymax + pad_y * 0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins='auto', prune='both'))

    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def generate(out_dir):
    """Generate all subplot PNGs for Figure 10.

    When *series* is ``\"dpmm\"``, only Best-DPMM is shown as the internal
    reference and the warm colour palette is used.  When ``\"topic\"``,
    only Best-Topic is shown with the cool blue/teal palette.
    """
    series = "topic"
    global _active_ext_colors

    print(f"\n  Figure 10 subplots ({series})")
    sub_dir = out_dir / "fig10"
    sub_dir.mkdir(parents=True, exist_ok=True)

    apply_subplot_style()

    # Select the internal reference and colour palette for this series
    if series == "topic":
        ref_model = "Best-Topic"
        _active_ext_colors = EXTERNAL_MODEL_COLORS_TOPIC
    else:
        ref_model = "Best-DPMM"
        _active_ext_colors = EXTERNAL_MODEL_COLORS_DPMM

    # Load data — only the relevant series
    ext_df = _load_external_results()
    int_df = _load_internal_best(series=series)
    int_all_df = _load_internal_all(series=series)

    # Panel A — series-specific workflow
    gen_workflow_png(_workflow_steps(series), sub_dir, filename="workflow.png")

    core_metrics_b = [(c, l, hb) for c, l, hb in get_core_metrics(series)]
    ext_metrics_b = [(c, l, hb) for c, l, hb in get_ext_metrics(series)]

    # --- Internal-only panels (6 models): clear, non-overcrowded ---
    for col, label, _ in core_metrics_b:
        if col in int_all_df.columns:
            safe = col.replace("/", "_")
            gen_internal_boxplot(int_all_df, col, label,
                                sub_dir / f"internal_core_{safe}.png", series=series)
    for col, label in EFFICIENCY_METRICS_FIG10:
        if col in int_all_df.columns:
            safe = col.replace("/", "_")
            gen_internal_boxplot(int_all_df, col, label,
                                sub_dir / f"internal_eff_{safe}.png", series=series)

    # --- External comparison panels (1 best internal + 11 external) ---
    if series == "topic":
        all_metrics = core_metrics_b + ext_metrics_b
        for col, label, hb in all_metrics:
            if col in ext_df.columns or col in int_df.columns:
                safe = col.replace("/", "_")
                gen_ext_boxplot(ext_df, int_df, col, label,
                                sub_dir / f"core_{safe}.png",
                                ref_model=ref_model)
    else:
        for col, label, hb in core_metrics_b:
            if col in ext_df.columns or col in int_df.columns:
                safe = col.replace("/", "_")
                gen_ext_boxplot(ext_df, int_df, col, label,
                                sub_dir / f"core_{safe}.png",
                                ref_model=ref_model)
        for col, label, hb in ext_metrics_b:
            if col in ext_df.columns or col in int_df.columns:
                safe = col.replace("/", "_")
                gen_ext_boxplot_compact(ext_df, int_df, col, label,
                                       sub_dir / f"ext_{safe}.png",
                                       higher_better=hb, ref_model=ref_model)

    for col, label in EFFICIENCY_METRICS_FIG10:
        if col in ext_df.columns or col in int_df.columns:
            safe = col.replace("/", "_")
            gen_ext_boxplot(ext_df, int_df, col, label,
                            sub_dir / f"eff_{safe}.png",
                            ref_model=ref_model)

    # Panel E — aggregate ranking (external comparison only)
    gen_ranking_bar(ext_df, int_df, sub_dir / "ranking.png",
                    ref_model=ref_model)

    # Legend strip
    gen_legend_strip(sub_dir / "legend.png", ref_model=ref_model)

    # Write manifest (internal vs external clearly separated)
    manifest = build_manifest(sub_dir, {
        "panelA": "workflow.png",
        "panelB_internal": sorted([f.name for f in sub_dir.glob("internal_core_*.png")]),
        "panelB_external": sorted([f.name for f in sub_dir.glob("core_*.png")]),
        "panelC": sorted([f.name for f in sub_dir.glob("ext_*.png")]),
        "panelD_internal": sorted([f.name for f in sub_dir.glob("internal_eff_*.png")]),
        "panelD_external": sorted([f.name for f in sub_dir.glob("eff_*.png")]),
        "panelE": "ranking.png",
        "legend": "legend.png",
        "series": series,
        "ref_model": ref_model,
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 10 subplots")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(out)
