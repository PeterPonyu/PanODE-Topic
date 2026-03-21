"""Generate Figure 2 (Base Ablation) using geometry-based layout.

Produces:
  Panel A — Cross-dataset UMAP embeddings (1 per model)
  Panel B — Core metric boxplots (NMI, ARI, ASW, DAV, DRE, LSE)
  Panel C — Extended metric boxplots (33 metrics)
  Panel D — Efficiency boxplots (s/epoch, GPU, params)

All figures use deterministic rectangle-based layout (no GridSpec/subplots).

Usage:
    python -m benchmarks.figure_generators.gen_fig2_subplots --series dpmm
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
    bind_figure_region, LayoutRegion,
    FONT_TITLE, FONT_TICK, FONT_LEGEND)
from benchmarks.figure_generators.subplot_style import (
    apply_subplot_style, save_subplot, build_manifest,
    SUBPLOT_DPI,
    FIGSIZE_BOXPLOT, FIGSIZE_UMAP,
    SCATTER_SIZE_BOXPLOT, SCATTER_SIZE_UMAP,
    LINE_WIDTH_MEDIAN, LINE_WIDTH_BOX,
    FONTSIZE_TITLE, FONTSIZE_TICK, FONTSIZE_LEGEND)
from benchmarks.figure_generators.common import (
    get_model_order, get_color, compute_umap,
    ALL_BOXPLOT_METRICS_CORE, ALL_BOXPLOT_METRICS_EXT,
    get_core_metrics, get_ext_metrics,
    MODEL_SHORT_NAMES, METRIC_DIRECTION, clip_extreme_outliers)
from benchmarks.figure_generators.data_loaders import (
    load_crossdata_per_dataset, load_cross_latent, load_joint_latent)
from benchmarks.figure_generators.significance_brackets import (
    draw_significance_brackets)


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_metric_matrix(per_ds, metric_col, series, seed_average=True):
    """Build per-model data arrays from per-dataset DataFrames.

    Parameters
    ----------
    per_ds : dict[str, DataFrame]
        Per-dataset DataFrames (one row per Model×seed when multi-seed).
    metric_col : str
        Column name of the metric to extract.
    series : str
        ``"dpmm"`` or ``"topic"`` — determines model ordering.
    seed_average : bool
        If True (default), average across seeds per (Model, Dataset) pair,
        yielding one value per dataset (n ≈ 12).  If False, retain all
        seed-level observations (n ≈ 60 for 5 seeds × 12 datasets).

    Returns
    -------
    data_per_model : list[np.ndarray]
        One array per model in ``order``.
    order : list[str]
        Model names in canonical order.
    datasets : list[str]
        Dataset names.
    """
    order = get_model_order(series)
    valid = set(order)
    datasets = sorted(per_ds.keys())

    model_vals = {m: [] for m in order}
    for ds in datasets:
        df_ds = per_ds[ds]
        df_ds = df_ds[df_ds["Model"].isin(valid)].copy()
        if metric_col not in df_ds.columns:
            continue
        for model in order:
            rows = df_ds[df_ds["Model"] == model]
            if len(rows) > 0:
                if seed_average:
                    mean_val = rows[metric_col].dropna().mean()
                    if pd.notna(mean_val):
                        model_vals[model].append(mean_val)
                else:
                    # Retain all seed-level values
                    vals = rows[metric_col].dropna().values
                    model_vals[model].extend(vals.tolist())

    data_per_model = [np.array(model_vals[m]) for m in order]
    return data_per_model, order, datasets


# ═══════════════════════════════════════════════════════════════════════════════
# Subplot generators
# ═══════════════════════════════════════════════════════════════════════════════

def gen_single_boxplot(per_ds, metric_col, metric_label, series,
                       out_path, show_friedman=True, seed_average=True):
    """Generate one boxplot subplot PNG.

    When ``seed_average=False``, uses all seed-level data points
    (5 seeds × 12 datasets = 60 per model).  Significance brackets
    are drawn for structured vs. pure counterpart pairs using
    pre-computed Wilcoxon signed-rank tests.
    """
    data_per_model, model_order, _ = _build_metric_matrix(
        per_ds, metric_col, series, seed_average=seed_average)
    # Clip extreme outliers (e.g. CAL = 83 M on teeth) to prevent
    # y-axis distortion that obscures real distributional differences.
    data_per_model = clip_extreme_outliers(data_per_model)
    short = [MODEL_SHORT_NAMES.get(m, m) for m in model_order]
    n_models = len(model_order)
    lower_better = {"SecPerEpoch", "PeakGPU_MB", "NumParams", "Time_s", "DAV"}
    higher_better = METRIC_DIRECTION.get(metric_col,
                                         metric_col not in lower_better)

    fig = plt.figure(figsize=FIGSIZE_BOXPLOT)
    layout = bind_figure_region(fig, (0.08, 0.15, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax, kind="boxplot")
    bp = ax.boxplot(data_per_model, vert=True, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=LINE_WIDTH_MEDIAN))
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(get_color(model_order[j]))
        patch.set_alpha(0.65)
        patch.set_edgecolor("gray")
        patch.set_linewidth(LINE_WIDTH_BOX)

    rng = np.random.RandomState(42)
    for j in range(n_models):
        vals = data_per_model[j]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(j + 1 + jitter, vals, s=SCATTER_SIZE_BOXPLOT,
                   c=[get_color(model_order[j])],
                   edgecolors="black", linewidths=0.2, zorder=5, alpha=0.85)

    title_str = metric_label

    # Highlight best-performing model with red border
    if higher_better:
        medians = [np.nanmedian(d) if len(d) else np.nan
                   for d in data_per_model]
        if all(np.isnan(m) for m in medians):
            best_idx = 0
        else:
            best_idx = int(np.nanargmax(medians))
    else:
        medians = [np.nanmedian(d) if len(d) else np.nan
                   for d in data_per_model]
        if all(np.isnan(m) for m in medians):
            best_idx = 0
        else:
            best_idx = int(np.nanargmin(medians))
    bp["boxes"][best_idx].set_edgecolor("red")
    bp["boxes"][best_idx].set_linewidth(1.3)

    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels(short, fontsize=FONTSIZE_TICK, rotation=75, ha="right")
    ax.set_title(title_str, fontsize=FONTSIZE_TITLE, pad=3,
                 loc="left", fontweight="normal")
    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(axis="y", alpha=0.2, lw=0.4)

    # Draw significance brackets (Wilcoxon signed-rank: structured vs pure)
    n_brackets = draw_significance_brackets(
        ax, model_order, metric_col, series,
        data_per_model=data_per_model,
        bracket_gap_frac=0.045, show_ns=False)

    # Add y-axis padding at top and left so ytick labels
    # don't extend beyond the figure border after tight_layout.
    # This also ensures significance brackets have headroom.
    ymin, ymax = ax.get_ylim()
    y_range = abs(ymax - ymin)
    pad_y = y_range * 0.08
    if ymax > ymin:
        ax.set_ylim(ymin - pad_y * 0.3, ymax + pad_y)
    else:
        ax.set_ylim(ymin - pad_y, ymax + pad_y * 0.3)

    # Prune upper y-tick to prevent the topmost label from clipping
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))

    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_cross_umap(model_name, datasets, out_path, show_legend=False):
    """Generate one cross-dataset UMAP subplot PNG."""
    import matplotlib as mpl
    blocks, ds_labels = [], []
    for ds in datasets:
        lat = load_cross_latent(model_name, ds)
        if lat is None or len(lat) == 0:
            continue
        blocks.append(lat)
        ds_labels.extend([ds] * len(lat))
    if not blocks:
        fig = plt.figure(figsize=FIGSIZE_UMAP)
        layout = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))
        ax = layout.add_axes(fig)
        style_axes(ax, kind="umap")
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=FONTSIZE_TITLE)
        save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)
        return
    X = np.vstack(blocks)
    emb = compute_umap(X)
    ds_labels = np.array(ds_labels)
    cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
    color_map = {ds: cmap(i / max(len(datasets) - 1, 1))
                 for i, ds in enumerate(datasets)}

    fig = plt.figure(figsize=FIGSIZE_UMAP)
    layout = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax, kind="umap")
    for ds in datasets:
        mask = ds_labels == ds
        if not np.any(mask):
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=SCATTER_SIZE_UMAP,
                   alpha=0.60, color=[color_map.get(ds, "gray")],
                   label=ds, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    short = MODEL_SHORT_NAMES.get(model_name, model_name)
    ax.set_title(short, fontsize=FONTSIZE_TITLE, pad=2,
                 loc="left", fontweight="normal")
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
    if show_legend:
        ax.legend(loc="best", fontsize=FONTSIZE_LEGEND,
                  framealpha=0.85, markerscale=2, handletextpad=0.2)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_joint_umap(model_name, out_path, show_legend=False):
    """Generate one UMAP subplot from the *joint* training NPZ.

    This differs from :func:`gen_cross_umap` in that all cells come from a
    single model trained on the combined dataset; the coloring by source
    dataset shows whether the model has learned a biologically meaningful
    shared latent space without any post-hoc alignment.
    """
    import matplotlib as mpl
    latent, ds_labels = load_joint_latent(model_name)
    if latent is None or len(latent) == 0:
        fig = plt.figure(figsize=FIGSIZE_UMAP)
        layout = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))
        ax = layout.add_axes(fig)
        style_axes(ax, kind="umap")
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=FONTSIZE_TITLE)
        save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)
        return

    datasets_present = list(dict.fromkeys(ds_labels))   # preserve encounter order
    emb = compute_umap(latent)
    cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
    color_map = {ds: cmap(i / max(len(datasets_present) - 1, 1))
                 for i, ds in enumerate(datasets_present)}

    fig = plt.figure(figsize=FIGSIZE_UMAP)
    layout = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax, kind="umap")
    for ds in datasets_present:
        mask = ds_labels == ds
        if not np.any(mask):
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=SCATTER_SIZE_UMAP,
                   alpha=0.60, color=[color_map.get(ds, "gray")],
                   label=ds, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    short = MODEL_SHORT_NAMES.get(model_name, model_name)
    ax.set_title(f"{short} (joint)", fontsize=FONTSIZE_TITLE, pad=2,
                 loc="left", fontweight="normal")
    if show_legend:
        ax.legend(loc="best", fontsize=FONTSIZE_LEGEND,
                  framealpha=0.85, markerscale=2, handletextpad=0.2)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_joint_umap_legend(model_name, out_path):
    """Shared legend for the joint UMAP panel (same style as per-dataset legend)."""
    import matplotlib as mpl
    _, ds_labels = load_joint_latent(model_name)
    if ds_labels is None:
        return
    datasets_present = list(dict.fromkeys(ds_labels))
    cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
    color_map = {ds: cmap(i / max(len(datasets_present) - 1, 1))
                 for i, ds in enumerate(datasets_present)}
    fig_w = FIGSIZE_UMAP[0] * 4   # extra-wide for VCD-safe legend
    fig = plt.figure(figsize=(fig_w, 0.65))
    layout = bind_figure_region(fig, (0.0, 0.0, 1.0, 1.0))
    ax = layout.add_axes(fig)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=color_map[ds],
                    markersize=5.0, alpha=0.75)
        for ds in datasets_present
    ]
    n_cols = min(len(datasets_present), 8)
    ax.legend(handles, datasets_present, loc="center",
              ncol=n_cols,
              fontsize=FONTSIZE_LEGEND, frameon=False,
              handletextpad=0.3, columnspacing=0.6, markerscale=1.0)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_umap_shared_legend(datasets, out_path):
    """Generate a standalone one-row legend image for the UMAP panel."""
    import matplotlib as mpl
    cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
    color_map = {ds: cmap(i / max(len(datasets) - 1, 1))
                 for i, ds in enumerate(datasets)}

    fig_w = FIGSIZE_UMAP[0] * 4   # extra-wide to accommodate all labels
    fig = plt.figure(figsize=(fig_w, 0.65))
    layout = bind_figure_region(fig, (0.0, 0.0, 1.0, 1.0))
    ax = layout.add_axes(fig)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=color_map[ds],
                    markersize=5.0, alpha=0.75)
        for ds in datasets if ds in color_map
    ]
    labels = [ds for ds in datasets if ds in color_map]
    n_cols = min(len(labels), 8)
    ax.legend(handles, labels, loc="center", ncol=n_cols,
              fontsize=FONTSIZE_LEGEND, frameon=False,
              handletextpad=0.3, columnspacing=0.6, markerscale=1.0)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate(out_dir, seed_average=True):
    """Generate all subplot PNGs for Figure 2.

    Parameters
    ----------
    seed_average : bool
        If True, boxplots show seed-averaged values (n ≈ 12 per model).
        If False (``--per-seed``), show all seed-level observations
        (n ≈ 60 per model).
    """
    series = "topic"
    print(f"\n  Figure 2 subplots ({series})"
          f"{' [per-seed mode]' if not seed_average else ''}")
    sub_dir = out_dir / "fig2"
    sub_dir.mkdir(parents=True, exist_ok=True)

    apply_subplot_style()

    per_ds = load_crossdata_per_dataset()
    valid_models = set(get_model_order(series))
    for ds_name in list(per_ds.keys()):
        per_ds[ds_name] = per_ds[ds_name][
            per_ds[ds_name]["Model"].isin(valid_models)
        ].copy()
        if per_ds[ds_name].empty:
            per_ds.pop(ds_name, None)
    datasets = sorted(per_ds.keys())
    order = get_model_order(series)

    # ── Data completeness diagnostic ──
    print(f"    Data completeness: {len(datasets)} datasets, "
          f"{len(order)} models")
    missing_entries = []
    for ds in datasets:
        df_ds = per_ds[ds]
        present = set(df_ds["Model"].unique()) & valid_models
        absent = valid_models - present
        n_seeds = df_ds.groupby("Model").size().max() if len(df_ds) else 0
        if absent:
            for m in sorted(absent):
                missing_entries.append((ds, m))
        if n_seeds < 5 and len(df_ds) > 0:
            print(f"    ⚠ {ds}: only {n_seeds} seed(s) per model "
                  f"(expected 5)")
    if missing_entries:
        print(f"    ⚠ {len(missing_entries)} missing (Model, Dataset) pairs:")
        for ds, m in missing_entries[:10]:
            print(f"      - {m} × {ds}")
        if len(missing_entries) > 10:
            print(f"      ... and {len(missing_entries) - 10} more")
    else:
        print(f"    ✓ All {len(order)} models present in all "
              f"{len(datasets)} datasets")
    mode_label = "seed-averaged (n≈12)" if seed_average else "per-seed (n≈60)"
    print(f"    Boxplot mode: {mode_label}")

    # Panel A — cross-dataset UMAPs (3 columns in frontend)
    # No per-UMAP legend; a shared legend row is generated separately.
    for i, m in enumerate(order):
        gen_cross_umap(m, datasets,
                       sub_dir / f"umap_{i}_{m.replace('/', '_')}.png",
                       show_legend=False)
    # Shared legend row for all UMAPs
    gen_umap_shared_legend(datasets, sub_dir / "umap_legend.png")

    # Panel A2 — joint-training UMAPs (same 3 columns; one model per subplot)
    for i, m in enumerate(order):
        gen_joint_umap(m, sub_dir / f"joint_umap_{i}_{m.replace('/', '_')}.png",
                       show_legend=False)
    # Joint legend (inferred from first available model with joint latent)
    for m in order:
        jl, _ = load_joint_latent(m)
        if jl is not None:
            gen_joint_umap_legend(m, sub_dir / "joint_umap_legend.png")
            break

    # Panel B — core metric boxplots (3 columns in frontend)
    # For Topic series (only 6 total metrics), merge core+ext into a
    # single "metrics" set — no need for separate panels.
    core_metrics = get_core_metrics(series)
    ext_metrics = get_ext_metrics(series)
    if series == "topic":
        # Merge all metrics under "core_" prefix for unified panel
        all_metrics = list(core_metrics) + list(ext_metrics)
        for col, label, _ in all_metrics:
            if any(col in per_ds[ds].columns for ds in per_ds):
                safe = col.replace("/", "_")
                gen_single_boxplot(per_ds, col, label, series,
                                   sub_dir / f"core_{safe}.png",
                                   seed_average=seed_average)
    else:
        for col, label, _ in core_metrics:
            if any(col in per_ds[ds].columns for ds in per_ds):
                safe = col.replace("/", "_")
                gen_single_boxplot(per_ds, col, label, series,
                                   sub_dir / f"core_{safe}.png",
                                   seed_average=seed_average)

        # Panel C — extended metric boxplots (4 columns in frontend)
        for col, label, _ in ext_metrics:
            if any(col in per_ds[ds].columns for ds in per_ds):
                safe = col.replace("/", "_")
                gen_single_boxplot(per_ds, col, label, series,
                                   sub_dir / f"ext_{safe}.png",
                                   show_friedman=False,
                                   seed_average=seed_average)

    # Panel D — efficiency boxplots (3 columns in frontend)
    eff_metrics = [
        ("SecPerEpoch", "Sec/Epoch \u2193", False),
        ("PeakGPU_MB", "Peak GPU (MB) \u2193", False),
        ("NumParams", "Num Params \u2193", False),
    ]
    for col, label, _ in eff_metrics:
        if any(col in per_ds[ds].columns for ds in per_ds):
            safe = col.replace("/", "_")
            gen_single_boxplot(per_ds, col, label, series,
                               sub_dir / f"eff_{safe}.png",
                               seed_average=seed_average)

    # Write manifest
    joint_umaps = sorted([f.name for f in sub_dir.glob("joint_umap_[0-9]*.png")])
    joint_legend_path = sub_dir / "joint_umap_legend.png"

    manifest_data = {
        "panelA": sorted([f.name for f in sub_dir.glob("umap_[0-9]*.png")]),
        "panelA_legend": "umap_legend.png",
        "panelB": sorted([f.name for f in sub_dir.glob("core_*.png")]),
        "panelC": sorted([f.name for f in sub_dir.glob("ext_*.png")]),
        "panelD": sorted([f.name for f in sub_dir.glob("eff_*.png")]),
    }
    if joint_umaps:
        manifest_data["panelA_joint"] = joint_umaps
        if joint_legend_path.exists():
            manifest_data["panelA_joint_legend"] = "joint_umap_legend.png"

    manifest = build_manifest(sub_dir, manifest_data)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 2 subplots")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--per-seed", action="store_true",
                        help="Show all seed-level data points (n≈60) "
                             "instead of seed-averaged values (n≈12)")
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(out, seed_average=not args.per_seed)
