"""
Visualization utilities for PanODE benchmarks.

Provides quick diagnostic plots used during benchmark runs:
- ``plot_umap_grid``              -- UMAP grid for multiple latent representations
- ``plot_all_metrics_barplot``    -- grouped horizontal bar charts for all metrics
- ``plot_core_metrics_barplot``   -- 2x3 bar chart of 6 core metrics

Note: publication-quality composite figures are produced by
``benchmarks/figure_generators/`` (separate pipeline).  The functions here
are for per-run diagnostics only.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Publication-quality defaults ──────────────────────────────────────────────
_FONT_CFG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# 12-color palette: 6 warm (DPMM/AE) + 6 cool (Topic/VAE)
_PALETTE_12 = [
    "#4C72B0", "#5DA5DA", "#7EB8DA",   # Pure-AE, Pure-Tfm-AE, Pure-Ctr-AE
    "#DD8452", "#E8A07E", "#C44E52",   # DPMM-Base, DPMM-Ctr, DPMM-Tfm
    "#55A868", "#8CC68B", "#B8D8B8",   # Pure-VAE, Pure-Tfm-VAE, Pure-Ctr-VAE
    "#8172B3", "#A899CC", "#937DB0",   # Topic-Base, Topic-Ctr, Topic-Tfm
]


def _apply_style():
    """Apply publication-quality style."""
    mpl.rcParams.update(_FONT_CFG)


# ═══════════════════════════════════════════════════════════════════════════════
# UMAP Grid
# ═══════════════════════════════════════════════════════════════════════════════

def plot_umap_grid(latent_dict, labels, title, save_path, n_neighbors=15, min_dist=0.5):
    """Create UMAP visualization grid for multiple latent representations.

    Args:
        latent_dict: Dict of {model_name: latent_array}
        labels: Cell type labels (strings or integers)
        title: Plot title
        save_path: Path to save the figure
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    import scanpy as sc
    from sklearn.preprocessing import LabelEncoder

    _apply_style()

    n = len(latent_dict)
    if n == 0:
        print("No latent representations to plot.")
        return

    if isinstance(labels[0], str):
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
    else:
        labels_encoded = np.array(labels)

    n_colors = len(np.unique(labels_encoded))
    cmap = mpl.colormaps.get_cmap("tab20" if n_colors <= 20 else "nipy_spectral")

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 4.5, 4.0

    fig, axes = plt.subplots(rows, cols, figsize=(cell_w * cols, cell_h * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, latent) in enumerate(latent_dict.items()):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        try:
            adata_temp = sc.AnnData(latent.astype(np.float32))
            sc.pp.neighbors(adata_temp, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.umap(adata_temp, min_dist=min_dist)
            umap_coords = adata_temp.obsm["X_umap"]

            ax.scatter(
                umap_coords[:, 0], umap_coords[:, 1],
                c=labels_encoded, cmap=cmap, s=18, alpha=0.7,
                edgecolors="none", rasterized=True)
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_xlabel("UMAP-1", fontsize=8)
            ax.set_ylabel("UMAP-2", fontsize=8)
            ax.tick_params(labelsize=7)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_title(name, fontsize=11, fontweight="bold")

    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FONT_CFG.get("savefig.dpi", 200), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics bar charts (horizontal, readable with 12 models)
# ═══════════════════════════════════════════════════════════════════════════════

# Consolidated metric groups — 8 panels (4×2 grid, no empty slots)
_METRIC_GROUPS = {
    "Clustering": {
        "cols": ["NMI", "ARI", "ASW"],
        "labels": ["NMI", "ARI", "ASW"],
        "note": None,
    },
    "Davies\u2013Bouldin \u2193": {
        "cols": ["DAV"],
        "labels": ["DAV"],
        "note": "Lower is better.",
    },
    "Calinski\u2013Harabasz \u2191": {
        "cols": ["CAL"],
        "labels": ["CAL"],
        "note": "Higher is better.",
    },
    "DRE \u2014 UMAP": {
        "cols": [
            "DRE_umap_distance_correlation", "DRE_umap_Q_local",
            "DRE_umap_Q_global", "DRE_umap_overall_quality",
        ],
        "labels": ["DistCorr", "Q_local", "Q_global", "Overall"],
        "note": None,
    },
    "DRE \u2014 t-SNE": {
        "cols": [
            "DRE_tsne_distance_correlation", "DRE_tsne_Q_local",
            "DRE_tsne_Q_global", "DRE_tsne_overall_quality",
        ],
        "labels": ["DistCorr", "Q_local", "Q_global", "Overall"],
        "note": None,
    },
    "DREX \u2014 UMAP": {
        "cols": [
            "DREX_trustworthiness", "DREX_continuity",
            "DREX_distance_spearman", "DREX_overall_quality",
        ],
        "labels": ["Trust", "Contin", "Spearman", "Overall"],
        "note": "Latent \u2192 UMAP fidelity.",
    },
    "LSE \u2014 Structure": {
        "cols": [
            "LSE_manifold_dimensionality", "LSE_spectral_decay_rate",
            "LSE_participation_ratio", "LSE_anisotropy_score",
        ],
        "labels": ["ManifDim", "SpectDecay", "Particip.", "Anisotropy"],
        "note": None,
    },
    "LSE \u2014 Trajectory & Quality": {
        "cols": [
            "LSE_trajectory_directionality", "LSE_noise_resilience",
            "LSE_overall_quality", "LSE_core_quality",
        ],
        "labels": ["TrajDir", "NoiseRes", "Overall", "CoreQual"],
        "note": None,
    },
}


def plot_all_metrics_barplot(df, save_path, title="Benchmark Results"):
    """Create grouped horizontal bar charts for all metric categories.

    Uses horizontal bars for readability with many models (up to 12).

    Args:
        df: DataFrame with Model column + metric columns
        save_path: Path to save figure
        title: Plot title
    """
    # Guard against swapped arguments
    if isinstance(save_path, str) and isinstance(title, str):
        if (os.sep in title or title.endswith(".png")) and not (os.sep in save_path or save_path.endswith(".png")):
            save_path, title = title, save_path

    _apply_style()

    df_valid = df[df.get("NMI", pd.Series(dtype=float)) > 0].copy() if "NMI" in df.columns else df.copy()
    if df_valid.empty:
        print("No valid results to plot.")
        return

    models = df_valid["Model"].tolist()
    n_models = len(models)

    n_groups = len(_METRIC_GROUPS)
    n_cols = 4
    n_rows = (n_groups + n_cols - 1) // n_cols
    row_height = max(3.5, 0.4 * n_models + 1.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, row_height * n_rows))
    axes = np.atleast_2d(axes).flatten()

    metric_cmap = mpl.colormaps["Set2"]

    for panel_idx, (group_name, group_info) in enumerate(_METRIC_GROUPS.items()):
        ax = axes[panel_idx]
        metric_cols = [c for c in group_info["cols"] if c in df_valid.columns]
        metric_labels = [group_info["labels"][i] for i, c in enumerate(group_info["cols"]) if c in df_valid.columns]

        if not metric_cols:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(group_name, fontweight="bold")
            continue

        vals = df_valid.set_index("Model")[metric_cols].reindex(models)
        y_pos = np.arange(n_models)
        n_metrics = len(metric_cols)
        bar_h = 0.8 / n_metrics

        for m_idx, (mcol, mlabel) in enumerate(zip(metric_cols, metric_labels)):
            offsets = y_pos + m_idx * bar_h - 0.4 + bar_h / 2
            values = vals[mcol].fillna(0).values

            bars = ax.barh(offsets, values, height=bar_h, label=mlabel,
                           color=metric_cmap(m_idx / max(n_metrics - 1, 1)),
                           edgecolor="white", linewidth=0.3)

            for bar, val in zip(bars, values):
                if not np.isnan(val) and val != 0:
                    fmt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                    x_offset = max(abs(values)) * 0.02 if max(abs(values)) > 0 else 0.01
                    ax.text(bar.get_width() + x_offset,
                            bar.get_y() + bar.get_height() / 2,
                            fmt, va="center", ha="left", fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(group_name, fontweight="bold", fontsize=12)

        # Place legend outside the plot area to avoid overlapping with bars
        ax.legend(fontsize=7, loc="upper right",
                  bbox_to_anchor=(1.0, 1.0), framealpha=0.85,
                  borderaxespad=0.3, handlelength=1.2, handletextpad=0.4)

        if group_info.get("note"):
            ax.annotate(group_info["note"], xy=(0.02, 0.02), xycoords="axes fraction",
                        fontsize=6.5, fontstyle="italic", color="gray")

    for idx in range(n_groups, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FONT_CFG.get("savefig.dpi", 200), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Core-metrics bar plot (6 panels — 2×3)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_core_metrics_barplot(df, save_path, title="Benchmark Results",
                              series="all", no_title=False, figformat="png"):
    """Horizontal bar plot of 6 core metrics (2×3) with per-model colours.

    Panels: NMI / ARI / ASW / DAV / DRE Overall / LSE Overall.

    Migrated from ``regenerate_figures.py`` so it can be imported without
    pulling in the full legacy script.

    Args:
        df: DataFrame with ``Model`` column + metric columns.
        save_path: Output path (extension overridden by *figformat*).
        title: Suptitle text.
        series: ``"all"`` / ``"dpmm"`` / ``"topic"`` — controls canonical
            model ordering via ``paper_style.sort_df_by_model_order``.
        no_title: If True, omit *suptitle* (useful for compositing).
        figformat: ``"png"`` | ``"pdf"`` | ``"svg"``.
    """
    from utils.paper_style import (
        sort_df_by_model_order, get_color, MODEL_SHORT_NAMES)

    _apply_style()

    df_s = sort_df_by_model_order(df, series)
    models = df_s["Model"].tolist()
    short = [MODEL_SHORT_NAMES.get(m, m) for m in models]
    n = len(models)

    fig, axes = plt.subplots(2, 3, figsize=(16, max(4, 0.55 * n + 1.5)))

    # Resolve DRE / LSE column names (naming may vary across runs)
    dre_col = next((c for c in df_s.columns
                    if c == "DRE_umap_overall_quality"), None)
    if dre_col is None:
        dre_col = next((c for c in df_s.columns
                        if "overall_quality" in c.lower() and "dre" in c.lower()), None)
    if dre_col is None:
        dre_col = "DRE_umap_overall_quality"
    lse_col = next((c for c in df_s.columns
                    if "overall_quality" in c.lower() and "lse" in c.lower()), None)
    if lse_col is None:
        lse_col = next((c for c in df_s.columns
                        if c.startswith("LSE") and "overall" in c),
                       "LSE_overall_quality")

    metrics_info = [
        ("NMI",     "NMI \u2191",                  True),
        ("ARI",     "ARI \u2191",                  True),
        ("ASW",     "ASW \u2191",                  True),
        ("DAV",     "Davies\u2013Bouldin \u2193",   False),
        (dre_col,   "DRE UMAP \u2191",             True),
        (lse_col,   "LSE Overall \u2191",           True),
    ]

    for ax, (col, label, higher_better) in zip(axes.flatten(), metrics_info):
        if col not in df_s.columns:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", fontsize=10)
            ax.set_title(label, fontweight="bold", fontsize=11)
            continue

        vals = df_s[col].fillna(0).values
        colors = [get_color(m) for m in models]
        y_pos = np.arange(n)

        bars = ax.barh(y_pos, vals, height=0.6, color=colors,
                       edgecolor="white", linewidth=0.5)

        max_val = max(abs(vals)) if len(vals) > 0 else 1
        for bar, val in zip(bars, vals):
            x_off = max_val * 0.02
            ax.text(bar.get_width() + x_off,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left",
                    fontsize=8, fontweight="medium")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlim(0, max_val * 1.20)

    clean = re.sub(r"\s*\d{8}_\d{6}", "", title)
    clean = re.sub(r"\s*\(.*?(setty|lung|endo).*?\)", "", clean, flags=re.I)
    if not no_title:
        fig.suptitle(clean.strip(), fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=_FONT_CFG.get("savefig.dpi", 200), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")