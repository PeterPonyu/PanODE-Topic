"""Significance bracket drawing for boxplot figures.

Draws bracket annotations (comparison pair bars with *, **, ns markers)
above boxplots based on pre-computed Wilcoxon signed-rank test results.

The module:
  1. Loads pairwise_wilcoxon.csv from statistical_exports/
  2. Filters to relevant model pairs and metric
  3. Draws non-overlapping brackets above boxplots
  4. Uses LINE_WIDTH_BRACKET from subplot_style.py

Design principles:
  - Brackets connect structured vs. pure counterpart pairs only
  - Stacking: multiple brackets are vertically stacked with consistent gap
  - Stars: * p<0.05, ** p<0.01, *** p<0.001, ns otherwise
  - Text uses FONTSIZE_ANNOTATION from subplot_style
  - Bracket line colour: dark gray (#333333), star colour: black
  - Effect size is shown as subscript when large (|d| > 0.5)

Usage:
    from benchmarks.figure_generators.significance_brackets import (
        draw_significance_brackets)
    draw_significance_brackets(ax, model_order, metric_col, series)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from benchmarks.figure_generators.subplot_style import (
    LINE_WIDTH_BRACKET, FONTSIZE_ANNOTATION)
from benchmarks.figure_generators.data_loaders import (
    load_pairwise_wilcoxon, load_pairwise_wilcoxon_external)

# ═══════════════════════════════════════════════════════════════════════════════
# Model pairing logic
# ═══════════════════════════════════════════════════════════════════════════════

# Structured ↔ Pure counterpart pairs (for Figure 2 internal ablation)
INTERNAL_PAIRS = {
    "dpmm": [
        ("DPMM-Base", "Pure-AE"),
        ("DPMM-Transformer", "Pure-Transformer-AE"),
        ("DPMM-Contrastive", "Pure-Contrastive-AE"),
    ],
    "topic": [
        ("Topic-Base", "Pure-VAE"),
        ("Topic-Transformer", "Pure-Transformer-VAE"),
        ("Topic-Contrastive", "Pure-Contrastive-VAE"),
    ],
}


def _p_to_stars(p_value):
    """Convert p-value to significance stars string."""
    if pd.isna(p_value) or p_value >= 0.05:
        return "ns"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    else:
        return "*"


def _get_sig_pairs(metric_col, series="topic", model_order=None):
    """Look up Wilcoxon test results for the given metric and series.

    Returns
    -------
    list[dict]
        Each dict has: idx_left, idx_right (0-based indices into
        model_order), stars, p_value, effect_size, cliffs_delta.
    """
    wilcox_df = load_pairwise_wilcoxon()
    if wilcox_df is None:
        return []

    # Direct lookup — the Wilcoxon CSV now contains all metrics by exact name
    lookup_metric = metric_col

    pairs = INTERNAL_PAIRS.get(series, [])
    results = []

    for struct, pure in pairs:
        if struct not in model_order or pure not in model_order:
            continue

        # Look up in Wilcoxon table by exact metric name
        row = wilcox_df[
            (wilcox_df["Structured"] == struct)
            & (wilcox_df["Pure"] == pure)
            & (wilcox_df["Metric"] == lookup_metric)
        ]
        if row.empty:
            continue

        r = row.iloc[0]
        p_val = r["p_value"]
        stars = _p_to_stars(p_val)
        cliffs = r.get("Cliffs_delta", np.nan)
        effect = r.get("Effect_size", "")

        idx_left = model_order.index(struct)
        idx_right = model_order.index(pure)
        # Ensure left < right for consistent bracket direction
        if idx_left > idx_right:
            idx_left, idx_right = idx_right, idx_left

        results.append({
            "idx_left": idx_left,
            "idx_right": idx_right,
            "stars": stars,
            "p_value": p_val,
            "effect_size": effect,
            "cliffs_delta": cliffs,
            "pair_label": f"{struct} vs {pure}",
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Bracket drawing
# ═══════════════════════════════════════════════════════════════════════════════

def draw_significance_brackets(ax, model_order, metric_col, series="topic",
                               data_per_model=None, bracket_gap_frac=0.04,
                               show_ns=False, show_effect=False):
    """Draw significance brackets above boxplots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes with boxplots already drawn.
    model_order : list[str]
        Full model names in boxplot order (x=1 is model_order[0]).
    metric_col : str
        Metric column name to look up in the Wilcoxon table.
    series : str
        ``"dpmm"`` or ``"topic"`` — determines which pairs to annotate.
    data_per_model : list[np.ndarray] or None
        Per-model data arrays. Used to compute bracket height above data.
        If None, uses current y-axis limits.
    bracket_gap_frac : float
        Fraction of y-range between stacked brackets.
    show_ns : bool
        If True, draw brackets for non-significant pairs too.
    show_effect : bool
        If True, show Cliff's delta value below the star text.

    Returns
    -------
    int
        Number of brackets drawn.
    """
    sig_pairs = _get_sig_pairs(metric_col, series, model_order)
    if not sig_pairs:
        return 0

    # Filter out ns if not requested
    if not show_ns:
        sig_pairs = [p for p in sig_pairs if p["stars"] != "ns"]
    if not sig_pairs:
        return 0

    # Determine y-range for bracket placement
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin

    # Find the maximum data value across all models to place brackets above
    if data_per_model is not None:
        data_max = max(
            (np.nanmax(d) if len(d) > 0 else ymin for d in data_per_model),
            default=ymax)
    else:
        data_max = ymax

    # Starting y for the first bracket: slightly above the highest data point
    bracket_y_start = data_max + y_range * 0.05
    bracket_step = y_range * bracket_gap_frac

    # Sort pairs by span width (narrower first → stacks nicely)
    sig_pairs.sort(key=lambda p: p["idx_right"] - p["idx_left"])

    n_drawn = 0
    for i, pair in enumerate(sig_pairs):
        x_left = pair["idx_left"] + 1     # 1-based boxplot positions
        x_right = pair["idx_right"] + 1
        y_bar = bracket_y_start + i * bracket_step

        # Draw the bracket: horizontal bar with short vertical ticks
        tick_h = y_range * 0.012
        bracket_color = "#333333"

        # Vertical ticks at each end
        ax.plot([x_left, x_left], [y_bar - tick_h, y_bar],
                color=bracket_color, lw=LINE_WIDTH_BRACKET,
                clip_on=False, zorder=10)
        ax.plot([x_right, x_right], [y_bar - tick_h, y_bar],
                color=bracket_color, lw=LINE_WIDTH_BRACKET,
                clip_on=False, zorder=10)
        # Horizontal bar
        ax.plot([x_left, x_right], [y_bar, y_bar],
                color=bracket_color, lw=LINE_WIDTH_BRACKET,
                clip_on=False, zorder=10)

        # Star text above the bar
        x_mid = (x_left + x_right) / 2
        stars_text = pair["stars"]

        # Annotation font size (must be ≥ 12 so 50 % composed scale → ≥ 6 pt)
        fs = max(FONTSIZE_ANNOTATION, 12)

        ax.text(x_mid, y_bar + tick_h * 0.05, stars_text,
                ha="center", va="bottom", fontsize=fs,
                fontweight="bold", color="black",
                clip_on=False, zorder=11)

        # Optional effect size annotation
        if show_effect and pair["effect_size"] and pair["effect_size"] != "":
            d_val = pair["cliffs_delta"]
            if pd.notna(d_val):
                eff_text = f"|d|={abs(d_val):.2f}"
                ax.text(x_mid, y_bar + tick_h * 0.05 + y_range * 0.015,
                        eff_text, ha="center", va="bottom",
                        fontsize=max(fs - 2, 12), color="#666666",
                        clip_on=False, zorder=11)

        n_drawn += 1

    # Expand y-axis upper limit to accommodate brackets
    new_ymax = bracket_y_start + len(sig_pairs) * bracket_step + y_range * 0.06
    if new_ymax > ymax:
        ax.set_ylim(ymin, new_ymax)

    return n_drawn


# ═══════════════════════════════════════════════════════════════════════════════
# External-model significance stars (compact: stars above each external box)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_external_significance_stars(ax, model_order, metric_col, series="topic",
                                     data_per_model=None):
    """Draw compact significance stars above external-model boxplots.

    Instead of connecting brackets (which would crowd the plot with 11
    external models), places significance stars directly above each
    external model's box, indicating Wilcoxon test result vs the
    internal reference (Best-DPMM or Best-Topic).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    model_order : list[str]
        Full model names in boxplot order (x=1 = model_order[0]).
    metric_col : str
    series : str
        ``"dpmm"`` or ``"topic"``
    data_per_model : list[np.ndarray] or None

    Returns
    -------
    int
        Number of stars drawn.
    """
    ext_df = load_pairwise_wilcoxon_external()
    if ext_df is None:
        return 0

    ref_label = "Best-Topic"

    n_drawn = 0
    for idx, model in enumerate(model_order):
        if model == ref_label:
            continue  # skip the internal reference itself

        row = ext_df[
            (ext_df["Structured"] == ref_label)
            & (ext_df["Pure"] == model)
            & (ext_df["Metric"] == metric_col)
        ]
        if row.empty:
            continue

        p_val = row.iloc[0]["p_value"]
        stars = _p_to_stars(p_val)
        if stars == "ns":
            continue  # skip non-significant

        # Place star above the box's whisker
        x_pos = idx + 1  # 1-based boxplot positions
        if data_per_model is not None and len(data_per_model[idx]) > 0:
            y_top = np.nanmax(data_per_model[idx])
        else:
            y_top = ax.get_ylim()[1] * 0.9

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        y_star = y_top + y_range * 0.03

        fs = max(FONTSIZE_ANNOTATION, 12)
        ax.text(x_pos, y_star, stars,
                ha="center", va="bottom", fontsize=fs,
                fontweight="bold", color="black",
                clip_on=False, zorder=11)
        n_drawn += 1

    # Expand y-axis if needed
    if n_drawn > 0 and data_per_model is not None:
        all_maxes = [np.nanmax(d) if len(d) > 0 else 0 for d in data_per_model]
        data_max = max(all_maxes)
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        needed = data_max + y_range * 0.12
        if needed > ymax:
            ax.set_ylim(ymin, needed)

    return n_drawn
