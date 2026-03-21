#!/usr/bin/env python
"""
Merge Internal + External Results and Visualise Comparison
==========================================================

Combines the **12 internal PanODE-LAB models** (from ``full_comparison``)
with all **external baselines** (from the external benchmark runner) into a
single merged experiment, then generates publication-quality statistical
comparison figures using the REA framework.

Architecture
------------
- ``MergedExperimentConfig`` reads per-dataset CSVs from both sources,
  filters to requested methods, concatenates, and writes to a new output
  directory so that REA can treat them as one experiment.
- The ``visualize()`` function mirrors ``visualize_experiment.py`` but is
  purpose-built for the merged (27-method) comparison, with appropriate
  adaptive layout parameters.

Output Layout
-------------
::

    experiments/results/full_vs_external/
        tables/      ← merged per-dataset CSVs (27 methods × 36 metrics)
        series/      ← merged training series
        figures/     ← publication figures (all_metrics.pdf, training_curves/)

Usage
-----
::

    # Step 1: Merge and visualise (full pipeline)
    python -m experiments.merge_and_visualize

    # Merge only — no figures
    python -m experiments.merge_and_visualize --merge-only

    # Figures only (tables already merged from previous run)
    python -m experiments.merge_and_visualize --figures-only

    # Summary statistics only (text output, no figures)
    python -m experiments.merge_and_visualize --summary-only

    # Override the external benchmark name
    python -m experiments.merge_and_visualize --external-name external

    # Visualise only specific metric groups
    python -m experiments.merge_and_visualize --groups clustering drex lsex

    # Training curves with reconstruction panels
    python -m experiments.merge_and_visualize --training-curves --show-recon
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Project imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_lib.viz.rea import (
    RigorousExperimentalAnalyzer,
    create_publication_figure,
    _apply_font,
    clamp_xtick_fontsize,
    needs_method_split,
    MIN_XTICK_FONTSIZE)
from eval_lib.experiment.merge import MergedExperimentConfig
from eval_lib.baselines.registry import EXTERNAL_MODELS

# Reuse helpers from visualize_experiment (DRY)
from experiments.visualize_experiment import (
    METRIC_GROUPS,
    DEFAULT_PALETTE,
    _MAX_HEIGHT_TO_WIDTH,
    _adaptive_ncols,
    _adaptive_params,
    _compute_hspace,
    _enforce_max_aspect,
    _build_sig_pairs,
    _smooth)


# ═══════════════════════════════════════════════════════════════════════════════
# Method ordering constants
# ═══════════════════════════════════════════════════════════════════════════════

# Internal models — same order as full_comparison preset
INTERNAL_METHODS = [
    "Pure-AE", "Pure-Trans-AE", "Pure-Contr-AE",
    "Pure-VAE", "Pure-Trans-VAE", "Pure-Contr-VAE",
    "DPMM-Base", "DPMM-Trans", "DPMM-Contr",
    "Topic-Base", "Topic-Trans", "Topic-Contr",
]

# External baselines — ordered from eval_lib.baselines.registry
EXTERNAL_METHODS = list(EXTERNAL_MODELS.keys())

# ── Logical method groups for multi-figure mode ──────────────────────────────
#
# When the total method count exceeds METHOD_GROUP_THRESHOLD, figures are
# automatically split into logical groups rather than crammed into one
# unreadable plot.  The "focal method" (best-performing proposed model) is
# included in every group as the comparison anchor.
#
# Groups:
#   1. internal  — all 12 proposed PanODE-LAB models
#   2. external  — all external baselines + the focal model as reference
#   3. topN      — top-K performers from both pools (cross-pool ranking)
#
METHOD_GROUP_THRESHOLD = 15   # auto-split when n_methods exceeds this

FOCAL_METHOD = "Pure-Trans-AE"  # best-performing proposed model (anchor)

# ═══════════════════════════════════════════════════════════════════════════════
# Merged experiment configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_INTERNAL_DIR = "mixed/full_comparison"
DEFAULT_EXTERNAL_DIR = "external"
DEFAULT_MERGED_NAME = "full_vs_external"
DEFAULT_OUTPUT_ROOT = Path("experiments/results")


def build_merged_config(
    internal_name: str = DEFAULT_INTERNAL_DIR,
    external_name: str = DEFAULT_EXTERNAL_DIR,
    merged_name: str = DEFAULT_MERGED_NAME,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    internal_methods: list = None,
    external_methods: list = None) -> MergedExperimentConfig:
    """Create a MergedExperimentConfig combining internal + external results.

    Parameters
    ----------
    internal_name : str
        Subdirectory name under ``output_root`` for internal models.
    external_name : str
        Subdirectory name under ``output_root`` for external baselines.
    merged_name : str
        Output subdirectory name for merged results.
    output_root : Path
        Root directory for experiments/results.
    internal_methods : list or None
        Internal method names (default: all 12).
    external_methods : list or None
        External method names (default: all 15).

    Returns
    -------
    MergedExperimentConfig
        Ready for ``build_merged_tables()``.
    """
    if internal_methods is None:
        internal_methods = list(INTERNAL_METHODS)
    if external_methods is None:
        external_methods = list(EXTERNAL_METHODS)

    return MergedExperimentConfig(
        name=merged_name,
        sources=[
            {
                "tables": str(output_root / internal_name / "tables"),
                "series": str(output_root / internal_name / "series"),
                "methods": internal_methods,
            },
            {
                "tables": str(output_root / external_name / "tables"),
                "series": str(output_root / external_name / "series"),
                "methods": external_methods,
            },
        ],
        output_root=output_root,
        description=(
            f"Merged comparison: {len(internal_methods)} internal PanODE-LAB "
            f"models + {len(external_methods)} external baselines"
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation entry point
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_merged(
    merged_cfg: MergedExperimentConfig,
    *,
    summary_only: bool = False,
    metric_groups: list = None,
    selected_metrics: list = None,
    excluded_metrics: list = None,
    per_group: bool = False,
    ncols: int = None,
    # Visual parameter overrides
    fig_height: float = None,
    fig_width_per_metric: float = 3.2,
    plot_type: str = "boxplot",
    palette_name: str = None,
    title_fontsize: float = 11,
    axis_label_fontsize: float = 11,
    tick_label_fontsize: float = 11,
    xtick_label_fontsize: float = None,
    significance_fontsize: float = 13,
    ns_fontsize: float = 10,
    xlabel_rotation: float = None,
    significance_marker_offset: float = -0.04,
    ns_offset: float = 0.0,
    significance_line_width: float = 1.5,
    bar_strip_size: float = 2,
    bar_strip_alpha: float = 0.5,
    font_family: str = "Arial",
    show_legend: bool = False,
    panel_labels: bool = False,
    dpi: int = 300):
    """Generate metric comparison figures for merged internal + external.

    Mirrors ``visualize_experiment.visualize()`` logic but operates on
    a ``MergedExperimentConfig`` directly instead of a preset key.
    """
    tables_dir = merged_cfg.tables_dir
    figures_dir = merged_cfg.figures_dir
    method_names = merged_cfg.method_names
    n_methods = len(method_names)

    # ── Validate ──────────────────────────────────────────────────────────
    if not tables_dir.exists() or not list(tables_dir.glob("*.csv")):
        print(f"ERROR: No merged tables found in {tables_dir}")
        print("Run with --merge-only first, or omit --figures-only.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Visualising merged experiment: {merged_cfg.name}")
    print(f"Tables dir : {tables_dir}")
    print(f"Methods ({n_methods}): {method_names}")
    print(f"Description: {merged_cfg.description}")
    print(f"{'='*70}\n")

    _apply_font(font_family)

    # ── Initialise REA ────────────────────────────────────────────────────
    analyzer = RigorousExperimentalAnalyzer(
        data_folder_path=str(tables_dir),
        method_names=method_names,
        method_order=method_names,
        verbose=True)
    analyzer.load_experimental_data()
    analyzer.preprocess_data()
    analyzer.print_comprehensive_summary()

    if summary_only:
        return

    # ── Layout parameters ─────────────────────────────────────────────────
    os.makedirs(figures_dir, exist_ok=True)

    palette = sns.color_palette(
        palette_name if palette_name else DEFAULT_PALETTE, n_methods
    )
    sig_pairs = _build_sig_pairs(method_names)
    auto_h, auto_rot, auto_xtick = _adaptive_params(n_methods)

    final_h = fig_height if fig_height is not None else auto_h
    final_rot = xlabel_rotation if xlabel_rotation is not None else auto_rot
    final_xtick = (xtick_label_fontsize
                   if xtick_label_fontsize is not None
                   else auto_xtick)
    # Clamp to guaranteed readable minimum
    final_xtick = clamp_xtick_fontsize(final_xtick)

    available = set(analyzer.metrics)
    groups_to_plot = metric_groups or list(METRIC_GROUPS.keys())

    # ── Flatten metrics ───────────────────────────────────────────────────
    flat_metrics = []
    flat_display = {}
    for group_key in groups_to_plot:
        if group_key not in METRIC_GROUPS:
            print(f"  WARNING: unknown metric group '{group_key}' — skipping")
            continue
        grp = METRIC_GROUPS[group_key]
        for m in grp["metrics"]:
            if m in available:
                flat_metrics.append(m)
                flat_display[m] = grp["display"].get(m, m)

    if not flat_metrics:
        print("  No matching metrics in data — skipping figures.")
        return

    # Apply individual selection/exclusion
    if selected_metrics:
        sel = set(selected_metrics)
        flat_metrics = [m for m in flat_metrics if m in sel]
        flat_display = {m: flat_display[m] for m in flat_metrics}
    if excluded_metrics:
        exc = set(excluded_metrics)
        flat_metrics = [m for m in flat_metrics if m not in exc]
        flat_display = {m: flat_display[m] for m in flat_metrics}

    if not flat_metrics:
        print("  No metrics after selection/exclusion — skipping.")
        return

    final_ncols = ncols if ncols is not None else _adaptive_ncols(n_methods)

    computed_hspace = _compute_hspace(
        method_names=method_names,
        rotation_deg=final_rot,
        xtick_fontsize=final_xtick,
        title_fontsize=title_fontsize,
        per_row_height=final_h)

    # Enforce aspect ratio cap (height/width ≤ 21:17)
    if ncols is None:
        final_ncols = _enforce_max_aspect(
            n_metrics=len(flat_metrics),
            per_row_height=final_h,
            hspace=computed_hspace,
            fig_width_per_metric=fig_width_per_metric,
            ncols_init=final_ncols)

    common_kwargs = dict(
        show_significance_pairs=sig_pairs,
        palette=palette,
        panel_labels=panel_labels,
        plot_type=plot_type,
        xlabel_rotation=final_rot,
        font_family=font_family,
        significance_line_width=significance_line_width,
        significance_marker_offset=significance_marker_offset,
        bar_strip_size=bar_strip_size,
        bar_strip_alpha=bar_strip_alpha,
        title_fontsize=title_fontsize,
        title_fontweight="normal",
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        significance_fontsize=significance_fontsize,
        ns_fontsize=ns_fontsize,
        ns_offset=ns_offset,
        show_legend=show_legend,
        dpi=dpi,
        hspace=computed_hspace)

    def _post_hoc_xtick(fig, axes, save_path):
        if xtick_label_fontsize is not None:
            for ax in axes:
                plt.setp(ax.xaxis.get_majorticklabels(), fontsize=final_xtick)
            fig.savefig(str(save_path), dpi=dpi, facecolor="white",
                        bbox_inches="tight", pad_inches=0.05)

    figure_paths = []

    if per_group:
        for group_key in groups_to_plot:
            if group_key not in METRIC_GROUPS:
                continue
            grp = METRIC_GROUPS[group_key]
            grp_metrics = [m for m in grp["metrics"] if m in available]
            if not grp_metrics:
                continue

            grp_display = {m: grp["display"].get(m, m) for m in grp_metrics}
            grp_ncols = min(final_ncols, len(grp_metrics))
            grp_nrows = math.ceil(len(grp_metrics) / grp_ncols)
            grp_w = fig_width_per_metric * grp_ncols
            grp_h = (final_h * grp_nrows
                     + final_h * computed_hspace * max(grp_nrows - 1, 0))
            save_path = figures_dir / f"{group_key}.pdf"

            print(f"  Generating: {group_key}  ({len(grp_metrics)} metrics)")
            try:
                fig, axes = create_publication_figure(
                    analyzer,
                    metrics=grp_metrics,
                    metric_display_names=grp_display,
                    figsize=(grp_w, grp_h),
                    ncols=grp_ncols,
                    save_path=str(save_path),
                    **common_kwargs)
                _post_hoc_xtick(fig, axes, save_path)
                plt.close(fig)
                figure_paths.append(save_path)
                print(f"    -> {save_path}")
            except Exception as exc:
                print(f"    ERROR: {exc}")
                traceback.print_exc()
    else:
        n_rows = math.ceil(len(flat_metrics) / final_ncols)
        total_w = fig_width_per_metric * final_ncols
        total_h = (final_h * n_rows
                   + final_h * computed_hspace * max(n_rows - 1, 0))
        save_path = figures_dir / "all_metrics.pdf"

        print(f"  Generating unified grid: {len(flat_metrics)} metrics, "
              f"{final_ncols} cols x {n_rows} rows  "
              f"({total_w:.1f} x {total_h:.1f} in @ {dpi} dpi)")

        try:
            fig, axes = create_publication_figure(
                analyzer,
                metrics=flat_metrics,
                metric_display_names=flat_display,
                figsize=(total_w, total_h),
                ncols=final_ncols,
                save_path=str(save_path),
                **common_kwargs)
            _post_hoc_xtick(fig, axes, save_path)
            plt.close(fig)
            figure_paths.append(save_path)
            print(f"    -> {save_path}")
        except Exception as exc:
            print(f"    ERROR generating unified figure: {exc}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Figures saved to: {figures_dir}")
    print(f"{'='*70}\n")
    return figure_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-figure grouped visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_method_groups(
    all_methods: list[str],
    tables_dir,
    *,
    focal_method: str = FOCAL_METHOD,
    top_n: int = 10,
    ranking_metric: str = "NMI") -> dict[str, list[str]]:
    """Split methods into logical groups for multi-figure mode.

    Returns a dict  ``{ group_name: [method_names_in_order] }``.

    Groups produced:
      - ``internal``  : all methods from INTERNAL_METHODS (present in data)
      - ``external``  : external baselines + the focal/proposed model
      - ``topN``      : the top-N performers across both pools, ranked by
                        cross-dataset mean of *ranking_metric*
    """
    internal_set = set(INTERNAL_METHODS)
    present_internal = [m for m in all_methods if m in internal_set]
    present_external = [m for m in all_methods if m not in internal_set]

    groups: dict[str, list[str]] = {}

    # Group 1: Internal models
    if present_internal:
        groups["internal"] = list(present_internal)

    # Group 2: External baselines + focal reference
    if present_external:
        ext_list = list(present_external)
        if focal_method and focal_method in all_methods and focal_method not in ext_list:
            ext_list.append(focal_method)          # anchor last
        groups["external"] = ext_list

    # Group 3: Top-N cross-pool (ranked by mean of ranking_metric)
    try:
        import glob as _g
        csvs = sorted(_g.glob(str(tables_dir / "*.csv")))
        if csvs:
            scores: dict[str, list[float]] = {m: [] for m in all_methods}
            for csv_path in csvs:
                df = pd.read_csv(csv_path, index_col=0)
                if ranking_metric in df.columns:
                    for method in all_methods:
                        if method in df.index:
                            val = df.loc[method, ranking_metric]
                            if pd.notna(val):
                                scores[method].append(float(val))
            # Mean score per method
            mean_scores = {m: np.mean(vals) if vals else -999
                           for m, vals in scores.items()}
            ranked = sorted(mean_scores, key=lambda m: mean_scores[m],
                            reverse=True)[:top_n]
            # Preserve logical order: internals first, then externals
            ranked_internal = [m for m in present_internal if m in ranked]
            ranked_external = [m for m in present_external if m in ranked]
            top_list = ranked_internal + ranked_external
            if top_list:
                groups["topN"] = top_list
    except Exception as exc:
        print(f"  WARNING: Could not build top-N group: {exc}")

    return groups


def visualize_merged_grouped(
    merged_cfg: MergedExperimentConfig,
    *,
    metric_groups: list = None,
    selected_metrics: list = None,
    excluded_metrics: list = None,
    ncols: int = None,
    top_n: int = 10,
    ranking_metric: str = "NMI",
    fig_width_per_metric: float = 3.2,
    plot_type: str = "boxplot",
    palette_name: str = None,
    title_fontsize: float = 11,
    axis_label_fontsize: float = 11,
    tick_label_fontsize: float = 11,
    xtick_label_fontsize: float = None,
    significance_fontsize: float = 13,
    ns_fontsize: float = 10,
    xlabel_rotation: float = None,
    significance_marker_offset: float = -0.04,
    ns_offset: float = 0.0,
    significance_line_width: float = 1.5,
    bar_strip_size: float = 2,
    bar_strip_alpha: float = 0.5,
    font_family: str = "Arial",
    show_legend: bool = False,
    panel_labels: bool = False,
    dpi: int = 300):
    """Generate **multiple** comparison figures, one per logical method group.

    When the merged experiment has too many methods (>METHOD_GROUP_THRESHOLD)
    for a single readable figure, this function splits them into:

      1. **internal** — all 12 proposed PanODE-LAB models
      2. **external** — all external baselines + the focal model as reference
      3. **topN**     — cross-pool top performers ranked by *ranking_metric*

    Each sub-figure is independently sized and laid-out according to its own
    method count, ensuring proper hspace, font sizes, and rotation.
    """
    tables_dir = merged_cfg.tables_dir
    figures_dir = merged_cfg.figures_dir
    all_methods = merged_cfg.method_names

    if not tables_dir.exists() or not list(tables_dir.glob("*.csv")):
        print(f"ERROR: No merged tables found in {tables_dir}")
        sys.exit(1)

    _apply_font(font_family)

    method_groups = _build_method_groups(
        all_methods, tables_dir, top_n=top_n, ranking_metric=ranking_metric)

    print(f"\n{'='*70}")
    print(f"Multi-figure grouped visualisation: {merged_cfg.name}")
    print(f"Total methods: {len(all_methods)}")
    for gname, gmethods in method_groups.items():
        print(f"  {gname:10s} ({len(gmethods):2d}): {gmethods}")
    print(f"{'='*70}\n")

    os.makedirs(figures_dir, exist_ok=True)

    # Build flat metric list (shared across all groups)
    # Load one analyzer briefly just to get available metrics
    _tmp_analyzer = RigorousExperimentalAnalyzer(
        data_folder_path=str(tables_dir),
        method_names=all_methods,   # must match CSV row count
        verbose=False)
    _tmp_analyzer.load_experimental_data()
    _tmp_analyzer.preprocess_data()
    available = set(_tmp_analyzer.metrics)
    del _tmp_analyzer

    groups_to_plot = metric_groups or list(METRIC_GROUPS.keys())
    flat_metrics = []
    flat_display = {}
    for group_key in groups_to_plot:
        if group_key not in METRIC_GROUPS:
            continue
        grp = METRIC_GROUPS[group_key]
        for m in grp["metrics"]:
            if m in available:
                flat_metrics.append(m)
                flat_display[m] = grp["display"].get(m, m)

    if selected_metrics:
        sel = set(selected_metrics)
        flat_metrics = [m for m in flat_metrics if m in sel]
        flat_display = {m: flat_display[m] for m in flat_metrics}
    if excluded_metrics:
        exc = set(excluded_metrics)
        flat_metrics = [m for m in flat_metrics if m not in exc]
        flat_display = {m: flat_display[m] for m in flat_metrics}

    if not flat_metrics:
        print("  No matching metrics — skipping.")
        return

    GROUP_TITLES = {
        "internal": "Proposed Models — Internal Comparison",
        "external": "External Baselines vs. Proposed Focal Model",
        "topN": f"Top-{top_n} Cross-Pool Ranking (by {ranking_metric})",
    }

    all_figure_paths = []

    for group_name, group_methods in method_groups.items():
        n_methods = len(group_methods)
        if n_methods < 2:
            print(f"  Skipping {group_name}: only {n_methods} methods")
            continue

        print(f"\n  ── {GROUP_TITLES.get(group_name, group_name)} "
              f"({n_methods} methods) ──")

        # Build a dedicated REA analyzer for this subset
        # NOTE: method_names must match CSV row count (all methods),
        # selected_methods + method_order filter/reorder the display
        analyzer = RigorousExperimentalAnalyzer(
            data_folder_path=str(tables_dir),
            method_names=all_methods,
            selected_methods=group_methods,
            method_order=group_methods,
            verbose=False)
        analyzer.load_experimental_data()
        analyzer.preprocess_data()

        # Adaptive layout for THIS group's method count
        auto_h, auto_rot, auto_xtick = _adaptive_params(n_methods)
        final_h = auto_h
        final_rot = xlabel_rotation if xlabel_rotation is not None else auto_rot
        final_xtick = (xtick_label_fontsize
                       if xtick_label_fontsize is not None
                       else auto_xtick)
        # Clamp to guaranteed readable minimum
        final_xtick = clamp_xtick_fontsize(final_xtick)
        final_ncols = ncols if ncols is not None else _adaptive_ncols(n_methods)

        palette = sns.color_palette(
            palette_name if palette_name else DEFAULT_PALETTE, n_methods
        )
        sig_pairs = _build_sig_pairs(group_methods)

        computed_hspace = _compute_hspace(
            method_names=group_methods,
            rotation_deg=final_rot,
            xtick_fontsize=final_xtick,
            title_fontsize=title_fontsize,
            per_row_height=final_h)

        # Enforce aspect ratio cap (height/width ≤ 21:17)
        if ncols is None:
            final_ncols = _enforce_max_aspect(
                n_metrics=len(flat_metrics),
                per_row_height=final_h,
                hspace=computed_hspace,
                fig_width_per_metric=fig_width_per_metric,
                ncols_init=final_ncols)

        n_rows = math.ceil(len(flat_metrics) / final_ncols)
        total_w = fig_width_per_metric * final_ncols
        total_h = (final_h * n_rows
                   + final_h * computed_hspace * max(n_rows - 1, 0))

        save_path = figures_dir / f"{group_name}_metrics.pdf"
        sup_title = GROUP_TITLES.get(group_name, group_name)

        print(f"    {len(flat_metrics)} metrics, {final_ncols} cols × "
              f"{n_rows} rows  ({total_w:.1f}×{total_h:.1f} in)")

        try:
            fig, axes = create_publication_figure(
                analyzer,
                metrics=flat_metrics,
                metric_display_names=flat_display,
                figsize=(total_w, total_h),
                ncols=final_ncols,
                show_significance_pairs=sig_pairs,
                palette=palette,
                panel_labels=panel_labels,
                plot_type=plot_type,
                xlabel_rotation=final_rot,
                font_family=font_family,
                significance_line_width=significance_line_width,
                significance_marker_offset=significance_marker_offset,
                bar_strip_size=bar_strip_size,
                bar_strip_alpha=bar_strip_alpha,
                title_fontsize=title_fontsize,
                title_fontweight="normal",
                axis_label_fontsize=axis_label_fontsize,
                tick_label_fontsize=tick_label_fontsize,
                significance_fontsize=significance_fontsize,
                ns_fontsize=ns_fontsize,
                ns_offset=ns_offset,
                show_legend=show_legend,
                dpi=dpi,
                hspace=computed_hspace,
                suptitle=sup_title,
                save_path=str(save_path))
            # Apply xtick fontsize post-hoc
            for ax in axes:
                plt.setp(ax.xaxis.get_majorticklabels(), fontsize=final_xtick)
            fig.savefig(str(save_path), dpi=dpi, facecolor="white",
                        bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            all_figure_paths.append(save_path)
            print(f"    -> {save_path}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Grouped figures saved to: {figures_dir}")
    for p in all_figure_paths:
        print(f"  {p}")
    print(f"{'='*70}\n")
    return all_figure_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Training curve visualisation for merged experiment
# ═══════════════════════════════════════════════════════════════════════════════

def plot_merged_training_curves(
    merged_cfg: MergedExperimentConfig,
    smoothing_window: int = None,
    fill_alpha: float = 0.15,
    line_alpha: float = 0.85,
    per_dataset: bool = False,
    show_recon: bool = False,
    figsize: tuple = None,
    dpi: int = 200,
    font_family: str = "Arial",
    palette_name: str = None):
    """Plot aggregated loss curves for the merged experiment.

    Cross-dataset mean +/- std shading, with separate lines per method.
    """
    _apply_font(font_family)

    series_dir = merged_cfg.series_dir
    figures_dir = merged_cfg.figures_dir / "training_curves"
    os.makedirs(figures_dir, exist_ok=True)

    series_files = sorted(series_dir.glob("*_dfs.csv"))
    if not series_files:
        print(f"No training series found in {series_dir}")
        return

    all_dfs = []
    for sf in series_files:
        ds_name = sf.stem.replace("_dfs", "")
        df = pd.read_csv(sf)
        df["dataset"] = ds_name
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)

    has_val = ("val_loss" in combined.columns
               and combined["val_loss"].notna().any())
    has_recon = ("recon_loss" in combined.columns
                 and combined["recon_loss"].notna().any())
    has_val_recon = ("val_recon_loss" in combined.columns
                     and combined["val_recon_loss"].notna().any())

    methods = merged_cfg.method_names
    # Only keep methods that have series data
    methods_with_data = [m for m in methods if m in combined["hue"].unique()]
    n_methods = len(methods_with_data)

    if n_methods == 0:
        print("  No training series data found for any method.")
        return

    palette_colors = sns.color_palette(
        palette_name if palette_name else DEFAULT_PALETTE, n_methods
    )
    color_map = dict(zip(methods_with_data, palette_colors))

    loss_cols = ["train_loss"]
    panel_titles = ["Training Loss"]
    if has_val:
        loss_cols.append("val_loss")
        panel_titles.append("Validation Loss")
    if show_recon and has_recon:
        loss_cols.append("recon_loss")
        panel_titles.append("Reconstruction Loss")
    if show_recon and has_val_recon:
        loss_cols.append("val_recon_loss")
        panel_titles.append("Val. Reconstruction Loss")

    n_panels = len(loss_cols)
    _figsize = figsize or (7.0 * n_panels, 5.5)  # taller for legend

    fig, axes = plt.subplots(1, n_panels, figsize=_figsize)
    if n_panels == 1:
        axes = [axes]

    for method_name in methods_with_data:
        method_df = combined[combined["hue"] == method_name]
        datasets_in = method_df["dataset"].unique()
        epoch_counts = [
            int(method_df[method_df["dataset"] == d]["epoch"].max())
            for d in datasets_in
        ]
        max_ep = min(epoch_counts)
        epochs = np.arange(1, max_ep + 1)
        w = smoothing_window or max(1, max_ep // 20)

        for pi, col in enumerate(loss_cols):
            matrices = []
            for d in datasets_in:
                sub = method_df[method_df["dataset"] == d].sort_values("epoch")
                vals = sub[col].values[:max_ep]
                if np.isnan(vals).all():
                    continue
                matrices.append(_smooth(vals, w))

            if not matrices:
                continue
            arr = np.array(matrices)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            c = color_map[method_name]
            axes[pi].plot(epochs, mean, color=c, alpha=line_alpha,
                          label=method_name, linewidth=1.0)
            axes[pi].fill_between(epochs, mean - std, mean + std,
                                  color=c, alpha=fill_alpha)

    for ax, title in zip(axes, panel_titles):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

    fig.tight_layout()
    agg_path = figures_dir / "aggregated_loss.pdf"
    fig.savefig(str(agg_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Aggregated loss -> {agg_path}")

    # Per-dataset plots
    if per_dataset:
        for sf in series_files:
            ds_name = sf.stem.replace("_dfs", "")
            df = pd.read_csv(sf)
            ds_cols = ["train_loss"]
            ds_titles = ["Training"]
            if "val_loss" in df.columns and df["val_loss"].notna().any():
                ds_cols.append("val_loss")
                ds_titles.append("Validation")
            if show_recon:
                if "recon_loss" in df.columns:
                    ds_cols.append("recon_loss")
                    ds_titles.append("Recon.")
                if "val_recon_loss" in df.columns:
                    ds_cols.append("val_recon_loss")
                    ds_titles.append("Val. Recon.")

            n_p = len(ds_cols)
            fig_d, axs_d = plt.subplots(1, n_p, figsize=(7.0 * n_p, 4))
            if n_p == 1:
                axs_d = [axs_d]

            for method_name in df["hue"].unique():
                subset = df[df["hue"] == method_name].sort_values("epoch")
                w = smoothing_window or max(1, len(subset) // 20)
                c = color_map.get(method_name, None)
                for pi, col in enumerate(ds_cols):
                    sm = _smooth(subset[col].values, w)
                    axs_d[pi].plot(subset["epoch"], sm, label=method_name,
                                   alpha=0.8, color=c, linewidth=0.8)

            for ax, t in zip(axs_d, ds_titles):
                ax.set(xlabel="Epoch", ylabel="Loss",
                       title=f"{ds_name} — {t}")
                ax.legend(fontsize=5, loc="upper right", ncol=2)
                ax.grid(True, alpha=0.3)
                sns.despine(ax=ax)

            fig_d.tight_layout()
            save_p = figures_dir / f"{ds_name}_loss.pdf"
            fig_d.savefig(str(save_p), dpi=dpi, bbox_inches="tight")
            plt.close(fig_d)

    print(f"  Training curves saved to: {figures_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Merge internal + external benchmark results and visualise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: merge + visualise
  python -m experiments.merge_and_visualize

  # Merge tables only (no figures)
  python -m experiments.merge_and_visualize --merge-only

  # Figures only (tables already merged)
  python -m experiments.merge_and_visualize --figures-only

  # Summary statistics only
  python -m experiments.merge_and_visualize --summary-only

  # Override external experiment name
  python -m experiments.merge_and_visualize --external-name my_baselines

  # Specific metric groups
  python -m experiments.merge_and_visualize --groups clustering drex lsex

  # Training curves
  python -m experiments.merge_and_visualize --training-curves

  # Custom merged name
  python -m experiments.merge_and_visualize --merged-name dpmm_vs_baselines
        """)

    # ── Pipeline mode flags ───────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--merge-only", action="store_true",
                            help="Only merge tables, skip visualisation")
    mode_group.add_argument("--figures-only", action="store_true",
                            help="Skip merge, only generate figures "
                                 "(assumes tables already exist)")
    mode_group.add_argument("--summary-only", action="store_true",
                            help="Merge + print statistics, no figures")

    # ── Source names ──────────────────────────────────────────────────────
    parser.add_argument("--internal-name", type=str,
                        default=DEFAULT_INTERNAL_DIR,
                        help=f"Internal experiment name "
                             f"(default: {DEFAULT_INTERNAL_DIR})")
    parser.add_argument("--external-name", type=str,
                        default=DEFAULT_EXTERNAL_DIR,
                        help=f"External experiment name "
                             f"(default: {DEFAULT_EXTERNAL_DIR})")
    parser.add_argument("--merged-name", type=str,
                        default=DEFAULT_MERGED_NAME,
                        help=f"Merged output name "
                             f"(default: {DEFAULT_MERGED_NAME})")
    parser.add_argument("--output-root", type=str,
                        default=str(DEFAULT_OUTPUT_ROOT),
                        help="Output root directory")

    # ── Method selection ──────────────────────────────────────────────────
    parser.add_argument("--internal-methods", nargs="+", default=None,
                        help="Internal methods to include (default: all 12)")
    parser.add_argument("--external-methods", nargs="+", default=None,
                        help="External methods to include (default: all 15)")

    # ── Metric selection ──────────────────────────────────────────────────
    parser.add_argument("--groups", nargs="+", default=None,
                        help="Metric groups to plot "
                             f"(default: all; available: "
                             f"{list(METRIC_GROUPS.keys())})")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Specific metric names to include")
    parser.add_argument("--exclude-metrics", nargs="+", default=None,
                        help="Metric names to exclude")

    # ── Visual parameters ─────────────────────────────────────────────────
    parser.add_argument("--per-group", action="store_true",
                        help="One PDF per metric group (default: unified grid)")
    parser.add_argument("--grouped", action="store_true", default=None,
                        help="Split methods into logical sub-figures "
                             "(internal / external / top-N). Auto-enabled "
                             f"when n_methods > {METHOD_GROUP_THRESHOLD}")
    parser.add_argument("--no-grouped", dest="grouped", action="store_false",
                        help="Force single unified figure even if many methods")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top methods for the cross-pool figure "
                             "(default: 10)")
    parser.add_argument("--ranking-metric", type=str, default="NMI",
                        help="Metric for ranking in top-N figure (default: NMI)")
    parser.add_argument("--ncols", type=int, default=None,
                        help="Override column count")
    parser.add_argument("--fig-height", type=float, default=None,
                        help="Per-row figure height")
    parser.add_argument("--fig-width", type=float, default=3.2,
                        help="Width per metric column (default: 3.2)")
    parser.add_argument("--plot-type", type=str, default="boxplot",
                        choices=["boxplot", "violin", "strip", "barplot",
                                 "paired_lines"],
                        help="Plot type (default: boxplot)")
    parser.add_argument("--palette", type=str, default=None,
                        help="Seaborn palette name")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure DPI (default: 300)")
    parser.add_argument("--xlabel-rotation", type=float, default=None,
                        help="X-label rotation degrees")
    parser.add_argument("--xtick-fontsize", type=float, default=None,
                        help="X-tick label fontsize")
    parser.add_argument("--show-legend", action="store_true",
                        help="Show figure legend")

    # ── Training curves ───────────────────────────────────────────────────
    parser.add_argument("--training-curves", action="store_true",
                        help="Generate training/validation loss curves")
    parser.add_argument("--show-recon", action="store_true",
                        help="Include reconstruction loss panels")
    parser.add_argument("--per-dataset-curves", action="store_true",
                        help="Generate per-dataset loss plots")
    parser.add_argument("--smoothing-window", type=int, default=None,
                        help="Loss curve smoothing window")

    args = parser.parse_args()
    output_root = Path(args.output_root)

    # ── Build merged config ───────────────────────────────────────────────
    merged_cfg = build_merged_config(
        internal_name=args.internal_name,
        external_name=args.external_name,
        merged_name=args.merged_name,
        output_root=output_root,
        internal_methods=args.internal_methods,
        external_methods=args.external_methods)

    print(f"\n{merged_cfg.summary()}\n")

    # ── Step 1: Merge tables ──────────────────────────────────────────────
    if not args.figures_only:
        print("=" * 70)
        print("Step 1: Merging result tables")
        print("=" * 70)
        merged_cfg.build_merged_tables()

    # Early exit for merge-only
    if args.merge_only:
        print("\nMerge complete. Use --figures-only to generate figures later.")
        return

    # ── Step 2: Visualise metrics ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 2: Generating comparison figures")
    print("=" * 70)

    n_total_methods = len(merged_cfg.method_names)

    # Decide grouped vs unified mode
    use_grouped = args.grouped
    if use_grouped is None:
        # Auto-detect: enable grouped mode when too many methods
        use_grouped = n_total_methods > METHOD_GROUP_THRESHOLD
        if use_grouped:
            print(f"\n  Auto-grouped mode: {n_total_methods} methods "
                  f"> threshold ({METHOD_GROUP_THRESHOLD})")

    if use_grouped and not args.summary_only:
        visualize_merged_grouped(
            merged_cfg,
            metric_groups=args.groups,
            selected_metrics=args.metrics,
            excluded_metrics=args.exclude_metrics,
            ncols=args.ncols,
            top_n=args.top_n,
            ranking_metric=args.ranking_metric,
            fig_width_per_metric=args.fig_width,
            plot_type=args.plot_type,
            palette_name=args.palette,
            xtick_label_fontsize=args.xtick_fontsize,
            xlabel_rotation=args.xlabel_rotation,
            show_legend=args.show_legend,
            dpi=args.dpi)
    else:
        visualize_merged(
            merged_cfg,
            summary_only=args.summary_only,
            metric_groups=args.groups,
            selected_metrics=args.metrics,
            excluded_metrics=args.exclude_metrics,
            per_group=args.per_group,
            ncols=args.ncols,
            fig_height=args.fig_height,
            fig_width_per_metric=args.fig_width,
            plot_type=args.plot_type,
            palette_name=args.palette,
            xtick_label_fontsize=args.xtick_fontsize,
            xlabel_rotation=args.xlabel_rotation,
            show_legend=args.show_legend,
            dpi=args.dpi)

    # ── Step 3 (optional): Training curves ────────────────────────────────
    if args.training_curves:
        print("\n" + "=" * 70)
        print("Step 3: Generating training curves")
        print("=" * 70)

        plot_merged_training_curves(
            merged_cfg,
            smoothing_window=args.smoothing_window,
            per_dataset=args.per_dataset_curves,
            show_recon=args.show_recon,
            palette_name=args.palette,
            dpi=args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
