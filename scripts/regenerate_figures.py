#!/usr/bin/env python
"""
Series-aware figure regenerator for PanODE-Topic.

Generates publication-quality **per-group** figures (one wide, single-row
PDF per metric group) for every experiment, and organises results into
the Topic-FM article series:

    Topic series ─  topic/ablation + topic/vs_external

Also regenerates per-group figures for the mixed full_comparison and
full_vs_external experiments under ``mixed/``.

Directory layout (article-oriented)::

    experiments/results/
    ├── dpmm/
    │   ├── ablation/       ← DPMM component ablation
    │   └── vs_external/    ← DPMM vs 15 external baselines
    ├── topic/
    │   ├── ablation/       ← Topic component ablation
    │   └── vs_external/    ← Topic vs 15 external baselines
    ├── external/           ← external benchmark raw results
    └── mixed/
        ├── full_comparison/
        └── full_vs_external/

Semantic splitting
------------------
When vs_external experiments have > 10 methods, they are automatically
split into 3 panels:

    proposed    (4)  — ablation variants
    classical   (7)  — generative/reconstruction baselines + focal
    deep        (10) — deep clustering/graph baselines + focal

Usage
-----
    python scripts/regenerate_figures.py              # everything
    python scripts/regenerate_figures.py --series dpmm
    python scripts/regenerate_figures.py --series topic
    python scripts/regenerate_figures.py --experiment full_comparison
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
import seaborn as sns

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_lib.viz.rea import (
    RigorousExperimentalAnalyzer,
    create_publication_figure,
    _apply_font,
    _safe_save_figure)
from eval_lib.viz.layout import (
    FIG_WIDTH_PER_METRIC,
    MAX_METHODS_PER_FIGURE,
    adaptive_params,
    per_row_height,
    rotation_fontsize,
    clamp_aspect_ratio,
    compute_hspace,
    clamp_xtick_fontsize,
    needs_method_split,
    assert_no_label_overlap)
from eval_lib.baselines.registry import CLASSICAL_BASELINES, DEEP_GRAPH_BASELINES
from eval_lib.experiment.merge import MergedExperimentConfig
from eval_lib.metrics.battery import PUBLICATION_METRIC_GROUPS as METRIC_GROUPS

# Font-size constants from the subplot style system (used scaled-down
# for composed per-group figures).
sys.path.insert(0, str(Path(PROJECT_ROOT) / "benchmarks"))
from figure_generators.subplot_style import (   # noqa: E402
    FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_LEGEND)

# ── Distinct color themes for semantic split groups (Figures 10–12) ──────────
# Import theme palettes from the subplot_style module used by the Next.js
# figure pipeline.  Each semantic group (proposed / classical / deep) gets
# a visually distinct palette so readers can immediately tell panels apart.
try:
    from figure_generators.subplot_style import get_theme_palette
except ImportError:
    # Fallback if benchmarks package is not on sys.path
    def get_theme_palette(theme_name: str) -> list:  # type: ignore[misc]
        return []

DEFAULT_PALETTE = "husl"
RESULTS_ROOT = Path(PROJECT_ROOT) / "experiments" / "results"

# Short display names for x-tick labels to prevent row overlap in grid figures.
# Full names are kept in method_names (data columns); only the rendered ticks
# are shortened.
TICK_SHORT_NAMES: dict[str, str] = {
    "Topic-FM-Base":        "TFM-Base",
    "Topic-FM-Transformer": "TFM-Trans",
    "Topic-FM-Contrastive": "TFM-Contr",
    "Pure-Trans-VAE":       "Pure-T-VAE",
    "Pure-Contr-VAE":       "Pure-C-VAE",
}


def _shorten_tick_labels(axes):
    """Replace long method names on x-tick labels with shorter versions."""
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        new_labels = []
        changed = False
        for lab in ax.get_xticklabels():
            txt = lab.get_text()
            short = TICK_SHORT_NAMES.get(txt, txt)
            if short != txt:
                changed = True
            new_labels.append(short)
        if changed:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(new_labels)

def _per_row_height(n_methods: int) -> float:
    """Thin wrapper over layout.per_row_height for local readability."""
    return per_row_height(n_methods)


def _rotation_fontsize(n_methods: int) -> tuple[float, float]:
    """Return (rotation_deg, xtick_fontsize) for per-group figures."""
    return rotation_fontsize(n_methods, per_group=True)


def _compute_hspace(method_names, rotation_deg, xtick_fontsize,
                    title_fontsize, per_row_height):
    """Delegate to the shared layout.compute_hspace helper."""
    return compute_hspace(
        method_names=method_names,
        rotation_deg=rotation_deg,
        xtick_fontsize=xtick_fontsize,
        title_fontsize=title_fontsize,
        per_row_h=per_row_height)


def _build_sig_pairs(method_names):
    if len(method_names) < 2:
        return []
    focal = method_names[-1]
    return [(m, focal) for m in method_names[:-1]]


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic method splitting — 3-group strategy
# ═══════════════════════════════════════════════════════════════════════════════

# Taxonomy of external baselines into two semantic tiers.
# classical   — generative / reconstruction / geometric / disentanglement (14 methods)
# deep_graph  — deep clustering, graph, contrastive, and scVI-family (12 methods)
# Imported from eval_lib.baselines.registry: CLASSICAL_BASELINES, DEEP_GRAPH_BASELINES


def split_methods_semantically(
    method_names: list[str],
    focal_method: str | None = None,
    *,
    max_per_panel: int = MAX_METHODS_PER_FIGURE) -> dict[str, list[str]]:
    """Split *method_names* into ≤ 3 semantic groups for multi-figure output.

    **Groups produced** (when splitting is triggered):

    +--------------+----------------------------------------------+----------+
    | Group tag    | Content                                      | Typical  |
    +==============+==============================================+==========+
    | ``proposed`` | ablation / internal methods (family variants) | 4        |
    +--------------+----------------------------------------------+----------+
    | ``classical``| generative / reconstruction ext. baselines   | 6 + focal|
    +--------------+----------------------------------------------+----------+
    | ``deep``     | deep clustering / graph / prob. ext.baselines| 9 + focal|
    +--------------+----------------------------------------------+----------+

    The *focal_method* is **appended last** in every group (so that
    significance pairs are always ``(baseline, focal)``).  The ``proposed``
    group already contains the focal method in natural order; the two
    external groups get it as an anchor for cross-group comparison.

    When the total method count fits within *max_per_panel* the input is
    returned as a single ``"all"`` group with no splitting.
    """
    if not needs_method_split(method_names, max_methods_per_panel=max_per_panel):
        return {"all": list(method_names)}

    classical_set = set(CLASSICAL_BASELINES)
    deep_set = set(DEEP_GRAPH_BASELINES)
    ext_set = classical_set | deep_set

    # Separate internal (proposed) vs external
    internals = [m for m in method_names if m not in ext_set]
    classicals = [m for m in method_names if m in classical_set]
    deeps = [m for m in method_names if m in deep_set]

    groups: dict[str, list[str]] = {}

    # 1. Proposed / internal group (ablation variants)
    if internals:
        ordered = [m for m in internals if m != focal_method]
        if focal_method and focal_method in internals:
            ordered.append(focal_method)
        groups["proposed"] = ordered

    # 2. Classical baselines + focal anchor
    if classicals:
        ordered = list(classicals)
        if focal_method and focal_method not in ordered:
            ordered.append(focal_method)
        groups["classical"] = ordered

    # 3. Deep / graph baselines + focal anchor
    if deeps:
        ordered = list(deeps)
        if focal_method and focal_method not in ordered:
            ordered.append(focal_method)
        groups["deep"] = ordered

    # Safety: if for some reason we still have a group > max_per_panel,
    # split that group into equal-sized sub-chunks.
    final: dict[str, list[str]] = {}
    for tag, methods in groups.items():
        if len(methods) <= max_per_panel:
            final[tag] = methods
        else:
            n_chunks = math.ceil(len(methods) / max_per_panel)
            chunk_sz = math.ceil(len(methods) / n_chunks)
            for i in range(n_chunks):
                chunk = methods[i * chunk_sz : (i + 1) * chunk_sz]
                if focal_method and focal_method not in chunk:
                    chunk.append(focal_method)
                final[f"{tag}_{i+1}"] = chunk

    return final


# ═══════════════════════════════════════════════════════════════════════════════
# Per-group figure generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_per_group_figures(
    tables_dir: str,
    figures_dir: str,
    method_names: list[str],
    title_fontsize: float = FONTSIZE_LABEL,
    dpi: int = 300,
    palette_name: str | None = None,
    save_png: bool = True,
    all_method_names: list[str] | None = None):
    """Generate one wide, single-row PDF per metric group.

    This produces figures with the CORRECT publication ratio: each metric
    group gets exactly ``ncols = len(group_metrics)`` columns and 1 row,
    matching the reference per-group style.  Aspect-ratio clamping and
    label-overlap assertions are applied automatically.

    Parameters
    ----------
    all_method_names : list[str] | None
        If given, this is the **full** ordered list of methods present in
        the CSV tables.  ``method_names`` is then treated as the subset
        to display.  Required when operating on a semantically-split
        sub-group whose source tables contain methods from all groups.
    """
    n_methods = len(method_names)
    row_h = _per_row_height(n_methods)
    rot, xtick_fs = _rotation_fontsize(n_methods)
    # Make sure x-tick labels are never smaller than y-ticks (11 pt),
    # while still respecting the global minimum from layout.
    xtick_fs = max(clamp_xtick_fontsize(xtick_fs, per_group=True), 11.0)
    palette = sns.color_palette(palette_name or DEFAULT_PALETTE, n_methods)
    sig_pairs = _build_sig_pairs(method_names)
    # Allow richer comparisons when the panel is small; gently cap for
    # larger panels to avoid a forest of brackets.
    if n_methods <= 5:
        max_sig = min(len(sig_pairs), 6)
    elif n_methods <= 10:
        max_sig = min(len(sig_pairs), 5)
    else:
        max_sig = min(len(sig_pairs), 5)

    _apply_font("Arial")
    os.makedirs(figures_dir, exist_ok=True)

    # Build display labels for spacing calculations
    display_labels = [TICK_SHORT_NAMES.get(m, m) for m in method_names]

    # When all_method_names is provided the CSVs contain more methods than
    # what we want to plot.  Pass the full list to the analyzer and select
    # a subset for display.
    if all_method_names is not None:
        analyzer = RigorousExperimentalAnalyzer(
            data_folder_path=tables_dir,
            method_names=all_method_names,
            selected_methods=method_names,
            method_order=method_names,
            method_display_names=TICK_SHORT_NAMES,
            verbose=False)
    else:
        analyzer = RigorousExperimentalAnalyzer(
            data_folder_path=tables_dir,
            method_names=method_names,
            method_order=method_names,
            method_display_names=TICK_SHORT_NAMES,
            verbose=False)
    analyzer.load_experimental_data()
    analyzer.preprocess_data()

    available = set(analyzer.metrics)
    generated = []

    for group_key, grp in METRIC_GROUPS.items():
        grp_metrics = [m for m in grp["metrics"] if m in available]
        if not grp_metrics:
            continue

        grp_display = {m: grp["display"].get(m, m) for m in grp_metrics}
        ncols = len(grp_metrics)
        fig_w = FIG_WIDTH_PER_METRIC * ncols
        fig_h = clamp_aspect_ratio(fig_w, row_h)

        # Safety: warn if labels are likely to overlap
        assert_no_label_overlap(
            display_labels, fig_w, ncols, xtick_fs, rot)

        hspace = _compute_hspace(display_labels, rot, xtick_fs,
                                 title_fontsize, fig_h)

        save_path = Path(figures_dir) / f"{group_key}.pdf"
        print(f"  {group_key:20s}: {ncols} metrics, "
              f"{fig_w:.1f}x{fig_h:.1f} in, rot={rot}°")

        try:
            fig, axes = create_publication_figure(
                analyzer,
                metrics=grp_metrics,
                metric_display_names=grp_display,
                method_display_names=TICK_SHORT_NAMES,
                figsize=(fig_w, fig_h),
                ncols=ncols,
                save_path=str(save_path),
                show_significance_pairs=sig_pairs,
                max_significance_pairs=max_sig,
                palette=palette,
                panel_labels=False,
                plot_type="boxplot",
                xlabel_rotation=rot,
                font_family="Arial",
                significance_line_width=1.5,
                significance_marker_offset=-0.04,
                bar_strip_size=2,
                bar_strip_alpha=0.5,
                title_fontsize=title_fontsize,
                title_fontweight="normal",
                axis_label_fontsize=FONTSIZE_TICK,
                tick_label_fontsize=FONTSIZE_LEGEND,
                significance_fontsize=13,
                ns_fontsize=10,
                ns_offset=0.0,
                show_legend=False,
                dpi=dpi,
                hspace=hspace)
            for ax in axes:
                plt.setp(ax.xaxis.get_majorticklabels(), fontsize=xtick_fs)
            _safe_save_figure(fig, save_path, dpi=dpi)

            if save_png:
                png_path = save_path.with_suffix(".png")
                _safe_save_figure(fig, png_path, dpi=min(dpi, 200))

            plt.close(fig)
            generated.append(save_path)
            print(f"    -> {save_path}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()

    return generated


# ═══════════════════════════════════════════════════════════════════════════════
# Uniform grid figure generation
# ═══════════════════════════════════════════════════════════════════════════════

UNIFORM_NCOLS = 6  # columns per row for the uniform boxplot grid


def generate_uniform_grid_figure(
    tables_dir: str,
    figures_dir: str,
    method_names: list[str],
    title_fontsize: float = FONTSIZE_LABEL,
    dpi: int = 300,
    palette_name: str | None = None,
    save_png: bool = True,
    all_method_names: list[str] | None = None):
    """Generate a single figure with ALL metrics in a uniform NxM grid.

    Unlike ``generate_per_group_figures`` which creates one PDF per metric
    group (causing uneven row widths), this function places every metric
    into a fixed-column grid so every row has the same number of subpanels.
    """
    import math

    n_methods = len(method_names)
    row_h = _per_row_height(n_methods)
    rot, xtick_fs = _rotation_fontsize(n_methods)
    xtick_fs = max(clamp_xtick_fontsize(xtick_fs, per_group=True), 11.0)
    palette = sns.color_palette(palette_name or DEFAULT_PALETTE, n_methods)
    sig_pairs = _build_sig_pairs(method_names)
    max_sig = min(len(sig_pairs), 6 if n_methods <= 5 else 5)

    _apply_font("Arial")
    os.makedirs(figures_dir, exist_ok=True)

    # Build display names: use SHORT labels for ticks, FULL names for data
    display_labels = [TICK_SHORT_NAMES.get(m, m) for m in method_names]

    # Build analyzer with native display name support
    if all_method_names is not None:
        analyzer = RigorousExperimentalAnalyzer(
            data_folder_path=tables_dir,
            method_names=all_method_names,
            selected_methods=method_names,
            method_order=method_names,
            method_display_names=TICK_SHORT_NAMES,
            verbose=False)
    else:
        analyzer = RigorousExperimentalAnalyzer(
            data_folder_path=tables_dir,
            method_names=method_names,
            method_order=method_names,
            method_display_names=TICK_SHORT_NAMES,
            verbose=False)
    analyzer.load_experimental_data()
    analyzer.preprocess_data()

    available = set(analyzer.metrics)

    # Flatten all metrics from all groups into a single ordered list
    all_metrics = []
    all_display = {}
    for grp in METRIC_GROUPS.values():
        for m in grp["metrics"]:
            if m in available and m not in all_metrics:
                all_metrics.append(m)
                all_display[m] = grp["display"].get(m, m)

    if not all_metrics:
        print("  SKIP: no metrics available")
        return []

    ncols = UNIFORM_NCOLS
    nrows = math.ceil(len(all_metrics) / ncols)
    fig_w = FIG_WIDTH_PER_METRIC * ncols
    fig_h = row_h * nrows + 3.0

    # Compute hspace using SHORT display labels for accurate spacing
    hspace = _compute_hspace(display_labels, rot, xtick_fs,
                             title_fontsize, row_h)

    save_path = Path(figures_dir) / "uniform_grid.pdf"
    print(f"  uniform_grid: {len(all_metrics)} metrics, "
          f"{ncols}x{nrows} grid, {fig_w:.1f}x{fig_h:.1f} in, rot={rot}°")

    try:
        fig, axes = create_publication_figure(
            analyzer,
            metrics=all_metrics,
            metric_display_names=all_display,
            method_display_names=TICK_SHORT_NAMES,
            figsize=(fig_w, fig_h),
            ncols=ncols,
            save_path=str(save_path),
            show_significance_pairs=sig_pairs,
            max_significance_pairs=max_sig,
            palette=palette,
            panel_labels=False,
            plot_type="boxplot",
            xlabel_rotation=rot,
            font_family="Arial",
            significance_line_width=1.5,
            significance_marker_offset=-0.04,
            bar_strip_size=2,
            bar_strip_alpha=0.5,
            title_fontsize=title_fontsize,
            title_fontweight="normal",
            axis_label_fontsize=FONTSIZE_TICK,
            tick_label_fontsize=FONTSIZE_LEGEND,
            significance_fontsize=13,
            ns_fontsize=10,
            ns_offset=0.0,
            show_legend=False,
            dpi=dpi,
            hspace=hspace,
            wspace=0.25)
        for ax in (axes if hasattr(axes, '__iter__') else [axes]):
            plt.setp(ax.xaxis.get_majorticklabels(), fontsize=xtick_fs)
        _safe_save_figure(fig, save_path, dpi=dpi)
        if save_png:
            _safe_save_figure(fig, save_path.with_suffix(".png"),
                              dpi=min(dpi, 200))
        plt.close(fig)
        print(f"    -> {save_path}")
        return [save_path]
    except Exception as exc:
        print(f"    ERROR: {exc}")
        traceback.print_exc()
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Method constants
# ═══════════════════════════════════════════════════════════════════════════════

DPMM_ABLATION_METHODS = [
    "Pure-AE", "DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
]

TOPIC_ABLATION_METHODS = [
    "Pure-VAE", "Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive",
]

FULL_COMPARISON_METHODS = [
    "Pure-AE", "Pure-Trans-AE", "Pure-Contr-AE",
    "Pure-VAE", "Pure-Trans-VAE", "Pure-Contr-VAE",
    "DPMM-Base", "DPMM-Trans", "DPMM-Contr",
    "Topic-FM-Base", "Topic-FM-Trans", "Topic-FM-Contr",
]

EXTERNAL_METHODS = [
    "CellBLAST", "GMVAE", "GMVAE-Poincare", "GMVAE-PGM",
    "GMVAE-LearnablePGM", "GMVAE-HW", "SCALEX", "scDiffusion", "siVAE",
    "VAE-DIP", "VAE-TC", "InfoVAE", "BetaVAE",
    "CLEAR", "scDAC", "scDeepCluster", "scDHMap", "scGNN",
    "scGCC", "scSMD", "scVI", "PeakVI", "PoissonVI",
]

DPMM_FOCAL = "DPMM-Contrastive"
TOPIC_FOCAL = "Topic-FM-Contrastive"


# ═══════════════════════════════════════════════════════════════════════════════
# Series merge + generate
# ═══════════════════════════════════════════════════════════════════════════════

def _order_for_comparison(
    ablation_methods: list[str],
    focal: str) -> list[str]:
    """Order methods for comparison: externals first, then ablation with focal last."""
    internals_except_focal = [m for m in ablation_methods if m != focal]
    return EXTERNAL_METHODS + internals_except_focal + [focal]


def build_series_merge(
    series_name: str,
    ablation_methods: list[str],
    ablation_dir: str,
    focal: str,
    external_dir: str = "external_full") -> tuple[MergedExperimentConfig, list[str]]:
    """Build a MergedExperimentConfig for <series>_vs_external.

    Returns (config, ordered_method_names) where the focal method is last.
    """
    ordered = _order_for_comparison(ablation_methods, focal)
    cfg = MergedExperimentConfig(
        name=f"{series_name}_vs_external",
        sources=[
            {
                "tables": str(RESULTS_ROOT / external_dir / "tables"),
                "series": str(RESULTS_ROOT / external_dir / "series"),
                "methods": EXTERNAL_METHODS,
            },
            {
                "tables": str(RESULTS_ROOT / ablation_dir / "tables"),
                "series": str(RESULTS_ROOT / ablation_dir / "series"),
                "methods": ablation_methods,
            },
        ],
        output_root=RESULTS_ROOT,
        description=(
            f"{series_name.upper()} ablation ({len(ablation_methods)} methods) "
            f"vs {len(EXTERNAL_METHODS)} external baselines "
            f"[focal: {focal}]"
        ))
    return cfg, ordered


def process_experiment(
    name: str,
    tables_dir: Path,
    figures_dir: Path,
    method_names: list[str],
    focal_method: str | None = None,
    dpi: int = 300,
    save_png: bool = True):
    """Generate per-group figures for a single experiment.

    When the method count exceeds ``MAX_METHODS_PER_FIGURE``, the methods
    are automatically split into semantically meaningful sub-figures
    (proposed / classical / deep) via ``split_methods_semantically``.
    Each sub-figure is written into a subfolder of *figures_dir*.
    """
    if not tables_dir.exists() or not list(tables_dir.glob("*.csv")):
        print(f"  SKIP {name}: no tables at {tables_dir}")
        return

    groups = split_methods_semantically(
        method_names, focal_method=focal_method)

    # When we split, sub-groups are subsets of the full method list.
    # The CSVs still contain all methods, so pass the full list for indexing.
    is_split = len(groups) > 1 or "all" not in groups

    for group_tag, group_methods in groups.items():
        tag_suffix = f" [{group_tag}]" if group_tag != "all" else ""
        sub_figures = (
            figures_dir / group_tag if group_tag != "all" else figures_dir
        )
        print(f"\n{'='*70}")
        print(f"  {name}{tag_suffix} ({len(group_methods)} methods)")
        print(f"  Methods: {group_methods}")
        print(f"{'='*70}")

        # TODO(color-themes): When the seaborn palette is replaced with
        # explicit hex lists, pass theme_colors to generate_per_group_figures
        # via a new `palette_colors` parameter so that each semantic group
        # (proposed / classical / deep) uses its own distinct color scheme.
        # The palettes are defined in benchmarks.figure_generators.subplot_style
        # and retrieved via get_theme_palette(group_tag).
        theme_colors = get_theme_palette(group_tag)  # noqa: F841 — reserved for future palette_colors param

        generate_uniform_grid_figure(
            tables_dir=str(tables_dir),
            figures_dir=str(sub_figures),
            method_names=group_methods,
            dpi=dpi,
            save_png=save_png,
            all_method_names=method_names if is_split else None)


def process_series(series_name: str, dpi: int = 300, save_png: bool = True):
    """Process one article series (Topic): ablation + vs_external."""
    ablation_methods = TOPIC_ABLATION_METHODS
    ablation_dir = "topic/ablation"
    focal = TOPIC_FOCAL

    print(f"\n{'#'*70}")
    print(f"  SERIES: {series_name.upper()}")
    print(f"  Focal method: {focal}")
    print(f"{'#'*70}")

    # 1. Ablation figures
    process_experiment(
        name=f"{series_name}_ablation",
        tables_dir=RESULTS_ROOT / ablation_dir / "tables",
        figures_dir=RESULTS_ROOT / ablation_dir / "figures",
        method_names=ablation_methods,
        focal_method=focal,
        dpi=dpi,
        save_png=save_png)

    # 2. VS external — merge then generate
    merge_cfg, ordered_methods = build_series_merge(
        series_name=series_name,
        ablation_methods=ablation_methods,
        ablation_dir=ablation_dir,
        focal=focal)
    print(f"\n{merge_cfg.summary()}")
    merge_cfg.build_merged_tables()

    process_experiment(
        name=f"{series_name}_vs_external",
        tables_dir=merge_cfg.tables_dir,
        figures_dir=merge_cfg.figures_dir,
        method_names=ordered_methods,
        focal_method=focal,
        dpi=dpi,
        save_png=save_png)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate per-group publication figures for PanODE-Topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/regenerate_figures.py                          # everything
  python scripts/regenerate_figures.py --series dpmm            # DPMM series only
  python scripts/regenerate_figures.py --series topic           # Topic series only
  python scripts/regenerate_figures.py --experiment full_comparison
  python scripts/regenerate_figures.py --no-png --dpi 400       # PDF only, hi-res
        """)
    parser.add_argument("--series", type=str, nargs="+", default=["topic"],
                        choices=["topic"],
                        help="Article series to regenerate (default: topic)")
    parser.add_argument("--experiment", type=str, nargs="+", default=None,
                        choices=["full_comparison", "full_vs_external"],
                        help="Mixed experiments to also regenerate")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (default: 300)")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip PNG generation (PDF only)")
    parser.add_argument("--compose", action="store_true",
                        help="After regenerating, compose per-group PDFs into a multi-panel paper layout")
    parser.add_argument(
        "--compose-layout",
        type=str,
        default="auto",
        choices=["auto", "paper", "portrait_3x2", "landscape_2x3"],
        help="Layout used by --compose (default: auto; ablation=portrait_3x2, others=paper)")
    args = parser.parse_args()

    save_png = not args.no_png
    run_all = args.series is None and args.experiment is None

    # Series
    series_list = args.series or (["topic"] if run_all else [])
    for s in series_list:
        process_series(s, dpi=args.dpi, save_png=save_png)

    # Mixed experiments
    experiments = args.experiment or (
        ["full_comparison", "full_vs_external"] if run_all else []
    )
    for exp_name in experiments:
        if exp_name == "full_comparison":
            process_experiment(
                name="full_comparison",
                tables_dir=RESULTS_ROOT / "mixed" / "full_comparison" / "tables",
                figures_dir=RESULTS_ROOT / "mixed" / "full_comparison" / "figures",
                method_names=FULL_COMPARISON_METHODS,
                focal_method="DPMM-Contr",  # reference only
                dpi=args.dpi,
                save_png=save_png)
        elif exp_name == "full_vs_external":
            fve_tables = RESULTS_ROOT / "mixed" / "full_vs_external" / "tables"
            if fve_tables.exists():
                process_experiment(
                    name="full_vs_external",
                    tables_dir=fve_tables,
                    figures_dir=RESULTS_ROOT / "mixed" / "full_vs_external" / "figures",
                    method_names=FULL_COMPARISON_METHODS + EXTERNAL_METHODS,
                    focal_method="DPMM-Contr",
                    dpi=args.dpi,
                    save_png=save_png)

    # Optional: compose per-group PDFs into a multi-panel figure
    if args.compose:
        import subprocess
        compose_script = Path(PROJECT_ROOT) / "scripts" / "compose_experiment_figures.py"
        try:
            print(f"\n--- Composing multi-panel figures ({args.compose_layout}) ---")
            subprocess.run(
                [
                    sys.executable,
                    str(compose_script),
                    "--all",
                    "--layout",
                    args.compose_layout,
                    "--dpi",
                    str(args.dpi),
                ],
                cwd=PROJECT_ROOT,
                check=False)
        except Exception as e:
            print(f"  Compose skipped: {e}")

    print("\n\nDone. All figures regenerated in per-group style.\n")


if __name__ == "__main__":
    main()
