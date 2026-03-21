#!/usr/bin/env python
"""
Experiment Visualisation Template
====================================

Project-agnostic visualisation runner using eval_lib.viz.rea for statistical
analysis and publication-quality figures.  Copy into your project's
``experiments/`` directory and customise only the ``# ═══ PROJECT-SPECIFIC``
sections.

Portable sections (do NOT modify):
  - Layout helper functions
  - visualize() main function
  - plot_training_curves()
  - CLI argument parser

Project-specific sections (MUST be customised):
  - PRESETS import (from your experiment_config.py)
  - DEFAULT_PALETTE choice (cosmetic only)

Usage
-----
    python -m experiments.visualize_experiment --preset my_ablation
    python -m experiments.visualize_experiment --preset my_ablation --summary-only
    python -m experiments.visualize_experiment --preset my_ablation --training-curves
"""

import argparse
import math
import os
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── eval_lib imports (portable) ───────────────────────────────────────────────
from eval_lib.viz.rea import (
    RigorousExperimentalAnalyzer,
    create_publication_figure,
    _apply_font)
from eval_lib.experiment.merge import MergedExperimentConfig
from eval_lib.experiment.config import ExperimentConfig
from eval_lib.metrics.battery import METRIC_GROUPS

try:
    from src.visualization import save_with_vcd, apply_style, style_axes
except ImportError:
    save_with_vcd = None

# ── Project imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PROJECT-SPECIFIC: Import PRESETS from your experiment_config
# ═══════════════════════════════════════════════════════════════════════════════

# TODO: Import your PRESETS dict.
# from experiments.experiment_config import PRESETS
PRESETS = {}  # placeholder


# ═══════════════════════════════════════════════════════════════════════════════
# ██  PROJECT-SPECIFIC: Default palette (cosmetic, safe to change)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PALETTE = "husl"


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: Layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _adaptive_ncols(n_methods: int) -> int:
    if n_methods <= 3:
        return 10
    elif n_methods <= 5:
        return 8
    elif n_methods <= 8:
        return 6
    else:
        return 5


def _adaptive_params(n_methods: int) -> tuple:
    if n_methods <= 3:
        return 3.0, 25, 11
    elif n_methods <= 6:
        return 3.5, 35, 9
    elif n_methods <= 8:
        return 3.8, 40, 8
    else:
        return 4.2, 50, 7


def _compute_hspace(
    method_names: list,
    rotation_deg: float,
    xtick_fontsize: float,
    title_fontsize: float,
    per_row_height: float) -> float:
    max_label_len = max((len(m) for m in method_names), default=5)
    theta = math.radians(rotation_deg)
    char_w_in = 0.55 * xtick_fontsize / 72.0
    char_h_in = xtick_fontsize / 72.0
    label_drop = (math.sin(theta) * max_label_len * char_w_in
                  + math.cos(theta) * char_h_in)
    title_h = title_fontsize * 1.4 / 72.0
    padding = 0.08
    gap_needed = label_drop + title_h + padding
    hspace = gap_needed / max(per_row_height, 0.5)
    return max(0.15, min(hspace, 0.90))


def _build_sig_pairs(method_names: list) -> list:
    if len(method_names) < 2:
        return []
    focal = method_names[-1]
    return [(m, focal) for m in method_names[:-1]]


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: Main metric-figure generation
# ═══════════════════════════════════════════════════════════════════════════════

def visualize(
    preset_name,
    summary_only=False,
    output_root=None,
    metric_groups=None,
    selected_metrics=None,
    excluded_metrics=None,
    per_group=False,
    ncols=None,
    fig_height=None,
    fig_width_per_metric=3.2,
    plot_type="boxplot",
    palette_name=None,
    title_fontsize=11,
    axis_label_fontsize=11,
    tick_label_fontsize=11,
    xtick_label_fontsize=None,
    significance_fontsize=13,
    ns_fontsize=10,
    xlabel_rotation=None,
    significance_marker_offset=-0.04,
    ns_offset=0.0,
    significance_line_width=1.5,
    bar_strip_size=2,
    bar_strip_alpha=0.5,
    font_family="Arial",
    show_legend=False,
    panel_labels=False,
    dpi=300):
    """Load experiment results and generate metric comparison figures."""
    cfg = PRESETS[preset_name]
    if output_root is not None:
        cfg = cfg.with_overrides(output_root=Path(output_root))

    tables_dir = cfg.tables_dir
    figures_dir = cfg.figures_dir
    method_names = cfg.method_names
    n_methods = len(method_names)

    if not tables_dir.exists() or not list(tables_dir.glob("*.csv")):
        print(f"ERROR: No results found in {tables_dir}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Visualising: {cfg.name}")
    print(f"Tables dir : {tables_dir}")
    print(f"Methods ({n_methods}): {method_names}")
    print(f"{'='*70}\n")

    _apply_font(font_family)

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

    os.makedirs(figures_dir, exist_ok=True)

    palette = (
        sns.color_palette(palette_name, n_methods)
        if palette_name is not None
        else sns.color_palette(DEFAULT_PALETTE, n_methods)
    )
    sig_pairs = _build_sig_pairs(method_names)
    auto_h, auto_rot, auto_xtick = _adaptive_params(n_methods)

    final_h = fig_height if fig_height is not None else auto_h
    final_rot = xlabel_rotation if xlabel_rotation is not None else auto_rot
    final_xtick = (xtick_label_fontsize
                   if xtick_label_fontsize is not None
                   else auto_xtick)

    available = set(analyzer.metrics)
    groups_to_plot = metric_groups or list(METRIC_GROUPS.keys())

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
        print("  No matching metrics found in the data — skipping figures.")
        return

    if selected_metrics is not None:
        sel_set = set(selected_metrics)
        flat_metrics = [m for m in flat_metrics if m in sel_set]
        flat_display = {m: flat_display[m] for m in flat_metrics}

    if excluded_metrics is not None:
        exc_set = set(excluded_metrics)
        flat_metrics = [m for m in flat_metrics if m not in exc_set]
        flat_display = {m: flat_display[m] for m in flat_metrics}

    if not flat_metrics:
        print("  No metrics remaining after selection/exclusion — skipping.")
        return

    final_ncols = ncols if ncols is not None else _adaptive_ncols(n_methods)

    computed_hspace = _compute_hspace(
        method_names=method_names,
        rotation_deg=final_rot,
        xtick_fontsize=final_xtick,
        title_fontsize=title_fontsize,
        per_row_height=final_h)

    if n_methods <= 5:
        max_sig_pairs = min(len(sig_pairs), 4)
    elif n_methods <= 8:
        max_sig_pairs = 3
    else:
        max_sig_pairs = 3

    common_kwargs = dict(
        show_significance_pairs=sig_pairs,
        max_significance_pairs=max_sig_pairs,
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
            if save_with_vcd is not None:
                save_with_vcd(fig, Path(save_path), dpi=dpi)
            else:
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
            except Exception as exc:
                print(f"    ERROR generating {group_key}: {exc}")
                traceback.print_exc()
    else:
        n_rows = math.ceil(len(flat_metrics) / final_ncols)
        total_w = fig_width_per_metric * final_ncols
        total_h = (final_h * n_rows
                   + final_h * computed_hspace * max(n_rows - 1, 0))
        save_path = figures_dir / "all_metrics.pdf"

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
        except Exception as exc:
            print(f"    ERROR generating unified figure: {exc}")
            traceback.print_exc()

    print(f"\nFigures saved to: {figures_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: Training / validation loss curve visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _smooth(arr, window):
    return pd.Series(arr).rolling(window=max(1, window),
                                   min_periods=1).mean().values


def plot_training_curves(
    preset_name,
    output_root=None,
    smoothing_window=None,
    fill_alpha=0.15,
    line_alpha=0.85,
    per_dataset=False,
    show_recon=False,
    figsize=None,
    dpi=200,
    font_family="Arial",
    palette_name=None):
    """Plot aggregated loss curves with cross-dataset mean +/- std shading."""
    _apply_font(font_family)

    cfg = PRESETS[preset_name]
    if output_root is not None:
        cfg = cfg.with_overrides(output_root=Path(output_root))

    series_dir = cfg.series_dir
    figures_dir = cfg.figures_dir / "training_curves"
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

    methods = list(dict.fromkeys(combined["hue"]))
    n_methods = len(methods)

    if palette_name is not None:
        palette_colors = sns.color_palette(palette_name, n_methods)
    else:
        palette_colors = sns.color_palette(DEFAULT_PALETTE, n_methods)
    color_map = dict(zip(methods, palette_colors))

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
    _figsize = figsize or (7.0 * n_panels, 4.5)

    fig, axes = plt.subplots(1, n_panels, figsize=_figsize)
    if n_panels == 1:
        axes = [axes]

    for method_name in methods:
        method_df = combined[combined["hue"] == method_name]
        datasets_in = method_df["dataset"].unique()
        epoch_counts = [
            int(method_df[method_df["dataset"] == d]["epoch"].max())
            for d in datasets_in
        ]
        max_ep = min(epoch_counts)
        epochs = np.arange(1, max_ep + 1)
        w = smoothing_window or max(1, max_ep // 20)

        for panel_idx, col in enumerate(loss_cols):
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

            axes[panel_idx].plot(
                epochs, mean, color=c, alpha=line_alpha,
                label=method_name, linewidth=1.2)
            axes[panel_idx].fill_between(
                epochs, mean - std, mean + std,
                color=c, alpha=fill_alpha)

    for ax, title in zip(axes, panel_titles):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

    fig.tight_layout()
    agg_path = figures_dir / "aggregated_loss.pdf"
    if save_with_vcd is not None:
        save_with_vcd(fig, agg_path, dpi=dpi, close=True)
    else:
        fig.savefig(str(agg_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"  Aggregated loss → {agg_path}")

    if per_dataset:
        for sf in series_files:
            ds_name = sf.stem.replace("_dfs", "")
            df = pd.read_csv(sf)

            ds_cols = ["train_loss"]
            ds_titles = ["Training"]
            if "val_loss" in df.columns and df["val_loss"].notna().any():
                ds_cols.append("val_loss")
                ds_titles.append("Validation")

            n_p = len(ds_cols)
            fig_d, axs_d = plt.subplots(1, n_p, figsize=(7.0 * n_p, 4))
            if n_p == 1:
                axs_d = [axs_d]

            for mn in df["hue"].unique():
                subset = df[df["hue"] == mn].sort_values("epoch")
                w = smoothing_window or max(1, len(subset) // 20)
                c = color_map.get(mn, None)
                for pi, col in enumerate(ds_cols):
                    sm = _smooth(subset[col].values, w)
                    axs_d[pi].plot(
                        subset["epoch"], sm, label=mn,
                        alpha=0.8, color=c)

            for ax, t in zip(axs_d, ds_titles):
                ax.set(xlabel="Epoch", ylabel="Loss",
                       title=f"{ds_name} — {t}")
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.3)
                sns.despine(ax=ax)

            fig_d.tight_layout()
            save_p = figures_dir / f"{ds_name}_loss.pdf"
            if save_with_vcd is not None:
                save_with_vcd(fig_d, save_p, dpi=dpi, close=True)
            else:
                fig_d.savefig(str(save_p), dpi=dpi, bbox_inches="tight")
                plt.close(fig_d)

    print(f"\n  Training curves saved to: {figures_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# ████  PORTABLE: CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualise experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.visualize_experiment --preset my_ablation
  python -m experiments.visualize_experiment --preset my_ablation --summary-only
  python -m experiments.visualize_experiment --preset my_ablation --training-curves
  python -m experiments.visualize_experiment \\
      --merge my_comparison \\
      --merge-source experiments/results/ablation/tables ModelA ModelB \\
      --merge-source experiments/results/external/tables CellBLAST GMVAE
        """)

    parser.add_argument("--preset", type=str, required=False, default=None,
                        choices=list(PRESETS.keys()))
    parser.add_argument("--merge", type=str, default=None, metavar="NAME")
    parser.add_argument("--merge-source", type=str, nargs="+", action="append",
                        default=None, metavar="TABLES_DIR")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--training-curves", action="store_true")
    parser.add_argument("--per-dataset-curves", action="store_true")
    parser.add_argument("--show-recon", action="store_true")
    parser.add_argument("--groups", type=str, nargs="+", default=None,
                        choices=list(METRIC_GROUPS.keys()))
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-metrics", type=str, nargs="+", default=None)
    parser.add_argument("--per-group", action="store_true")
    parser.add_argument("--ncols", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)

    viz = parser.add_argument_group("Visual tuning")
    viz.add_argument("--fig-height", type=float, default=None)
    viz.add_argument("--fig-width-per-metric", type=float, default=3.2)
    viz.add_argument("--plot-type", type=str, default="boxplot",
                     choices=["boxplot", "violin", "strip", "barplot",
                              "paired_lines"])
    viz.add_argument("--palette", type=str, default=None)
    viz.add_argument("--title-fontsize", type=float, default=11)
    viz.add_argument("--axis-fontsize", type=float, default=11)
    viz.add_argument("--tick-fontsize", type=float, default=11)
    viz.add_argument("--xtick-fontsize", type=float, default=None)
    viz.add_argument("--sig-fontsize", type=float, default=13)
    viz.add_argument("--ns-fontsize", type=float, default=10)
    viz.add_argument("--xlabel-rotation", type=float, default=None)
    viz.add_argument("--sig-offset", type=float, default=-0.04)
    viz.add_argument("--ns-offset", type=float, default=0.0)
    viz.add_argument("--sig-linewidth", type=float, default=1.5)
    viz.add_argument("--strip-size", type=float, default=2)
    viz.add_argument("--strip-alpha", type=float, default=0.5)
    viz.add_argument("--font-family", type=str, default="Arial")
    viz.add_argument("--show-legend", action="store_true")
    viz.add_argument("--panel-labels", action="store_true")
    viz.add_argument("--dpi", type=int, default=300)

    loss_grp = parser.add_argument_group("Loss curve tuning")
    loss_grp.add_argument("--smoothing-window", type=int, default=None)
    loss_grp.add_argument("--fill-alpha", type=float, default=0.15)
    loss_grp.add_argument("--line-alpha", type=float, default=0.85)
    loss_grp.add_argument("--loss-figsize", type=float, nargs=2, default=None)
    loss_grp.add_argument("--loss-dpi", type=int, default=200)
    loss_grp.add_argument("--loss-palette", type=str, default=None)

    args = parser.parse_args()

    if args.preset is None and args.merge is None:
        parser.error("Either --preset or --merge is required.")

    preset_name_for_viz = args.preset
    merge_cfg = None

    if args.merge is not None:
        if not args.merge_source:
            parser.error("--merge requires at least one --merge-source.")

        sources = []
        for src_args in args.merge_source:
            tables_dir = src_args[0]
            methods = src_args[1:] if len(src_args) > 1 else None
            tables_path = Path(tables_dir)
            series_path = tables_path.parent / "series"
            source = {"tables": str(tables_path)}
            if series_path.exists():
                source["series"] = str(series_path)
            if methods:
                source["methods"] = methods
            sources.append(source)

        merge_cfg = MergedExperimentConfig(
            name=args.merge,
            sources=sources,
            output_root=(Path(args.output_root) if args.output_root
                         else Path("experiments/results")))
        print(f"\n{merge_cfg.summary()}\n")
        merge_cfg.build_merged_tables()

        temp_preset = ExperimentConfig(
            name=merge_cfg.name,
            models={m: {} for m in merge_cfg.method_names},
            output_root=merge_cfg.output_root)
        PRESETS[merge_cfg.name] = temp_preset
        preset_name_for_viz = merge_cfg.name

    visualize(
        preset_name_for_viz,
        summary_only=args.summary_only,
        output_root=args.output_root,
        metric_groups=args.groups,
        selected_metrics=args.metrics,
        excluded_metrics=args.exclude_metrics,
        per_group=args.per_group,
        ncols=args.ncols,
        fig_height=args.fig_height,
        fig_width_per_metric=args.fig_width_per_metric,
        plot_type=args.plot_type,
        palette_name=args.palette,
        title_fontsize=args.title_fontsize,
        axis_label_fontsize=args.axis_fontsize,
        tick_label_fontsize=args.tick_fontsize,
        xtick_label_fontsize=args.xtick_fontsize,
        significance_fontsize=args.sig_fontsize,
        ns_fontsize=args.ns_fontsize,
        xlabel_rotation=args.xlabel_rotation,
        significance_marker_offset=args.sig_offset,
        ns_offset=args.ns_offset,
        significance_line_width=args.sig_linewidth,
        bar_strip_size=args.strip_size,
        bar_strip_alpha=args.strip_alpha,
        font_family=args.font_family,
        show_legend=args.show_legend,
        panel_labels=args.panel_labels,
        dpi=args.dpi)

    if args.training_curves:
        plot_training_curves(
            preset_name_for_viz,
            output_root=args.output_root,
            smoothing_window=args.smoothing_window,
            fill_alpha=args.fill_alpha,
            line_alpha=args.line_alpha,
            per_dataset=args.per_dataset_curves,
            show_recon=args.show_recon,
            figsize=(tuple(args.loss_figsize)
                     if args.loss_figsize else None),
            dpi=args.loss_dpi,
            font_family=args.font_family,
            palette_name=args.loss_palette)


if __name__ == "__main__":
    main()
