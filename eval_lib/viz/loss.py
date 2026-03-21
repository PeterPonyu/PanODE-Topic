"""
LossViz  —  Project-adaptive training / validation loss curve visualisation.

A standalone, project-agnostic module that produces publication-quality
aggregated loss curves (mean ± std across datasets) from per-dataset series
CSV files.

Companion to `REA.py` in the results-visualizer skill.  While REA handles
metric bar/box/violin plots, LossViz handles temporal loss curves.

## Expected CSV format

Each series CSV must have **at minimum**:

    epoch, hue, train_loss

Optional columns (auto-detected):

    val_loss, recon_loss, val_recon_loss

``hue`` identifies the method name.  Multiple CSVs (one per dataset) are
combined for cross-dataset aggregation.

## Quick start

```python
from LossViz import plot_aggregated_loss, plot_individual_loss

# Cross-dataset aggregated loss (mean ± std)
plot_aggregated_loss(
    series_dir="results/series",
    output_dir="results/figures/training_curves",
    methods=["Pure-AE", "DPMM-Base", "DPMM-Transformer"],
    palette="husl")

# Per-dataset individual loss curves
plot_individual_loss(
    series_dir="results/series",
    output_dir="results/figures/training_curves",
    methods=["Pure-AE", "DPMM-Base", "DPMM-Transformer"],
    palette="husl")
```
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Geometry-based layout system
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.visualization import (
    apply_style, style_axes, save_with_vcd,
    bind_figure_region, LayoutRegion)

__all__ = [
    "smooth",
    "build_color_map",
    "plot_aggregated_loss",
    "plot_individual_loss",
    "plot_training_curves",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing for a 1-D array."""
    return pd.Series(arr).rolling(window=max(1, window), min_periods=1).mean().values


def build_color_map(
    methods: Sequence[str],
    palette: str = "husl") -> Dict[str, Tuple[float, ...]]:
    """Build a {method_name: colour} dict from a named seaborn palette."""
    colors = sns.color_palette(palette, len(methods))
    return dict(zip(methods, colors))


def _apply_font(font_family: str = "Arial") -> None:
    """Best-effort font override."""
    try:
        matplotlib.rcParams["font.family"] = font_family
    except Exception:
        pass


def _safe_save_figure(fig: plt.Figure, save_path: Path, dpi: int, max_pixels: int = 20000) -> None:
    """Save figure via geometry-based ``save_with_vcd``, clamping DPI first.

    When very large multi-panel figures are created at high DPI, some Matplotlib
    backends can silently truncate or error once any dimension exceeds their
    internal pixel limits. This helper downscales the DPI as needed while
    preserving the aspect ratio.
    """
    width_in, height_in = fig.get_size_inches()
    width_px = width_in * dpi
    height_px = height_in * dpi

    max_dim_px = max(width_px, height_px)
    if max_dim_px > max_pixels:
        scale = max_pixels / max_dim_px
        safe_dpi = max(72, int(dpi * scale))
        print(
            f"[eval_lib.viz.loss] Auto-clamping DPI from {dpi} to {safe_dpi} "
            f"to keep figure within {max_pixels}px (w={width_px:.0f}, "
            f"h={height_px:.0f})."
        )
    else:
        safe_dpi = dpi

    save_with_vcd(fig, save_path, dpi=safe_dpi)


def _detect_columns(df: pd.DataFrame):
    """Return booleans for optional columns present with real data."""
    has_val = ("val_loss" in df.columns
               and df["val_loss"].notna().any())
    has_recon = ("recon_loss" in df.columns
                 and df["recon_loss"].notna().any())
    has_val_recon = ("val_recon_loss" in df.columns
                     and df["val_recon_loss"].notna().any())
    return has_val, has_recon, has_val_recon


def _load_series(
    series_dir: Union[str, Path],
    glob_pattern: str = "*_dfs.csv") -> pd.DataFrame:
    """Load and concatenate all series CSVs from *series_dir*."""
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No series CSVs matching '{glob_pattern}' in {series_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["dataset"] = f.stem.replace("_dfs", "")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregated loss curves  (cross-dataset mean ± std)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_aggregated_loss(
    series_dir: Union[str, Path],
    output_dir: Union[str, Path],
    methods: Optional[List[str]] = None,
    palette: str = "husl",
    color_map: Optional[Dict[str, Tuple]] = None,
    smoothing_window: Optional[int] = None,
    fill_alpha: float = 0.15,
    line_alpha: float = 0.85,
    show_recon: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    font_family: str = "Arial",
    filename: str = "aggregated_loss.pdf",
    glob_pattern: str = "*_dfs.csv") -> Path:
    """Aggregated 2+-panel loss figure with mean ± std across datasets.

    Parameters
    ----------
    series_dir : path
        Directory containing per-dataset series CSVs.
    output_dir : path
        Directory to save the output figure.
    methods : list[str] or None
        Ordered method names.  If None, discovered from the ``hue`` column.
    palette : str
        Seaborn palette name (used when *color_map* is None).
    color_map : dict or None
        Pre-built ``{method: colour}`` dict.  Overrides *palette*.
    smoothing_window : int or None
        Moving-average window.  ``None`` → auto (max_epochs / 20).
    fill_alpha, line_alpha : float
        Alpha for shaded band / mean line.
    show_recon : bool
        Include reconstruction-loss panels.
    figsize : tuple or None
        Override figure size (auto-calculated if None).
    dpi : int
        Output resolution.
    font_family : str
        Font family (best-effort).
    filename : str
        Output filename.
    glob_pattern : str
        Glob for series CSV filenames.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    _apply_font(font_family)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    combined = _load_series(series_dir, glob_pattern)

    # Resolve method order
    if methods is None:
        methods = list(dict.fromkeys(combined["hue"]))

    n_methods = len(methods)
    if color_map is None:
        color_map = build_color_map(methods, palette)

    has_val, has_recon, has_val_recon = _detect_columns(combined)

    # Panel layout
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

    fig = plt.figure(figsize=_figsize)
    _root = bind_figure_region(fig, (0.06, 0.10, 0.96, 0.92))
    _cols = _root.split_cols([1] * n_panels, gap=0.06)
    axes = [c.add_axes(fig) for c in _cols]
    for _ax in axes:
        style_axes(_ax)

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
                matrices.append(smooth(vals, w))

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
    save_path = output_dir / filename
    _safe_save_figure(fig, save_path, dpi=dpi)
    plt.close(fig)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# Per-dataset individual loss curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_individual_loss(
    series_dir: Union[str, Path],
    output_dir: Union[str, Path],
    methods: Optional[List[str]] = None,
    palette: str = "husl",
    color_map: Optional[Dict[str, Tuple]] = None,
    smoothing_window: Optional[int] = None,
    show_recon: bool = False,
    dpi: int = 200,
    font_family: str = "Arial",
    glob_pattern: str = "*_dfs.csv") -> List[Path]:
    """Generate individual loss-curve plots for each dataset.

    Parameters
    ----------
    series_dir, output_dir, methods, palette, color_map, smoothing_window,
    show_recon, dpi, font_family, glob_pattern :
        Same semantics as ``plot_aggregated_loss``.

    Returns
    -------
    list[Path]
        Paths to saved per-dataset figures.
    """
    _apply_font(font_family)
    series_dir = Path(series_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(series_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No series CSVs in {series_dir}")

    # Resolve methods from first file if needed
    if methods is None:
        first = pd.read_csv(files[0])
        methods = list(dict.fromkeys(first["hue"]))

    if color_map is None:
        color_map = build_color_map(methods, palette)

    saved: List[Path] = []

    for sf in files:
        ds_name = sf.stem.replace("_dfs", "")
        df = pd.read_csv(sf)

        ds_cols = ["train_loss"]
        ds_titles = ["Training"]
        has_val, has_recon_flag, has_val_recon = _detect_columns(df)
        if has_val:
            ds_cols.append("val_loss")
            ds_titles.append("Validation")
        if show_recon:
            if has_recon_flag:
                ds_cols.append("recon_loss")
                ds_titles.append("Recon.")
            if has_val_recon:
                ds_cols.append("val_recon_loss")
                ds_titles.append("Val. Recon.")

        n_p = len(ds_cols)
        fig = plt.figure(figsize=(7.0 * n_p, 4))
        _root = bind_figure_region(fig, (0.06, 0.10, 0.96, 0.92))
        _cols = _root.split_cols([1] * n_p, gap=0.06)
        axs = [c.add_axes(fig) for c in _cols]
        for _ax in axs:
            style_axes(_ax)

        for method_name in df["hue"].unique():
            subset = df[df["hue"] == method_name].sort_values("epoch")
            w = smoothing_window or max(1, len(subset) // 20)
            c = color_map.get(method_name, None)
            for pi, col in enumerate(ds_cols):
                sm = smooth(subset[col].values, w)
                axs[pi].plot(
                    subset["epoch"], sm, label=method_name,
                    alpha=0.8, color=c)

        for ax, t in zip(axs, ds_titles):
            ax.set(xlabel="Epoch", ylabel="Loss",
                   title=f"{ds_name} — {t}")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

        fig.tight_layout()
        save_p = output_dir / f"{ds_name}_loss.pdf"
        _safe_save_figure(fig, save_p, dpi=dpi)
        plt.close(fig)
        saved.append(save_p)

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper  (mirrors visualize_experiment.py interface)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    series_dir: Union[str, Path],
    output_dir: Union[str, Path],
    methods: Optional[List[str]] = None,
    palette: str = "husl",
    color_map: Optional[Dict[str, Tuple]] = None,
    smoothing_window: Optional[int] = None,
    fill_alpha: float = 0.15,
    line_alpha: float = 0.85,
    per_dataset: bool = False,
    show_recon: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    font_family: str = "Arial",
    glob_pattern: str = "*_dfs.csv") -> List[Path]:
    """One-call convenience — aggregated + optional per-dataset loss curves.

    Parameters
    ----------
    All parameters follow the same semantics as ``plot_aggregated_loss``
    and ``plot_individual_loss``.

    Returns
    -------
    list[Path]
        All saved figure paths.
    """
    saved = []

    agg = plot_aggregated_loss(
        series_dir=series_dir,
        output_dir=output_dir,
        methods=methods,
        palette=palette,
        color_map=color_map,
        smoothing_window=smoothing_window,
        fill_alpha=fill_alpha,
        line_alpha=line_alpha,
        show_recon=show_recon,
        figsize=figsize,
        dpi=dpi,
        font_family=font_family,
        glob_pattern=glob_pattern)
    saved.append(agg)
    print(f"  Aggregated loss → {agg}")

    if per_dataset:
        indiv = plot_individual_loss(
            series_dir=series_dir,
            output_dir=output_dir,
            methods=methods,
            palette=palette,
            color_map=color_map,
            smoothing_window=smoothing_window,
            show_recon=show_recon,
            dpi=dpi,
            font_family=font_family,
            glob_pattern=glob_pattern)
        saved.extend(indiv)
        for p in indiv:
            print(f"    {p}")

    return saved
