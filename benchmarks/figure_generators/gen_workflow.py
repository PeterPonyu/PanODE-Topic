"""Generate horizontal workflow diagram PNGs for Figures 2–7.

Each figure gets a compact left-to-right flowchart showing:
    data source → processing stages → methodological output

The function ``gen_workflow_png`` is called from each per-figure generator
module.  The resulting PNG (``workflow.png``) is placed in the figure's
subplot directory and referenced by the manifest.

Each step includes a small Unicode icon for visual clarity.
The last step describes the *methodological output*, not panel labels.

Style is consistent with the rest of the subplot pipeline (``subplot_style``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from .subplot_style import (
    SUBPLOT_DPI, CONTAINER_CSS_PX, DPR, save_subplot)

# ═══════════════════════════════════════════════════════════════════════════════
# Visual constants
# ═══════════════════════════════════════════════════════════════════════════════

_FULL_W_PX = CONTAINER_CSS_PX * DPR          # device-px width
_FULL_W_IN = _FULL_W_PX / SUBPLOT_DPI        # inches

_BOX_HEIGHT = 0.50                            # fraction of axes height (taller for icons)
_BOX_RADIUS = 0.015                           # rounded-rect radius
_ARROW_HEAD_W = 0.008                         # arrow head width
_ARROW_HEAD_L = 0.012                         # arrow head length

_COLOR_FILL = "#E8F0FE"                       # light blue fill
_COLOR_EDGE = "#4A7CBF"                       # medium blue border
_COLOR_ARROW = "#6A6A6A"                      # dark grey arrows
_COLOR_TEXT = "#1A1A1A"                        # near-black text
_COLOR_ICON = "#4A7CBF"                        # icon colour

_FONT_SIZE = 13                                # base text inside boxes
_FONT_SIZE_SMALL = 12                          # secondary line inside boxes
_FONT_SIZE_ICON = 15                           # icon symbol size

# Unicode schematic icons for each step type
_STEP_ICONS: dict[str, str] = {
    "cells":      "\U0001F9EC",   # 🧬 DNA/cell
    "preprocess": "\U0001F527",   # 🔧 wrench
    "train":      "\U0001F9E0",   # 🧠 brain
    "evaluate":   "\U0001F4CA",   # 📊 chart
    "compare":    "\u2696",       # ⚖ balance
    "sweep":      "\U0001F50D",   # 🔍 magnifier
    "metrics":    "\U0001F4C8",   # 📈 chart up
    "umap":       "\u2726",       # ✦ sparkle
    "scatter":    "\u25C8",       # ◈ diamond
    "gradient":   "\u2593",       # ▓ shade block
    "enrichment": "\U0001F4A1",   # 💡 light bulb
    "latent":     "\U0001F300",   # 🌀 cyclone
    "perturb":    "\u2300",       # ⌀ diameter
    "diff":       "\u0394",       # Δ delta
    "heatmap":    "\u2588",       # █ full block
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def gen_workflow_png(
    steps: list[dict],
    out_dir: Path | str,
    filename: str = "workflow.png",
    aspect: float = 0.22) -> str:
    """Render a horizontal workflow strip and save it.

    Parameters
    ----------
    steps : list[dict]
        Each dict has keys:
        - ``"label"`` (str): Main text (required).
        - ``"sub"``   (str): Optional small secondary line.
        - ``"icon"``  (str): Optional icon key (see ``_STEP_ICONS``).
    out_dir : Path
        Directory to save the PNG into.
    filename : str
        Output filename.
    aspect : float
        Height / width ratio for the figure.

    Returns
    -------
    str
        The filename of the saved PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(steps)
    fig_w = _FULL_W_IN
    fig_h = fig_w * aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # ── Layout geometry ───────────────────────────────────────────────────
    margin_x = 0.02
    gap_frac = 0.03                         # gap between boxes
    total_gap = gap_frac * (n - 1)
    total_box_w = 1.0 - 2 * margin_x - total_gap
    box_w = total_box_w / n
    box_h = _BOX_HEIGHT
    cy = 0.50                               # vertical centre

    for i, step in enumerate(steps):
        x0 = margin_x + i * (box_w + gap_frac)
        y0 = cy - box_h / 2

        # ── Rounded-rect box ──────────────────────────────────────────────
        fancy = mpatches.FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle=f"round,pad=0,rounding_size={_BOX_RADIUS}",
            facecolor=_COLOR_FILL,
            edgecolor=_COLOR_EDGE,
            linewidth=1.0,
            transform=ax.transAxes,
            clip_on=False,
            label=f"_workflow_box_{i}",   # underscore prefix → VCD skips
        )
        ax.add_patch(fancy)

        tx = x0 + box_w / 2
        label = step["label"]
        sub = step.get("sub", "")
        icon_key = step.get("icon", "")
        icon_char = _STEP_ICONS.get(icon_key, "")

        # ── Icon above text ──────────────────────────────────────────────
        if icon_char and sub:
            ax.text(tx, cy + 0.16, icon_char,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE_ICON,
                    color=_COLOR_ICON,
                    transform=ax.transAxes)
            ax.text(tx, cy + 0.01, label,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE, fontweight="bold",
                    color=_COLOR_TEXT,
                    transform=ax.transAxes)
            ax.text(tx, cy - 0.14, sub,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE_SMALL,
                    color="#555555",
                    transform=ax.transAxes)
        elif icon_char:
            ax.text(tx, cy + 0.10, icon_char,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE_ICON,
                    color=_COLOR_ICON,
                    transform=ax.transAxes)
            ax.text(tx, cy - 0.06, label,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE, fontweight="bold",
                    color=_COLOR_TEXT,
                    transform=ax.transAxes)
        elif sub:
            ax.text(tx, cy + 0.05, label,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE, fontweight="bold",
                    color=_COLOR_TEXT,
                    transform=ax.transAxes)
            ax.text(tx, cy - 0.12, sub,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE_SMALL,
                    color="#555555",
                    transform=ax.transAxes)
        else:
            ax.text(tx, cy, label,
                    ha="center", va="center",
                    fontsize=_FONT_SIZE, fontweight="bold",
                    color=_COLOR_TEXT,
                    transform=ax.transAxes)

        # ── Arrow between this box and the next ──────────────────────────
        if i < n - 1:
            ax_start = x0 + box_w + 0.003
            ax_end = x0 + box_w + gap_frac - 0.003
            ax.annotate(
                "", xy=(ax_end, cy), xytext=(ax_start, cy),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=_COLOR_ARROW,
                    lw=1.2,
                    mutation_scale=10))

    path = out_dir / filename
    save_subplot(fig, path)
    return filename


# ═══════════════════════════════════════════════════════════════════════════════
# Per-figure workflow step definitions
# NOTE: Last step = methodological output, NOT panel labels.
# ═══════════════════════════════════════════════════════════════════════════════

def get_workflow_steps(fig_num: int, series: str = "dpmm") -> list[dict]:
    """Return workflow step definitions appropriate for *fig_num* and *series*.

    Series-aware: adjusts model names and metric counts for DPMM vs Topic.
    """
    is_dpmm = series == "dpmm"
    model_family = "DPMM" if is_dpmm else "Topic"
    baseline_family = "Pure-AE" if is_dpmm else "Pure-VAE"
    base_model = "DPMM-Base" if is_dpmm else "Topic-Base"
    n_metrics = "41" if is_dpmm else "6"
    n_core = "6" if is_dpmm else "4"

    STEPS: dict[int, list[dict]] = {
        2: [
            {"label": "12 scRNA-seq", "sub": "datasets", "icon": "cells"},
            {"label": "Preprocess", "sub": "HVG · norm · log1p", "icon": "preprocess"},
            {"label": "Train 6 Models",
             "sub": f"3 {model_family} + 3 {baseline_family}", "icon": "train"},
            {"label": "Evaluate",
             "sub": f"{n_metrics} metrics × 12 datasets", "icon": "evaluate"},
            {"label": "Ablation Analysis",
             "sub": "UMAP · metrics · efficiency", "icon": "compare"},
        ],
        3: [
            {"label": "12 scRNA-seq", "sub": "datasets", "icon": "cells"},
            {"label": base_model, "sub": "single architecture", "icon": "train"},
            {"label": "Sweep 10 HPs",
             "sub": "one factor at a time", "icon": "sweep"},
            {"label": f"{n_core} Core Metrics",
             "sub": "per sweep value", "icon": "metrics"},
            {"label": "Sensitivity Profile",
             "sub": "parameter robustness", "icon": "evaluate"},
        ],
        4: [
            {"label": "3 Repr. Datasets",
             "sub": "setty · endo · dentate", "icon": "cells"},
            {"label": "HP Sweep", "sub": "10 parameters", "icon": "sweep"},
            {"label": "Latent Extract",
             "sub": "per sweep point", "icon": "latent"},
            {"label": "KMeans + UMAP",
             "sub": "cluster & project", "icon": "umap"},
            {"label": "Geometry Trends",
             "sub": "embedding evolution", "icon": "scatter"},
        ],
        5: [
            {"label": "12 scRNA-seq", "sub": "datasets", "icon": "cells"},
            {"label": "Train 6 Models",
             "sub": f"3 {model_family} + 3 {baseline_family}", "icon": "train"},
            {"label": f"{n_metrics} Metrics",
             "sub": "per model × dataset", "icon": "metrics"},
            {"label": "Pairwise Scatter",
             "sub": "convex hulls", "icon": "scatter"},
            {"label": "Trade-off Landscape",
             "sub": "multi-objective", "icon": "compare"},
        ],
        6: [
            {"label": "3 Repr. Datasets",
             "sub": "setty · endo · dentate", "icon": "cells"},
            {"label": f"Train 3 {model_family}",
             "sub": "Base · Trans · Contr", "icon": "train"},
            {"label": "Gradient Saliency",
             "sub": "∂loss / ∂gene", "icon": "gradient"},
            {"label": "Gene Ranking",
             "sub": "top-K per component", "icon": "heatmap"},
        ],
        7: [
            {"label": "Salient Genes",
             "sub": "per component", "icon": "gradient"},
            {"label": "GO / BP Query",
             "sub": "Enrichr API", "icon": "enrichment"},
            {"label": "Filter adj. p < 0.05",
             "sub": "Bonferroni", "icon": "evaluate"},
            {"label": "Enrichment Map",
             "sub": "dotplot & terms", "icon": "enrichment"},
        ],
    }
    return STEPS.get(fig_num, [])


def gen_figure_workflow(
    fig_num: int, out_dir: Path | str, series: str = "dpmm") -> str:
    """Generate the workflow PNG for a given figure number and series.

    Returns the filename.
    """
    steps = get_workflow_steps(fig_num, series)
    if not steps:
        raise ValueError(f"No workflow defined for Figure {fig_num}")
    return gen_workflow_png(steps, out_dir)
