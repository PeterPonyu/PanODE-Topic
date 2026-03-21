"""
Centralised layout constants and adaptive sizing helpers.

All layout-related constants that govern figure dimensions, font sizes,
rotation angles, and spacing are defined here so that both
``experiments/visualize_experiment.py``, ``experiments/merge_and_visualize.py``,
and ``scripts/regenerate_figures.py`` share a single source of truth.

Import paths
------------
::

    from eval_lib.viz.layout import (
        MIN_XTICK_FONTSIZE,
        MIN_XTICK_FONTSIZE_PER_GROUP,
        FIG_WIDTH_PER_METRIC,
        MAX_METHODS_PER_FIGURE,
        MAX_HEIGHT_TO_WIDTH,
        adaptive_params,
        per_row_height,
        rotation_fontsize,
        compute_hspace,
        clamp_xtick_fontsize,
        needs_method_split,
        assert_no_label_overlap)
"""

from __future__ import annotations

import math
import warnings

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Absolute minimum x-tick font size (unified / multi-row grids).
MIN_XTICK_FONTSIZE: float = 7.0

# Minimum x-tick font size for per-group, single-row figures (stricter).
MIN_XTICK_FONTSIZE_PER_GROUP: float = 8.0

# Default width (inches) allocated per metric column.
FIG_WIDTH_PER_METRIC: float = 3.2

# When method count exceeds this, semantic splitting is triggered.
# Lowered from 15 → 10 to keep per-panel figures readable and to
# ensure statistical comparison marks don't collide.
MAX_METHODS_PER_FIGURE: int = 10

# Maximum height-to-width ratio for per-group (single-row) figures,
# enforcing a landscape-biased layout.  Range 0.25–0.45 keeps data
# axes compact while leaving room for rotated x-labels.
MAX_HEIGHT_TO_WIDTH: float = 0.45

# Minimum height-to-width ratio — prevents figures that are too flat
# to display box-plots legibly.  Set low (0.12) because 8-metric groups
# are inherently very wide (25+ inches) and don't need tall data axes.
MIN_HEIGHT_TO_WIDTH: float = 0.12

# Paper-style composed metrics figures are width-locked to the article
# container (17 cm) and content-fitted up to a 21 cm maximum height.
COMPOSED_FIG_WIDTH_CM: float = 17.0
COMPOSED_FIG_MAX_HEIGHT_CM: float = 21.0
COMPOSED_FIG_WIDTH_IN: float = COMPOSED_FIG_WIDTH_CM / 2.54
COMPOSED_FIG_MAX_HEIGHT_IN: float = COMPOSED_FIG_MAX_HEIGHT_CM / 2.54
COMPOSED_FIG_TARGET_RATIO: float = COMPOSED_FIG_WIDTH_CM / COMPOSED_FIG_MAX_HEIGHT_CM


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def clamp_xtick_fontsize(fs: float, *, per_group: bool = False) -> float:
    """Return *fs* clamped to the appropriate minimum x-tick size."""
    floor = MIN_XTICK_FONTSIZE_PER_GROUP if per_group else MIN_XTICK_FONTSIZE
    return max(fs, floor)


def needs_method_split(
    method_names: list[str],
    max_methods_per_panel: int = MAX_METHODS_PER_FIGURE,
    max_avg_label_len: int = 18) -> bool:
    """Return ``True`` when *method_names* should be split across figures."""
    n = len(method_names)
    if n > max_methods_per_panel:
        return True
    avg_len = sum(len(m) for m in method_names) / max(n, 1)
    if avg_len > max_avg_label_len and n > max_methods_per_panel // 2:
        return True
    return False


def per_row_height(n_methods: int) -> float:
    """Row height (inches) keyed on method count.

    Compressed relative to earlier versions to produce landscape-biased
    per-group figures (H/W typically in 0.25 – 0.45).
    """
    if n_methods <= 3:
        return 2.4
    elif n_methods <= 6:
        return 2.8
    elif n_methods <= 8:
        return 3.2
    elif n_methods <= 10:
        return 3.5
    elif n_methods <= 14:
        return 3.8
    elif n_methods <= 20:
        return 4.8
    else:
        return 5.5


def adaptive_params(n_methods: int, *, per_group: bool = False) -> tuple[float, float, float]:
    """Return ``(fig_height, xlabel_rotation, xtick_fontsize)`` by method count.

    For **per-group, single-row figures** we bias toward slightly larger
    x-tick fonts so that method names remain comparable in size to y-tick
    labels, while still staying within the overlap-safe region enforced
    by :func:`compute_hspace` and :func:`assert_no_label_overlap`.
    """
    if n_methods <= 3:
        # Small comparisons → generous height and large fonts.
        h, r, fs = 2.4, 25, 11
    elif n_methods <= 6:
        # Moderate method counts: keep x-ticks close to y-ticks (≈10 pt).
        h, r, fs = 2.8, 35, 10 if per_group else 9
    elif n_methods <= 8:
        # Up to 8 methods per panel is still comfortable at 9 pt.
        h, r, fs = 3.2, 40, 9 if per_group else 8
    elif n_methods <= 10:
        # 9–10 methods: slight increase in height and rotation; keep 9 pt.
        h, r, fs = 3.5, 45, 9 if per_group else 8
    elif n_methods <= 14:
        h, r, fs = 3.8, 50, 9 if per_group else 7
    elif n_methods <= 20:
        h, r, fs = 4.8, 60, 9 if per_group else 7
    else:
        # Very large unified grids (rare in the new workflow) fall back to
        # the global minimum; per-group mode should have split them already.
        h, r, fs = 5.5, 70, 8 if per_group else 7

    return h, r, clamp_xtick_fontsize(fs, per_group=per_group)


def rotation_fontsize(n_methods: int, *, per_group: bool = True) -> tuple[float, float]:
    """Return ``(rotation_deg, xtick_fontsize)`` — per-group default."""
    _, r, fs = adaptive_params(n_methods, per_group=per_group)
    return r, fs


def clamp_aspect_ratio(fig_w: float, fig_h: float) -> float:
    """Clamp *fig_h* so that H/W stays within ``[MIN_HEIGHT_TO_WIDTH, MAX_HEIGHT_TO_WIDTH]``.

    Returns the (possibly reduced) figure height.
    """
    ratio = fig_h / max(fig_w, 0.1)
    if ratio > MAX_HEIGHT_TO_WIDTH:
        return fig_w * MAX_HEIGHT_TO_WIDTH
    if ratio < MIN_HEIGHT_TO_WIDTH:
        return fig_w * MIN_HEIGHT_TO_WIDTH
    return fig_h


def assert_no_label_overlap(
    method_names: list[str],
    fig_width: float,
    n_cols: int,
    xtick_fontsize: float,
    rotation_deg: float,
    *,
    strict: bool = False) -> None:
    """Warn (or raise) if rotated x-tick labels are likely to overlap.

    The check estimates the projected horizontal footprint of every label
    and compares it to the per-column width budget.  When *strict* is
    ``True`` a ``ValueError`` is raised instead of a warning.
    """
    if not method_names:
        return
    col_width_in = fig_width / max(n_cols, 1)
    slot_per_method = col_width_in / len(method_names)

    theta = math.radians(rotation_deg)
    char_w_in = 0.55 * xtick_fontsize / 72.0
    max_label_len = max(len(m) for m in method_names)

    # Horizontal footprint of longest label after rotation
    h_footprint = (math.cos(theta) * max_label_len * char_w_in
                   + math.sin(theta) * (xtick_fontsize / 72.0))

    if h_footprint > slot_per_method * 1.05:
        msg = (
            f"Label overlap likely: longest label projects ~{h_footprint:.2f} in "
            f"but only {slot_per_method:.2f} in available per method "
            f"({len(method_names)} methods, {n_cols} cols, "
            f"fig_w={fig_width:.1f} in, rot={rotation_deg}°, "
            f"fs={xtick_fontsize}pt)"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)


def compute_hspace(
    method_names: list[str],
    rotation_deg: float,
    xtick_fontsize: float,
    title_fontsize: float,
    per_row_h: float) -> float:
    """Dynamic vertical spacing between subplot rows."""
    n_methods = len(method_names)
    max_label_len = max((len(m) for m in method_names), default=5)
    theta = math.radians(rotation_deg)
    char_w_in = 0.55 * xtick_fontsize / 72.0
    char_h_in = xtick_fontsize / 72.0

    label_drop = (math.sin(theta) * max_label_len * char_w_in
                  + math.cos(theta) * char_h_in)
    title_h = title_fontsize * 1.4 / 72.0
    padding = 0.08

    if n_methods > 20:
        padding += 0.25
    elif n_methods > 14:
        padding += 0.15
    elif n_methods > 8:
        padding += 0.05

    gap = label_drop + title_h + padding
    return max(0.15, min(gap / max(per_row_h, 0.5), 1.40))
