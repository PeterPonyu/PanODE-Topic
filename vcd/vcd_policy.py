"""Figure-quality policy for PanODE-Topic publication figures.

Encodes typography, spacing, density, legend, and truncation rules that
every generated panel must satisfy before it is considered publication-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------

@dataclass
class FigurePolicy:
    """Central knob-file for every visual-conflict rule.

    Instantiate with defaults for the PanODE-Topic paper, or override individual
    fields for a different publication style.
    """

    # -- Typography rules ---------------------------------------------------
    allowed_fonts: set = field(
        default_factory=lambda: {
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
            "sans-serif",
        }
    )
    min_body_pt: float = 10.0
    min_dense_pt: float = 8.0
    composed_scale: float = 0.70
    max_title_label_diff: float = 2.0

    # -- Density rules ------------------------------------------------------
    max_xtick_labels: int = 25
    max_ytick_labels: int = 25
    heatmap_max_ticks: int = 15
    bar_max_categories: int = 30
    rotation_threshold: int = 15  # start rotating labels above this count

    # -- Legend rules --------------------------------------------------------
    preferred_locations: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "line": ["best", "upper left"],
            "bar": ["upper right", "upper left"],
            "heatmap": ["outside_right"],
            "scatter": ["upper left", "best"],
            "polar": ["upper right"],
        }
    )
    max_legend_entries_inside: int = 6
    legend_fontsize_min: int = 8

    # -- Truncation / sizing rules ------------------------------------------
    border_tolerance_px: float = 3.0
    max_figsize: Tuple[float, float] = (14.0, 10.0)
    figsize_increment: Tuple[float, float] = (0.5, 0.5)

    # -- Annotation rules ---------------------------------------------------
    max_annotations_per_axes: int = 5
    max_heatmap_annotations: int = 50
    annotation_min_fontsize: int = 8

    # -- Height / whitespace compaction targets ----------------------------
    target_height_width_ratio: float = 0.50   # h/w for full-width standalone panels
    hspace_compact_target: float = 0.35       # aim for this hspace after cleanup
    hspace_excess_threshold: float = 0.70     # hspace above this triggers compaction
    height_compact_min_inches: float = 4.0    # never shrink figure below this height

    # -- Semantic integrity guards ----------------------------------------
    min_label_display_chars: int = 12         # never shorten a label below this
    label_ellipsis_mode: str = "end"          # "end" or "middle"
    protected_label_prefixes: set = field(
        default_factory=lambda: {"↑", "↓", "+", "-", "−", "∗"}
    )
    annotation_drop_order: list = field(
        default_factory=lambda: ["redundant", "secondary", "primary"]
    )

    # -- Panel complexity thresholds --------------------------------------
    max_legend_series: int = 10               # legend entries per axes
    max_numeric_bar_labels: int = 30          # per-bar value-label count
    max_annotations_complexity: int = 20      # text elements per axes
    complexity_score_threshold: float = 15.0  # weighted sum to trigger info


# ---------------------------------------------------------------------------
# Singleton default
# ---------------------------------------------------------------------------

DEFAULT_POLICY: FigurePolicy = FigurePolicy()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _resolve(policy: Optional[FigurePolicy]) -> FigurePolicy:
    """Return *policy* if given, else the module-level default."""
    return policy if policy is not None else DEFAULT_POLICY


def effective_pt(fontsize: float, policy: Optional[FigurePolicy] = None) -> float:
    """Return the effective point size after composed-figure scaling.

    When a panel is composed into a multi-panel figure the on-screen size
    shrinks by ``policy.composed_scale``.  This helper makes the maths
    explicit so callers do not have to remember the multiplication.
    """
    p = _resolve(policy)
    return fontsize * p.composed_scale


def is_font_adequate(
    fontsize: float,
    is_dense: bool = False,
    policy: Optional[FigurePolicy] = None) -> bool:
    """Check whether *fontsize* meets the minimum after composition scaling.

    Parameters
    ----------
    fontsize:
        Nominal font size in points as set by the plotting code.
    is_dense:
        If ``True`` the check uses the relaxed ``min_dense_pt`` threshold
        (e.g. for heatmap annotations or small-multiple tick labels).
    policy:
        Override the default policy if needed.

    Returns
    -------
    bool
        ``True`` when the effective size is at or above the minimum.
    """
    p = _resolve(policy)
    eff = effective_pt(fontsize, p)
    threshold = p.min_dense_pt if is_dense else p.min_body_pt
    return eff >= threshold


def suggest_max_ticks(
    plot_kind: str,
    n_items: int,
    policy: Optional[FigurePolicy] = None) -> int:
    """Return the recommended maximum number of visible tick labels.

    Parameters
    ----------
    plot_kind:
        One of ``"bar"``, ``"heatmap"``, ``"line"``, ``"scatter"``, etc.
    n_items:
        The actual number of data items along the axis.
    policy:
        Override the default policy if needed.

    Returns
    -------
    int
        The maximum number of ticks that should be displayed.  If *n_items*
        is already within the limit, *n_items* is returned unchanged.
    """
    p = _resolve(policy)

    if plot_kind == "heatmap":
        cap = p.heatmap_max_ticks
    elif plot_kind == "bar":
        cap = p.bar_max_categories
    else:
        cap = p.max_xtick_labels

    return min(n_items, cap)


def suggest_legend_loc(
    plot_kind: str,
    n_entries: int,
    policy: Optional[FigurePolicy] = None) -> str:
    """Return the recommended ``loc`` string for ``ax.legend()``.

    If the number of entries exceeds ``max_legend_entries_inside`` the
    function returns ``"outside_right"`` so the caller can switch to
    ``bbox_to_anchor`` placement.

    Parameters
    ----------
    plot_kind:
        One of ``"line"``, ``"bar"``, ``"heatmap"``, ``"scatter"``,
        ``"polar"``, etc.
    n_entries:
        How many legend entries will be displayed.
    policy:
        Override the default policy if needed.

    Returns
    -------
    str
        A matplotlib-compatible ``loc`` string (or ``"outside_right"``
        as a sentinel the caller must handle).
    """
    p = _resolve(policy)

    if n_entries > p.max_legend_entries_inside:
        return "outside_right"

    prefs = p.preferred_locations.get(plot_kind, ["best"])
    return prefs[0]


def should_rotate_labels(
    n_labels: int,
    policy: Optional[FigurePolicy] = None) -> Tuple[bool, int]:
    """Decide whether axis tick labels should be rotated.

    Parameters
    ----------
    n_labels:
        Number of tick labels along the axis.
    policy:
        Override the default policy if needed.

    Returns
    -------
    tuple[bool, int]
        A pair ``(should_rotate, angle)`` where *angle* is ``0`` when no
        rotation is needed, ``45`` for moderate crowding, or ``90`` for
        extreme crowding.
    """
    p = _resolve(policy)

    if n_labels <= p.rotation_threshold:
        return False, 0

    if n_labels <= p.rotation_threshold * 2:
        return True, 45

    return True, 90
