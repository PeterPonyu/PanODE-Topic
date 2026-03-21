"""Helpers for deterministic figure geometry.

This module centralises subplot margins and export padding so figure modules
can freeze their layout before save time instead of relying on matplotlib's
automatic layout engines.
"""

from __future__ import annotations

from typing import Iterable, Sequence

DEFAULT_LAYOUT_RECT: tuple[float, float, float, float] = (0.02, 0.03, 0.98, 0.95)
DEFAULT_EXPORT_PAD_INCHES: float = 0.10


def normalize_layout_rect(rect: Sequence[float] | None) -> tuple[float, float, float, float]:
    """Validate and normalise a subplot layout rectangle.

    The rectangle is expressed in figure coordinates as
    ``(left, bottom, right, top)``.
    """
    if rect is None:
        return DEFAULT_LAYOUT_RECT
    if len(rect) != 4:
        raise ValueError(f"layout rect must contain 4 floats, got {rect!r}")
    left, bottom, right, top = map(float, rect)
    if not (0.0 <= left < right <= 1.0 and 0.0 <= bottom < top <= 1.0):
        raise ValueError(f"invalid layout rect {rect!r}")
    return left, bottom, right, top


def apply_layout_rect(
    fig,
    rect: Sequence[float] | None = None,
    *,
    wspace: float | None = None,
    hspace: float | None = None,
) -> tuple[float, float, float, float]:
    """Apply a deterministic subplot layout to ``fig``.

    This only affects subplot-based axes; manually positioned axes created via
    ``fig.add_axes(...)`` remain untouched.
    """
    left, bottom, right, top = normalize_layout_rect(rect)
    kwargs: dict[str, float] = {
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
    }
    if wspace is not None:
        kwargs["wspace"] = float(wspace)
    if hspace is not None:
        kwargs["hspace"] = float(hspace)
    fig.subplots_adjust(**kwargs)
    fig._panode_layout_rect = (left, bottom, right, top)
    fig._panode_layout_managed = True
    return fig._panode_layout_rect


def set_export_pad_inches(fig, pad_inches: float = DEFAULT_EXPORT_PAD_INCHES) -> float:
    """Store the export padding to reuse across raster/vector saves."""
    fig._panode_export_pad_inches = float(pad_inches)
    return fig._panode_export_pad_inches


def get_export_pad_inches(fig, fallback: float = DEFAULT_EXPORT_PAD_INCHES) -> float:
    """Return the per-figure export padding in inches."""
    return float(getattr(fig, "_panode_export_pad_inches", fallback))


def mark_export_artists(fig, artists: Iterable[object]) -> None:
    """Register extra artists that must be considered when computing export crops."""
    current = list(getattr(fig, "_panode_export_artists", []) or [])
    current.extend(list(artists))
    fig._panode_export_artists = current