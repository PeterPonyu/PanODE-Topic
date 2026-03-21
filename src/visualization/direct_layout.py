"""Direct rectangle-based figure layout primitives.

This module provides a small geometry engine for creating matplotlib axes via
``fig.add_axes`` only. It intentionally avoids ``GridSpec`` and ``plt.subplots``
for final subplot placement so figure geometry is fully controlled by this
codebase rather than matplotlib's subplot template machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .panel_geometry import normalize_layout_rect


NumberList = Sequence[float] | int



def _normalize_weights(spec: NumberList) -> list[float]:
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("count must be positive")
        return [1.0] * int(spec)
    weights = [float(v) for v in spec]
    if not weights or any(v <= 0 for v in weights):
        raise ValueError(f"weights must be positive, got {spec!r}")
    return weights



def _normalize_gaps(gap: float | Sequence[float] | None, n_items: int) -> list[float]:
    n_gaps = max(n_items - 1, 0)
    if n_gaps == 0:
        return []
    if gap is None:
        return [0.0] * n_gaps
    if isinstance(gap, (int, float)):
        return [float(gap)] * n_gaps
    gaps = [float(v) for v in gap]
    if len(gaps) != n_gaps:
        raise ValueError(f"expected {n_gaps} gap values, got {gap!r}")
    return gaps



def _gap_from_space(total: float, weights: list[float], space: float | None) -> float | None:
    """Translate GridSpec-like ``wspace``/``hspace`` into an absolute gap.

    Matplotlib defines these values as a fraction of the average axis size.
    We mirror that math directly so existing figure numbers can migrate without
    visual drift.
    """
    if space is None:
        return None
    n_items = len(weights)
    if n_items <= 1:
        return 0.0
    total_weight = sum(weights)
    factor = total_weight * (1.0 + ((n_items - 1) * float(space) / n_items))
    base = float(total) / factor
    avg_item = base * total_weight / n_items
    return float(space) * avg_item


@dataclass(frozen=True)
class LayoutRegion:
    """A rectangle in figure coordinates with helpers for explicit subdivision."""

    left: float
    bottom: float
    width: float
    height: float

    @classmethod
    def from_bounds(cls, left: float, bottom: float, right: float, top: float) -> "LayoutRegion":
        left = float(left)
        bottom = float(bottom)
        right = float(right)
        top = float(top)
        if not (0.0 <= left < right <= 1.0 and 0.0 <= bottom < top <= 1.0):
            raise ValueError(f"invalid bounds {(left, bottom, right, top)!r}")
        return cls(left=left, bottom=bottom, width=right - left, height=top - bottom)

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def top(self) -> float:
        return self.bottom + self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.left, self.bottom, self.width, self.height)

    def inset(
        self,
        *,
        left: float = 0.0,
        bottom: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
    ) -> "LayoutRegion":
        return LayoutRegion.from_bounds(
            self.left + float(left),
            self.bottom + float(bottom),
            self.right - float(right),
            self.top - float(top),
        )

    def split_cols(
        self,
        widths: NumberList,
        *,
        gap: float | Sequence[float] | None = None,
        wspace: float | None = None,
    ) -> list["LayoutRegion"]:
        weights = _normalize_weights(widths)
        if gap is not None and wspace is not None:
            raise ValueError("provide either gap or wspace, not both")
        resolved_gap = _gap_from_space(self.width, weights, wspace)
        gaps = _normalize_gaps(resolved_gap if resolved_gap is not None else gap, len(weights))
        usable_width = self.width - sum(gaps)
        total_weight = sum(weights)
        scale = usable_width / total_weight

        x = self.left
        regions: list[LayoutRegion] = []
        for idx, weight in enumerate(weights):
            width = weight * scale
            regions.append(LayoutRegion(x, self.bottom, width, self.height))
            if idx < len(gaps):
                x += width + gaps[idx]
        return regions

    def split_rows(
        self,
        heights: NumberList,
        *,
        gap: float | Sequence[float] | None = None,
        hspace: float | None = None,
        top_to_bottom: bool = True,
    ) -> list["LayoutRegion"]:
        weights = _normalize_weights(heights)
        if gap is not None and hspace is not None:
            raise ValueError("provide either gap or hspace, not both")
        resolved_gap = _gap_from_space(self.height, weights, hspace)
        gaps = _normalize_gaps(resolved_gap if resolved_gap is not None else gap, len(weights))
        usable_height = self.height - sum(gaps)
        total_weight = sum(weights)
        scale = usable_height / total_weight

        regions: list[LayoutRegion] = []
        if top_to_bottom:
            y_top = self.top
            for idx, weight in enumerate(weights):
                height = weight * scale
                bottom = y_top - height
                regions.append(LayoutRegion(self.left, bottom, self.width, height))
                y_top = bottom - (gaps[idx] if idx < len(gaps) else 0.0)
        else:
            y = self.bottom
            for idx, weight in enumerate(weights):
                height = weight * scale
                regions.append(LayoutRegion(self.left, y, self.width, height))
                y += height + (gaps[idx] if idx < len(gaps) else 0.0)
        return regions

    def grid(
        self,
        rows: int,
        cols: int,
        *,
        row_heights: Sequence[float] | None = None,
        col_widths: Sequence[float] | None = None,
        hgap: float | Sequence[float] | None = None,
        wgap: float | Sequence[float] | None = None,
        hspace: float | None = None,
        wspace: float | None = None,
    ) -> list[list["LayoutRegion"]]:
        row_regions = self.split_rows(row_heights or rows, gap=hgap, hspace=hspace)
        return [row.split_cols(col_widths or cols, gap=wgap, wspace=wspace) for row in row_regions]

    def add_axes(self, fig, **kwargs):
        return fig.add_axes(self.as_tuple(), **kwargs)



def bind_figure_region(fig, rect: Sequence[float] | None = None) -> LayoutRegion:
    """Attach canonical layout metadata to ``fig`` and return the main region.

    ``rect`` uses subplot-layout semantics: ``(left, bottom, right, top)`` in
    figure coordinates. The returned region stores explicit geometry as
    ``(left, bottom, width, height)``.
    """
    left, bottom, right, top = normalize_layout_rect(rect)
    fig._panode_layout_rect = (left, bottom, right, top)
    fig._panode_layout_managed = True
    return LayoutRegion.from_bounds(left, bottom, right, top)
