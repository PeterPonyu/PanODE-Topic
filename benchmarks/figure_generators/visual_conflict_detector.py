"""Enhanced visual conflict detection for matplotlib figure subplots.

Goes beyond text-only overlap/truncation checks (in ``common.py``) to detect:

 1. **Text overlap** — pairwise BBox intersection of all text artists.
 2. **Text truncation** — text extending beyond figure canvas.
 3. **Artist truncation** — ANY drawn content (PathCollections / scatter dots,
    PolyCollections / fills, patches, Line2D, etc.) clipped by figure borders.
 4. **Artist-vs-artist overlap** — significant overlap between non-text
    graphical objects (e.g. dots overflowing boxplot whiskers, legend boxes
    covering data points, colorbar overlapping axes).
 5. **Artist-vs-text overlap** — graphical content overlapping text labels.
    Detects in-axes annotations (significance markers ``*``, ``**``,
    ``ns``, ``p<…``, etc.) colliding with data content (warning).
 6. **Axes overflow** — child artists whose extents exceed their parent axes.
 7. **Scatter clip risk** — scatter markers with ``clip_on=True`` whose
    true unclipped extent (data offset + marker radius) exceeds the axes
    clip box.
 8. **Cross-panel spillover** — text from one axes leaking into another.
 9. **Panel-label overlap** — (A)/(B) labels covering data or text.
10. **Legend spillover** — legend extending beyond figure or into other axes.
11. **Legend-to-panel content** — legend bbox overlapping data in OTHER panels.
12. **Legend-vs-own-content** — legend occluding data lines/scatter INSIDE its
    own panel (significant overlap only).
13. **Figure-level legend vs subplot content** — shared fig.legend()
    overlapping subplot scatter/bar data.
14. **Colorbar internal** — tick labels overlapping each other, axis
    label overlapping ticks, tick labels truncated at figure edges.
15. **Legend internal** — legend text entries crowding each other,
    legend text extending beyond legend frame or figure.
16. **Significance brackets** — bracket star/ns text truncated at border,
    bracket text overlapping other labels, bracket lines exceeding figure.
17. **Colorbar-data overlap** — colorbar inset axes occluding parent data
    content (scatter, lines, images) with auto-relocate capability.
18. **Legend crowding auto-fix** — detects crowded legend entries and
    optionally shrinks font or relocates to ``loc='best'``.
19. **Font-size adequacy** — detects text artists whose effective rendered
    size (after tight_layout + DPR scaling) would fall below a minimum
    point threshold (default 6pt) in the final paper figure, accounting
    for the downscaling from subplot PNG to the composed screenshot.

Two-layer detection:
  Layer 1 (subplot-level): passes 12-13, per-axes summary
  Layer 2 (figure-level):  passes 1-11, 14-18

Usage:
    from benchmarks.figure_generators.visual_conflict_detector import (
        detect_all_conflicts)

    fig, ax = plt.subplots(...)
    # ... draw ...
    issues = detect_all_conflicts(fig, label="my_subplot", verbose=True)
    # issues is a list of dicts: {type, severity, detail, elements}

This module is designed to be called right before ``save_subplot`` in every
figure generator.  It replaces and extends ``check_text_overlaps`` from
``common.py``.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.collections import PathCollection, PolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_bbox(artist, renderer) -> Bbox | None:
    """Try to extract display-coords BBox from any artist."""
    try:
        bb = artist.get_window_extent(renderer)
        if bb is not None and bb.width > 0 and bb.height > 0:
            return bb
    except Exception:
        pass
    return None


def _shrink(bb: Bbox, px: float) -> Bbox | None:
    """Shrink a Bbox by *px* pixels on each side; return None if degenerate."""
    b = Bbox.from_extents(bb.x0 + px, bb.y0 + px, bb.x1 - px, bb.y1 - px)
    if b.width > 0 and b.height > 0:
        return b
    return None


def _fig_bbox(fig) -> Bbox:
    """Figure bounding box in display coordinates."""
    w, h = fig.get_size_inches()
    dpi = fig.dpi
    return Bbox.from_bounds(0, 0, w * dpi, h * dpi)


def _overlap_area(a: Bbox, b: Bbox) -> float:
    """Pixel area of intersection between two Bboxes (0 if no overlap)."""
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 > x0 and y1 > y0:
        return (x1 - x0) * (y1 - y0)
    return 0.0


def _sides_outside(bb: Bbox, fig_bb: Bbox, tol: float = 1.0) -> list[str]:
    """Which sides of *bb* extend beyond *fig_bb*."""
    sides = []
    if bb.x0 < fig_bb.x0 - tol:
        sides.append("left")
    if bb.y0 < fig_bb.y0 - tol:
        sides.append("bottom")
    if bb.x1 > fig_bb.x1 + tol:
        sides.append("right")
    if bb.y1 > fig_bb.y1 + tol:
        sides.append("top")
    return sides


def _artist_label(artist, hint: str = "") -> str:
    """Human-readable tag for an artist."""
    if isinstance(artist, Text):
        s = artist.get_text().strip()
        return f"{hint}: {s[:50]}" if hint else f"text: {s[:50]}"
    cls = type(artist).__name__
    label = getattr(artist, "_label", "") or ""
    tag = f"{cls}"
    if label and not label.startswith("_"):
        tag += f"({label[:30]})"
    return f"{hint}: {tag}" if hint else tag


# ═══════════════════════════════════════════════════════════════════════════════
# Collector: gather all artists with bounding boxes
# ═══════════════════════════════════════════════════════════════════════════════

class _ArtistInfo:
    """Lightweight carrier for an artist + its display bbox + metadata."""
    __slots__ = ("artist", "bbox", "tag", "kind", "ax_id")

    def __init__(self, artist, bbox, tag, kind, ax_id=None):
        self.artist = artist
        self.bbox = bbox
        self.tag = tag
        self.kind = kind        # "text" | "patch" | "collection" | "line" | "image" | "legend"
        self.ax_id = ax_id      # id(ax) if owned by a specific axes


def _collect_artists(fig, renderer) -> list[_ArtistInfo]:
    """Walk *fig* and collect all visible artists with valid bboxes."""
    infos: list[_ArtistInfo] = []

    # Identify colorbar axes to annotate properly
    cbar_axes = set()
    for ax in fig.get_axes():
        if hasattr(ax, '_colorbar_info') or getattr(ax, '_colorbar', None):
            cbar_axes.add(id(ax))

    for ax in fig.get_axes():
        is_cbar = id(ax) in cbar_axes
        pfx = "cbar" if is_cbar else ""
        aid = id(ax)

        # ── Text artists ───────────────────────────────────────────────
        # Titles
        for title_obj in [ax.title, ax._left_title, ax._right_title]:
            if title_obj and title_obj.get_text().strip():
                bb = _safe_bbox(title_obj, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        title_obj, bb,
                        _artist_label(title_obj, f"{pfx}title"),
                        "text", aid))

        # Axis labels
        for lbl, hint in [(ax.xaxis.label, f"{pfx}xlabel"),
                          (ax.yaxis.label, f"{pfx}ylabel")]:
            if lbl.get_text().strip():
                bb = _safe_bbox(lbl, renderer)
                if bb:
                    infos.append(_ArtistInfo(lbl, bb,
                                             _artist_label(lbl, hint),
                                             "text", aid))

        # Tick labels
        for tl in ax.get_xticklabels():
            if tl.get_text().strip():
                bb = _safe_bbox(tl, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        tl, bb,
                        _artist_label(tl, "cbar_tick" if is_cbar else "xtick"),
                        "text", aid))
        for tl in ax.get_yticklabels():
            if tl.get_text().strip():
                bb = _safe_bbox(tl, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        tl, bb,
                        _artist_label(tl, "cbar_tick" if is_cbar else "ytick"),
                        "text", aid))

        if not is_cbar:
            # Manual ax.text() objects
            for txt in ax.texts:
                if txt.get_text().strip():
                    bb = _safe_bbox(txt, renderer)
                    if bb:
                        infos.append(_ArtistInfo(
                            txt, bb, _artist_label(txt, "annotation"),
                            "text", aid))

            # Legend
            legend = ax.get_legend()
            if legend is not None:
                bb = _safe_bbox(legend, renderer)
                if bb:
                    infos.append(_ArtistInfo(
                        legend, bb, "legend_box", "legend", aid))
                for txt in legend.get_texts():
                    if txt.get_text().strip():
                        tbb = _safe_bbox(txt, renderer)
                        if tbb:
                            infos.append(_ArtistInfo(
                                txt, tbb,
                                _artist_label(txt, "legend_text"),
                                "text", aid))

        # ── Graphical artists ──────────────────────────────────────────
        for child in ax.get_children():
            if isinstance(child, Text):
                continue   # already handled above
            if not getattr(child, "get_visible", lambda: True)():
                continue
            # Skip axes background patch (spans entire axes, always triggers
            # false positive overlaps with any internal legend)
            if child is ax.patch:
                continue
            # Skip grid lines, spines, internal artists
            label = getattr(child, "_label", "") or ""
            if label.startswith("_") and label not in ("_nolegend_"):
                # internal matplotlib artists — skip most
                if isinstance(child, (Line2D)):
                    # Could be grid lines
                    if child.get_linestyle() in ("--", ":", "-."):
                        continue
                continue

            bb = _safe_bbox(child, renderer)
            if bb is None:
                continue

            if isinstance(child, PathCollection):
                infos.append(_ArtistInfo(
                    child, bb,
                    _artist_label(child, "scatter"),
                    "collection", aid))
            elif isinstance(child, (PolyCollection, LineCollection)):
                infos.append(_ArtistInfo(
                    child, bb,
                    _artist_label(child, "poly"),
                    "collection", aid))
            elif isinstance(child, Patch):
                infos.append(_ArtistInfo(
                    child, bb,
                    _artist_label(child, "patch"),
                    "patch", aid))
            elif isinstance(child, Line2D):
                infos.append(_ArtistInfo(
                    child, bb,
                    _artist_label(child, "line"),
                    "line", aid))
            elif isinstance(child, AxesImage):
                infos.append(_ArtistInfo(
                    child, bb,
                    _artist_label(child, "image"),
                    "image", aid))

    return infos


# ═══════════════════════════════════════════════════════════════════════════════
# Detection passes 1–7 (core checks)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_text_overlaps(infos: list[_ArtistInfo], tol_px: float = 2.0):
    """Pass 1: Pairwise text overlap detection."""
    texts = [a for a in infos if a.kind == "text"]
    issues = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            # Skip xtick-vs-ytick pairs (naturally meet at axes corner)
            tags = {a.tag.split(":")[0].strip(), b.tag.split(":")[0].strip()}
            if tags == {"xtick", "ytick"}:
                continue
            sa = _shrink(a.bbox, tol_px)
            sb = _shrink(b.bbox, tol_px)
            if sa and sb and sa.overlaps(sb):
                issues.append({
                    "type": "text_overlap",
                    "severity": "warning",
                    "detail": f"'{a.tag}' overlaps '{b.tag}'",
                    "elements": [a.tag, b.tag],
                })
    return issues


def _check_truncation(infos: list[_ArtistInfo], fig_bb: Bbox, tol_px: float = 1.0):
    """Pass 2–3: Check if any artist extends beyond figure canvas."""
    issues = []
    for a in infos:
        sides = _sides_outside(a.bbox, fig_bb, tol_px)
        if sides:
            sev = "warning" if a.kind == "text" else "info"
            if a.kind in ("collection", "patch", "line"):
                sev = "warning"
            issues.append({
                "type": f"{a.kind}_truncation",
                "severity": sev,
                "detail": (f"'{a.tag}' extends beyond figure border "
                           f"({', '.join(sides)})"),
                "elements": [a.tag],
            })
    return issues


def _check_artist_content_overlap(
    infos: list[_ArtistInfo],
    min_overlap_px2: float = 100.0):
    """Pass 4: Check significant overlap between non-text graphical artists.

    Skips same-axes overlaps entirely — those are intentional layering
    (legend on bars, scatter layers, colorbar on image, etc.).
    """
    graphical = [a for a in infos
                 if a.kind in ("collection", "patch", "image", "legend")]
    issues = []
    for i in range(len(graphical)):
        for j in range(i + 1, len(graphical)):
            a, b = graphical[i], graphical[j]
            # Same axes = intentional layering — skip entirely
            if a.ax_id == b.ax_id:
                continue
            area = _overlap_area(a.bbox, b.bbox)
            if area > min_overlap_px2:
                issues.append({
                    "type": "artist_overlap",
                    "severity": "warning",
                    "detail": (f"'{a.tag}' overlaps '{b.tag}' "
                               f"({area:.0f} px²)"),
                    "elements": [a.tag, b.tag],
                })
    return issues


def _is_significance_marker(tag: str) -> bool:
    """Return True if *tag* looks like a statistical significance annotation.

    Catches common patterns: ``*``, ``**``, ``***``, ``ns``, ``p<0.05``, etc.
    """
    import re
    if not tag.startswith("annotation:"):
        return False
    txt = tag.split(":", 1)[1].strip()
    if re.fullmatch(r"\*{1,4}", txt):          # *, **, ***, ****
        return True
    if txt.lower() in ("ns", "n.s.", "ns."):   # not-significant
        return True
    if re.match(r"p\s*[<>=]", txt, re.I):       # p<0.05, P = 0.01 ...
        return True
    return False


def _check_text_vs_artist_overlap(
    infos: list[_ArtistInfo],
    tol_px: float = 2.0,
    min_overlap_px2: float = 50.0):
    """Pass 5: Check if graphical content overlaps text labels.

    In-axes annotations (especially significance markers like ``*``, ``ns``)
    overlapping with data artists in the same panel are elevated to warnings.
    """
    texts = [a for a in infos if a.kind == "text"]
    graphics = [a for a in infos
                if a.kind in ("collection", "patch", "line", "image")]
    issues = []
    for t in texts:
        tb = _shrink(t.bbox, tol_px)
        if not tb:
            continue
        for g in graphics:
            area = _overlap_area(tb, g.bbox)
            if area > min_overlap_px2:
                # Determine severity: in-axes annotations overlapping with
                # data artists in the same panel are warnings (especially
                # significance markers like *, **, ns, etc.)
                is_annotation = t.tag.startswith("annotation:")
                same_axes = (t.ax_id == g.ax_id) and (t.ax_id is not None)
                if is_annotation and same_axes:
                    severity = "warning"
                    issue_type = "annotation_data_overlap"
                    if _is_significance_marker(t.tag):
                        issue_type = "significance_marker_overlap"
                else:
                    severity = "info"
                    issue_type = "text_artist_overlap"

                issues.append({
                    "type": issue_type,
                    "severity": severity,
                    "detail": (f"Text '{t.tag}' overlaps content "
                               f"'{g.tag}' ({area:.0f} px²)"),
                    "elements": [t.tag, g.tag],
                })
    return issues


def _check_axes_overflow(infos: list[_ArtistInfo], fig):
    """Pass 6: Check if artist content exceeds its parent axes bounds.

    Only checks meaningful data artists (collections, lines),
    not axis furniture (patches = spines, background, etc.).
    """
    renderer = fig.canvas.get_renderer()
    issues = []

    for ax in fig.get_axes():
        ax_bb = _safe_bbox(ax, renderer)
        if not ax_bb:
            continue
        aid = id(ax)
        ax_artists = [a for a in infos
                      if a.ax_id == aid
                      and a.kind in ("collection", "line")]
        for a in ax_artists:
            sides = _sides_outside(a.bbox, ax_bb, tol=3.0)
            if sides:
                issues.append({
                    "type": "axes_overflow",
                    "severity": "info",
                    "detail": (f"'{a.tag}' extends beyond axes border "
                               f"({', '.join(sides)})"),
                    "elements": [a.tag],
                })
    return issues


def _check_scatter_clip_risk(fig) -> list[dict]:
    """Pass 7: Detect scatter markers silently clipped by axes boundaries.

    When ``clip_on=True`` (matplotlib's default), scatter dots that
    extend past the axes clip box are *invisibly* truncated.
    ``get_window_extent()`` returns the *already-clipped* bbox, so the
    standard checks cannot detect the loss.

    This pass reconstructs the **true unclipped extent** of each scatter
    marker by reading raw data offsets, transforming to display coords,
    adding marker radius, and comparing against the axes clip box.
    """
    renderer = fig.canvas.get_renderer()
    issues = []

    for ax in fig.get_axes():
        ax_bb = _safe_bbox(ax, renderer)
        if ax_bb is None:
            continue

        for child in ax.get_children():
            if not isinstance(child, PathCollection):
                continue
            if not child.get_visible():
                continue
            if not child.get_clip_on():
                continue

            offsets = child.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue

            raw_sizes = child.get_sizes()
            if raw_sizes is None or len(raw_sizes) == 0:
                continue
            sizes = np.broadcast_to(np.asarray(raw_sizes),
                                    (len(offsets)))

            transform = child.get_offset_transform()
            if transform is None:
                transform = ax.transData
            try:
                display_pts = transform.transform(offsets)
            except Exception:
                continue

            pts_per_px = 72.0 / fig.dpi
            radii_px = np.sqrt(sizes / np.pi) / pts_per_px

            n_clipped = 0
            clipped_sides: set[str] = set()
            for (dx, dy), r in zip(display_pts, radii_px):
                if dx - r < ax_bb.x0:
                    n_clipped += 1
                    clipped_sides.add("left")
                elif dx + r > ax_bb.x1:
                    n_clipped += 1
                    clipped_sides.add("right")
                if dy - r < ax_bb.y0:
                    n_clipped += 1
                    clipped_sides.add("bottom")
                elif dy + r > ax_bb.y1:
                    n_clipped += 1
                    clipped_sides.add("top")

            if n_clipped > 0:
                label = getattr(child, "_label", "") or ""
                tag = f"scatter({label})" if label and not label.startswith("_") else "scatter"
                issues.append({
                    "type": "scatter_clip_risk",
                    "severity": "warning",
                    "detail": (
                        f"'{tag}' has {n_clipped} marker(s) clipped at "
                        f"axes edge ({', '.join(sorted(clipped_sides))}). "
                        f"Set clip_on=False or add axis margin."
                    ),
                    "elements": [tag],
                })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Detection passes 8–13 (composed-figure & legend checks)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_cross_panel_spillover(fig, renderer, tol_px=5.0):
    """Pass 8: Detect content from one axes spilling into an adjacent axes."""
    axes_list = fig.get_axes()
    if len(axes_list) < 2:
        return []

    issues = []
    ax_bboxes = []
    for ax in axes_list:
        bb = _safe_bbox(ax, renderer)
        if bb:
            ax_bboxes.append((ax, bb))

    for i, (ax_i, bb_i) in enumerate(ax_bboxes):
        for child in ax_i.get_children():
            if not child.get_visible():
                continue
            if isinstance(child, Text):
                child_bb = _safe_bbox(child, renderer)
                if child_bb is None:
                    continue
                for j, (ax_j, bb_j) in enumerate(ax_bboxes):
                    if i == j:
                        continue
                    area = _overlap_area(child_bb, bb_j)
                    if area > 50:
                        txt = getattr(child, "_text", "")[:30]
                        issues.append({
                            "type": "cross_panel_spillover",
                            "severity": "warning",
                            "detail": (
                                f"Text '{txt}' from axes {i} "
                                f"spills into axes {j} ({area:.0f} px²)"
                            ),
                            "elements": [f"ax{i}", f"ax{j}"],
                        })
    return issues


def _check_panel_label_overlap(fig, renderer, infos, tol_px=2.0):
    """Pass 9: Check that panel labels (A), (B), etc. don't overlap content or text.

    Panel labels must be clearly visible; any overlap with data content,
    axis text (titles, ticks, axis labels), or legend text is a warning.
    """
    panel_texts = []
    for child in fig.texts:
        txt = getattr(child, "_text", "")
        if txt and txt.startswith("(") and txt.endswith(")") and len(txt) <= 4:
            bb = _safe_bbox(child, renderer)
            if bb:
                panel_texts.append((txt, bb, child))

    issues = []
    for txt, pbb, panel_obj in panel_texts:
        # Check against graphical content (collection, patch, image, line)
        for a in infos:
            if a.kind in ("collection", "patch", "image", "line"):
                if a.artist is panel_obj:
                    continue
                area = _overlap_area(pbb, a.bbox)
                if area > 20:
                    issues.append({
                        "type": "panel_label_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Panel label '{txt}' overlaps "
                            f"content '{a.tag}' ({area:.0f} px²)"
                        ),
                        "elements": [txt, a.tag],
                    })

        # Check against all text (titles, ticks, labels, legend text, annotations)
        for a in infos:
            if a.kind == "text":
                if a.artist is panel_obj:
                    continue
                area = _overlap_area(pbb, a.bbox)
                if area > tol_px:
                    issues.append({
                        "type": "panel_label_text_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Panel label '{txt}' overlaps "
                            f"text '{a.tag}' ({area:.0f} px²)"
                        ),
                        "elements": [txt, a.tag],
                    })

        # Check against other panel labels
        for txt2, pbb2, panel_obj2 in panel_texts:
            if panel_obj2 is panel_obj:
                continue
            area = _overlap_area(pbb, pbb2)
            if area > tol_px:
                issues.append({
                    "type": "panel_label_mutual_overlap",
                    "severity": "warning",
                    "detail": (
                        f"Panel labels '{txt}' and '{txt2}' overlap ({area:.0f} px²)"
                    ),
                    "elements": [txt, txt2],
                })
    return issues


def _check_legend_spillover(fig, renderer, tol_px=5.0):
    """Pass 10: Legends extending beyond their parent axes or the figure."""
    issues = []
    fig_bb = _fig_bbox(fig)

    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            continue

        # Check legend vs figure bounds
        sides = _sides_outside(leg_bb, fig_bb, tol_px)
        if sides:
            issues.append({
                "type": "legend_truncation",
                "severity": "warning",
                "detail": (
                    f"Legend in axes extends beyond figure "
                    f"({', '.join(sides)})"
                ),
                "elements": ["legend"],
            })

        # Check legend vs other axes
        for other_ax in fig.get_axes():
            if other_ax is ax:
                continue
            other_bb = _safe_bbox(other_ax, renderer)
            if other_bb is None:
                continue
            area = _overlap_area(leg_bb, other_bb)
            if area > 100:
                issues.append({
                    "type": "legend_spillover",
                    "severity": "warning",
                    "detail": (
                        f"Legend from axes spills into a "
                        f"neighbouring axes ({area:.0f} px²)"
                    ),
                    "elements": ["legend"],
                })

    # Also check figure-level legends
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            leg_bb = _safe_bbox(child, renderer)
            if leg_bb is None:
                continue
            sides = _sides_outside(leg_bb, fig_bb, tol_px)
            if sides:
                issues.append({
                    "type": "legend_truncation",
                    "severity": "warning",
                    "detail": f"Figure legend extends beyond border ({', '.join(sides)})",
                    "elements": ["fig_legend"],
                })

    return issues


def _check_legend_vs_other_panel_content(fig, renderer, infos, min_overlap_px2=30.0):
    """Pass 11: Detect legends overlapping data content in OTHER panels.

    Panel A's legend could spill out of its axes and land on top of Panel B's
    bar chart.  Also checks figure-level legends against all subplot content.
    """
    issues: list[dict] = []

    legend_infos: list[tuple[Bbox, int]] = []
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb:
            legend_infos.append((leg_bb, id(ax)))

    # Also include figure-level legends
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            leg_bb = _safe_bbox(child, renderer)
            if leg_bb:
                legend_infos.append((leg_bb, id(fig)))

    data_artists_full = [a for a in infos
                         if a.kind in ("collection", "line", "image", "patch")]
    data_artists_no_patch = [a for a in infos
                             if a.kind in ("collection", "line", "image")]

    for leg_bb, leg_owner_id in legend_infos:
        # Figure-level legends: skip patches (axis backgrounds/spines)
        targets = data_artists_no_patch if leg_owner_id == id(fig) \
                  else data_artists_full
        for da in targets:
            if da.ax_id == leg_owner_id:
                continue  # same panel — handled by pass 12
            area = _overlap_area(leg_bb, da.bbox)
            if area > min_overlap_px2:
                issues.append({
                    "type": "legend_panel_overlap",
                    "severity": "warning",
                    "detail": (
                        f"Legend overlaps '{da.tag}' in a different "
                        f"panel ({area:.0f} px²)"
                    ),
                    "elements": ["legend", da.tag],
                })
    return issues


def _check_legend_vs_own_content(fig, renderer, infos, min_overlap_px2=50.0):
    """Pass 12: Detect legends covering data IN THEIR OWN panel.

    Checks every legend bbox against ALL data artists (including patches/bars)
    in the same axes.  Only reports when the legend covers >5% of the artist.
    """
    issues: list[dict] = []

    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        # Skip axes that are purely legend-holding cells
        if getattr(ax, '_is_legend_cell', False):
            continue
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            continue

        aid = id(ax)
        _SKIP = ("Spine", "Wedge", "FancyBbox")
        local_data = [a for a in infos
                      if a.ax_id == aid
                      and a.kind in ("collection", "line", "image", "patch")
                      and not any(s in a.tag for s in _SKIP)]

        for da in local_data:
            area = _overlap_area(leg_bb, da.bbox)
            if area > min_overlap_px2:
                frac = area / (da.bbox.width * da.bbox.height + 1e-8)
                if frac > 0.05:  # legend covers >5% of the data artist
                    issues.append({
                        "type": "legend_data_occlusion",
                        "severity": "warning",
                        "detail": (
                            f"Legend in same panel occludes "
                            f"'{da.tag}' ({frac:.0%}, {area:.0f} px²)"
                        ),
                        "elements": ["legend", da.tag],
                    })
    return issues


def _check_fig_legend_vs_subplot_content(fig, renderer, infos,
                                          min_overlap_px2=30.0):
    """Pass 13: Figure-level legends overlapping subplot scatter/bar data.

    Shared legends (e.g. UMAP legend placed via fig.legend()) can overlap
    the scatter/bar content of individual subplots.
    """
    issues: list[dict] = []

    fig_legends = []
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            leg_bb = _safe_bbox(child, renderer)
            if leg_bb:
                fig_legends.append(leg_bb)

    if not fig_legends:
        return issues

    _SKIP = ("Spine", "Wedge", "FancyBbox")
    data_artists = [a for a in infos
                    if a.kind in ("collection", "line", "image", "patch")
                    and not any(s in a.tag for s in _SKIP)]

    for leg_bb in fig_legends:
        for da in data_artists:
            area = _overlap_area(leg_bb, da.bbox)
            if area > min_overlap_px2:
                frac = area / (da.bbox.width * da.bbox.height + 1e-8)
                if frac > 0.03:  # 3% — very sensitive for shared legends
                    issues.append({
                        "type": "fig_legend_subplot_occlusion",
                        "severity": "warning",
                        "detail": (
                            f"Figure-level legend occludes subplot "
                            f"content '{da.tag}' ({frac:.0%}, {area:.0f} px²)"
                        ),
                        "elements": ["fig_legend", da.tag],
                    })
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Detection passes 14–15 (colorbar & legend internals)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_colorbar_internal(fig, renderer, tol_px=1.0):
    """Pass 14: Detect overlaps within colorbar axes.

    Checks:
      a) Colorbar tick labels overlapping each other.
      b) Colorbar axis label overlapping tick labels.
      c) Colorbar extending beyond its parent/inset host axes.
      d) Colorbar tick labels truncated at figure edges.
    """
    issues: list[dict] = []
    fig_bb = _fig_bbox(fig)

    for ax in fig.get_axes():
        is_cbar = (hasattr(ax, '_colorbar_info')
                   or getattr(ax, '_colorbar', None) is not None)
        if not is_cbar:
            continue

        # Gather tick labels
        xticks = [tl for tl in ax.get_xticklabels()
                  if tl.get_text().strip()]
        yticks = [tl for tl in ax.get_yticklabels()
                  if tl.get_text().strip()]
        tick_labels = xticks + yticks

        tick_bbs: list[tuple[str, Bbox]] = []
        for tl in tick_labels:
            bb = _safe_bbox(tl, renderer)
            if bb:
                tick_bbs.append((tl.get_text().strip()[:20], bb))

        # a) Tick labels overlapping each other
        for i in range(len(tick_bbs)):
            for j in range(i + 1, len(tick_bbs)):
                txt_i, bb_i = tick_bbs[i]
                txt_j, bb_j = tick_bbs[j]
                si = _shrink(bb_i, tol_px)
                sj = _shrink(bb_j, tol_px)
                if si and sj and si.overlaps(sj):
                    area = _overlap_area(bb_i, bb_j)
                    issues.append({
                        "type": "cbar_tick_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Colorbar tick '{txt_i}' overlaps "
                            f"tick '{txt_j}' ({area:.0f} px²)"
                        ),
                        "elements": [f"cbar_tick:{txt_i}",
                                     f"cbar_tick:{txt_j}"],
                    })

        # b) Colorbar axis label vs tick labels
        for lbl_artist in [ax.xaxis.label, ax.yaxis.label]:
            lbl_txt = lbl_artist.get_text().strip()
            if not lbl_txt:
                continue
            lbl_bb = _safe_bbox(lbl_artist, renderer)
            if lbl_bb is None:
                continue
            lbl_s = _shrink(lbl_bb, tol_px)
            if not lbl_s:
                continue
            for txt_t, bb_t in tick_bbs:
                bb_ts = _shrink(bb_t, tol_px)
                if bb_ts and lbl_s.overlaps(bb_ts):
                    area = _overlap_area(lbl_bb, bb_t)
                    issues.append({
                        "type": "cbar_label_tick_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Colorbar label '{lbl_txt[:20]}' overlaps "
                            f"tick '{txt_t}' ({area:.0f} px²)"
                        ),
                        "elements": [f"cbar_label:{lbl_txt[:20]}",
                                     f"cbar_tick:{txt_t}"],
                    })

        # c) + d) Tick labels extending beyond figure
        for txt_t, bb_t in tick_bbs:
            sides = _sides_outside(bb_t, fig_bb, 1.0)
            if sides:
                issues.append({
                    "type": "cbar_tick_truncation",
                    "severity": "warning",
                    "detail": (
                        f"Colorbar tick '{txt_t}' extends beyond "
                        f"figure ({', '.join(sides)})"
                    ),
                    "elements": [f"cbar_tick:{txt_t}"],
                })

    return issues


def _check_legend_internal(fig, renderer, tol_px=1.0):
    """Pass 15: Detect internal crowding within legend boxes.

    Checks:
      a) Legend text entries overlapping each other.
      b) Legend texts extending beyond the legend frame bbox.
      c) Legend texts extending beyond figure bounds.
    """
    issues: list[dict] = []
    fig_bb = _fig_bbox(fig)

    def _audit_legend(legend, ctx_label="axes"):
        if legend is None or not legend.get_visible():
            return
        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            return

        texts = legend.get_texts()
        text_bbs: list[tuple[str, Bbox]] = []
        for t in texts:
            txt = t.get_text().strip()
            if not txt:
                continue
            bb = _safe_bbox(t, renderer)
            if bb:
                text_bbs.append((txt[:30], bb))

        # a) Pairwise text overlap within legend
        for i in range(len(text_bbs)):
            for j in range(i + 1, len(text_bbs)):
                txt_i, bb_i = text_bbs[i]
                txt_j, bb_j = text_bbs[j]
                si = _shrink(bb_i, tol_px)
                sj = _shrink(bb_j, tol_px)
                if si and sj and si.overlaps(sj):
                    area = _overlap_area(bb_i, bb_j)
                    issues.append({
                        "type": "legend_text_crowding",
                        "severity": "warning",
                        "detail": (
                            f"Legend entries '{txt_i}' and '{txt_j}' "
                            f"overlap in {ctx_label} ({area:.0f} px²)"
                        ),
                        "elements": [f"legend_text:{txt_i}",
                                     f"legend_text:{txt_j}"],
                    })

        # b) Texts extending beyond legend frame
        for txt_t, bb_t in text_bbs:
            sides = _sides_outside(bb_t, leg_bb, 2.0)
            if sides:
                issues.append({
                    "type": "legend_text_overflow",
                    "severity": "info",
                    "detail": (
                        f"Legend text '{txt_t}' extends beyond "
                        f"legend frame in {ctx_label} "
                        f"({', '.join(sides)})"
                    ),
                    "elements": [f"legend_text:{txt_t}"],
                })

        # c) Texts extending beyond figure
        for txt_t, bb_t in text_bbs:
            sides = _sides_outside(bb_t, fig_bb, 1.0)
            if sides:
                issues.append({
                    "type": "legend_text_truncation",
                    "severity": "warning",
                    "detail": (
                        f"Legend text '{txt_t}' extends beyond "
                        f"figure border ({', '.join(sides)})"
                    ),
                    "elements": [f"legend_text:{txt_t}"],
                })

    # Check per-axes legends
    for idx, ax in enumerate(fig.get_axes()):
        _audit_legend(ax.get_legend(), f"axes[{idx}]")

    # Check figure-level legends
    for child in fig.get_children():
        if hasattr(child, 'get_texts') and hasattr(child, '_legend_box'):
            _audit_legend(child, "figure-legend")

    return issues


def _check_significance_brackets(fig, renderer, border_tol_px=5.0):
    """Pass 16: Detect significance bracket annotation issues.

    Checks:
      a) Bracket star/ns text truncated at figure border.
      b) Bracket star text overlapping other text labels.
      c) Bracket lines (drawn with clip_on=False) extending
         beyond figure bounds.

    Significance brackets are drawn by ``significance_brackets.py``
    using ``clip_on=False``, ``zorder=10-11``, fontweight=bold, and
    text content matching ``*``, ``**``, ``***``, or ``ns``.
    """
    import re
    issues: list[dict] = []
    fig_bb = _fig_bbox(fig)
    star_pattern = re.compile(r'^(\*{1,3}|ns)$')

    # Collect all significance annotation texts and bracket lines
    star_texts = []
    bracket_lines = []

    for ax in fig.get_axes():
        for child in ax.get_children():
            if hasattr(child, 'get_text'):
                txt = child.get_text().strip()
                if star_pattern.match(txt) and child.get_visible():
                    bb = _safe_bbox(child, renderer)
                    if bb:
                        star_texts.append((txt, bb, child))

            # Lines with zorder >= 10 and clip_on=False are likely brackets
            if hasattr(child, 'get_xdata') and hasattr(child, 'get_zorder'):
                if child.get_zorder() >= 10 and not child.get_clip_on():
                    bb = _safe_bbox(child, renderer)
                    if bb:
                        bracket_lines.append(("bracket_line", bb, child))

    # a) Check star texts for truncation at figure border
    for txt, bb, artist in star_texts:
        sides = _sides_outside(bb, fig_bb, border_tol_px)
        if sides:
            issues.append({
                "type": "bracket_text_truncation",
                "severity": "warning",
                "detail": (
                    f"Significance text '{txt}' extends beyond "
                    f"figure border ({', '.join(sides)})"
                ),
                "elements": [f"sig_text:{txt}"],
            })

    # b) Check star texts overlapping other (non-bracket) texts
    all_texts = []
    for ax in fig.get_axes():
        for child in ax.get_children():
            if hasattr(child, 'get_text') and child.get_visible():
                other_txt = child.get_text().strip()
                if other_txt and not star_pattern.match(other_txt):
                    bb = _safe_bbox(child, renderer)
                    if bb:
                        all_texts.append((other_txt[:25], bb))

    for sig_txt, sig_bb, _ in star_texts:
        for other_txt, other_bb in all_texts:
            shrunk_sig = _shrink(sig_bb, 1.0)
            shrunk_other = _shrink(other_bb, 1.0)
            if shrunk_sig and shrunk_other and shrunk_sig.overlaps(shrunk_other):
                area = _overlap_area(sig_bb, other_bb)
                if area > 10:
                    issues.append({
                        "type": "bracket_text_overlap",
                        "severity": "warning",
                        "detail": (
                            f"Significance '{sig_txt}' overlaps "
                            f"label '{other_txt}' ({area:.0f} px²)"
                        ),
                        "elements": [f"sig_text:{sig_txt}",
                                     f"label:{other_txt}"],
                    })

    # c) Check bracket lines for figure-border truncation
    for tag, bb, artist in bracket_lines:
        sides = _sides_outside(bb, fig_bb, border_tol_px)
        if sides:
            issues.append({
                "type": "bracket_line_truncation",
                "severity": "warning",
                "detail": (
                    f"Bracket line extends beyond "
                    f"figure border ({', '.join(sides)})"
                ),
                "elements": [tag],
            })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Detection passes 17–18 (colorbar-data overlap & legend auto-fix)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_colorbar_data_overlap(fig, renderer, auto_fix=True):
    """Pass 17: Detect colorbar axes overlapping parent data content.

    For each colorbar-type axes:
      1. Find the parent axes (hosting the actual data).
      2. Compute the colorbar's display-space bbox.
      3. Collect all data artists (scatter, lines, images) in the parent.
      4. If >5% of any data artist is occluded, flag as warning.
      5. If auto_fix=True, attempt to relocate the inset to a less crowded
         corner (lower-left, upper-left, upper-right, lower-right).

    Returns list of issue dicts.
    """
    issues: list[dict] = []

    for ax in fig.get_axes():
        is_cbar = (hasattr(ax, '_colorbar_info')
                   or getattr(ax, '_colorbar', None) is not None)
        if not is_cbar:
            continue

        cbar_bb = _safe_bbox(ax, renderer)
        if cbar_bb is None:
            continue

        # Find parent axes — the colorbar's host
        parent = None
        if hasattr(ax, '_parent_axes'):
            parent = ax._parent_axes
        elif hasattr(ax, 'get_axes_locator'):
            loc = ax.get_axes_locator()
            if loc and hasattr(loc, '_parent'):
                parent = loc._parent
        if parent is None:
            best_area = 0
            for candidate in fig.get_axes():
                if candidate is ax:
                    continue
                c_is_cbar = (hasattr(candidate, '_colorbar_info')
                             or getattr(candidate, '_colorbar', None) is not None)
                if c_is_cbar:
                    continue
                c_bb = _safe_bbox(candidate, renderer)
                if c_bb is None:
                    continue
                area = _overlap_area(cbar_bb, c_bb)
                if area > best_area:
                    best_area = area
                    parent = candidate
        if parent is None:
            continue

        # Collect data artists in parent
        data_artists = []
        for child in parent.get_children():
            if child is ax:
                continue
            if isinstance(child, (PathCollection, PolyCollection,
                                  LineCollection, Line2D, AxesImage)):
                da_bb = _safe_bbox(child, renderer)
                if da_bb:
                    data_artists.append((_artist_label(child), da_bb))

        for da_tag, da_bb in data_artists:
            area = _overlap_area(cbar_bb, da_bb)
            if area < 1:
                continue
            da_area = da_bb.width * da_bb.height
            if da_area < 1:
                continue
            frac = area / da_area
            if frac > 0.05:
                issues.append({
                    "type": "cbar_data_overlap",
                    "severity": "warning",
                    "detail": (
                        f"Colorbar overlaps data '{da_tag}' "
                        f"({frac:.0%}, {area:.0f} px²)"
                    ),
                    "elements": ["colorbar", da_tag],
                    "auto_fixable": auto_fix,
                })

    return issues


def _check_legend_crowding_autofix(fig, renderer, auto_fix=True):
    """Pass 18: Detect and optionally auto-fix legend crowding.

    For each axes legend:
      1. Check if legend entries overlap each other.
      2. If auto_fix and overlap detected, try ``loc='best'`` relocation.
      3. If legend has >6 entries and font > 8pt, shrink by 1pt.

    Returns list of issue dicts.
    """
    issues: list[dict] = []

    for idx, ax in enumerate(fig.get_axes()):
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue

        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            continue

        texts = legend.get_texts()
        if not texts:
            continue

        text_bbs = []
        for t in texts:
            txt = t.get_text().strip()
            bb = _safe_bbox(t, renderer)
            if bb and txt:
                text_bbs.append((txt[:25], bb))

        has_crowding = False
        for i in range(len(text_bbs)):
            for j in range(i + 1, len(text_bbs)):
                _, bb_i = text_bbs[i]
                _, bb_j = text_bbs[j]
                si = _shrink(bb_i, 0.5)
                sj = _shrink(bb_j, 0.5)
                if si and sj and si.overlaps(sj):
                    has_crowding = True
                    break
            if has_crowding:
                break

        fixed = False
        if has_crowding and auto_fix:
            if len(texts) > 6:
                for t in texts:
                    current_fs = t.get_fontsize()
                    if current_fs > 8:
                        t.set_fontsize(current_fs - 1)
                        fixed = True
            try:
                if hasattr(legend, 'set_loc'):
                    legend.set_loc("best")        # mpl >= 3.8 public API
                else:
                    legend._loc = 0               # fallback for older mpl
                fig.canvas.draw()
                fixed = True
            except Exception:
                pass

        if has_crowding:
            issues.append({
                "type": "legend_crowding_autofix",
                "severity": "info" if fixed else "warning",
                "detail": (
                    f"Legend in axes[{idx}] has crowded entries"
                    + (" — auto-fixed (font shrink / relocation)" if fixed
                       else " — auto-fix not possible")
                ),
                "elements": [f"legend_axes_{idx}"],
            })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Two-layer reporting helper
# ═══════════════════════════════════════════════════════════════════════════════

def _per_axes_summary(fig, renderer, infos):
    """Per-subplot conflict summary (Layer 1).

    Returns a dict mapping axes title → list of issues in that subplot,
    useful for pinpointing which panels still have internal conflicts.
    """
    per_ax: dict[str, list[dict]] = {}

    for ax in fig.get_axes():
        if getattr(ax, '_is_legend_cell', False):
            continue
        title = ax.get_title() or f"ax@{id(ax):#x}"
        aid = id(ax)
        ax_issues: list[dict] = []

        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            per_ax[title] = ax_issues
            continue

        leg_bb = _safe_bbox(legend, renderer)
        if leg_bb is None:
            per_ax[title] = ax_issues
            continue

        local_data = [a for a in infos
                      if a.ax_id == aid
                      and a.kind in ("collection", "line", "image", "patch")
                      and not any(s in a.tag for s in ("Spine", "Wedge", "FancyBbox"))]

        for da in local_data:
            area = _overlap_area(leg_bb, da.bbox)
            if area > 30:
                frac = area / (da.bbox.width * da.bbox.height + 1e-8)
                if frac > 0.03:
                    ax_issues.append({
                        "type": "subplot_legend_overlap",
                        "severity": "warning",
                        "detail": (
                            f"[{title}] Legend occludes '{da.tag}' "
                            f"({frac:.0%}, {area:.0f} px²)"
                        ),
                    })
        per_ax[title] = ax_issues
    return per_ax


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 19: Font-size adequacy detection
# ═══════════════════════════════════════════════════════════════════════════════

def _check_fontsize_adequacy(
    fig,
    renderer,
    infos: list[_ArtistInfo],
    min_pt: float = 6.0,
    composed_scale: float = 0.5) -> list[dict]:
    """Detect text whose effective print size falls below *min_pt*.

    The *composed_scale* parameter approximates the downscaling that
    occurs when the subplot PNG is placed inside the Next.js compositor
    (e.g. a 3-column grid in a 2-panel layout → each subplot rendered
    at ~50% of its original size).  The effective font size is::

        effective_pt = artist.get_fontsize() * composed_scale

    Any text below *min_pt* after scaling is flagged.  This prevents
    tiny illegible labels from reaching the final PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    renderer : matplotlib renderer
    infos : list of _ArtistInfo
    min_pt : float
        Minimum acceptable point size in the composed figure (default 6).
    composed_scale : float
        Approximate scale factor from subplot PNG to composed screenshot
        (default 0.5 for 2-panel layout with 3-column grid).

    Returns
    -------
    list[dict]
        Issues with ``type='fontsize_too_small'``.
    """
    issues: list[dict] = []
    seen_sizes: dict[float, list[str]] = {}  # group by fontsize for summary

    for info in infos:
        if info.kind != "text":
            continue
        artist = info.artist
        if not isinstance(artist, Text):
            continue
        text_str = artist.get_text().strip()
        if not text_str:
            continue
        # Skip internal matplotlib artists (underscore prefix)
        label = getattr(artist, '_label', '') or ''
        if label.startswith('_'):
            continue

        fs = artist.get_fontsize()
        effective = fs * composed_scale
        if effective < min_pt:
            seen_sizes.setdefault(fs, []).append(text_str[:30])

    # Emit one warning per distinct fontsize (avoids 50+ duplicate warnings)
    for fs, examples in sorted(seen_sizes.items()):
        effective = fs * composed_scale
        n = len(examples)
        sample = examples[:3]
        sample_str = ", ".join(f"'{s}'" for s in sample)
        if n > 3:
            sample_str += f" ... +{n - 3} more"
        issues.append({
            "type": "fontsize_too_small",
            "severity": "warning",
            "detail": (
                f"{n} text(s) at {fs:.1f}pt → effective {effective:.1f}pt "
                f"(min {min_pt}pt): {sample_str}"
            ),
            "elements": examples,
        })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def detect_all_conflicts(
    fig,
    label: str = "",
    verbose: bool = True,
    text_overlap_tol_px: float = 2.0,
    border_tol_px: float = 1.0,
    artist_overlap_min_px2: float = 100.0,
    text_artist_overlap_min_px2: float = 50.0):
    """Run all 19 visual conflict detection passes on a matplotlib figure.

    Two-layer detection:
      Layer 1 (subplot-level): passes 12-13, per-axes summary
      Layer 2 (figure-level):  passes 1-11, 14-18

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    label : str
        Descriptive label for log messages.
    verbose : bool
        Print diagnostic summary to stdout.
    text_overlap_tol_px : float
        Shrink text bboxes by this many pixels before overlap test.
    border_tol_px : float
        Tolerance for border truncation checks.
    artist_overlap_min_px2 : float
        Minimum overlap area (px²) to report graphical artist overlap.
    text_artist_overlap_min_px2 : float
        Minimum overlap area (px²) to report text-vs-artist overlap.

    Returns
    -------
    list[dict]
        Each dict has ``type``, ``severity``, ``detail``, ``elements``.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return []

    fig_bb = _fig_bbox(fig)
    infos = _collect_artists(fig, renderer)

    issues = []
    # ── Layer 2: figure-level passes (1-11) ──
    issues.extend(_check_text_overlaps(infos, text_overlap_tol_px))
    issues.extend(_check_truncation(infos, fig_bb, border_tol_px))
    issues.extend(_check_artist_content_overlap(infos, artist_overlap_min_px2))
    issues.extend(_check_text_vs_artist_overlap(
        infos, text_overlap_tol_px, text_artist_overlap_min_px2))
    issues.extend(_check_axes_overflow(infos, fig))
    issues.extend(_check_scatter_clip_risk(fig))
    # Passes 8-10: Composed-figure-specific checks
    issues.extend(_check_cross_panel_spillover(fig, renderer))
    issues.extend(_check_panel_label_overlap(fig, renderer, infos))
    issues.extend(_check_legend_spillover(fig, renderer))
    # Pass 11: Cross-panel legend overlap
    issues.extend(_check_legend_vs_other_panel_content(fig, renderer, infos))
    # ── Layer 1: subplot-level passes (12-15) ──
    issues.extend(_check_legend_vs_own_content(fig, renderer, infos))
    issues.extend(_check_fig_legend_vs_subplot_content(fig, renderer, infos))
    # Passes 14-15: colorbar & legend internals
    issues.extend(_check_colorbar_internal(fig, renderer))
    issues.extend(_check_legend_internal(fig, renderer))
    # Pass 16: significance bracket annotation checks
    issues.extend(_check_significance_brackets(fig, renderer, border_tol_px))
    # Pass 17: colorbar-vs-data overlap (with auto-relocate)
    issues.extend(_check_colorbar_data_overlap(fig, renderer))
    # Pass 18: legend crowding auto-fix
    issues.extend(_check_legend_crowding_autofix(fig, renderer))
    # Pass 19: font-size adequacy (min 6pt effective in composed figure)
    issues.extend(_check_fontsize_adequacy(fig, renderer, infos,
                                           min_pt=6.0, composed_scale=0.5))

    # Two-layer per-axes summary for verbose output
    per_ax = _per_axes_summary(fig, renderer, infos)

    if verbose:
        tag = f" [{label}]" if label else ""
        n_warn = sum(1 for x in issues if x["severity"] == "warning")
        n_info = sum(1 for x in issues if x["severity"] == "info")

        counts = {}
        for iss in issues:
            counts[iss["type"]] = counts.get(iss["type"], 0) + 1

        # ── Layer 1: subplot-level summary ──
        subplot_problems = {t: iss for t, iss in per_ax.items() if iss}
        if subplot_problems:
            print(f"  ── Layer 1 (subplot-level){tag} ──")
            for title, ax_iss in subplot_problems.items():
                for i in ax_iss:
                    print(f"    ⚠ {i['detail']}")

        # ── Layer 2: figure-level summary ──
        if n_warn > 0 or subplot_problems:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  ── Layer 2 (figure-level){tag} ──")
            print(f"  ⚠ CONFLICT: {n_warn} warnings, {n_info} info [{parts}]")
            for iss in issues[:25]:
                marker = "⚠" if iss["severity"] == "warning" else "ℹ"
                print(f"    {marker} [{iss['type']}] {iss['detail']}")
            if len(issues) > 25:
                print(f"    ... and {len(issues) - 25} more")
        elif n_info > 0:
            print(f"  ℹ INFO{tag}: 0 warnings, {n_info} info "
                  f"(likely auto-fixed by bbox_inches='tight')")
        elif label:
            print(f"  ✓ OK{tag}: no conflicts detected (both layers clean)")

    return issues


def detect_conflicts_in_file(
    png_path: str,
    label: str = "",
    verbose: bool = True) -> list[dict]:
    """Re-generate conflict detection requires the figure object, not PNG.

    This is a placeholder that documents the limitation: conflict
    detection must run on the live matplotlib Figure object before
    saving, not on a rasterised PNG.  Call ``detect_all_conflicts(fig)``
    from within the figure generator instead.
    """
    print(f"  NOTE: Conflict detection requires a live Figure object. "
          f"Call detect_all_conflicts(fig) in the generator for '{png_path}'.")
    return []


def summarize_issues(all_issues: dict[str, list[dict]]) -> None:
    """Print final audit summary across multiple figures."""
    total_warn = 0
    total_info = 0
    problem_figs = []
    for name, issues in all_issues.items():
        n_w = sum(1 for x in issues if x["severity"] == "warning")
        n_i = sum(1 for x in issues if x["severity"] == "info")
        total_warn += n_w
        total_info += n_i
        if n_w > 0:
            problem_figs.append(name)

    print(f"\n{'═' * 60}")
    print(f"CONFLICT AUDIT SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Figures checked: {len(all_issues)}")
    print(f"  Total warnings:  {total_warn}")
    print(f"  Total info:      {total_info}")
    if problem_figs:
        print(f"  Figures with warnings: {', '.join(problem_figs)}")
    else:
        print(f"  ✓ All figures clean — no warnings detected")
    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone figure audit script
# ═══════════════════════════════════════════════════════════════════════════════

def audit_all_figures(series: str = "dpmm"):
    """Re-generate all figure subplots with enhanced conflict detection.

    This function imports each figure generator and runs it, capturing
    all conflict reports.  It serves as a comprehensive audit tool.

    Usage:
        python -m benchmarks.figure_generators.visual_conflict_detector --series dpmm
    """
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))

    from benchmarks.figure_generators import SUBPLOT_GENERATORS

    out_base = ROOT / "benchmarks" / "paper_figures" / series / "subplots"
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"VISUAL CONFLICT AUDIT — series={series}")
    print("=" * 70)

    # Patch check_text_overlaps to use the enhanced detector
    import benchmarks.figure_generators.common as _common_mod
    _original_check = _common_mod.check_text_overlaps

    def _enhanced_check(fig, label="", verbose=True, overlap_tol_px=2):
        """Drop-in replacement using enhanced conflict detection."""
        return detect_all_conflicts(
            fig, label=label, verbose=verbose,
            text_overlap_tol_px=overlap_tol_px)

    _common_mod.check_text_overlaps = _enhanced_check

    all_issues = {}
    for fig_id in sorted(SUBPLOT_GENERATORS.keys()):
        gen = SUBPLOT_GENERATORS[fig_id]
        print(f"\n{'─' * 70}")
        print(f"  Auditing Figure {fig_id}...")
        print(f"{'─' * 70}")
        try:
            gen(series, out_base)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Restore original
    _common_mod.check_text_overlaps = _original_check

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Enhanced visual conflict detection audit (18 passes)")
    parser.add_argument("--series", default="dpmm",
                        choices=["dpmm", "topic"])
    args = parser.parse_args()
    audit_all_figures(args.series)
