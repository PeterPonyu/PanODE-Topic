"""Shared subplot style constants — geometry-based figure system.

Uses the PanODE deterministic geometry engine (``src.visualization``)
instead of matplotlib's GridSpec/subplots for publication figures.
Layout is controlled via ``LayoutRegion.split_rows/split_cols/grid``
and ``bind_figure_region`` rather than ``plt.subplots()``.

MDPI column calibration: 17 cm text width on A4 (170 mm).
At 300 DPI the full-width figure is 6.7 inches.
"""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import sys

# Ensure src.visualization is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.visualization import (
    apply_style as _apply_geometry_style,
    style_axes,
    add_panel_label,
    save_with_vcd,
    bind_figure_region,
    LayoutRegion,
    VIS_STYLE,
    MODEL_COLORS as _MODEL_COLORS,
    COLORS as _COLORS,
    FONT_TITLE as _FONT_TITLE,
    FONT_LABEL as _FONT_LABEL,
    FONT_TICK as _FONT_TICK,
    FONT_LEGEND as _FONT_LEGEND,
    FONT_ANNOTATION as _FONT_ANNOTATION)

_log = logging.getLogger(__name__)

# ── Font fallback chain (strict Arial enforcement) ───────────────────────────
_FONT_FALLBACK_CHAIN = [
    "Arial", "DejaVu Sans", "Helvetica", "Liberation Sans", "sans-serif",
]


def _resolve_font_dir() -> Path | None:
    """Resolve the Arial font directory with a 3-tier fallback.

    1. ``$PANODE_FONT_DIR`` environment variable (CI / other machines)
    2. ``<repo_root>/fonts/`` (project-relative)
    3. ``~/Desktop/fonts`` (legacy developer path)

    Returns the first existing directory, or ``None``.
    """
    import os
    candidates = [
        os.environ.get("PANODE_FONT_DIR", ""),
        str(Path(__file__).resolve().parent.parent.parent / "fonts"),
        str(Path.home() / "Desktop" / "fonts"),
    ]
    for c in candidates:
        p = Path(c)
        if c and p.is_dir():
            return p
    return None


# ── Register Arial from the resolved fonts directory ─────────────────────────
_FONT_DIR = _resolve_font_dir()
if _FONT_DIR is not None and _FONT_DIR.is_dir():
    for _f in _FONT_DIR.glob("Arial*.ttf"):
        fm.fontManager.addfont(str(_f))
    mpl.rcParams["font.family"] = "Arial"


def validate_arial_available() -> bool:
    """Check whether Arial is available in matplotlib's font manager.

    Returns ``True`` if Arial can be resolved, ``False`` otherwise.
    A warning is logged (not raised) when Arial is missing so that
    downstream code can still run with the next font in the fallback chain.
    """
    try:
        match = fm.findfont("Arial", fallback_to_default=False)
        if match and "arial" in Path(match).stem.lower():
            return True
    except Exception:
        pass
    _log.warning(
        "Arial font not found in matplotlib font manager. "
        "Figures will fall back to: %s",
        " -> ".join(_FONT_FALLBACK_CHAIN[1:]))
    return False


# Run validation at import time (logs only, never crashes)
_ARIAL_AVAILABLE = validate_arial_available()

# ═══════════════════════════════════════════════════════════════════════════════
# Core constants
# ═══════════════════════════════════════════════════════════════════════════════

SUBPLOT_DPI = 300
SUBPLOT_FONTSIZE = 13          # Base font size in pt (up from 11)
CONTAINER_CSS_PX = 670         # FigureContainer maxWidth in CSS px (17cm)
DPR = 3                        # deviceScaleFactor for Playwright screenshots

# ═══════════════════════════════════════════════════════════════════════════════
# Figsize presets — calibrated to frontend grid columns
# Each preset matches the device pixel width / DPI for the given column count.
# "half" variants are for panels inside a 2-column panel layout (≈332px).
# ═══════════════════════════════════════════════════════════════════════════════

def _col_figsize(ncols, aspect=0.78, gap_css=3, container=None):
    """Compute (width_inches, height_inches) for a given column count."""
    ctn = container or CONTAINER_CSS_PX
    device_px = (ctn - (ncols - 1) * gap_css) * DPR / ncols
    w = round(device_px / SUBPLOT_DPI, 2)
    h = round(w * aspect, 2)
    return (w, h)

_HALF_CTN = (CONTAINER_CSS_PX - 4) // 2     # ≈333 CSS px per panel column

# Standard presets (full-width container)
# NOTE: bbox_inches='tight' in save_subplot auto-expands the canvas
# to include all text, so aspect ratios only set the base proportions.
FIGSIZE_2COL = _col_figsize(2, aspect=0.95)
FIGSIZE_3COL = _col_figsize(3, aspect=1.00)
FIGSIZE_4COL = _col_figsize(4, aspect=0.95)   # Compact for dense grids
FIGSIZE_6COL = _col_figsize(6, aspect=0.95)

# Special presets for specific subplot types (full-width)
FIGSIZE_BOXPLOT = _col_figsize(3, aspect=1.05)       # Room for rotated xlabels + title + brackets
FIGSIZE_UMAP = _col_figsize(3, aspect=1.00)          # Nearly square (no axis labels)
FIGSIZE_SCATTER = _col_figsize(3, aspect=1.00)       # Scatter plots
FIGSIZE_HEATMAP = _col_figsize(3, aspect=1.15)       # Room for rotated gene names + colorbar
FIGSIZE_ENRICHMENT = _col_figsize(3, aspect=1.25)    # Increased height to prevent clipping

# Half-width panel presets (for 2-column panel layout)
FIGSIZE_BOXPLOT_HALF = _col_figsize(3, aspect=1.10, container=_HALF_CTN)
FIGSIZE_UMAP_HALF = _col_figsize(3, aspect=1.05, container=_HALF_CTN)
FIGSIZE_HEATMAP_HALF = _col_figsize(3, aspect=1.25, container=_HALF_CTN)
FIGSIZE_ENRICHMENT_HALF = _col_figsize(3, aspect=1.35, container=_HALF_CTN)

# ═══════════════════════════════════════════════════════════════════════════════
# Scatter / line visual parameters
# Scaled up proportionally to maintain legibility at smaller display size.
# ═══════════════════════════════════════════════════════════════════════════════

SCATTER_SIZE_BOXPLOT = 16      # Jitter dots on boxplots (optim: reduced from 18)
SCATTER_SIZE_UMAP = 3          # UMAP scatter points (optim: reduced from 4)
SCATTER_SIZE_SWEEP = 14        # Sweep boxplot jitter (optim: reduced from 16)
SCATTER_SIZE_HULL = 25         # Scatter+hull trade-off plots (optim: reduced from 30)
LINE_WIDTH_BOX = 0.8           # Boxplot box edges (up from 0.6)
LINE_WIDTH_MEDIAN = 1.2        # Boxplot median line (up from 1.0)
LINE_WIDTH_BRACKET = 0.9       # Significance brackets (up from 0.7)
LINE_WIDTH_SPINE = 0.5         # Axes spine width (up from 0.4)
LINE_WIDTH_GRID = 0.5          # Grid line width (up from 0.4)

# ═══════════════════════════════════════════════════════════════════════════════
# Color themes for external benchmark figures (Figs 10-12)
# Each theme provides a distinct palette for visual separation.
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_THEMES = {
    "proposed":  ["#4E79A7", "#76B7B2", "#B07AA1", "#9C755F"],
    "classical": ["#59A14F", "#8CD17D", "#EDC948", "#F28E2B"],
    "deep":      ["#E15759", "#FF9D9A", "#FF6B6B", "#C75146"],
}


def get_theme_palette(theme_name: str) -> list:
    """Return color palette for the given theme. Falls back to proposed."""
    return COLOR_THEMES.get(theme_name, COLOR_THEMES["proposed"])


# Font size — graduated hierarchy for readability.
# After 2-column panel downscaling (~50%), effective sizes become
# approximately 7/6.5/6/5.5/5.5/5.5 pt → all above the 5pt paper minimum.
FONTSIZE_TITLE = 14          # Subplot titles (most prominent)
FONTSIZE_LABEL = 13          # Axis labels
FONTSIZE_TICK = 12           # Tick labels
FONTSIZE_LEGEND = 11         # Legend text
FONTSIZE_ANNOTATION = 11     # Significance stars, annotations
FONTSIZE_CAPTION = 11        # Captions (rarely used in subplots)

# ═══════════════════════════════════════════════════════════════════════════════
# Style application
# ═══════════════════════════════════════════════════════════════════════════════

def apply_subplot_style():
    """Apply clean publication style using the geometry-based system.

    Delegates to ``src.visualization.apply_style()`` and adds
    PanODE-specific overrides for figure background and spine widths.
    """
    _apply_geometry_style()
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "axes.linewidth": LINE_WIDTH_SPINE,
        "grid.linewidth": LINE_WIDTH_GRID,
    })


def save_subplot(fig, path, dpi=SUBPLOT_DPI):
    """Save and close a subplot figure, running VCD before save.

    Uses ``bbox_inches='tight'`` so matplotlib automatically expands the
    canvas to include all text decorations (titles, tick labels, axis
    labels).  This prevents clipping of long GO term labels or rotated
    gene names.  The frontend CSS (``w-full h-auto``) normalises the
    displayed widths regardless of minor pixel-size differences.

    After saving, a VCD check runs if the detector is available.
    """
    # Delegate to geometry-based save with VCD integration
    save_with_vcd(fig, path, dpi=dpi, close=True, run_vcd=True)
    print(f"    {Path(path).name}  [SAVED]")


def build_manifest(sub_dir, data):
    """Write a JSON manifest file for Next.js to discover subplots.

    Also records actual pixel dimensions of each PNG so the frontend
    can normalise display sizes across subplots of the same type.
    """
    import json
    import struct

    def _png_size(path):
        """Read PNG width×height from the IHDR chunk (no PIL needed)."""
        try:
            with open(path, "rb") as f:
                sig = f.read(8)
                if sig[:4] != b'\x89PNG':
                    return None
                _length = struct.unpack(">I", f.read(4))[0]
                _chunk_type = f.read(4)
                w = struct.unpack(">I", f.read(4))[0]
                h = struct.unpack(">I", f.read(4))[0]
                return {"w": w, "h": h}
        except Exception:
            return None

    # Scan all PNGs in the directory and record sizes
    sizes = {}
    for png in sub_dir.glob("*.png"):
        sz = _png_size(png)
        if sz:
            sizes[png.name] = sz
    data["image_sizes"] = sizes

    manifest_path = sub_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"    manifest: {manifest_path.name}")
    return data
