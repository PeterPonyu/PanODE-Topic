#!/usr/bin/env python
"""
Compose per-group metric PDFs into paper-ready multi-panel figures.

The source metric groups are the six per-group PDFs produced by
``regenerate_figures.py``:

    clustering, dre_umap, dre_tsne, lse_intrinsic, drex, lsex

The default ``auto`` layout applies policy by experiment role:

    ablation (Figure 2): strict portrait_3x2 (3 rows)
    vs_external (Figures 10–12 panels): semantic paper layout

The semantic ``paper`` layout is portrait-friendly:

    row 1: clustering
    row 2: dre_umap | dre_tsne
    row 3: lse_intrinsic
    row 4: drex | lsex

This keeps the composed metrics figure aligned with the 17 cm article width
used by the DPMM / Topic paper figures, while preserving legibility for the
wide source panels.  Two alternate layouts are also available:

    portrait_3x2   - strict 3x2 portrait grid
    landscape_2x3  - legacy wide grid

Use after ``regenerate_figures.py``. When experiments use semantic splitting
(``proposed/classical/deep``), each subfolder is composed independently.

Usage
-----
    python scripts/compose_experiment_figures.py
    python scripts/compose_experiment_figures.py --figures-dir experiments/results/dpmm/ablation/figures
    python scripts/compose_experiment_figures.py --all --layout auto
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_lib.viz.layout import COMPOSED_FIG_MAX_HEIGHT_IN, COMPOSED_FIG_WIDTH_IN

# NOTE: Distinct per-group color themes (proposed / classical / deep) are
# defined in benchmarks.figure_generators.subplot_style.COLOR_THEMES and
# retrieved via get_theme_palette().  The composition step here assembles
# pre-rendered PDFs and does not re-colour them; theme colours are applied
# upstream in regenerate_figures.py at figure generation time.

# Canonical metric-group order.
METRIC_GROUP_ORDER = [
    "clustering",
    "dre_umap",
    "dre_tsne",
    "lse_intrinsic",
    "drex",
    "lsex",
]

PANEL_TITLES = {
    "clustering": "Clustering",
    "dre_umap": "DRE-UMAP",
    "dre_tsne": "DRE-tSNE",
    "lse_intrinsic": "LSE-Intrinsic",
    "drex": "DREX",
    "lsex": "LSEX",
}

LAYOUT_SPECS = {
    # Best match for the paper container: portrait width, content-fitted height,
    # and semantic row grouping to avoid over-shrinking the widest panels.
    "paper": {
        "rows": [
            ["clustering"],
            ["dre_umap", "dre_tsne"],
            ["lse_intrinsic"],
            ["drex", "lsex"],
        ],
        "fig_width_in": COMPOSED_FIG_WIDTH_IN,
        "max_height_in": COMPOSED_FIG_MAX_HEIGHT_IN,
        "margins_in": {"left": 0.14, "right": 0.14, "top": 0.16, "bottom": 0.12},
        "row_gap_in": 0.13,
        "col_gap_in": 0.08,
        "title_pad_in": 0.15,
        "title_fontsize": 8.8,
    },
    # Strict portrait 3x2 grid for users who want a literal 3-row / 2-column
    # arrangement regardless of the input panel aspect ratios.
    "portrait_3x2": {
        "rows": [
            ["clustering", "dre_umap"],
            ["dre_tsne", "lse_intrinsic"],
            ["drex", "lsex"],
        ],
        "fig_width_in": COMPOSED_FIG_WIDTH_IN,
        "max_height_in": COMPOSED_FIG_MAX_HEIGHT_IN,
        "margins_in": {"left": 0.10, "right": 0.10, "top": 0.12, "bottom": 0.08},
        "row_gap_in": 0.10,
        "col_gap_in": 0.08,
        "title_pad_in": 0.14,
        "title_fontsize": 8.6,
    },
    "landscape_2x3": {
        "rows": [
            ["clustering", "dre_umap", "dre_tsne"],
            ["lse_intrinsic", "drex", "lsex"],
        ],
        "fig_width_in": 18.0,
        "fig_height_in": 10.0,
        "margins_in": {"left": 0.22, "right": 0.22, "top": 0.22, "bottom": 0.18},
        "row_gap_in": 0.24,
        "col_gap_in": 0.18,
        "title_pad_in": 0.24,
        "title_fontsize": 10.5,
    },
    "split_clustering_projection": {
        "rows": [
            ["clustering"],
            ["dre_umap", "dre_tsne"],
        ],
        "fig_width_in": COMPOSED_FIG_WIDTH_IN,
        "max_height_in": COMPOSED_FIG_MAX_HEIGHT_IN * 0.55,
        "margins_in": {"left": 0.14, "right": 0.14, "top": 0.16, "bottom": 0.12},
        "row_gap_in": 0.13,
        "col_gap_in": 0.08,
        "title_pad_in": 0.15,
        "title_fontsize": 8.8,
    },
    "split_latent_structure": {
        "rows": [
            ["lse_intrinsic"],
            ["drex", "lsex"],
        ],
        "fig_width_in": COMPOSED_FIG_WIDTH_IN,
        "max_height_in": COMPOSED_FIG_MAX_HEIGHT_IN * 0.55,
        "margins_in": {"left": 0.14, "right": 0.14, "top": 0.16, "bottom": 0.12},
        "row_gap_in": 0.13,
        "col_gap_in": 0.08,
        "title_pad_in": 0.15,
        "title_fontsize": 8.8,
    },
}

AUTO_LAYOUT = "auto"


def _render_pdf_to_array(pdf_path: Path, dpi: int = 150):
    """Render first page of PDF to numpy array (RGB).

    Uses PyMuPDF if available, else pdftoppm (poppler-utils).
    """
    import numpy as np
    import subprocess
    import tempfile

    # Try PyMuPDF first
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        page = doc[0]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = pix.samples
        h, w = pix.height, pix.width
        doc.close()
        return np.frombuffer(arr, dtype=np.uint8).reshape(h, w, 3)
    except ImportError:
        pass

    # Fallback: pdftoppm (poppler-utils)
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(out_prefix)],
            check=True,
            capture_output=True)
        png_path = Path(tmp) / "page-1.png"
        if not png_path.exists():
            raise RuntimeError("pdftoppm did not produce output")
        from PIL import Image
        img = np.array(Image.open(png_path).convert("RGB"))
    return img


def _crop_white_border(image, threshold: int = 249, pad: int = 10):
    """Trim surrounding white border to maximise usable panel area."""
    import numpy as np

    mask = np.any(image < threshold, axis=2)
    if not np.any(mask):
        return image

    ys, xs = np.where(mask)
    top = max(int(ys.min()) - pad, 0)
    bottom = min(int(ys.max()) + pad + 1, image.shape[0])
    left = max(int(xs.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, image.shape[1])
    return image[top:bottom, left:right]


def _resolve_layout(layout_name: str, width_cm: float | None, max_height_cm: float | None) -> dict:
    """Return a mutable layout specification with optional dimension overrides."""
    if layout_name not in LAYOUT_SPECS:
        raise ValueError(f"Unknown layout '{layout_name}'")

    base = LAYOUT_SPECS[layout_name]
    spec = {
        "rows": [list(row) for row in base["rows"]],
        "fig_width_in": base["fig_width_in"],
        "margins_in": dict(base["margins_in"]),
        "row_gap_in": base["row_gap_in"],
        "col_gap_in": base["col_gap_in"],
        "title_pad_in": base["title_pad_in"],
        "title_fontsize": base["title_fontsize"],
    }
    if "fig_height_in" in base:
        spec["fig_height_in"] = base["fig_height_in"]
    if "max_height_in" in base:
        spec["max_height_in"] = base["max_height_in"]

    if width_cm is not None:
        spec["fig_width_in"] = width_cm / 2.54
    if max_height_cm is not None and "max_height_in" in spec:
        spec["max_height_in"] = max_height_cm / 2.54
    return spec


def _auto_layout_for_dir(figures_dir: Path) -> str:
    """Choose layout by experiment role.

    Figure 2 (ablation) is forced to strict 3-row layout for readability.
    Figures 10–12 split panels (vs_external/proposed|classical|deep) remain in
    paper layout, since separation already reduces panel crowding.
    """
    p = figures_dir.as_posix()
    if "/ablation/figures" in p:
        return "portrait_3x2"
    return "paper"


def _row_wspace(fig_width_in: float, margins_in: dict, row: list[str], col_gap_in: float) -> float:
    """Convert a physical column gap into GridSpec wspace units."""
    if len(row) < 2:
        return 0.0
    usable_width = fig_width_in - margins_in["left"] - margins_in["right"]
    panel_width = (usable_width - col_gap_in * (len(row) - 1)) / len(row)
    return col_gap_in / max(panel_width, 0.1)


def _compute_row_heights(images: dict[str, object], layout_spec: dict) -> list[float]:
    """Estimate row heights (inches) from panel aspect ratios."""
    rows = layout_spec["rows"]
    fig_width_in = layout_spec["fig_width_in"]
    margins_in = layout_spec["margins_in"]
    col_gap_in = layout_spec["col_gap_in"]
    title_pad_in = layout_spec["title_pad_in"]
    usable_width = fig_width_in - margins_in["left"] - margins_in["right"]

    heights = []
    for row in rows:
        panel_width = (usable_width - col_gap_in * (len(row) - 1)) / len(row)
        image_heights = []
        for key in row:
            img = images[key]
            img_h, img_w = img.shape[:2]
            image_heights.append(panel_width * (img_h / max(img_w, 1)))
        heights.append(max(image_heights) + title_pad_in)
    return heights


def _resolve_figure_height(row_heights: list[float], layout_spec: dict) -> float:
    """Return a content-fitted figure height, capped when a max is defined."""
    margins = layout_spec["margins_in"]
    total = (
        margins["top"]
        + margins["bottom"]
        + sum(row_heights)
        + layout_spec["row_gap_in"] * max(len(row_heights) - 1, 0)
    )
    max_height = layout_spec.get("max_height_in")
    if max_height is not None:
        return min(total, max_height)
    return layout_spec["fig_height_in"]


def compose_figures_in_dir(
    figures_dir: Path,
    output_name: str = "composed_metrics",
    dpi: int = 300,
    save_png: bool = True,
    layout: str = AUTO_LAYOUT,
    width_cm: float | None = None,
    max_height_cm: float | None = None,
    metric_groups: list[str] | None = None,
    show_titles: bool = True,
    show_panel_labels: bool = False) -> Path | None:
    """Compose the 6 metric-group PDFs in *figures_dir* into a configured layout.

    Returns the path to the saved composed PDF, or None if any PDF is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved_layout = _auto_layout_for_dir(figures_dir) if layout == AUTO_LAYOUT else layout
    layout_spec = _resolve_layout(resolved_layout, width_cm=width_cm, max_height_cm=max_height_cm)

    pdfs = {}
    for key in (metric_groups if metric_groups else METRIC_GROUP_ORDER):
        pdf_path = figures_dir / f"{key}.pdf"
        if not pdf_path.exists():
            return None
        pdfs[key] = pdf_path

    images = {
        key: _crop_white_border(_render_pdf_to_array(pdf_path, dpi=dpi))
        for key, pdf_path in pdfs.items()
    }

    row_heights = _compute_row_heights(images, layout_spec)
    fig_width_in = layout_spec["fig_width_in"]
    fig_height_in = _resolve_figure_height(row_heights, layout_spec)
    margins = layout_spec["margins_in"]

    avg_row_height = sum(row_heights) / max(len(row_heights), 1)
    outer_hspace = 0.0
    if len(row_heights) > 1:
        outer_hspace = layout_spec["row_gap_in"] / max(avg_row_height, 0.1)

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), facecolor="white")
    fig.subplots_adjust(
        left=margins["left"] / fig_width_in,
        right=1.0 - (margins["right"] / fig_width_in),
        top=1.0 - (margins["top"] / fig_height_in),
        bottom=margins["bottom"] / fig_height_in)

    outer = fig.add_gridspec(
        nrows=len(layout_spec["rows"]),
        ncols=1,
        height_ratios=row_heights,
        hspace=outer_hspace)

    for row_idx, row in enumerate(layout_spec["rows"]):
        subgrid = outer[row_idx].subgridspec(
            1,
            len(row),
            wspace=_row_wspace(
                fig_width_in,
                margins,
                row,
                layout_spec["col_gap_in"]))
        for col_idx, key in enumerate(row):
            ax = fig.add_subplot(subgrid[0, col_idx])
            ax.imshow(images[key], interpolation="lanczos")
            if show_titles:
                ax.set_title(
                    PANEL_TITLES[key],
                    fontsize=layout_spec["title_fontsize"],
                    fontweight="normal",
                    pad=2.5)
            if show_panel_labels:
                panel_idx = sum(len(layout_spec["rows"][r]) for r in range(row_idx)) + col_idx
                panel_letter = chr(ord('A') + panel_idx)
                ax.text(-0.02, 1.02, f"({panel_letter})",
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        va='bottom', ha='right')
            ax.set_anchor("N")
            ax.axis("off")

    out_pdf = figures_dir / f"{output_name}.pdf"
    fig.savefig(out_pdf, dpi=dpi, facecolor="white")
    if save_png:
        out_png = figures_dir / f"{output_name}.png"
        fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_pdf


def _is_composable(root: Path) -> bool:
    """True if *root* contains all 6 metric-group PDFs."""
    return all((root / f"{key}.pdf").exists() for key in METRIC_GROUP_ORDER)


def main():
    parser = argparse.ArgumentParser(
        description="Compose per-group metric PDFs into paper-style multi-panel figures")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Single figures directory to compose (default: process all experiment figures)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all experiment result figures (dpmm, topic, mixed)")
    parser.add_argument(
        "--layout",
        type=str,
        default=AUTO_LAYOUT,
        choices=[AUTO_LAYOUT] + sorted(LAYOUT_SPECS.keys()),
        help="Composition layout: auto (default), paper, portrait_3x2, or landscape_2x3")
    parser.add_argument(
        "--width-cm",
        type=float,
        default=None,
        help="Override figure width in cm (default: layout preset width)")
    parser.add_argument(
        "--max-height-cm",
        type=float,
        default=None,
        help="Override max height in cm for portrait layouts (default: 21 cm)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-png", action="store_true", help="Skip PNG output")
    parser.add_argument("--split", action="store_true",
        help="Split into two figures: clustering+projection and latent structure")
    parser.add_argument("--no-titles", action="store_true",
        help="Omit panel titles from the composed figure (move to LaTeX caption)")
    parser.add_argument("--panel-labels", action="store_true",
        help="Add (A), (B), (C) panel labels to each subplot")
    args = parser.parse_args()

    results_root = PROJECT_ROOT / "experiments" / "results"
    save_png = not args.no_png

    if args.figures_dir:
        d = Path(args.figures_dir)
        if not d.is_absolute():
            d = PROJECT_ROOT / d
        if args.split:
            out1 = compose_figures_in_dir(
                d,
                output_name="composed_clustering_projection",
                dpi=args.dpi,
                save_png=save_png,
                layout="split_clustering_projection",
                width_cm=args.width_cm,
                max_height_cm=args.max_height_cm,
                metric_groups=["clustering", "dre_umap", "dre_tsne"],
                show_titles=not args.no_titles,
                show_panel_labels=args.panel_labels)
            out2 = compose_figures_in_dir(
                d,
                output_name="composed_latent_structure",
                dpi=args.dpi,
                save_png=save_png,
                layout="split_latent_structure",
                width_cm=args.width_cm,
                max_height_cm=args.max_height_cm,
                metric_groups=["lse_intrinsic", "drex", "lsex"],
                show_titles=not args.no_titles,
                show_panel_labels=args.panel_labels)
            for out in [out1, out2]:
                if out:
                    print(f"  Composed: {out}")
                else:
                    print(f"  SKIP {d}: missing metric-group PDFs")
        else:
            out = compose_figures_in_dir(
                d,
                dpi=args.dpi,
                save_png=save_png,
                layout=args.layout,
                width_cm=args.width_cm,
                max_height_cm=args.max_height_cm,
                show_titles=not args.no_titles,
                show_panel_labels=args.panel_labels)
            if out:
                print(f"  Composed: {out}")
            else:
                print(f"  SKIP {d}: missing metric-group PDFs")
        return

    # Discover all composable directories (figures/ and figures/proposed|classical|deep)
    dirs_to_process = []
    for base in [results_root / "dpmm", results_root / "topic", results_root / "mixed"]:
        if not base.exists():
            continue
        for sub in base.rglob("figures"):
            if not sub.is_dir():
                continue
            if _is_composable(sub):
                dirs_to_process.append(sub)
            for child in sub.iterdir():
                if child.is_dir() and _is_composable(child):
                    dirs_to_process.append(child)

    # Deduplicate by path
    seen = set()
    final_dirs = []
    for d in dirs_to_process:
        key = str(d.resolve())
        if key not in seen:
            final_dirs.append(d)
            seen.add(key)

    for d in sorted(final_dirs):
        rel = d.relative_to(PROJECT_ROOT)
        if args.split:
            out1 = compose_figures_in_dir(
                d,
                output_name="composed_clustering_projection",
                dpi=args.dpi,
                save_png=save_png,
                layout="split_clustering_projection",
                width_cm=args.width_cm,
                max_height_cm=args.max_height_cm,
                metric_groups=["clustering", "dre_umap", "dre_tsne"],
                show_titles=not args.no_titles,
                show_panel_labels=args.panel_labels)
            out2 = compose_figures_in_dir(
                d,
                output_name="composed_latent_structure",
                dpi=args.dpi,
                save_png=save_png,
                layout="split_latent_structure",
                width_cm=args.width_cm,
                max_height_cm=args.max_height_cm,
                metric_groups=["lse_intrinsic", "drex", "lsex"],
                show_titles=not args.no_titles,
                show_panel_labels=args.panel_labels)
            for out in [out1, out2]:
                if out:
                    print(f"  {rel} -> {out.name}")
                else:
                    print(f"  SKIP {rel}")
        else:
            out = compose_figures_in_dir(
                d,
                dpi=args.dpi,
                save_png=save_png,
                layout=args.layout,
                width_cm=args.width_cm,
                max_height_cm=args.max_height_cm,
                show_titles=not args.no_titles,
                show_panel_labels=args.panel_labels)
            if out:
                print(f"  {rel} -> {out.name}")
            else:
                print(f"  SKIP {rel}")

    print("\nDone. Composed figures saved alongside per-group PDFs.")


if __name__ == "__main__":
    main()
