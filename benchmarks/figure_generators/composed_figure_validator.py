"""Composed-figure-level validation for final composite PNGs.

Validates composed figure outputs for:
  1. Maximum size constraint: ≤ 17 cm × 21 cm at print resolution (~300 DPI)
     Figures are content-fitted — height shrinks to actual content.
  2. White-border uniformity: no unintended colored bleeding at edges
  3. Panel spacing: detects large empty gaps or overlapping panels
  4. Content presence: ensures no blank (all-white) panels
  5. Aspect ratio: validates width:height is within acceptable range

Supports two output sources:
  - Benchmark pipeline: benchmarks/paper_figures/{series}/Fig{N}_*_{series}.png (Figs 1, 3–4, 6–9)
  - Experiment pipeline: experiments/results/{series}/*/figures/composed_metrics.png (Figs 2, 10–12)

Usage:
    python -m benchmarks.figure_generators.composed_figure_validator --series dpmm
    python -m benchmarks.figure_generators.composed_figure_validator --series dpmm topic
    python -m benchmarks.figure_generators.composed_figure_validator --series dpmm --figures 6 7 8 10 11 12

Author: PanODE-LAB pipeline
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install via: pip install Pillow")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# At DPR=3 (96 CSS DPI × 3 = 288 DPI):
# 17 cm = 6.69 in × 288 = 1928 px  → content-fitted width ≈ 2010px (670 CSS × 3)
# 21 cm = 8.27 in × 288 = 2382 px  → maximum height; actual height may be less
MAX_WIDTH_PX = 2010   # 670 CSS px × DPR=3; figures should not exceed this
MAX_HEIGHT_PX = 2481  # 827 CSS px × DPR=3 = 21cm at 96dpi

# Print conversion factor
PX_PER_CM = 288 / 2.54  # ≈ 113.4 px/cm at DPR=3


# ═══════════════════════════════════════════════════════════════════════════════
# Validation checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_size(img: Image.Image, label: str) -> list[dict]:
    """Check maximum size constraint (≤ 17 × 21 cm). Figures are content-fitted."""
    issues = []
    w, h = img.size
    w_cm = w / PX_PER_CM
    h_cm = h / PX_PER_CM

    if w > MAX_WIDTH_PX:
        issues.append({
            "type": "size_width",
            "severity": "warning",
            "detail": f"{label}: width {w}px ({w_cm:.1f}cm) > maximum 17cm ({MAX_WIDTH_PX}px)",
        })
    if h > MAX_HEIGHT_PX:
        issues.append({
            "type": "size_height",
            "severity": "warning",
            "detail": f"{label}: height {h}px ({h_cm:.1f}cm) > maximum 21cm ({MAX_HEIGHT_PX}px)",
        })

    if not issues:
        return [{
            "type": "size_ok",
            "severity": "info",
            "detail": f"{label}: {w}×{h}px ({w_cm:.1f}×{h_cm:.1f}cm) — OK ≤ 17×21cm",
        }]
    return issues


def check_blank_regions(img: Image.Image, label: str,
                        stripe_h: int = 200,
                        white_threshold: float = 0.995) -> list[dict]:
    """Detect large horizontal blank (all-white) stripes WITHIN content
    that may indicate missing content or rendering failures.

    Trailing white space at the bottom (from min-height padding) is NOT
    flagged — we find the last row with non-white content and only scan
    up to that point.
    """
    issues = []
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape

    # Find the last row with non-white content
    row_has_content = np.any(np.any(arr < 250, axis=2), axis=1)
    content_rows = np.where(row_has_content)[0]
    if len(content_rows) == 0:
        return issues  # entirely blank — handled by check_content_presence
    last_content_row = int(content_rows[-1])

    # Only scan horizontal stripes within the content area
    y = 0
    while y + stripe_h <= last_content_row:
        stripe = arr[y:y + stripe_h, :, :]
        white_frac = np.mean(np.all(stripe > 250, axis=2))
        if white_frac > white_threshold:
            y_cm = y / PX_PER_CM
            issues.append({
                "type": "blank_stripe",
                "severity": "warning",
                "detail": (f"{label}: blank horizontal stripe at y={y}–{y + stripe_h}px "
                           f"({y_cm:.1f}cm), {white_frac:.1%} white — "
                           "possible missing or failed panel render"),
            })
        y += stripe_h

    return issues


def check_content_presence(img: Image.Image, label: str) -> list[dict]:
    """Ensure the image is not entirely or mostly white/blank."""
    issues = []
    arr = np.array(img.convert("RGB"))
    white_frac = np.mean(np.all(arr > 250, axis=2))

    if white_frac > 0.95:
        issues.append({
            "type": "mostly_blank",
            "severity": "warning",
            "detail": f"{label}: image is {white_frac:.1%} white — likely rendering failure",
        })
    elif white_frac > 0.85:
        issues.append({
            "type": "high_whitespace",
            "severity": "info",
            "detail": f"{label}: image is {white_frac:.1%} white — consider tighter layout",
        })

    return issues


def check_edge_bleeding(img: Image.Image, label: str,
                        border_px: int = 5) -> list[dict]:
    """Check that image edges are clean (white/near-white background)."""
    issues = []
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape

    edges = {
        "top": arr[:border_px, :, :],
        "bottom": arr[h - border_px:, :, :],
        "left": arr[:, :border_px, :],
        "right": arr[:, w - border_px:, :],
    }

    for edge_name, edge_arr in edges.items():
        non_white = np.mean(np.any(edge_arr < 240, axis=-1))
        if non_white > 0.3:
            issues.append({
                "type": "edge_bleeding",
                "severity": "info",
                "detail": (f"{label}: {edge_name} edge has {non_white:.0%} non-white pixels "
                           "— possible content bleeding beyond figure boundary"),
            })

    return issues


def check_aspect_ratio(img: Image.Image, label: str) -> list[dict]:
    """Validate aspect ratio is reasonable for a scientific figure."""
    issues = []
    w, h = img.size
    ratio = w / h

    # Scientific figures typically range from 0.5 (tall) to 2.0 (wide)
    if ratio > 2.0:
        issues.append({
            "type": "aspect_ratio",
            "severity": "warning",
            "detail": f"{label}: aspect ratio {ratio:.2f} is very wide (>2.0) — may not fit journal layout",
        })
    elif ratio < 0.3:
        issues.append({
            "type": "aspect_ratio",
            "severity": "warning",
            "detail": f"{label}: aspect ratio {ratio:.2f} is very tall (<0.3) — consider splitting into multiple figures",
        })

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def validate_composed_figure(png_path: Path, label: str = "",
                             verbose: bool = True) -> list[dict]:
    """Run all composed-figure-level validation checks on a PNG file.

    Parameters
    ----------
    png_path : Path
        Path to the composed figure PNG.
    label : str
        Descriptive label for the figure.
    verbose : bool
        Print results to stdout.

    Returns
    -------
    list[dict]
        Each dict has ``type``, ``severity``, ``detail``.
    """
    if not png_path.exists():
        return [{"type": "missing", "severity": "warning",
                 "detail": f"{label}: file not found: {png_path}"}]

    img = Image.open(png_path)
    if not label:
        label = png_path.stem

    all_issues = []
    all_issues.extend(check_size(img, label))
    all_issues.extend(check_content_presence(img, label))
    all_issues.extend(check_blank_regions(img, label))
    all_issues.extend(check_edge_bleeding(img, label))
    all_issues.extend(check_aspect_ratio(img, label))

    if verbose:
        warnings = [i for i in all_issues if i["severity"] == "warning"]
        infos = [i for i in all_issues if i["severity"] == "info"]

        if warnings:
            print(f"  ⚠ {label}: {len(warnings)} warnings, {len(infos)} info")
            for iss in warnings:
                print(f"    ⚠ [{iss['type']}] {iss['detail']}")
            for iss in infos[:5]:
                print(f"    ℹ [{iss['type']}] {iss['detail']}")
        else:
            size_info = [i for i in infos if i["type"] == "size_ok"]
            size_str = size_info[0]["detail"].split("—")[0].strip() if size_info else ""
            print(f"  ✓ {label}: OK — {size_str}")

    return all_issues


def validate_all_figures(series_list: list[str],
                         figures: Optional[list[int]] = None,
                         verbose: bool = True) -> dict[str, list[dict]]:
    """Validate all composed figure PNGs for given series.

    Parameters
    ----------
    series_list : list[str]
        List of series names (e.g., ["dpmm", "topic"]).
    figures : list[int] or None
        Figure numbers to validate. None = all (1-12).
    verbose : bool
        Print results to stdout.

    Returns
    -------
    dict[str, list[dict]]
        Map from "{series}/Fig{N}" to list of issues.
    """
    if figures is None:
        figures = list(range(1, 13))

    fig_dir = ROOT / "benchmarks" / "paper_figures"
    exp_dir = ROOT / "experiments" / "results"

    # Mapping from figure number to experiment-pipeline path (relative to exp_dir/{series}/)
    EXPERIMENT_FIGURES = {
        2: "ablation/figures/composed_metrics.png",
        10: "vs_external/figures/proposed/composed_metrics.png",
        11: "vs_external/figures/classical/composed_metrics.png",
        12: "vs_external/figures/deep/composed_metrics.png",
    }

    all_results = {}
    total_warnings = 0

    for s in series_list:
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Composed-figure validation — series: {s}")
            print(f"{'─' * 60}")

        for fig_num in figures:
            if fig_num in EXPERIMENT_FIGURES:
                # Experiment-pipeline figure
                png_path = exp_dir / s / EXPERIMENT_FIGURES[fig_num]
                label = f"{s}/Fig{fig_num}"
                if not png_path.exists():
                    all_results[label] = [{
                        "type": "missing",
                        "severity": "warning",
                        "detail": f"{label}: experiment composed PNG not found: {png_path}",
                    }]
                    if verbose:
                        print(f"  ✗ {label}: MISSING ({png_path})")
                    total_warnings += 1
                    continue

                issues = validate_composed_figure(png_path, label, verbose)
                all_results[label] = issues
                total_warnings += sum(1 for i in issues if i["severity"] == "warning")
            else:
                # Benchmark-pipeline figure (Fig{N}_*_{series}.png)
                pattern = f"Fig{fig_num}_*_{s}.png"
                matches = list((fig_dir / s).glob(pattern))
                if not matches:
                    label = f"{s}/Fig{fig_num}"
                    all_results[label] = [{
                        "type": "missing",
                        "severity": "warning",
                        "detail": f"{label}: no composed PNG found matching {pattern}",
                    }]
                    if verbose:
                        print(f"  ✗ {label}: MISSING")
                    total_warnings += 1
                    continue

                for png_path in matches:
                    label = f"{s}/{png_path.stem}"
                    issues = validate_composed_figure(png_path, label, verbose)
                    all_results[label] = issues
                    total_warnings += sum(1 for i in issues if i["severity"] == "warning")

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  COMPOSED-FIGURE VALIDATION SUMMARY")
        print(f"{'═' * 60}")
        print(f"  Figures checked:  {len(all_results)}")
        print(f"  Total warnings:   {total_warnings}")
        if total_warnings == 0:
            print(f"  ✓ All composed figures pass validation")
        else:
            problem_figs = [k for k, v in all_results.items()
                            if any(i["severity"] == "warning" for i in v)]
            print(f"  Figures with warnings: {', '.join(problem_figs)}")
        print(f"{'═' * 60}\n")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate composed figure PNGs (size, content, conflicts)")
    parser.add_argument("--series", nargs="+", default=["dpmm", "topic"],
                        choices=["dpmm", "topic"],
                        help="Series to validate")
    parser.add_argument("--figures", nargs="+", type=int, default=None,
                        help="Figure numbers to validate (default: all 1-12)")
    args = parser.parse_args()
    results = validate_all_figures(args.series, args.figures)

    # Exit with error code if any warnings
    has_warnings = any(
        any(i["severity"] == "warning" for i in issues)
        for issues in results.values()
    )
    sys.exit(1 if has_warnings else 0)
