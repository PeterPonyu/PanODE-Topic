"""Refined Figure 1 — Architecture Overview (CLOP-DiT style).

Generates a publication-quality architecture diagram showing all three
variants side-by-side using matplotlib patches.

  DPMM series: DPMM-Base | DPMM-Transformer | DPMM-Contrastive
  Topic series: Topic-Base | Topic-Transformer | Topic-Contrastive

Each variant shows: Input → Encoder → Latent → Structured Prior → Decoder → Output
with variant-specific modules highlighted.

Data source: None (static architecture diagram)

Usage:
    python -m benchmarks.refined_figures.fig01_architecture --series dpmm
"""

import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, add_panel_label, save_with_vcd,
    bind_figure_region, COLORS)

apply_style()
matplotlib.rcParams.update({
    "axes.grid": False,
    "figure.constrained_layout.use": False,
})

DPI = 300

# ── Font sizes (matching CLOP-DiT) ───────────────────────────────────────────
FONT_LABEL = 11
FONT_SUBLABEL = 9
FONT_TITLE = 12

# ── Colour palette ───────────────────────────────────────────────────────────
C_WHITE = "#FFFFFF"
C_GREY = "#9E9E9E"
C_MID_GREY = "#607D8B"

# Input / output
C_INPUT = "#E3F2FD"
C_INPUT_E = "#1565C0"
C_OUTPUT = "#E8EAF6"
C_OUTPUT_E = "#283593"

# Encoder
C_ENC = "#E8F5E9"
C_ENC_E = "#2E7D32"
C_ENC_LIGHT = "#A5D6A7"

# Latent
C_LAT = "#FFF3E0"
C_LAT_E = "#E65100"
C_LAT_LIGHT = "#FFCCBC"

# Prior
C_PRIOR = "#F3E5F5"
C_PRIOR_E = "#7B1FA2"

# Decoder
C_DEC = "#E0F2F1"
C_DEC_E = "#004D40"

# Variant-specific modules
C_ATTN = "#FFF9C4"     # Transformer attention
C_ATTN_E = "#F9A825"
C_CONTRA = "#FFEBEE"   # Contrastive
C_CONTRA_E = "#C62828"

# ── Architecture specs per series ─────────────────────────────────────────────
_ARCH = {
    "dpmm": {
        "prior_name": "DPMM",
        "prior_detail": "n_comp=50",
        "latent_label": "z (32-d)",
        "enc_dims": "256→128→32",
        "dec_dims": "32→128→256",
        "variants": [
            {
                "name": "DPMM-Base",
                "subtitle": "MLP encoder + online DPMM refitting",
                "encoder": "MLP Encoder",
                "enc_sub": "256→128→32",
                "extra": None,
            },
            {
                "name": "DPMM-Transformer",
                "subtitle": "Multi-head projection encoder",
                "encoder": "MH Proj. Enc.",
                "enc_sub": "8 tokens, d=128\n4 heads, 2 layers",
                "extra": ("Self-Attention", "cross-attn agg."),
            },
            {
                "name": "DPMM-Contrastive",
                "subtitle": "MoCo projection head",
                "encoder": "MLP Encoder",
                "enc_sub": "256→128→32",
                "extra": ("MoCo Head", "q=4096, τ=0.2"),
            },
        ],
    },
    "topic": {
        "prior_name": "Dirichlet",
        "prior_detail": "n_topics=10",
        "latent_label": "θ (simplex)",
        "enc_dims": "128→128→K",
        "dec_dims": "β: K×V",
        "variants": [
            {
                "name": "Topic-Base",
                "subtitle": "Logistic-Normal encoder + KL reg.",
                "encoder": "LogNorm Enc.",
                "enc_sub": "128→128→(μ, σ²)",
                "extra": None,
            },
            {
                "name": "Topic-Transformer",
                "subtitle": "Cell-as-token + self-attention",
                "encoder": "Cell-Token Enc.",
                "enc_sub": "d=128, 4 heads\n2 layers",
                "extra": ("Self-Attention", "token aggregation"),
            },
            {
                "name": "Topic-Contrastive",
                "subtitle": "MoCo for topic representations",
                "encoder": "LogNorm Enc.",
                "enc_sub": "128→128→(μ, σ²)",
                "extra": ("MoCo Head", "topic contrast"),
            },
        ],
    },
}


# ── Drawing helpers (matching CLOP-DiT pattern) ──────────────────────────────

def _draw_box(ax, xy, w, h, label, sublabel=None, facecolor=C_WHITE,
              edgecolor=C_GREY, fontsize=FONT_LABEL, sublabel_size=FONT_SUBLABEL,
              textcolor="black", bold=False, linewidth=1.0, zorder=3,
              boxstyle="round,pad=0.08"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder, mutation_scale=0.5)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    y_off = h * 0.12 if sublabel else 0
    ax.text(x + w / 2, y + h / 2 + y_off, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=zorder + 1)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - h * 0.22, sublabel,
                ha="center", va="center", fontsize=sublabel_size,
                color=C_MID_GREY, zorder=zorder + 1)
    return box


def _draw_arrow(ax, start, end, color=C_GREY, linewidth=1.2,
                style="->", connectionstyle="arc3,rad=0", zorder=2):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, color=color, linewidth=linewidth,
        connectionstyle=connectionstyle, zorder=zorder,
        shrinkA=2, shrinkB=2, mutation_scale=10)
    ax.add_patch(arrow)
    return arrow


def _draw_stage_bg(ax, xy, w, h, label, color, alpha=0.10):
    x, y = xy
    bg = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="none", linewidth=0,
        alpha=alpha, zorder=0)
    ax.add_patch(bg)
    ax.text(x + w / 2, y + h + 0.02, label,
            ha="center", va="bottom", fontsize=10,
            fontweight="normal", color=color, zorder=1)


def _draw_variant_row(ax, y_base, variant, prior_name, prior_detail,
                      latent_label, panel_letter=None):
    """Draw one model variant as a horizontal pipeline."""
    BW = 1.00   # Box width
    BH = 0.50   # Box height
    SBW = 0.70  # Small box width
    SBH = 0.45  # Small box height
    gap = 0.15

    # Variant title
    ax.text(0.0, y_base + BH + 0.08, variant["name"],
            ha="left", va="bottom", fontsize=FONT_TITLE,
            fontweight="bold", color="black", zorder=5)
    ax.text(0.0, y_base + BH + 0.02, variant["subtitle"],
            ha="left", va="top", fontsize=FONT_SUBLABEL,
            color=C_MID_GREY, zorder=5)

    x = 0.0

    # Input box
    _draw_box(ax, (x, y_base), SBW, SBH, "Gene\nExpression",
              facecolor=C_INPUT, edgecolor=C_INPUT_E, fontsize=FONT_SUBLABEL)
    if panel_letter:
        ax.text(x - 0.08, y_base + SBH + 0.15, f"({panel_letter})",
                ha="left", va="bottom", fontsize=14, fontweight="bold",
                color="black", zorder=10)

    # Arrow → Encoder
    x_end_input = x + SBW
    x_enc = x_end_input + gap
    _draw_arrow(ax, (x_end_input, y_base + SBH / 2),
                (x_enc, y_base + BH / 2), color=C_ENC_E)

    # Encoder box
    _draw_box(ax, (x_enc, y_base), BW, BH, variant["encoder"],
              sublabel=variant["enc_sub"],
              facecolor=C_ENC, edgecolor=C_ENC_E, linewidth=1.3)

    # Arrow → Latent
    x_end_enc = x_enc + BW
    x_lat = x_end_enc + gap
    _draw_arrow(ax, (x_end_enc, y_base + BH / 2),
                (x_lat, y_base + BH / 2), color=C_LAT_E)

    # Latent box
    _draw_box(ax, (x_lat, y_base), SBW + 0.1, SBH, latent_label,
              facecolor=C_LAT, edgecolor=C_LAT_E)

    # Arrow → Prior (dashed upward)
    x_mid_lat = x_lat + (SBW + 0.1) / 2
    prior_y = y_base + BH + 0.22
    prior_w = 0.90
    prior_h = 0.35
    _draw_box(ax, (x_mid_lat - prior_w / 2, prior_y), prior_w, prior_h,
              f"{prior_name} Prior", sublabel=prior_detail,
              facecolor=C_PRIOR, edgecolor=C_PRIOR_E, linewidth=1.0)
    # Dashed line connecting prior to latent
    prior_arrow = FancyArrowPatch(
        (x_mid_lat, y_base + SBH), (x_mid_lat, prior_y),
        arrowstyle="<->", color=C_PRIOR_E, linewidth=0.8,
        linestyle="--", zorder=2, shrinkA=2, shrinkB=2, mutation_scale=8)
    ax.add_patch(prior_arrow)

    # Arrow → Decoder
    x_end_lat = x_lat + SBW + 0.1
    x_dec = x_end_lat + gap
    _draw_arrow(ax, (x_end_lat, y_base + SBH / 2),
                (x_dec, y_base + BH / 2), color=C_DEC_E)

    # Decoder box
    _draw_box(ax, (x_dec, y_base), BW, BH, "Decoder",
              sublabel="MLP", facecolor=C_DEC, edgecolor=C_DEC_E, linewidth=1.3)

    # Arrow → Output
    x_end_dec = x_dec + BW
    x_out = x_end_dec + gap
    _draw_arrow(ax, (x_end_dec, y_base + BH / 2),
                (x_out, y_base + BH / 2), color=C_OUTPUT_E)

    # Output box
    _draw_box(ax, (x_out, y_base), SBW, SBH, "Recon.\nOutput",
              facecolor=C_OUTPUT, edgecolor=C_OUTPUT_E, fontsize=FONT_SUBLABEL)

    # Variant-specific extra module
    if variant["extra"]:
        extra_name, extra_sub = variant["extra"]
        if "MoCo" in extra_name:
            # Place below encoder
            ex_x = x_enc
            ex_y = y_base - 0.55
            _draw_box(ax, (ex_x, ex_y), BW, 0.40, extra_name,
                      sublabel=extra_sub,
                      facecolor=C_CONTRA, edgecolor=C_CONTRA_E, linewidth=1.0)
            _draw_arrow(ax, (x_enc + BW / 2, y_base),
                        (ex_x + BW / 2, ex_y + 0.40),
                        color=C_CONTRA_E, linewidth=0.8)
        elif "Attention" in extra_name:
            # Place above encoder
            ex_x = x_enc + 0.05
            ex_y = y_base + BH + 0.22
            ex_w = BW - 0.10
            _draw_box(ax, (ex_x, ex_y), ex_w, 0.30, extra_name,
                      sublabel=extra_sub,
                      facecolor=C_ATTN, edgecolor=C_ATTN_E, linewidth=0.8)
            _draw_arrow(ax, (x_enc + BW / 2, y_base + BH),
                        (ex_x + ex_w / 2, ex_y),
                        color=C_ATTN_E, linewidth=0.8)


def generate(series, out_dir):
    """Generate refined Figure 1 — architecture overview."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _ARCH[series]
    variants = spec["variants"]
    n_variants = len(variants)

    fig = plt.figure(figsize=(10.0, 3.2 * n_variants))
    ax = bind_figure_region(fig, (0.01, 0.02, 0.99, 0.97)).add_axes(fig)
    ax.set_xlim(-0.20, 6.80)
    ax.set_ylim(-1.0, 3.2 * n_variants + 0.3)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor(C_WHITE)

    # Draw stage backgrounds
    _draw_stage_bg(ax, (-0.10, -0.80), 1.00, 3.2 * n_variants + 0.8,
                   "Input", C_INPUT_E, alpha=0.06)
    _draw_stage_bg(ax, (1.00, -0.80), 1.40, 3.2 * n_variants + 0.8,
                   "Encoder", C_ENC_E, alpha=0.06)
    _draw_stage_bg(ax, (2.55, -0.80), 1.20, 3.2 * n_variants + 0.8,
                   "Latent + Prior", C_LAT_E, alpha=0.06)
    _draw_stage_bg(ax, (3.85, -0.80), 1.40, 3.2 * n_variants + 0.8,
                   "Decoder", C_DEC_E, alpha=0.06)
    _draw_stage_bg(ax, (5.40, -0.80), 1.20, 3.2 * n_variants + 0.8,
                   "Output", C_OUTPUT_E, alpha=0.06)

    # Draw each variant from top to bottom
    letters = "abc"
    for i, variant in enumerate(variants):
        y_base = (n_variants - 1 - i) * 3.2 + 0.2
        _draw_variant_row(
            ax, y_base, variant,
            prior_name=spec["prior_name"],
            prior_detail=spec["prior_detail"],
            latent_label=spec["latent_label"],
            panel_letter=letters[i] if i < len(letters) else None)

    # Legend
    legend_items = [
        (C_INPUT, C_INPUT_E, "Input/Output"),
        (C_ENC, C_ENC_E, "Encoder"),
        (C_LAT, C_LAT_E, "Latent"),
        (C_PRIOR, C_PRIOR_E, "Prior"),
        (C_DEC, C_DEC_E, "Decoder"),
        (C_ATTN, C_ATTN_E, "Attention"),
        (C_CONTRA, C_CONTRA_E, "Contrastive"),
    ]
    lx = 0.5
    ly = -0.70
    for fc, ec, label in legend_items:
        box = FancyBboxPatch(
            (lx, ly), 0.18, 0.14, boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=ec, linewidth=0.6, zorder=5)
        ax.add_patch(box)
        ax.text(lx + 0.22, ly + 0.06, label,
                fontsize=FONT_SUBLABEL, va="center", color=ec, zorder=5)
        lx += 0.90

    out_path = out_dir / f"Fig1_architecture_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "refined_figures" / "output" / args.series)
    generate(args.series, out)
