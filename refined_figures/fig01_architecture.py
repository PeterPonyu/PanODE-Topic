"""Refined Figure 1 — compact publication architecture overview.

Generates a clean 2×2 layout for the four Topic-FM variants.

Each panel shows the shared Topic-FM backbone with only the
variant-specific module emphasized, which keeps the figure readable in the
final manuscript without the excessive whitespace of the original tall stack.

Usage:
        python -m refined_figures.fig01_architecture
"""

import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, save_with_vcd, set_export_pad_inches)

apply_style()
matplotlib.rcParams.update({
    "axes.grid": False,
    "figure.constrained_layout.use": False,
})

DPI = 300

FONT_LABEL = 10.5
FONT_SUBLABEL = 8.6
FONT_TITLE = 13
FONT_STAGE = 9.5

# ── Colour palette ───────────────────────────────────────────────────────────
C_WHITE = "#FFFFFF"
C_GREY = "#8A8F98"
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
C_ATTN = "#FFF4CC"
C_ATTN_E = "#C48A00"
C_CONTRA = "#FCE4EC"
C_CONTRA_E = "#C62828"
C_GRAPH = "#E0F7FA"
C_GRAPH_E = "#00838F"
C_FLOW = "#FCE4EC"
C_FLOW_E = "#AD1457"

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
        "variants": [
            {
                "name": "Topic-FM-Base",
                "subtitle": "Logistic-normal encoder + KL regularization",
                "encoder": "LogNorm Enc.",
                "enc_sub": "128→128→(μ, σ²)",
                "extra": None,
                "has_fm": True,
            },
            {
                "name": "Topic-FM-Transformer",
                "subtitle": "Cell-as-token encoder with self-attention",
                "encoder": "Cell-Token Enc.",
                "enc_sub": "d=128, 4 heads\n2 layers",
                "extra": ("Self-Attention", "token aggregation"),
                "has_fm": True,
            },
            {
                "name": "Topic-FM-Contrastive",
                "subtitle": "MoCo augmentation for topic representations",
                "encoder": "LogNorm Enc.",
                "enc_sub": "128→128→(μ, σ²)",
                "extra": ("MoCo Head", "topic contrast"),
                "has_fm": True,
            },
            {
                "name": "Topic-FM-GAT",
                "subtitle": "Graph attention over a batch-wise kNN graph",
                "encoder": "GAT Encoder",
                "enc_sub": "2 layers · 4 heads\nresidual + LN",
                "extra": ("kNN Graph", "k=15, batch-wise"),
                "has_fm": True,
            },
        ],
    },
}


_BOX = {
    "input": (0.03, 0.39, 0.14, 0.14),
    "encoder": (0.23, 0.36, 0.21, 0.18),
    "latent": (0.50, 0.39, 0.15, 0.14),
    "decoder": (0.71, 0.36, 0.16, 0.18),
    "output": (0.90, 0.39, 0.09, 0.14),
    "prior": (0.46, 0.66, 0.23, 0.11),
    "flow": (0.60, 0.16, 0.17, 0.11),
    "extra_top": (0.23, 0.61, 0.21, 0.11),
    "extra_bottom": (0.23, 0.14, 0.21, 0.11),
}

_STAGES = [
    (0.01, 0.30, 0.18, 0.33, "Input", C_INPUT_E, C_INPUT),
    (0.21, 0.30, 0.26, 0.33, "Encoder", C_ENC_E, C_ENC),
    (0.49, 0.30, 0.18, 0.33, "Latent + Prior", C_LAT_E, C_LAT),
    (0.69, 0.30, 0.20, 0.33, "Decoder", C_DEC_E, C_DEC),
    (0.90, 0.30, 0.09, 0.33, "Output", C_OUTPUT_E, C_OUTPUT),
]


def _draw_box(ax, xy, w, h, label, sublabel=None, facecolor=C_WHITE,
              edgecolor=C_GREY, fontsize=FONT_LABEL, sublabel_size=FONT_SUBLABEL,
              textcolor="black", bold=False, linewidth=1.0, zorder=3,
              boxstyle="round,pad=0.018"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder, mutation_scale=0.5)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    y_off = h * 0.11 if sublabel else 0
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


def _draw_stage_headers(ax):
    for x, y, w, h, label, label_color, fill_color in _STAGES:
        band = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012",
            facecolor=fill_color,
            edgecolor="none",
            alpha=0.18,
            zorder=0,
        )
        ax.add_patch(band)


def _center(box_key, edge=None):
    x, y, w, h = _BOX[box_key]
    if edge == "left":
        return x, y + h / 2
    if edge == "right":
        return x + w, y + h / 2
    if edge == "top":
        return x + w / 2, y + h
    if edge == "bottom":
        return x + w / 2, y
    return x + w / 2, y + h / 2


def _draw_variant_panel(ax, variant, prior_name, prior_detail, latent_label, panel_letter):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    _draw_stage_headers(ax)

    ax.text(
        0.00, 0.98, f"({panel_letter})",
        ha="left", va="top", fontsize=13, fontweight="bold", color="black")
    ax.text(
        0.08, 0.95, variant["name"],
        ha="left", va="top", fontsize=FONT_TITLE, fontweight="bold", color="black")
    ax.text(
        0.08, 0.885, variant["subtitle"],
        ha="left", va="top", fontsize=FONT_SUBLABEL, color=C_MID_GREY)

    _draw_box(ax, _BOX["input"][:2], _BOX["input"][2], _BOX["input"][3], "Gene\nExpression",
              facecolor=C_INPUT, edgecolor=C_INPUT_E, fontsize=FONT_SUBLABEL)
    _draw_box(ax, _BOX["encoder"][:2], _BOX["encoder"][2], _BOX["encoder"][3], variant["encoder"],
              sublabel=variant["enc_sub"], facecolor=C_ENC, edgecolor=C_ENC_E, linewidth=1.2)
    _draw_box(ax, _BOX["latent"][:2], _BOX["latent"][2], _BOX["latent"][3], latent_label,
              facecolor=C_LAT, edgecolor=C_LAT_E, linewidth=1.2)
    _draw_box(ax, _BOX["decoder"][:2], _BOX["decoder"][2], _BOX["decoder"][3], "Decoder",
              sublabel="MLP", facecolor=C_DEC, edgecolor=C_DEC_E, linewidth=1.2)
    _draw_box(ax, _BOX["output"][:2], _BOX["output"][2], _BOX["output"][3], "Recon.\nOutput",
              facecolor=C_OUTPUT, edgecolor=C_OUTPUT_E, fontsize=FONT_SUBLABEL)
    _draw_box(ax, _BOX["prior"][:2], _BOX["prior"][2], _BOX["prior"][3], f"{prior_name} Prior",
              sublabel=prior_detail, facecolor=C_PRIOR, edgecolor=C_PRIOR_E, linewidth=1.0)

    _draw_arrow(ax, _center("input", "right"), _center("encoder", "left"), color=C_INPUT_E, linewidth=1.0)
    _draw_arrow(ax, _center("encoder", "right"), _center("latent", "left"), color=C_ENC_E, linewidth=1.0)
    _draw_arrow(ax, _center("latent", "right"), _center("decoder", "left"), color=C_LAT_E, linewidth=1.0)
    _draw_arrow(ax, _center("decoder", "right"), _center("output", "left"), color=C_DEC_E, linewidth=1.0)

    prior_arrow = FancyArrowPatch(
        _center("latent", "top"), _center("prior", "bottom"),
        arrowstyle="<->", color=C_PRIOR_E, linewidth=0.9, linestyle="--",
        zorder=2, shrinkA=2, shrinkB=2, mutation_scale=8)
    ax.add_patch(prior_arrow)

    if variant.get("has_fm", False):
        _draw_box(ax, _BOX["flow"][:2], _BOX["flow"][2], _BOX["flow"][3], "Flow Match",
                  sublabel="OT refine", facecolor=C_FLOW, edgecolor=C_FLOW_E,
                  linewidth=0.9, fontsize=FONT_SUBLABEL, boxstyle="round,pad=0.016")
        flow_from_latent = FancyArrowPatch(
            _center("latent", "bottom"), _center("flow", "top"),
            arrowstyle="->", color=C_FLOW_E, linewidth=0.8, linestyle="--",
            zorder=2, shrinkA=2, shrinkB=2, mutation_scale=8)
        flow_to_decoder = FancyArrowPatch(
            (_BOX["flow"][0] + _BOX["flow"][2], _BOX["flow"][1] + _BOX["flow"][3] / 2),
            (_BOX["decoder"][0], _BOX["decoder"][1] + 0.03),
            arrowstyle="->", color=C_FLOW_E, linewidth=0.8, linestyle="--",
            zorder=2, shrinkA=2, shrinkB=2, mutation_scale=8)
        ax.add_patch(flow_from_latent)
        ax.add_patch(flow_to_decoder)

    if variant["extra"]:
        extra_name, extra_sub = variant["extra"]
        extra_key = "extra_top" if "Attention" in extra_name else "extra_bottom"
        extra_xy = _BOX[extra_key][:2]
        extra_w = _BOX[extra_key][2]
        extra_h = _BOX[extra_key][3]
        extra_face = C_ATTN if "Attention" in extra_name else (C_GRAPH if "kNN" in extra_name else C_CONTRA)
        extra_edge = C_ATTN_E if "Attention" in extra_name else (C_GRAPH_E if "kNN" in extra_name else C_CONTRA_E)
        _draw_box(ax, extra_xy, extra_w, extra_h, extra_name, sublabel=extra_sub,
                  facecolor=extra_face, edgecolor=extra_edge,
                  linewidth=0.9, fontsize=FONT_LABEL if "Attention" in extra_name else FONT_SUBLABEL)
        if extra_key == "extra_top":
            _draw_arrow(ax, _center("encoder", "top"), (extra_xy[0] + extra_w / 2, extra_xy[1]),
                        color=extra_edge, linewidth=0.8)
        else:
            _draw_arrow(ax, _center("encoder", "bottom"), (extra_xy[0] + extra_w / 2, extra_xy[1] + extra_h),
                        color=extra_edge, linewidth=0.8)


def generate(out_dir):
    """Generate refined Figure 1 — compact architecture overview."""
    series = "topic"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _ARCH[series]
    variants = spec["variants"]
    fig = plt.figure(figsize=(10.8, 6.4))
    fig.patch.set_facecolor(C_WHITE)
    set_export_pad_inches(fig, 0.02)

    grid = fig.add_gridspec(
        2, 2,
        left=0.03, right=0.99,
        bottom=0.06, top=0.97,
        wspace=0.06, hspace=0.10,
    )

    letters = "abcd"
    for idx, variant in enumerate(variants):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        _draw_variant_panel(
            ax,
            variant,
            prior_name=spec["prior_name"],
            prior_detail=spec["prior_detail"],
            latent_label=spec["latent_label"],
            panel_letter=letters[idx] if idx < len(letters) else "?",
        )

    fig.text(
        0.5, 0.02,
        "Shared Topic-FM backbone with compact variant-specific modules; the dashed pink box marks the flow-matching refinement.",
        ha="center", va="bottom", fontsize=FONT_SUBLABEL, color=C_MID_GREY,
    )

    out_path_png = out_dir / "Fig1_arch_topic.png"
    save_with_vcd(fig, out_path_png, dpi=DPI, close=False)
    print(f"  ✓ {out_path_png.name}")
    out_path_pdf = out_dir / "Fig1_arch_topic.pdf"
    save_with_vcd(fig, out_path_pdf, dpi=DPI, close=True)
    print(f"  ✓ {out_path_pdf.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "refined_figures" / "output" / series)
    generate(out)
