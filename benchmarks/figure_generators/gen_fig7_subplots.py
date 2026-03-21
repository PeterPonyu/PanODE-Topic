"""Generate individual subplot PNGs for Figure 7 (Latent–Gene Correlation).

Produces:
  Panel A — Latent–gene Pearson correlation heatmaps (datasets × models)

Correlation data is loaded from pre-computed NPZ files in the biological
validation results directory.

Output: benchmarks/paper_figures/{series}/subplots/fig7/

Usage:
    python -m benchmarks.figure_generators.gen_fig7_subplots --series dpmm
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion)

from benchmarks.figure_generators.subplot_style import (
    apply_subplot_style, save_subplot, build_manifest,
    FIGSIZE_HEATMAP, SUBPLOT_DPI,
    FONTSIZE_TITLE, FONTSIZE_TICK)
from benchmarks.figure_generators.common import (
    MODEL_SHORT_NAMES,
    PRIOR_MODELS_DPMM, PRIOR_MODELS_TOPIC,
    REPRESENTATIVE_DATASETS, BIO_RESULTS
)


# ═══════════════════════════════════════════════════════════════════════════════
# Subplot generator
# ═══════════════════════════════════════════════════════════════════════════════

def gen_correlation_heatmap(corr, gene_names, comp_prefix, ds_name, model,
                            out_path, top_genes=30):
    """Generate a latent–gene Pearson correlation heatmap.

    Selects the top correlated genes per component (by absolute
    correlation) and displays as a diverging heatmap.  Both positive
    and negative correlations are informative.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    n_comp = min(corr.shape[0], 10)
    corr_sub = corr[:n_comp, :]

    # Select top genes per component by |r|, then re-sort by dominant
    # component (argmax) to create a block-diagonal layout that visually
    # groups genes by the latent dimension they most strongly correlate with.
    top_idx = set()
    for k in range(n_comp):
        top_idx.update(np.argsort(np.abs(corr_sub[k]))[::-1][:top_genes])
    top_idx_ranked = sorted(top_idx,
                            key=lambda i: -np.abs(corr_sub[:, i]).max())[:top_genes]

    def _sort_key_corr(i):
        abs_col = np.abs(corr_sub[:, i])
        dom = int(np.argmax(abs_col))
        return (dom, -abs_col[dom])

    top_idx = sorted(top_idx_ranked, key=_sort_key_corr)
    sub_corr = corr_sub[:, top_idx]
    sub_genes = ([str(gene_names[i]) for i in top_idx]
                 if gene_names is not None
                 else [f"g{i}" for i in top_idx])

    fig_w = FIGSIZE_HEATMAP[0] * 2.2
    fig_h = FIGSIZE_HEATMAP[1] * 1.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)
    vlim = max(abs(sub_corr.min()), abs(sub_corr.max()), 0.2)
    im = ax.imshow(sub_corr, aspect="auto", cmap="coolwarm",
                   interpolation="nearest", vmin=-vlim, vmax=vlim)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([f"{comp_prefix}{k+1}" for k in range(n_comp)],
                       fontsize=FONTSIZE_TITLE)
    ax.set_xticks(range(len(sub_genes)))
    # Truncate long gene names to 8 chars to reduce overlap
    display_genes = [g[:8] for g in sub_genes]
    ax.set_xticklabels(display_genes, rotation=90, ha="center",
                       fontsize=max(FONTSIZE_TICK - 1, 12))
    ax.set_title(f"{short} — {ds_name}  (latent–gene corr.)",
                 fontsize=FONTSIZE_TITLE, loc="left",
                 fontweight="normal")
    # Colorbar placed OUTSIDE the heatmap (below gene names) to avoid
    # overlapping data content.  Uses make_axes_locatable to carve out
    # a dedicated thin axes strip beneath the main axes.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.12)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=max(FONTSIZE_TICK - 1, 12), length=2, pad=1)
    cb.set_ticks([-vlim, 0, vlim])
    cb.set_ticklabels([f"{-vlim:.2f}", "0", f"{vlim:.2f}"])
    cb.outline.set_linewidth(0.4)
    cb.set_label("Pearson r", fontsize=max(FONTSIZE_TICK - 1, 12), labelpad=3)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate(out_dir):
    """Generate all subplot PNGs for Figure 7 (Latent–Gene Correlation)."""
    series = "topic"
    print(f"\n  Figure 7 subplots ({series})")
    sub_dir = out_dir / "fig7"
    sub_dir.mkdir(parents=True, exist_ok=True)
    apply_subplot_style()

    models = PRIOR_MODELS_TOPIC if series == "topic" else PRIOR_MODELS_DPMM
    comp_prefix = "Topic" if series == "topic" else "Dim"
    results_dir = BIO_RESULTS

    avail_models = []
    for m in models:
        for ds in REPRESENTATIVE_DATASETS:
            corr_path = results_dir / f"{m}_{ds}_correlation.npz"
            if corr_path.exists() and m not in avail_models:
                avail_models.append(m)

    if not avail_models:
        print("    No correlation data found.")
        return build_manifest(sub_dir, {})

    corr_files = {}
    for model in avail_models:
        safe_m = model.replace("/", "_")
        for ds_name in REPRESENTATIVE_DATASETS:
            corr_path = results_dir / f"{model}_{ds_name}_correlation.npz"
            if not corr_path.exists():
                continue
            cd = np.load(corr_path, allow_pickle=True)
            corr = cd["correlation"]
            gn = cd.get("gene_names")
            fname = f"corr_{ds_name}_{safe_m}.png"
            gen_correlation_heatmap(
                corr, gn, comp_prefix, ds_name, model, sub_dir / fname)
            corr_files.setdefault(ds_name, []).append(
                {"file": fname, "model": model})

    manifest = build_manifest(sub_dir, {
        "panelA": corr_files,
        "models": avail_models,
        "datasets": list(corr_files.keys()),
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 7 subplots")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(out)
