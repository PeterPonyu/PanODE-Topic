"""Generate individual subplot PNGs for Figure 8 (Latent UMAP Projections).

Produces:
  Per dataset, per model:
    Panel — Component projection intensity UMAP (first 6 components)
    Panel — Top positively correlated gene expression UMAP (first 6 components)

All 3 representative datasets (setty, endo, dentate) × 3 models.
Each subplot image shows a 2×3 grid of UMAP panels (6 components),
with proper colorbar placement that avoids overlap.

Layout: Per-dataset grouping — top row = intensity, bottom row = expression.

Output: benchmarks/paper_figures/{series}/subplots/fig8/

Usage:
    python -m benchmarks.figure_generators.gen_fig8_subplots --series dpmm
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
    FIGSIZE_UMAP, SCATTER_SIZE_UMAP, LINE_WIDTH_SPINE,
    FONTSIZE_TITLE, FONTSIZE_TICK, SUBPLOT_DPI)
from benchmarks.figure_generators.common import (
    MODEL_SHORT_NAMES,
    PRIOR_MODELS_DPMM, PRIOR_MODELS_TOPIC,
    REPRESENTATIVE_DATASETS, BIO_RESULTS
)

# Increased font sizes for Figure 8 subplots
_FS_TITLE = max(FONTSIZE_TITLE + 1, 9)
_FS_TICK = max(FONTSIZE_TICK, 7)


# ═══════════════════════════════════════════════════════════════════════════════
# Subplot generators
# ═══════════════════════════════════════════════════════════════════════════════

def gen_component_umap(umap_emb, latent, labels, comp_prefix, ds_name,
                       model, out_path, n_comp=3):
    """Generate UMAP panels colored by latent component intensity.

    Layout: 1×3 grid for 3 components (compact for per-dataset grouping).
    Each cell has a small colorbar placed outside the plot area to avoid
    overlap with scatter data.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    K = min(latent.shape[1], n_comp)
    nrows = 2 if K > 3 else 1
    ncols = min(K, 3)

    cell_w = FIGSIZE_UMAP[0] * 1.10
    cell_h = FIGSIZE_UMAP[1] * 0.95
    fig = plt.figure(figsize=(cell_w * ncols, cell_h * nrows))
    _root = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))
    _grid = _root.grid(nrows, ncols, hgap=0.04, wgap=0.04)
    axes = [[_grid[r][c].add_axes(fig) for c in range(ncols)] for r in range(nrows)]
    axes = np.array(axes).reshape(nrows, ncols)
    for _ax in np.atleast_1d(axes).flat: style_axes(_ax)

    for k in range(K):
        row, col = divmod(k, ncols)
        ax = axes[row, col]
        vals = latent[:, k]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                        c=vals, s=SCATTER_SIZE_UMAP, alpha=0.70,
                        cmap="viridis", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{comp_prefix}{k+1}", fontsize=_FS_TITLE,
                     pad=1, fontweight="normal")
        for sp in ax.spines.values():
            sp.set_linewidth(LINE_WIDTH_SPINE)
        # Colorbar outside each cell (right edge)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.06,
                          aspect=18)
        cb.ax.tick_params(labelsize=max(_FS_TICK - 2, 12), length=1.5)
        cb.ax.yaxis.set_major_locator(
            __import__('matplotlib.ticker', fromlist=['MaxNLocator'])
            .MaxNLocator(nbins=4, prune='both'))
        cb.outline.set_linewidth(0.3)

    # Hide unused axes
    for k in range(K, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"{short} — {ds_name}",
                 fontsize=_FS_TITLE + 0.5, y=1.01, fontweight="normal")
    fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.5)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_gene_umap(umap_emb, top_gene_expr, top_gene_names, labels,
                  comp_prefix, ds_name, model, out_path, n_comp=3):
    """Generate UMAP panels colored by top positively correlated gene expression.

    Layout: 1×3 grid for 3 components (compact for per-dataset grouping).
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    K = min(top_gene_expr.shape[1], n_comp)
    nrows = 2 if K > 3 else 1
    ncols = min(K, 3)

    cell_w = FIGSIZE_UMAP[0] * 1.10
    cell_h = FIGSIZE_UMAP[1] * 0.95
    fig = plt.figure(figsize=(cell_w * ncols, cell_h * nrows))
    _root = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))
    _grid = _root.grid(nrows, ncols, hgap=0.04, wgap=0.04)
    axes = [[_grid[r][c].add_axes(fig) for c in range(ncols)] for r in range(nrows)]
    axes = np.array(axes).reshape(nrows, ncols)
    for _ax in np.atleast_1d(axes).flat: style_axes(_ax)

    for k in range(K):
        row, col = divmod(k, ncols)
        ax = axes[row, col]
        expr = top_gene_expr[:, k]
        gene_name = str(top_gene_names[k])[:12]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                        c=expr, s=SCATTER_SIZE_UMAP, alpha=0.70,
                        cmap="magma", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{comp_prefix}{k+1}: {gene_name}",
                     fontsize=_FS_TITLE, pad=1, fontweight="normal")
        for sp in ax.spines.values():
            sp.set_linewidth(LINE_WIDTH_SPINE)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.06,
                          aspect=18)
        cb.ax.tick_params(labelsize=max(_FS_TICK - 2, 12), length=1.5)
        cb.ax.yaxis.set_major_locator(
            __import__('matplotlib.ticker', fromlist=['MaxNLocator'])
            .MaxNLocator(nbins=4, prune='both'))
        cb.outline.set_linewidth(0.3)

    # Hide unused axes
    for k in range(K, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"{short} — {ds_name}",
                 fontsize=_FS_TITLE + 0.5, y=1.01, fontweight="normal")
    fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.5)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate(out_dir):
    """Generate all subplot PNGs for Figure 8 (Latent UMAP Projections).

    Per-dataset grouping: for each dataset, component intensity (top row)
    and top positively correlated gene expression (bottom row), 3 models each.
    """
    series = "topic"
    print(f"\n  Figure 8 subplots ({series})")
    sub_dir = out_dir / "fig8"
    sub_dir.mkdir(parents=True, exist_ok=True)
    apply_subplot_style()

    models = PRIOR_MODELS_TOPIC if series == "topic" else PRIOR_MODELS_DPMM
    comp_prefix = "Topic" if series == "topic" else "Dim"
    results_dir = BIO_RESULTS

    # Per-dataset dict: { ds_name: [{"file": ..., "model": ...}, ...] }
    comp_umap_files = {}
    gene_umap_files = {}

    for model in models:
        safe_m = model.replace("/", "_")
        for ds_name in REPRESENTATIVE_DATASETS:
            umap_path = results_dir / f"{model}_{ds_name}_umap_data.npz"
            if not umap_path.exists():
                continue
            ud = np.load(umap_path, allow_pickle=True)
            umap_emb = ud["umap_emb"]
            latent = ud["latent"]
            labels = ud.get("labels")
            top_gene_expr = ud["top_gene_expr"]
            top_gene_names = ud["top_gene_names"]

            # Component intensity UMAP (2×3 grid)
            fname_comp = f"comp_umap_{ds_name}_{safe_m}.png"
            gen_component_umap(
                umap_emb, latent, labels, comp_prefix, ds_name,
                model, sub_dir / fname_comp)
            comp_umap_files.setdefault(ds_name, []).append(
                {"file": fname_comp, "model": model})

            # Gene expression UMAP (2×3 grid)
            fname_gene = f"gene_umap_{ds_name}_{safe_m}.png"
            gen_gene_umap(
                umap_emb, top_gene_expr, top_gene_names, labels,
                comp_prefix, ds_name, model, sub_dir / fname_gene)
            gene_umap_files.setdefault(ds_name, []).append(
                {"file": fname_gene, "model": model})

    avail_models = list({e["model"]
                         for ds_files in comp_umap_files.values()
                         for e in ds_files})

    manifest = build_manifest(sub_dir, {
        "comp_umap": comp_umap_files,
        "gene_umap": gene_umap_files,
        "models": avail_models,
        "datasets": list(comp_umap_files.keys()),
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 8 subplots")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(out)
