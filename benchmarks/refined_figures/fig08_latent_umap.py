"""Refined Figure 8 — Latent UMAP Projections (composed figure).

Per dataset × model: component projection intensity UMAPs (first 6 components)
and top positively-correlated gene expression UMAPs.

Data source: benchmarks/biological_validation/results/

Usage:
    python -m benchmarks.refined_figures.fig08_latent_umap --series dpmm
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
    bind_figure_region, LayoutRegion, add_colorbar_safe)
from benchmarks.figure_generators.common import (
    MODEL_SHORT_NAMES, compute_umap,
    PRIOR_MODELS_DPMM, PRIOR_MODELS_TOPIC,
    REPRESENTATIVE_DATASETS, BIO_RESULTS)

DPI = 300
N_COMPONENTS = 6  # Show first 6 latent components


def _load_latent_and_umap(ds_name, model_name):
    """Load latent vectors and precomputed UMAP for a model × dataset."""
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    bio_dir = BIO_RESULTS / ds_name / safe_model
    latent, umap_emb = None, None
    for fname in ["latent.npz", "latent_vectors.npz"]:
        p = bio_dir / fname
        if p.exists():
            data = np.load(p, allow_pickle=True)
            latent = data.get("latent", data.get("z", None))
            umap_emb = data.get("umap", data.get("umap_emb", None))
            break
    if latent is not None and umap_emb is None:
        umap_emb = compute_umap(latent)
    return latent, umap_emb


def _draw_component_umaps(ax_grid_row, latent, umap_emb, title_prefix):
    """Draw component intensity UMAPs across a row of axes."""
    for i, ax in enumerate(ax_grid_row):
        style_axes(ax, kind="umap")
        if latent is None or i >= latent.shape[1]:
            ax.axis("off")
            continue
        vals = latent[:, i]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=vals,
                        s=1.5, alpha=0.55, cmap="viridis", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title_prefix} z{i}", fontsize=8, pad=2,
                     loc="left")
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)


def generate(series, out_dir):
    """Generate refined Figure 8."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    prior_models = (PRIOR_MODELS_DPMM if series == "dpmm"
                    else PRIOR_MODELS_TOPIC)
    datasets = REPRESENTATIVE_DATASETS
    n_ds = len(datasets)
    n_models = len(prior_models)

    # Layout: for each dataset, one row of 6 component UMAPs per model
    total_rows = n_ds * n_models
    fig = plt.figure(figsize=(17.0, 2.8 * total_rows + 1.0))
    root = bind_figure_region(fig, (0.03, 0.02, 0.97, 0.97))

    # Split by dataset blocks
    ds_blocks = root.split_rows(n_ds, gap=0.03)
    first_label = True
    for d_idx, ds in enumerate(datasets):
        model_rows = ds_blocks[d_idx].split_rows(n_models, gap=0.02)
        for m_idx, model in enumerate(prior_models):
            cols = model_rows[m_idx].split_cols(N_COMPONENTS, gap=0.01)
            latent, umap_emb = _load_latent_and_umap(ds, model)
            short = MODEL_SHORT_NAMES.get(model, model)
            ax_list = [c.add_axes(fig) for c in cols]
            _draw_component_umaps(ax_list, latent, umap_emb,
                                  f"{short}/{ds}")
            if first_label and ax_list:
                add_panel_label(ax_list[0], "a")
                first_label = False

    out_path = out_dir / f"Fig8_umap_{series}.png"
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
