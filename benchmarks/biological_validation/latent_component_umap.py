#!/usr/bin/env python
"""
Latent component visualization: project each component's strength onto UMAP.

For DPMM models:
  - Each latent dim is one component; project its value onto UMAP as a heatmap
  - Also show DPMM mixture weights / per-component responsibility

For Topic models:
  - Each topic proportion (theta_k) is projected onto UMAP
  - Shows which cells have high loading on each topic

Usage:
    # Requires a trained model saved by training_dynamics.py or benchmark
    python benchmarks/biological_validation/latent_component_umap.py \
        --model-path benchmarks/training_dynamics_results/DPMM-Base_setty_model.pt \
        --dataset setty --series dpmm

    python benchmarks/biological_validation/latent_component_umap.py \
        --model-path benchmarks/training_dynamics_results/Topic-Base_setty_model.pt \
        --dataset setty --series topic

Requires GPU for model inference (lightweight).
"""

import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.paper_style import apply_style, apply_cli_overrides, add_style_args
from benchmarks.biological_validation import load_model  # shared helper
from benchmarks.config import BIO_RESULTS_DIR


def extract_latent_components(model, data_loader, device):
    """Extract per-component latent activations.

    Returns:
        latent: [N, D] full latent (for UMAP computation)
        components: [N, K] per-component values
            - For DPMM: latent z values (D=K=latent_dim)
            - For Topic: topic proportions theta [N, n_topics]
        component_names: list of str
    """
    is_topic = hasattr(model, 'n_topics')

    all_latent = []
    all_components = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device).float()
            else:
                x = batch.to(device).float()

            if is_topic:
                # Get topic proportions (simplex)
                theta = model.encode(x)  # [B, n_topics]
                latent = theta.cpu().numpy()
                components = theta.cpu().numpy()
            else:
                # For DPMM/AE: use raw latent z
                z = model.encode(x)  # [B, latent_dim]
                latent = z.cpu().numpy()
                components = z.cpu().numpy()

            all_latent.append(latent)
            all_components.append(components)

    latent = np.concatenate(all_latent, axis=0)
    components = np.concatenate(all_components, axis=0)

    K = components.shape[1]
    if is_topic:
        names = [f"Topic {k+1}" for k in range(K)]
    else:
        names = [f"Dim {k+1}" for k in range(K)]

    return latent, components, names


def compute_dpmm_responsibilities(model, components):
    """Compute per-cell DPMM component responsibilities (soft assignment).

    Returns [N, n_active_components] responsibility matrix.
    """
    if not hasattr(model, 'dpmm_params') or not model.dpmm_params:
        return None, None

    dp = model.dpmm_params
    means = dp["means"].cpu().numpy()     # [K, D]
    weights = dp["weights"].cpu().numpy()  # [K]
    covs = dp["covariances"].cpu().numpy() # [K, D]

    # Active components
    active = weights > 0.01
    if active.sum() == 0:
        return None, None

    means = means[active]
    weights = weights[active]
    covs = covs[active]
    K_active = active.sum()

    # Compute log-responsibilities
    N = components.shape[0]
    log_resp = np.zeros((N, K_active))

    for k in range(K_active):
        diff = components - means[k]  # [N, D]
        # Diagonal Gaussian: -0.5 * sum((x-mu)^2 / sigma^2)
        log_resp[:, k] = np.log(weights[k] + 1e-10) - 0.5 * np.sum(diff**2 / (covs[k] + 1e-10), axis=1)

    # Normalize (log-sum-exp)
    log_resp -= np.max(log_resp, axis=1, keepdims=True)
    resp = np.exp(log_resp)
    resp /= resp.sum(axis=1, keepdims=True)

    comp_names = [f"DPMM Comp {i+1}" for i in range(K_active)]
    return resp, comp_names


def plot_component_umap(latent, components, comp_names, labels,
                        save_path, title="Latent Components on UMAP",
                        no_title=False, figformat="png", max_components=10):
    """Project each component's intensity onto UMAP space.

    Layout: grid of UMAP plots (one per component) + one with cell-type labels.
    """
    try:
        import scanpy as sc
        use_scanpy = True
    except ImportError:
        from umap import UMAP
        use_scanpy = False
    from sklearn.preprocessing import LabelEncoder

    apply_style()

    # Compute UMAP once
    if use_scanpy:
        adata = sc.AnnData(latent.astype(np.float32))
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)
        sc.tl.umap(adata, min_dist=0.5)
        umap_xy = adata.obsm["X_umap"]
    else:
        reducer = UMAP(n_neighbors=15, min_dist=0.5, random_state=42)
        umap_xy = reducer.fit_transform(latent)

    # Encode labels
    if isinstance(labels[0], str):
        le = LabelEncoder()
        labels_enc = le.fit_transform(labels)
    else:
        labels_enc = np.array(labels)

    K = min(components.shape[1], max_components)
    n_panels = K + 1  # +1 for cell-type reference
    cols = min(4, n_panels)
    rows = (n_panels + cols - 1) // cols
    cell_w, cell_h = 5.0, 4.5

    fig, axes = plt.subplots(rows, cols, figsize=(cell_w * cols, cell_h * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Panel 0: cell-type labels
    ax = axes[0, 0]
    n_cls = len(np.unique(labels_enc))
    cmap_ct = mpl.colormaps.get_cmap("tab20" if n_cls <= 20 else "nipy_spectral")
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], c=labels_enc, cmap=cmap_ct,
               s=12, alpha=0.7, edgecolors="none", rasterized=True)
    ax.set_title("Cell Types", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.tick_params(labelsize=9)

    # Panels 1–K: component intensities
    for k in range(K):
        idx = k + 1
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        vals = components[:, k]
        sc_plot = ax.scatter(umap_xy[:, 0], umap_xy[:, 1],
                             c=vals, cmap="viridis", s=12, alpha=0.7,
                             edgecolors="none", rasterized=True)
        ax.set_title(comp_names[k], fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP-1", fontsize=11)
        ax.set_ylabel("UMAP-2", fontsize=11)
        ax.tick_params(labelsize=9)
        plt.colorbar(sc_plot, ax=ax, shrink=0.7, pad=0.02)

    # Hide unused subplots
    for idx in range(n_panels, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    if not no_title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=mpl.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Latent component UMAP projection")
    parser.add_argument("--model-path", required=True,
                        help="Path to saved model .pt (from training_dynamics.py)")
    parser.add_argument("--dataset", required=True, choices=["setty", "lung", "endo", "dentate"])
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--max-components", type=int, default=10,
                        help="Max components to visualize")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    add_style_args(parser)
    args = parser.parse_args()

    apply_style()
    apply_cli_overrides(args)

    out_root = Path(args.output_dir) if args.output_dir else BIO_RESULTS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, model_name = load_model(args.model_path, device)

    # Per-model sub-directory keeps outputs organized
    out_dir = out_root / model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    from benchmarks.data_utils import DATASET_PATHS, load_data
    from utils.data import DataSplitter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_info = DATASET_PATHS[args.dataset]
    adata = load_data(ds_info["path"], max_cells=3000, hvg_top_genes=3000, seed=args.seed)
    splitter = DataSplitter(
        adata=adata,
        batch_size=128, random_seed=args.seed)

    # Extract components
    print("Extracting latent components...")
    latent, components, comp_names = extract_latent_components(
        model, splitter.test_loader, device
    )

    # Also compute DPMM responsibilities if applicable
    if args.series == "dpmm":
        resp, resp_names = compute_dpmm_responsibilities(model, components)
        if resp is not None:
            print(f"  DPMM active components: {resp.shape[1]}")
            tag = f"{model_name}_{args.dataset}_dpmm_resp"
            plot_component_umap(
                latent, resp, resp_names, splitter.labels_test,
                out_dir / tag,
                title=f"DPMM Responsibilities — {model_name} ({args.dataset})",
                max_components=args.max_components,
                no_title=getattr(args, 'no_title', False),
                figformat=args.fig_format)

    # Plot component intensities
    tag = f"{model_name}_{args.dataset}_components"
    plot_component_umap(
        latent, components, comp_names, splitter.labels_test,
        out_dir / tag,
        title=f"{'Topic Proportions' if args.series == 'topic' else 'Latent Dimensions'} "
              f"— {model_name} ({args.dataset})",
        max_components=args.max_components,
        no_title=getattr(args, 'no_title', False),
        figformat=args.fig_format)

    # Save intermediate data for downstream perturbation analysis
    np.savez(out_dir / f"{model_name}_{args.dataset}_latent_data.npz",
             latent=latent, components=components,
             labels=np.array(splitter.labels_test),
             gene_names=np.array(splitter.var_names if hasattr(splitter, 'var_names') else []))
    print(f"  Latent data saved for downstream analysis.")


if __name__ == "__main__":
    main()
