"""Pre-compute per-cell expression for top-correlated genes (UMAP coloring).

For each (model, dataset) pair with latent_data + correlation, this script:
1. Loads the dataset (same seed → same test split as latent extraction).
2. Identifies the top-1 correlated gene per latent component (by |r|).
3. Extracts the per-cell normalised expression for those genes.
4. Computes UMAP embedding from the latent space.
5. Saves ``{model}_{dataset}_umap_data.npz`` with keys:
   - ``umap_emb`` [N, 2]  — 2D UMAP embedding
   - ``latent`` [N, K]    — latent representation per cell
   - ``top_gene_expr`` [N, K] — expression of top-correlated gene per dim
   - ``top_gene_names`` [K]   — name of top gene per dim
   - ``labels`` [N]           — cell cluster labels

Usage
-----
    python -m benchmarks.biological_validation.precompute_umap_data
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.config import BIO_RESULTS_DIR
BIO_RESULTS = BIO_RESULTS_DIR

REPRESENTATIVE_DATASETS = ["setty", "endo", "dentate"]
ALL_MODELS = [
    "DPMM-Base", "DPMM-Contrastive", "DPMM-Transformer",
    "Topic-Base", "Topic-Contrastive", "Topic-Transformer",
]


def compute_all(datasets=None, models=None, force=False):
    """Pre-compute UMAP + gene expression data for figure generation."""
    # Delayed imports — only needed when actually computing
    try:
        from benchmarks.biological_validation import load_data_with_genes
    except ImportError:
        print("  Cannot import load_data_with_genes. "
              "Run from PanODE-LAB root.")
        return

    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed. pip install umap-learn")
        return

    datasets = datasets or REPRESENTATIVE_DATASETS
    models = models or ALL_MODELS

    for dataset in datasets:
        # Load dataset once per dataset (same seed as bio validation)
        splitter = None
        gene_names = None
        X_test = None

        for model in models:
            tag = f"{model}_{dataset}"
            out_path = BIO_RESULTS / f"{tag}_umap_data.npz"
            if out_path.exists() and not force:
                print(f"    {tag}: already exists (skip)")
                continue

            # Check required inputs — check root AND model subdir
            latent_path = BIO_RESULTS / f"{tag}_latent_data.npz"
            if not latent_path.exists():
                latent_path = (BIO_RESULTS / model.replace("/", "_")
                               / f"{tag}_latent_data.npz")
            if not latent_path.exists():
                continue
            corr_path = BIO_RESULTS / f"{tag}_correlation.npz"
            if not corr_path.exists():
                corr_path = (BIO_RESULTS / model.replace("/", "_")
                             / f"{tag}_correlation.npz")
            if not corr_path.exists():
                continue

            # Lazy-load dataset
            if splitter is None:
                try:
                    splitter, gene_names = load_data_with_genes(
                        dataset, seed=42, max_cells=3000, hvg=3000)
                    X_test = splitter.X_test_norm  # [N, G]
                    print(f"\n  Dataset {dataset}: {X_test.shape[0]} cells, "
                          f"{X_test.shape[1]} genes")
                except Exception as e:
                    print(f"  Could not load dataset {dataset}: {e}")
                    break

            ld = np.load(latent_path, allow_pickle=True)
            latent = ld["latent"]       # [N, K]
            labels = ld.get("labels")   # [N]

            cd = np.load(corr_path, allow_pickle=True)
            corr = cd["correlation"]    # [K, G]
            corr_genes = cd.get("gene_names")

            # Verify dimensions
            if latent.shape[0] != X_test.shape[0]:
                print(f"    {tag}: latent {latent.shape[0]} != X_test "
                      f"{X_test.shape[0]} — skipping")
                continue

            K = corr.shape[0]

            # Top-1 positively correlated gene per component
            # Use positive correlation only (not absolute) to identify
            # genes that increase with component activation.
            top_gene_idx = np.argmax(corr, axis=1)  # [K]
            top_gene_names = np.array(
                [str(corr_genes[i]) for i in top_gene_idx]
                if corr_genes is not None
                else [f"gene_{i}" for i in top_gene_idx]
            )
            # Per-cell expression of top genes
            top_gene_expr = X_test[:, top_gene_idx].copy()  # [N, K]
            if hasattr(top_gene_expr, 'toarray'):
                top_gene_expr = top_gene_expr.toarray()
            top_gene_expr = np.asarray(top_gene_expr, dtype=np.float32)

            # Compute UMAP from latent
            reducer = UMAP(n_components=2, n_neighbors=15,
                           min_dist=0.3, random_state=42)
            umap_emb = reducer.fit_transform(latent).astype(np.float32)

            np.savez(out_path,
                     umap_emb=umap_emb,
                     latent=latent,
                     top_gene_expr=top_gene_expr,
                     top_gene_names=top_gene_names,
                     labels=labels)
            print(f"    {tag}: saved  umap={umap_emb.shape}  "
                  f"genes={top_gene_names.tolist()}")

    print("\n  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute UMAP + gene expression data for Figure 6")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if output exists")
    args = parser.parse_args()
    compute_all(args.datasets, args.models, args.force)
