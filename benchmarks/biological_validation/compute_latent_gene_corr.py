"""Compute Pearson correlation between latent dimensions and HVG expression.

For each model whose ``*_latent_data.npz`` exists in the bio results
directory, this script:

1. Reloads the corresponding dataset (same seed → same test split).
2. Extracts ``X_test_norm  [N, G]`` — the normalised expression matrix
   for the same test cells that produced ``latent  [N, K]``.
3. Computes ``corr  [K, G]`` — the Pearson correlation between each
   latent dimension (or topic proportion) and each gene.
4. Saves ``{model}_{dataset}_correlation.npz`` with keys
   ``correlation``, ``gene_names``.

Usage
-----
    python -m benchmarks.biological_validation.compute_latent_gene_corr
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.config import BIO_RESULTS_DIR
BIO_RESULTS = BIO_RESULTS_DIR

# Datasets used in the biological validation
REPRESENTATIVE_DATASETS = ["setty", "endo", "dentate"]
# Models to process
ALL_MODELS = [
    "DPMM-Base", "DPMM-Contrastive", "DPMM-Transformer",
    "Topic-Base", "Topic-Contrastive", "Topic-Transformer",
]


def _pearson_cols(X, Y):
    """Row-wise Pearson correlation: correlate each row of Y with each column of X.

    X: [N, G]  (observations × genes)
    Y: [N, K]  (observations × latent dims)

    Returns: [K, G] correlation matrix.
    """
    N = X.shape[0]
    # Centre
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # Covariance [K, G]
    cov = (Yc.T @ Xc) / (N - 1)
    # Standard deviations
    sx = Xc.std(axis=0, ddof=1) + 1e-12  # [G]
    sy = Yc.std(axis=0, ddof=1) + 1e-12  # [K]
    corr = cov / np.outer(sy, sx)
    return np.clip(corr, -1.0, 1.0)


def compute_all(datasets=None, models=None):
    """Compute correlation for all available (model, dataset) pairs."""
    from benchmarks.biological_validation import load_data_with_genes

    datasets = datasets or REPRESENTATIVE_DATASETS
    models = models or ALL_MODELS

    for dataset in datasets:
        # Load dataset once per dataset
        try:
            splitter, gene_names = load_data_with_genes(
                dataset, seed=42, max_cells=3000, hvg=3000)
        except Exception as e:
            print(f"  Could not load dataset {dataset}: {e}")
            continue
        X_test = splitter.X_test_norm  # [N_test, G]
        N_test = X_test.shape[0]
        print(f"\n  Dataset {dataset}: {N_test} test cells, "
              f"{X_test.shape[1]} genes")

        for model in models:
            tag = f"{model}_{dataset}"
            latent_path = BIO_RESULTS / f"{tag}_latent_data.npz"
            # Also check model subdirectory
            if not latent_path.exists():
                latent_path = (BIO_RESULTS / model.replace("/", "_")
                               / f"{tag}_latent_data.npz")
            if not latent_path.exists():
                continue

            ld = np.load(latent_path, allow_pickle=True)
            latent = ld["latent"]  # [N, K]

            # Verify cell count match
            if latent.shape[0] != N_test:
                print(f"    {tag}: latent {latent.shape[0]} != test "
                      f"{N_test} cells — skipping")
                continue

            corr = _pearson_cols(X_test, latent)  # [K, G]

            out_path = BIO_RESULTS / f"{tag}_correlation.npz"
            np.savez(out_path,
                     correlation=corr,
                     gene_names=np.array(gene_names))
            print(f"    {tag}: corr {corr.shape}  "
                  f"range [{corr.min():.3f}, {corr.max():.3f}]")

    print("\n  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    compute_all(args.datasets, args.models)
