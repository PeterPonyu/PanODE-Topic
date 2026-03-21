"""Data loading and preprocessing utilities for benchmarks.

Centralises the data pipeline (cache key, raw-count detection,
normalisation, HVG selection) that was previously embedded in
``benchmark_base.py``.

Exports
-------
load_data                — convenience wrapper with sensible defaults
load_or_preprocess_adata — full pipeline with caching
DATASET_PATHS            — legacy dict (thin wrapper around DATASET_REGISTRY)
"""

import hashlib
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from benchmarks.config import DEFAULT_OUTPUT_DIR, ensure_dirs
from benchmarks.dataset_registry import DATASET_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy compatibility bridge
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATHS = {
    k: {**v, "type": v["data_type"]}   # add legacy "type" alias
    for k, v in DATASET_REGISTRY.items()
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(data_path, max_cells=3000, hvg_top_genes=3000, seed=42,
              use_cache=True):
    """Convenience wrapper around :func:`load_or_preprocess_adata`.

    Returns an AnnData with HVGs selected, counts layer, and cell-type
    labels.
    """
    cache_dir = DEFAULT_OUTPUT_DIR / "cache"
    return load_or_preprocess_adata(
        data_path, max_cells, hvg_top_genes, seed, cache_dir,
        use_cache=use_cache)


def load_or_preprocess_adata(data_path, max_cells, hvg_top_genes, seed,
                             cache_dir, use_cache=True):
    """Load h5ad, subsample, normalise, select HVGs, and cache result.

    Parameters
    ----------
    data_path : str or Path
        Path to the source ``.h5ad`` file.
    max_cells : int or None
        Subsample to at most this many cells (``None`` = keep all).
    hvg_top_genes : int or None
        Number of highly variable genes to select (``None`` = keep all).
    seed : int
        Random seed for subsampling reproducibility.
    cache_dir : str or Path
        Directory for the preprocessed-data cache.
    use_cache : bool
        If True, return a cached file when the cache-key matches.

    Returns
    -------
    AnnData
    """
    cache_dir = Path(cache_dir)
    ensure_dirs(cache_dir)
    cache_name = f"base_preproc_{_cache_key(data_path, max_cells, hvg_top_genes, seed)}.h5ad"
    cache_path = cache_dir / cache_name

    if use_cache and cache_path.exists():
        print(f"Loading cached preprocessed data: {cache_path}")
        return sc.read_h5ad(str(cache_path))

    print(f"\nLoading {data_path}...")
    adata = sc.read_h5ad(str(data_path))
    print(f"  Original shape: {adata.shape}")

    # Sub-sample before heavy processing
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.RandomState(seed)
        keep_idx = rng.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[keep_idx].copy()
        print(f"  Subsampled to {adata.n_obs} cells")

    # Normalisation (only if data contains raw counts)
    has_counts_layer = 'counts' in adata.layers

    if has_counts_layer:
        is_raw = _is_raw_counts(adata.layers['counts'])
        print(f"  Found 'counts' layer: "
              f"{'raw counts' if is_raw else 'already processed'}")
        if is_raw:
            adata.X = adata.layers['counts'].copy()
            if sp.issparse(adata.X):
                adata.X = adata.X.toarray()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            print(f"  Applied normalize_total + log1p from counts layer")
        else:
            print(f"  Using existing X matrix (already normalized)")
    else:
        is_raw = _is_raw_counts(adata.X)
        print(f"  No 'counts' layer found. X matrix: "
              f"{'raw counts' if is_raw else 'already processed'}")
        if is_raw:
            if sp.issparse(adata.X):
                adata.layers['counts'] = adata.X.copy()
            else:
                adata.layers['counts'] = sp.csr_matrix(adata.X.copy())
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            print(f"  Applied normalize_total + log1p")
        else:
            if sp.issparse(adata.X):
                adata.layers['counts'] = adata.X.copy()
            else:
                adata.layers['counts'] = sp.csr_matrix(adata.X.copy())
            print(f"  Data already normalized, skipping normalization")

    # HVG selection
    if hvg_top_genes is not None and adata.n_vars > hvg_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top_genes)
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"  Selected {adata.n_vars} HVGs")

    if use_cache:
        print(f"Saving preprocessed cache: {cache_path}")
        adata.write_h5ad(str(cache_path))

    return adata


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_key(data_path, max_cells, hvg_top_genes, seed):
    payload = f"{data_path}|{max_cells}|{hvg_top_genes}|{seed}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _is_raw_counts(X, threshold: float = 0.5) -> bool:
    """Heuristic: check whether *X* contains raw integer counts."""
    if sp.issparse(X):
        sample_data = X.data[:min(10000, len(X.data))]
    else:
        flat_data = np.asarray(X).flatten()
        sample_data = flat_data[np.random.choice(
            len(flat_data), min(10000, len(flat_data)), replace=False)]

    sample_data = sample_data[sample_data > 0]
    if len(sample_data) == 0:
        return False

    frac_small = np.mean((sample_data > 0) & (sample_data < 1))
    if frac_small > 0.1:
        return False
    if np.any(sample_data < 0):
        return False

    integer_like = np.abs(sample_data - np.round(sample_data)) < 1e-6
    return np.mean(integer_like) >= threshold
