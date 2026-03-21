"""
Data Utilities for PanODE Benchmark

Provides standalone data preprocessing, normalization, and splitting utilities
for single-cell gene expression data.

Functions:
- is_raw_counts: Check if data contains raw integer counts
- compute_dataset_stats: Compute statistics for adaptive normalization
- adaptive_normalize: Apply adaptive normalization based on dataset characteristics

Classes:
- DataSplitter: Handles data extraction, normalization, and train/val/test splitting
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Dict, Any


# ============================================================
# Normalization Utilities
# ============================================================

def is_raw_counts(X, threshold: float = 0.5) -> bool:
    """
    Check if data contains raw integer counts.
    
    Args:
        X: Input data matrix (dense or sparse)
        threshold: Fraction of values that must be integer-like
        
    Returns:
        True if data appears to be raw counts, False otherwise
    """
    if sp.issparse(X):
        sample_data = X.data[:min(10000, len(X.data))]
    else:
        flat_data = X.flatten()
        sample_data = flat_data[np.random.choice(
            len(flat_data), min(10000, len(flat_data)), replace=False
        )]
    
    sample_data = sample_data[sample_data > 0]
    if len(sample_data) == 0:
        return False
    
    # Check for normalized data (values between 0 and 1)
    if np.mean((sample_data > 0) & (sample_data < 1)) > 0.1:
        return False
    # Check for negative values
    if np.any(sample_data < 0):
        return False
    
    # Check if values are integer-like
    integer_like = np.abs(sample_data - np.round(sample_data)) < 1e-6
    return np.mean(integer_like) >= threshold


def compute_dataset_stats(X) -> Dict[str, float]:
    """
    Compute statistics for adaptive normalization.
    
    Args:
        X: Input data matrix (dense or sparse)
        
    Returns:
        Dictionary with sparsity, lib_size_mean, lib_size_std, max_val
    """
    X_dense = X.toarray() if sp.issparse(X) else X
    return {
        'sparsity': np.mean(X_dense == 0),
        'lib_size_mean': X_dense.sum(axis=1).mean(),
        'lib_size_std': X_dense.sum(axis=1).std(),
        'max_val': X_dense.max()
    }


def adaptive_normalize(X_log, stats: Dict[str, float], verbose: bool = True) -> np.ndarray:
    """
    Apply adaptive normalization based on dataset characteristics.
    
    Args:
        X_log: Log-transformed data matrix
        stats: Dataset statistics from compute_dataset_stats()
        verbose: Whether to print normalization details
        
    Returns:
        Normalized data matrix as float32
    """
    if stats['sparsity'] > 0.95:
        if verbose:
            print("  -> High sparsity: applying conservative clipping")
        X_norm = np.clip(X_log, -5, 5).astype(np.float32)
    elif stats['lib_size_std'] / stats['lib_size_mean'] > 2.0:
        if verbose:
            print("  -> High variance: applying per-cell standardization")
        cell_means = X_log.mean(axis=1, keepdims=True)
        cell_stds = X_log.std(axis=1, keepdims=True) + 1e-6
        X_norm = np.clip((X_log - cell_means) / cell_stds, -10, 10).astype(np.float32)
    elif stats['max_val'] > 10000:
        if verbose:
            print("  -> Extreme values: applying scaled normalization")
        scale = min(1.0, 10.0 / X_log.max())
        X_norm = np.clip(X_log * scale, -10, 10).astype(np.float32)
    else:
        if verbose:
            print("  -> Standard normalization")
        X_norm = np.clip(X_log, -10, 10).astype(np.float32)
    return X_norm


def log_transform(X, offset: float = 1.0) -> np.ndarray:
    """
    Apply log1p transformation.
    
    Args:
        X: Input data matrix
        offset: Offset for log transform (default: 1.0 for log1p)
        
    Returns:
        Log-transformed data
    """
    return np.log1p(X)


def validate_data(X_norm: np.ndarray, raise_error: bool = True) -> bool:
    """
    Validate normalized data for NaN and Inf values.
    
    Args:
        X_norm: Normalized data matrix
        raise_error: Whether to raise ValueError on invalid data
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If data contains NaN or Inf values and raise_error=True
    """
    has_nan = np.isnan(X_norm).any()
    has_inf = np.isinf(X_norm).any()
    
    if has_nan or has_inf:
        if raise_error:
            raise ValueError(f"Invalid values in data: NaN={has_nan}, Inf={has_inf}")
        return False
    return True


# ============================================================
# Data Splitting
# ============================================================

class DataSplitter:
    """
    Standalone data splitter for single-cell data.
    
    Handles:
    - Raw count extraction and validation
    - Adaptive log-normalization
    - Train/val/test splitting
    - DataLoader creation
    - Label encoding
    
    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> splitter = DataSplitter(adata, layer='counts')
        >>> # Access dataloaders
        >>> train_loader = splitter.train_loader
        >>> # Access labels
        >>> labels = splitter.labels_test
    """
    
    def __init__(
        self,
        adata,
        layer: str = 'counts',
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_size: int = 128,
        random_seed: int = 42,
        latent_dim: int = 10,
        adaptive_norm: bool = True,
        verbose: bool = True):
        """
        Initialize DataSplitter and process data.
        
        Args:
            adata: AnnData object with count data
            layer: Layer name for raw counts (default: 'counts')
            train_size: Fraction for training set
            val_size: Fraction for validation set
            test_size: Fraction for test set
            batch_size: Batch size for DataLoaders
            random_seed: Random seed for reproducibility
            latent_dim: Number of clusters for pseudo-labels if no cell_type
            adaptive_norm: Whether to use adaptive normalization
            verbose: Whether to print processing details
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.latent_dim = latent_dim
        self.adaptive_norm = adaptive_norm
        self.verbose = verbose
        
        self._process_data(adata, layer)
    
    def _process_data(self, adata, layer: str):
        """Process AnnData and create splits."""
        # Extract raw counts
        if layer in adata.layers:
            X = adata.layers[layer]
        else:
            X = adata.X
        
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X_raw = X.astype(np.float32)
        
        # Compute statistics
        stats = compute_dataset_stats(X)
        if self.verbose:
            print("Dataset statistics:")
            print(f"  Cells: {X.shape[0]:,}, Genes: {X.shape[1]:,}")
            print(f"  Sparsity: {stats['sparsity']:.2%}, "
                  f"Lib size: {stats['lib_size_mean']:.0f}+/-{stats['lib_size_std']:.0f}, "
                  f"Max value: {stats['max_val']:.0f}")
        
        # Apply library size normalization before log transform (important for scRNA-seq!)
        # This normalizes each cell to have the same total count (target_sum)
        lib_sizes = X.sum(axis=1, keepdims=True)
        lib_sizes = np.clip(lib_sizes, 1e-10, None)  # Avoid division by zero
        target_sum = 1e4  # Standard target for scRNA-seq
        X_libsize_norm = X * target_sum / lib_sizes
        
        # Log-transform (now properly normalized)
        X_log = log_transform(X_libsize_norm)
        
        # Adaptive normalization (clipping/scaling)
        if self.adaptive_norm:
            X_norm = adaptive_normalize(X_log, stats, verbose=self.verbose)
        else:
            X_norm = np.clip(X_log, -10, 10).astype(np.float32)
        
        # Validate
        validate_data(X_norm, raise_error=True)
        
        self.n_obs, self.n_var = X.shape
        
        # Store gene names for downstream analysis
        self.var_names = list(adata.var_names) if hasattr(adata, 'var_names') else [f"gene_{i}" for i in range(self.n_var)]
        
        # Generate or extract labels
        self._extract_labels(adata, X_norm)
        
        # Create splits
        self._create_splits(X_norm, X_raw)
        
        # Create DataLoaders
        self._create_dataloaders()
    
    def _extract_labels(self, adata, X_norm: np.ndarray):
        """Extract or generate cell type labels."""
        if 'cell_type' in adata.obs.columns:
            le = LabelEncoder()
            self.labels = le.fit_transform(adata.obs['cell_type'])
            self.label_encoder = le
            if self.verbose:
                print(f"  Using 'cell_type' labels: {len(np.unique(self.labels))} types")
        else:
            try:
                self.labels = KMeans(
                    n_clusters=self.latent_dim,
                    n_init=10,
                    max_iter=300,
                    random_state=self.random_seed
                ).fit_predict(X_norm)
                self.label_encoder = None
                if self.verbose:
                    print(f"  Generated KMeans pseudo-labels: {self.latent_dim} clusters")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: KMeans failed ({e}), using random labels")
                self.labels = np.random.randint(0, self.latent_dim, size=self.n_obs)
                self.label_encoder = None
    
    def _create_splits(self, X_norm: np.ndarray, X_raw: np.ndarray):
        """Create train/val/test splits."""
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.n_obs)
        
        n_train = int(self.train_size * self.n_obs)
        n_val = int(self.val_size * self.n_obs)
        
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]
        
        # Split data
        self.X_train_norm = X_norm[self.train_idx]
        self.X_train_raw = X_raw[self.train_idx]
        self.X_val_norm = X_norm[self.val_idx]
        self.X_val_raw = X_raw[self.val_idx]
        self.X_test_norm = X_norm[self.test_idx]
        self.X_test_raw = X_raw[self.test_idx]
        
        # Keep full arrays for reference
        self.X_norm = X_norm
        self.X_raw = X_raw
        
        # Split labels
        self.labels_train = self.labels[self.train_idx]
        self.labels_val = self.labels[self.val_idx]
        self.labels_test = self.labels[self.test_idx]
        
        if self.verbose:
            print(f"\nData split:")
            print(f"  Train: {len(self.train_idx):,} cells ({len(self.train_idx)/self.n_obs*100:.1f}%)")
            print(f"  Val:   {len(self.val_idx):,} cells ({len(self.val_idx)/self.n_obs*100:.1f}%)")
            print(f"  Test:  {len(self.test_idx):,} cells ({len(self.test_idx)/self.n_obs*100:.1f}%)")
    
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders.
        
        IMPORTANT: 
        - train_loader uses shuffle=True for stochastic training
        - val_loader and test_loader use shuffle=False to preserve order
          for correct label alignment during evaluation/visualization
        """
        X_train_norm_t = torch.FloatTensor(self.X_train_norm)
        X_train_raw_t = torch.FloatTensor(self.X_train_raw)
        X_val_norm_t = torch.FloatTensor(self.X_val_norm)
        X_val_raw_t = torch.FloatTensor(self.X_val_raw)
        X_test_norm_t = torch.FloatTensor(self.X_test_norm)
        X_test_raw_t = torch.FloatTensor(self.X_test_raw)
        
        train_dataset = TensorDataset(X_train_norm_t, X_train_raw_t)
        val_dataset = TensorDataset(X_val_norm_t, X_val_raw_t)
        test_dataset = TensorDataset(X_test_norm_t, X_test_raw_t)
        
        # Training: shuffle=True for stochastic gradient descent
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        # Validation: shuffle=False to maintain order for metrics
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        # Test: shuffle=False to maintain order for latent extraction & label alignment
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        
        if self.verbose:
            print(f"  Batch size: {self.batch_size}, Batches/epoch: {len(self.train_loader)}")
    
    def get_all_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Create a DataLoader for all data (train + val + test).
        
        Args:
            batch_size: Batch size (default: same as splitter batch_size)
            
        Returns:
            DataLoader for all data
        """
        bs = batch_size or self.batch_size
        X_all_norm_t = torch.FloatTensor(self.X_norm)
        X_all_raw_t = torch.FloatTensor(self.X_raw)
        all_dataset = TensorDataset(X_all_norm_t, X_all_raw_t)
        return DataLoader(all_dataset, batch_size=bs, shuffle=False, drop_last=False)
