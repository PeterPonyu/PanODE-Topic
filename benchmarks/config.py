"""
Centralized configuration for all benchmark scripts.
Provides consistent defaults and directory management.

Path policy
-----------
Every script that reads or writes benchmark artefacts should import paths
from this module rather than constructing its own.  The five canonical
directories are:

    DEFAULT_OUTPUT_DIR   – benchmarks/benchmark_results/
    CACHE_DIR            – benchmarks/benchmark_results/cache/
    STATISTICAL_EXPORTS_DIR – benchmarks/benchmark_results/statistical_exports/
    PAPER_FIGURES_DIR    – benchmarks/paper_figures/
    BIO_RESULTS_DIR      – benchmarks/biological_validation/results/
    DYNAMICS_DIR         – benchmarks/training_dynamics_results/
"""

import os
import random
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Data (override via PANODE_DATASETS_ROOT; default ~/datasets)
    data_path: Path = (
        Path(os.environ.get("PANODE_DATASETS_ROOT", str(Path.home() / "datasets")))
        .expanduser()
        .resolve()
        / "DevelopmentDatasets"
        / "setty.h5ad"
    )
    data_type: str = "trajectory"
    
    # Model architecture
    latent_dim: int = 10  # Standard latent dimension for all models (allows fair comparison)
    
    # Training
    lr: float = 1e-3  # Standard learning rate for all models
    batch_size: int = 128  # Consistent batch size for ALL models (including attention)
    epochs: int = 1000  # Canonical refresh setting (article: 1000 epochs)
    patience: int = 100  # Early stopping patience (set to 0 or negative to disable)
    early_stopping: bool = False  # Disable by default for epoch series experiments
    
    # Data preprocessing  
    hvg_top_genes: int = 3000
    max_cells: int = 3000  # Subsample to 3000 cells for benchmarking
    
    # Metrics
    dre_k: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    verbose_every: int = 50


# Global config instance
BASE_CONFIG = BenchmarkConfig()

# ──────────────────────────────────────────────────────────────────────────────
# Canonical directory paths (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
_BENCHMARKS_ROOT = Path(__file__).resolve().parent          # benchmarks/

DEFAULT_OUTPUT_DIR      = _BENCHMARKS_ROOT / "benchmark_results"
CACHE_DIR               = DEFAULT_OUTPUT_DIR / "cache"
STATISTICAL_EXPORTS_DIR = DEFAULT_OUTPUT_DIR / "statistical_exports"
MODELS_DIR              = DEFAULT_OUTPUT_DIR / "models"

PAPER_FIGURES_DIR       = _BENCHMARKS_ROOT / "paper_figures"
BIO_RESULTS_DIR         = _BENCHMARKS_ROOT / "biological_validation" / "results"
DYNAMICS_DIR            = _BENCHMARKS_ROOT / "training_dynamics_results"


def result_subdir(benchmark_name: str, *subdirs: str) -> Path:
    """Return ``DEFAULT_OUTPUT_DIR / benchmark_name / *subdirs``.

    Example::

        csv_dir = result_subdir("warmup", "csv")
        # → benchmarks/benchmark_results/warmup/csv
    """
    return DEFAULT_OUTPUT_DIR.joinpath(benchmark_name, *subdirs)


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
