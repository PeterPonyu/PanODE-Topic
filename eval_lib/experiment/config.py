"""eval_lib.experiment.config — Portable experiment configuration dataclass.

``ExperimentConfig`` specifies an experiment fully: which models to train,
which datasets to use, and all hyper-parameters. It has NO project-specific
content — projects build their own ``PRESETS`` dict from this dataclass.

Usage in a project::

    from eval_lib.experiment.config import ExperimentConfig

    ABLATION = ExperimentConfig(
        name="ablation",
        models=my_model_dict(),
        datasets=MY_DATASETS,
        epochs=100,
        description="Component ablation study")
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ExperimentConfig:
    """Fully parameterised experiment specification.

    Attributes
    ----------
    name : str
        Experiment identifier (used as output sub-directory name).
    models : dict
        ``{display_name: {"class": ModelCls, "params": {...}}}``.
    datasets : dict
        ``{short_name: {"path": ..., "label_key": ..., "data_type": ...}}``.
    epochs, max_cells, n_hvg, latent_dim, batch_size, seed : int
        Training and data budget parameters.
    patience : int
        Early-stopping patience (large value = effectively disabled).
    verbose_every : int
        Print training progress every N epochs.
    dre_k : int
        k for DRE k-nearest-neighbour quality (default 15).
    output_root : Path
        Root directory for all experiment outputs.
    description : str
        Human-readable experiment description.
    """

    name: str = "experiment"
    models: Dict[str, dict] = field(default_factory=dict)
    datasets: Dict[str, dict] = field(default_factory=dict)
    epochs: int = 100
    max_cells: int = 3000
    n_hvg: int = 2000
    latent_dim: int = 10
    batch_size: int = 128
    seed: int = 42
    patience: int = 9999          # effectively disabled for fixed-epoch runs
    verbose_every: int = 25
    dre_k: int = 15               # k for DRE k-nearest-neighbour quality
    output_root: Path = Path("experiments/results")
    description: str = ""

    # ── Derived paths ─────────────────────────────────────────────────────

    @property
    def tables_dir(self) -> Path:
        return self.output_root / self.name / "tables"

    @property
    def series_dir(self) -> Path:
        return self.output_root / self.name / "series"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / self.name / "figures"

    @property
    def method_names(self) -> List[str]:
        return list(self.models.keys())

    @property
    def dataset_keys(self) -> List[str]:
        return list(self.datasets.keys())

    # ── Utilities ─────────────────────────────────────────────────────────

    def with_overrides(self, **kwargs) -> "ExperimentConfig":
        """Return a shallow copy with selected fields overridden."""
        cfg = copy.copy(self)
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown config field: {k}")
        return cfg

    def summary(self) -> str:
        """One-line-per-field experiment description."""
        lines = [
            f"Experiment : {self.name}",
            f"Description: {self.description}",
            f"Models     : {len(self.models)} variants → {self.method_names}",
            f"Datasets   : {len(self.datasets)} → {self.dataset_keys}",
            f"Epochs     : {self.epochs}",
            f"Cells      : {self.max_cells}",
            f"HVGs       : {self.n_hvg}",
            f"Latent dim : {self.latent_dim}",
            f"Batch size : {self.batch_size}",
            f"Seed       : {self.seed}",
            f"Output     : {self.output_root / self.name}",
        ]
        return "\n".join(lines)
