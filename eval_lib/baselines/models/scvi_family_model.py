"""
scVI-family baseline wrappers: SCVI, PeakVI, PoissonVI

Bridges the scvi-tools AnnData-based API to our BaseModel DataLoader interface.

scVI (Lopez et al., 2018) — Deep generative model for scRNA-seq with negative
    binomial likelihood.
PeakVI (Ashuach et al., 2022) — Variational inference for scATAC-seq peak data.
PoissonVI — Poisson-likelihood variant for scATAC-seq count data.

Requirements:
    pip install scvi-tools  (includes scvi.model.SCVI, PEAKVI, POISSONVI)

Each wrapper:
    1. Overrides fit() to reconstruct AnnData from DataLoader tensors,
       then delegates to scvi-tools' native training loop.
    2. Overrides extract_latent() to use scvi-tools' get_latent_representation().
    3. Implements encode/decode/forward/compute_loss as thin wrappers around
       the internal scvi-tools model for compatibility.
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

from .base_model import BaseModel

try:
    import anndata as ad
    import scvi as scvi_pkg
    _SCVI_AVAILABLE = True
except ImportError:
    _SCVI_AVAILABLE = False


def _check_scvi():
    """Raise helpful error if scvi-tools is not installed."""
    if not _SCVI_AVAILABLE:
        raise ImportError(
            "scvi-tools is required for scVI-family models. "
            "Install with: pip install scvi-tools"
        )


def _loader_to_array(data_loader: DataLoader) -> np.ndarray:
    """
    Extract all data from a DataLoader into a single numpy array.

    Handles common DataLoader formats:
      - Tensor batches
      - (x_norm, x_raw) tuples  (our Env DataLoader format)
      - (x, y) or (x, y, ...) tuples (uses first element)
    """
    arrays = []
    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        arrays.append(x.astype(np.float32))
    return np.concatenate(arrays, axis=0)


def _array_to_anndata(X: np.ndarray, layer_name: str = "counts") -> "ad.AnnData":
    """
    Wrap a numpy array as AnnData with a counts layer.

    scvi-tools models expect an AnnData object with:
      - .X  set to the expression matrix
      - .layers["counts"]  set for setup_anndata()
    """
    adata = ad.AnnData(X=X.copy())
    adata.layers[layer_name] = X.copy()
    return adata


# ===========================================================================
#  Base adapter for all scVI-family models
# ===========================================================================

class _ScVIFamilyBase(BaseModel):
    """
    Shared adapter logic for scVI-family wrappers.

    Subclasses must set:
        _scvi_model_cls : the scvi.model class (SCVI, PEAKVI, etc.)
        _model_label    : str used in print messages
    """
    _scvi_model_cls = None   # set by subclass
    _model_label = "scVI"    # set by subclass

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dims: list = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        max_epochs: int = 400,
        layer_key: str = "counts",
        model_name: str = "scvi_family",
        **extra_model_kwargs):
        """
        Args:
            input_dim: Number of input features (genes / peaks).
            latent_dim: Latent space dimensionality.
            hidden_dims: Ignored (kept for BaseModel signature compat).
            n_layers: Number of hidden layers in encoder/decoder.
            n_hidden: Width of each hidden layer.
            dropout_rate: Dropout probability.
            max_epochs: Default max training epochs.
            layer_key: AnnData layer key holding count data.
            model_name: Model identifier.
            **extra_model_kwargs: Forwarded to the scvi-tools model constructor.
        """
        _check_scvi()
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims or [n_hidden] * n_layers,
            model_name=model_name)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs
        self.layer_key = layer_key
        self.extra_model_kwargs = extra_model_kwargs

        # Populated after fit()
        self._scvi_model = None
        self._adata = None

    # -- BaseModel abstract methods (thin wrappers) -------------------------

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode via scvi-tools internal model (requires fit first)."""
        if self._scvi_model is None:
            raise RuntimeError(f"{self._model_label} model not trained yet. Call fit() first.")
        arr = x.detach().cpu().numpy().astype(np.float32)
        adata = _array_to_anndata(arr, self.layer_key)
        latent = self._scvi_model.get_latent_representation(adata)
        return torch.from_numpy(latent).to(x.device)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode is not directly supported by scvi-tools' public API.
        Returns zeros matching input_dim for interface compatibility.
        """
        batch_size = z.shape[0]
        return torch.zeros(batch_size, self.input_dim, device=z.device)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        z = self.encode(x, **kwargs)
        return {"latent": z, "reconstruction": self.decode(z)}

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        **kwargs) -> Dict[str, torch.Tensor]:
        """Training loss is handled internally by scvi-tools."""
        zero = torch.tensor(0.0, device=x.device)
        return {"total_loss": zero, "recon_loss": zero}

    # -- fit() override: use scvi-tools native training ---------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = None,
        lr: float = 1e-3,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 25,
        verbose: int = 1,
        verbose_every: int = 1,
        **kwargs) -> Dict[str, list]:
        """
        Train the scVI-family model.

        Reconstructs AnnData from the provided DataLoader(s), sets up the
        scvi-tools model, and runs its native training loop.

        Returns:
            history dict with 'train_loss' (from scvi-tools training plan).
        """
        _check_scvi()
        max_epochs = epochs if epochs is not None else self.max_epochs

        # ----- reconstruct AnnData -----
        X_train = _loader_to_array(train_loader)
        if val_loader is not None:
            X_val = _loader_to_array(val_loader)
            X_all = np.concatenate([X_train, X_val], axis=0)
        else:
            X_all = X_train

        adata = _array_to_anndata(X_all, self.layer_key)
        self._adata = adata

        # ----- setup + construct model -----
        self._setup_anndata(adata)
        self._scvi_model = self._create_scvi_model(adata)

        # ----- validation split -----
        train_size = 1.0
        validation_size = None
        if val_loader is not None:
            n_train = X_train.shape[0]
            n_total = X_all.shape[0]
            train_size = n_train / n_total
            validation_size = 1.0 - train_size

        # ----- train -----
        use_gpu = device.startswith("cuda") and torch.cuda.is_available()
        if verbose >= 1:
            print(f"[{self._model_label}] Training on {X_all.shape[0]} cells, "
                  f"input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
                  f"max_epochs={max_epochs}")

        plan_kwargs = {"lr": lr}
        train_kwargs = {
            "max_epochs": max_epochs,
            "train_size": train_size,
            "early_stopping": val_loader is not None,
            "early_stopping_patience": patience,
            "plan_kwargs": plan_kwargs,
            "check_val_every_n_epoch": verbose_every,
        }

        # scvi-tools uses accelerator kwarg in newer versions
        try:
            self._scvi_model.train(
                accelerator="gpu" if use_gpu else "cpu",
                devices=1 if use_gpu else "auto",
                **train_kwargs)
        except TypeError:
            # Fallback for older scvi-tools API
            self._scvi_model.train(
                use_gpu=use_gpu,
                **train_kwargs)

        # ----- extract training history -----
        try:
            hist = self._scvi_model.history
            train_loss = hist.get("train_loss_epoch", hist.get("elbo_train", []))
            if hasattr(train_loss, "values"):
                train_loss = train_loss.values.flatten().tolist()
            else:
                train_loss = list(train_loss) if train_loss is not None else []
        except Exception:
            train_loss = []

        if verbose >= 1:
            print(f"✓ {self._model_label} training finished!")

        return {"train_loss": train_loss, "val_loss": []}

    # -- extract_latent() override ------------------------------------------

    def extract_latent(
        self,
        data_loader: DataLoader,
        device: str = "cuda",
        return_reconstructions: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract latent representations using scvi-tools' native method.
        """
        if self._scvi_model is None:
            raise RuntimeError(
                f"{self._model_label} model not trained. Call fit() first."
            )

        X = _loader_to_array(data_loader)
        adata = _array_to_anndata(X, self.layer_key)
        self._setup_anndata(adata)

        latent = self._scvi_model.get_latent_representation(adata)
        result: Dict[str, Any] = {"latent": latent}

        if return_reconstructions:
            warnings.warn(
                f"{self._model_label}: reconstruction extraction not supported "
                f"via scvi-tools public API. Returning zeros.",
                stacklevel=2)
            result["reconstruction"] = np.zeros(
                (X.shape[0], self.input_dim), dtype=np.float32
            )

        return result

    # -- private helpers (overridden per model) -----------------------------

    def _setup_anndata(self, adata: "ad.AnnData"):
        """Call the appropriate setup_anndata for this model family."""
        self._scvi_model_cls.setup_anndata(adata, layer=self.layer_key)

    def _create_scvi_model(self, adata: "ad.AnnData"):
        """Instantiate the scvi-tools model. Override for model-specific kwargs."""
        return self._scvi_model_cls(
            adata,
            n_latent=self.latent_dim,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            **self.extra_model_kwargs)


# ===========================================================================
#  Concrete model classes
# ===========================================================================

class SCVIModel(_ScVIFamilyBase):
    """
    scVI wrapper (Lopez et al., 2018).

    Deep generative model for scRNA-seq data with negative binomial
    likelihood, variational inference, and library size correction.

    Usage:
        >>> model = SCVIModel(input_dim=2000, latent_dim=10)
        >>> history = model.fit(train_loader, val_loader)
        >>> result = model.extract_latent(test_loader)
        >>> Z = result['latent']  # (n_cells, latent_dim)
    """
    _model_label = "scVI"

    def __init__(self, input_dim: int, latent_dim: int = 10, **kwargs):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            model_name="scVI",
            **kwargs)

    @property
    def _scvi_model_cls(self):
        _check_scvi()
        return scvi_pkg.model.SCVI


class PeakVIModel(_ScVIFamilyBase):
    """
    PeakVI wrapper (Ashuach et al., 2022).

    Variational inference for scATAC-seq peak-level data.
    Models binary accessibility with a Bernoulli likelihood.

    Usage:
        >>> model = PeakVIModel(input_dim=50000, latent_dim=10)
        >>> history = model.fit(train_loader)
        >>> result = model.extract_latent(test_loader)
    """
    _model_label = "PeakVI"

    def __init__(self, input_dim: int, latent_dim: int = 10, **kwargs):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            model_name="PeakVI",
            **kwargs)

    @property
    def _scvi_model_cls(self):
        _check_scvi()
        return scvi_pkg.model.PEAKVI

    def _create_scvi_model(self, adata):
        """PeakVI uses different parameter names."""
        return scvi_pkg.model.PEAKVI(
            adata,
            n_latent=self.latent_dim,
            **self.extra_model_kwargs)


class PoissonVIModel(_ScVIFamilyBase):
    """
    PoissonVI wrapper.

    scVI variant using Poisson likelihood for scATAC-seq count data
    (e.g., fragment counts per peak).

    Usage:
        >>> model = PoissonVIModel(input_dim=50000, latent_dim=10)
        >>> history = model.fit(train_loader)
        >>> result = model.extract_latent(test_loader)
    """
    _model_label = "PoissonVI"

    def __init__(self, input_dim: int, latent_dim: int = 10, **kwargs):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            model_name="PoissonVI",
            **kwargs)

    @property
    def _scvi_model_cls(self):
        _check_scvi()
        # PoissonVI was added in scvi-tools >=1.1; fall back to SCVI with gene_likelihood="poisson"
        if hasattr(scvi_pkg.model, "POISSONVI"):
            return scvi_pkg.model.POISSONVI
        return None

    def _create_scvi_model(self, adata):
        """PoissonVI or SCVI with Poisson likelihood fallback."""
        if hasattr(scvi_pkg.model, "POISSONVI"):
            return scvi_pkg.model.POISSONVI(
                adata,
                n_latent=self.latent_dim,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                dropout_rate=self.dropout_rate,
                **self.extra_model_kwargs)
        # Fallback: use SCVI with Poisson likelihood
        warnings.warn(
            "POISSONVI not found in scvi-tools. "
            "Falling back to SCVI with gene_likelihood='poisson'.",
            stacklevel=2)
        return scvi_pkg.model.SCVI(
            adata,
            n_latent=self.latent_dim,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            gene_likelihood="poisson",
            **self.extra_model_kwargs)

    def _setup_anndata(self, adata):
        """PoissonVI may use its own setup or fall back to SCVI's."""
        if hasattr(scvi_pkg.model, "POISSONVI"):
            scvi_pkg.model.POISSONVI.setup_anndata(adata, layer=self.layer_key)
        else:
            scvi_pkg.model.SCVI.setup_anndata(adata, layer=self.layer_key)


# ===========================================================================
#  Factory functions (consistent with other baselines)
# ===========================================================================

def create_scvi_model(input_dim: int, **kwargs) -> SCVIModel:
    """Factory: create scVI model instance."""
    return SCVIModel(input_dim=input_dim, **kwargs)


def create_peakvi_model(input_dim: int, **kwargs) -> PeakVIModel:
    """Factory: create PeakVI model instance."""
    return PeakVIModel(input_dim=input_dim, **kwargs)


def create_poissonvi_model(input_dim: int, **kwargs) -> PoissonVIModel:
    """Factory: create PoissonVI model instance."""
    return PoissonVIModel(input_dim=input_dim, **kwargs)
