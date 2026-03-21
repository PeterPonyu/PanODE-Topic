"""External model registry for cross-method benchmarking.

.. deprecated::
    This module is **superseded** by ``eval_lib.baselines.registry``
    which contains 21+ models in 5 groups.  Use::

        from eval_lib.baselines.registry import EXTERNAL_MODELS, MODEL_GROUPS

    This file is kept only for backward compatibility with
    ``benchmarks/runners/benchmark_external.py``.

Maps each external baseline (from Liora unified_models) to its factory
function and default hyper-parameters, mirroring the internal
``model_registry.py`` layout.

The 12 external models:
  CellBLAST, GMVAE, SCALEX, scDiffusion, siVAE, CLEAR,
  scDAC, scDeepCluster, scDHMap, scGNN, scGCC, scSMD

Each entry specifies:
  - factory : callable — creates model given input_dim
  - params  : dict     — keyword arguments for the factory
  - notes   : str      — brief description
"""

import warnings as _warnings
_warnings.warn(
    "benchmarks.external_model_registry is deprecated — "
    "use eval_lib.baselines.registry (21+ models, 5 groups) instead.",
    DeprecationWarning,
    stacklevel=2)

from benchmarks.external_models import (
    create_cellblast_model,
    create_gmvae_model,
    create_scalex_model,
    create_scdiffusion_model,
    create_sivae_model,
    create_clear_model,
    create_scdac_model,
    create_scdeepcluster_model,
    create_scdhmap_model,
    create_scgnn_model,
    create_scgcc_model,
    create_scsmd_model)

# ═══════════════════════════════════════════════════════════════════════════════
# External model configurations
# Latent dim = 10 for fair comparison with internal models
# ═══════════════════════════════════════════════════════════════════════════════

EXTERNAL_MODELS = {
    "CellBLAST": {
        "factory": create_cellblast_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "output_distribution": "gaussian",
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "VAE with adversarial batch correction (batch correction disabled)",
    },
    "GMVAE": {
        "factory": create_gmvae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "distribution": "euclidean",
            "loss_type": "MSE",
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "Geometric Manifold VAE (Euclidean mode)",
    },
    "SCALEX": {
        "factory": create_scalex_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "n_domains": 1,
            "recon_type": "mse",
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "VAE with domain-specific batch normalization",
    },
    "scDiffusion": {
        "factory": create_scdiffusion_model,
        "params": {
            "latent_dim": 64,        # UNet base channels
            "embedding_dim": 10,     # actual latent dim
            "n_timesteps": 500,
            "beta_schedule": "linear",
        },
        "fit_params": {
            "epochs": 500,           # diffusion models need fewer epochs
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "Diffusion-based generative model",
    },
    "siVAE": {
        "factory": create_sivae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "use_interpretable": True,
            "output_distribution": "gaussian",
            "dropout": 0.1,
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "Sparse interpretable VAE for GRN inference",
    },
    "CLEAR": {
        "factory": create_clear_model,
        "params": {
            "latent_dim": 128,
            "hidden_dim": 256,
            "queue_size": 1024,
            "temperature": 0.2,
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "MoCo contrastive learning (no decoder)",
    },
    "scDAC": {
        "factory": create_scdac_model,
        "params": {
            "latent_dim": 10,
            "encoder_dims": [256, 128],
            "decoder_dims": [128, 256],
            "dpmm_warmup_ratio": 0.6,
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "Deep AE with DPMM clustering",
    },
    "scDeepCluster": {
        "factory": create_scdeepcluster_model,
        "params": {
            "latent_dim": 10,
            "n_clusters": 10,
            "hidden_dims": [256, 64],
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "ZINB AE + DEC clustering",
    },
    "scDHMap": {
        "factory": create_scdhmap_model,
        "params": {
            "latent_dim": 10,
            "encoder_layers": [128, 64, 32, 16],
            "decoder_layers": [16, 32, 64, 128],
            "likelihood": "zinb",
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "Hyperbolic deep embedding (Lorentz model)",
    },
    "scGNN": {
        "factory": create_scgnn_model,
        "params": {
            "latent_dim": 10,
            "hidden_dim": 32,
            "dropout": 0.1,
            "k_neighbors": 10,
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "Graph neural network with kNN graph",
    },
    "scGCC": {
        "factory": create_scgcc_model,
        "params": {
            "latent_dim": 128,
            "queue_size": 512,
            "heads": 4,
        },
        "fit_params": {
            "epochs": 500,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "GAT + MoCo contrastive (requires torch_geometric)",
    },
    "scSMD": {
        "factory": create_scsmd_model,
        "params": {
            "latent_dim": 10,
            "n_clusters": 10,
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "ResNet CNN + manifold clustering",
    },
}


def list_external_models():
    """Print summary of all external models."""
    print(f"{'Model':<18} {'Latent':<8} {'Notes'}")
    print("-" * 70)
    for name, cfg in EXTERNAL_MODELS.items():
        ld = cfg["params"].get("latent_dim",
                               cfg["params"].get("embedding_dim", "?"))
        print(f"{name:<18} {str(ld):<8} {cfg['notes']}")


if __name__ == "__main__":
    list_external_models()
