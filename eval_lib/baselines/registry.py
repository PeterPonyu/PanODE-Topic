"""External model registry for cross-method benchmarking.

Aligned with LAIOR-style categories (https://peterponyu.github.io/liora-ui/):
  - Generative: CellBLAST, SCALEX, scDiffusion, siVAE, scDAC, scDeepCluster, scDHMap, scSMD
  - Gaussian geometric (GM-VAE series): GMVAE, GMVAE-Poincare, GMVAE-PGM, GMVAE-LearnablePGM, GMVAE-HW
  - Disentanglement: VAE-DIP, VAE-TC, InfoVAE, BetaVAE (default VAE + DIP/TC/MMD/beta)
  - Graph & contrastive: CLEAR, scGCC, scGNN (predictive / encoder-only style)
  - Optional scVI-family (scvi-tools): scVI, PeakVI, PoissonVI

Each entry specifies:
  - factory : callable — creates model given input_dim
  - params  : dict     — keyword arguments for the factory
  - notes   : str      — brief description
"""

from .models import (
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
    create_scsmd_model,
    create_disentanglement_vae_model)

# scVI-family (optional — requires scvi-tools)
try:
    from .models import (
        create_scvi_model,
        create_peakvi_model,
        create_poissonvi_model)
    _SCVI_AVAILABLE = True
except ImportError:
    _SCVI_AVAILABLE = False

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
    "GMVAE-Poincare": {
        "factory": create_gmvae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "distribution": "poincare",
            "loss_type": "MSE",
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "GM-VAE with Poincaré ball latent (figure group: GM-VAE series)",
    },
    "GMVAE-PGM": {
        "factory": create_gmvae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "distribution": "pgm",
            "loss_type": "MSE",
        },
        "fit_params": {"epochs": 1000, "lr": 1e-3, "patience": 100, "verbose_every": 50},
        "notes": "GM-VAE with PGM latent (figure group: GM-VAE series)",
    },
    "GMVAE-LearnablePGM": {
        "factory": create_gmvae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "distribution": "learnable_pgm",
            "loss_type": "MSE",
        },
        "fit_params": {"epochs": 1000, "lr": 1e-3, "patience": 100, "verbose_every": 50},
        "notes": "GM-VAE with learnable PGM (figure group: GM-VAE series)",
    },
    "GMVAE-HW": {
        "factory": create_gmvae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "distribution": "hw",
            "loss_type": "MSE",
        },
        "fit_params": {"epochs": 1000, "lr": 1e-3, "patience": 100, "verbose_every": 50},
        "notes": "GM-VAE with hyperbolic (HW) latent (figure group: GM-VAE series)",
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
        "notes": "MoCo contrastive learning (no decoder) [figure group: graph & contrastive]",
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
        "notes": "Graph neural network with kNN graph [figure group: graph & contrastive]",
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
        "notes": "GAT + MoCo contrastive (requires torch_geometric) [figure group: graph & contrastive]",
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
    # Disentanglement group: default VAE + DIP or TC loss (figure group)
    "VAE-DIP": {
        "factory": create_disentanglement_vae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "dip_weight": 1.0,
            "tc_weight": 0.0,
            "beta": 1.0,
        },
        "fit_params": {
            "epochs": 1000,
            "lr": 1e-3,
            "patience": 100,
            "verbose_every": 50,
        },
        "notes": "VAE + DIP-VAE covariance regularization (identity prior on cov of mean)",
    },
    "VAE-TC": {
        "factory": create_disentanglement_vae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "dip_weight": 0.0,
            "tc_weight": 1.0,
            "beta": 1.0,
        },
        "fit_params": {"epochs": 1000, "lr": 1e-3, "patience": 100, "verbose_every": 50},
        "notes": "VAE + total-correlation penalty (factorized latent)",
    },
    "InfoVAE": {
        "factory": create_disentanglement_vae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "dip_weight": 0.0,
            "tc_weight": 0.0,
            "infovae_mmd_weight": 1.0,
            "beta": 1.0,
        },
        "fit_params": {"epochs": 1000, "lr": 1e-3, "patience": 100, "verbose_every": 50},
        "notes": "Info-VAE: MMD between aggregated posterior and prior (Zhao et al.)",
    },
    "BetaVAE": {
        "factory": create_disentanglement_vae_model,
        "params": {
            "latent_dim": 10,
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "dip_weight": 0.0,
            "tc_weight": 0.0,
            "beta": 4.0,
        },
        "fit_params": {"epochs": 1000, "lr": 1e-3, "patience": 100, "verbose_every": 50},
        "notes": "Beta-VAE (beta=4) for stronger disentanglement (Higgins et al.)",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# scVI-family models (optional, requires scvi-tools)
# ═══════════════════════════════════════════════════════════════════════════════

if _SCVI_AVAILABLE:
    EXTERNAL_MODELS["scVI"] = {
        "factory": create_scvi_model,
        "params": {
            "latent_dim": 10,
            "n_layers": 2,
            "n_hidden": 128,
            "dropout_rate": 0.1,
        },
        "fit_params": {
            "epochs": 400,
            "lr": 1e-3,
            "patience": 45,
            "verbose_every": 50,
        },
        "notes": "Deep generative model with negative binomial likelihood (Lopez 2018)",
    }
    EXTERNAL_MODELS["PeakVI"] = {
        "factory": create_peakvi_model,
        "params": {
            "latent_dim": 10,
        },
        "fit_params": {
            "epochs": 400,
            "lr": 1e-3,
            "patience": 45,
            "verbose_every": 50,
        },
        "notes": "Variational inference for scATAC-seq peaks (Ashuach 2022)",
    }
    EXTERNAL_MODELS["PoissonVI"] = {
        "factory": create_poissonvi_model,
        "params": {
            "latent_dim": 10,
            "n_layers": 2,
            "n_hidden": 128,
            "dropout_rate": 0.1,
        },
        "fit_params": {
            "epochs": 400,
            "lr": 1e-3,
            "patience": 45,
            "verbose_every": 50,
        },
        "notes": "Poisson likelihood variant for scATAC-seq counts",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Model taxonomy for semantic figure splitting
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_TAXONOMY = {
    "CellBLAST": "classical",
    "GMVAE": "classical",
    "GMVAE-Poincare": "classical",
    "GMVAE-PGM": "classical",
    "GMVAE-LearnablePGM": "classical",
    "GMVAE-HW": "classical",
    "SCALEX": "classical",
    "siVAE": "classical",
    "scDiffusion": "classical",
    "VAE-DIP": "classical",
    "VAE-TC": "classical",
    "InfoVAE": "classical",
    "BetaVAE": "classical",
    "CLEAR": "classical",
    "scDAC": "deep",
    "scDeepCluster": "deep",
    "scDHMap": "deep",
    "scGNN": "deep",
    "scGCC": "deep",
    "scSMD": "deep",
    "scVI": "deep",
    "PeakVI": "deep",
    "PoissonVI": "deep",
}

CLASSICAL_BASELINES = sorted(k for k, v in MODEL_TAXONOMY.items() if v == "classical")
DEEP_GRAPH_BASELINES = sorted(k for k, v in MODEL_TAXONOMY.items() if v == "deep")


# Model groups for figure panels (LAIOR-style categories)
MODEL_GROUPS = {
    "generative": [
        "CellBLAST", "SCALEX", "scDiffusion", "siVAE",
        "scDAC", "scDeepCluster", "scDHMap", "scSMD",
    ],
    "gaussian_geometric": [
        "GMVAE", "GMVAE-Poincare", "GMVAE-PGM", "GMVAE-LearnablePGM", "GMVAE-HW",
    ],
    "disentanglement": ["VAE-DIP", "VAE-TC", "InfoVAE", "BetaVAE"],
    "graph_contrastive": ["CLEAR", "scGCC", "scGNN"],
    "scvi_family": ["scVI", "PeakVI", "PoissonVI"],
}


def list_external_models():
    """Print summary of all external models."""
    print(f"{'Model':<22} {'Latent':<8} {'Notes'}")
    print("-" * 72)
    for name, cfg in EXTERNAL_MODELS.items():
        ld = cfg["params"].get("latent_dim",
                               cfg["params"].get("embedding_dim", "?"))
        print(f"{name:<22} {str(ld):<8} {cfg['notes'][:45]}")


def list_model_groups():
    """Print model groups for figure panels."""
    print("Model groups (for figure panels):")
    for group, models in MODEL_GROUPS.items():
        available = [m for m in models if m in EXTERNAL_MODELS]
        if available:
            print(f"  {group}: {available}")


if __name__ == "__main__":
    list_external_models()
    print()
    list_model_groups()
