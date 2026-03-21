"""Biological validation tools for PanODE latent representations.

Modules
-------
latent_component_umap
    Project each latent component’s strength onto UMAP (requires GPU).
perturbation_analysis
    Perturb latent dims → identify responsive genes → GO enrichment.
compose_figures
    Compose multi-panel bio validation figures from pre-computed data.

Shared helpers (``load_model``, ``load_data_with_genes``) are defined here
and imported by the sub-modules to avoid duplication.
"""

import torch
import numpy as np


def load_model(model_path, device="cpu"):
    """Load a model checkpoint saved by training_dynamics.py / benchmark_base.

    Re-used by both ``latent_component_umap`` and ``perturbation_analysis``.
    """
    from benchmarks.model_registry import MODELS

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model_name = config["model_name"]
    input_dim = config["input_dim"]
    params = dict(config["params"])

    # Remove fit-specific params not accepted by model __init__
    params.pop("fit_lr", None)
    params.pop("fit_weight_decay", None)
    params.pop("fit_epochs", None)

    model_info = MODELS[model_name]
    model = model_info["class"](input_dim=input_dim, **params)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


def load_data_with_genes(dataset, seed=42, max_cells=3000, hvg=3000):
    """Load dataset and return (DataSplitter, gene_names).

    Re-used by ``perturbation_analysis`` (and potentially others).
    """
    from benchmarks.data_utils import load_data
    from benchmarks.dataset_registry import DATASET_REGISTRY
    from utils.data import DataSplitter

    ds_info = DATASET_REGISTRY[dataset]
    data_path = ds_info["path"]
    adata = load_data(data_path, max_cells=max_cells, hvg_top_genes=hvg, seed=seed)
    gene_names = list(adata.var_names)

    splitter = DataSplitter(
        adata=adata, batch_size=128, random_seed=seed, verbose=False)
    return splitter, gene_names
