#!/usr/bin/env python
"""
Joint Multi-Dataset Training Benchmark

Trains all 6 core model variants (3 DPMM + 3 Topic) on a single joint
dataset formed by concatenating all 12 development datasets.

Motivation
----------
The cross-dataset UMAP in Figure 2 Panel B currently concatenates the
latent spaces of models trained *separately* on each dataset.  That
approach measures per-dataset quality but does not test whether a model
can learn a *shared* latent space across datasets simultaneously.

This script provides the comparison: one model instance, all 12 datasets,
one joint latent space.  The resulting UMAP (colored by dataset) reveals
whether the model has learned biologically meaningful structure that
generalises across tissues and studies, without any post-hoc alignment.

Gene Intersection Strategy
---------------------------
Each dataset undergoes independent HVG selection (top 3000 genes).
The *intersection* of all HVG sets is then used as the joint gene space.
This ensures every cell is represented by the same gene features, while
keeping only genes informative in at least one dataset.

If the intersection is too small (< 300 genes), a fallback strategy
uses the top-N genes by frequency of appearance across datasets until
at least 300 genes are obtained.

Outputs (under benchmark_results/joint/)
-----------------------------------------
  latents/   — per-model NPZ with ``latent`` (N×D) and ``dataset_labels``
  csv/        — per-model performance rows (one row per model)
  meta/       — JSON with gene list, dataset cell counts, timestamp

Usage
-----
  python benchmarks/benchmark_joint.py
  python benchmarks/benchmark_joint.py --models DPMM-Base Topic-Base
  python benchmarks/benchmark_joint.py --epochs 200 --max-cells 1500
"""

import sys
import os
import json
import gc
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata

warnings.filterwarnings("ignore")

from benchmarks.config import (
    BASE_CONFIG, set_global_seed, DEFAULT_OUTPUT_DIR, CACHE_DIR, result_subdir,
    ensure_dirs)
from benchmarks.dataset_registry import DATASET_REGISTRY
from benchmarks.model_registry import MODELS
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.train_utils import train_and_evaluate

# ──────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ──────────────────────────────────────────────────────────────────────────────
JOINT_MODELS = [
    "DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
    "Topic-Base", "Topic-Transformer", "Topic-Contrastive",
]
HVG_PER_DS   = 3000   # HVGs to select per dataset before intersection
MIN_SHARED   = 300    # minimum acceptable shared genes (fallback if < this)
MAX_CELLS    = 2000   # cells per dataset (downsampled for tractability)
EPOCHS       = 300    # training epochs for joint model
LR           = BASE_CONFIG.lr
BATCH_SIZE   = BASE_CONFIG.batch_size
DEVICE       = BASE_CONFIG.device
SEED         = BASE_CONFIG.seed

RESULTS_DIR  = result_subdir("joint")

# ──────────────────────────────────────────────────────────────────────────────
# Gene-intersection helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_shared_genes(adatas, n_hvg=HVG_PER_DS, min_shared=MIN_SHARED):
    """Find a shared gene set across datasets using a common-then-variable strategy.

    Strategy
    --------
    1. Compute the intersection of ALL gene names across datasets (common genes).
    2. For each dataset, subset to common genes and rank by variability (dispersion).
    3. Score each common gene by how often it appears in per-dataset top-N HVG lists.
    4. Select the top genes by frequency score, breaking ties by mean dispersion.

    This avoids the failure mode where per-dataset HVG filtering produces
    tissue-specific gene sets with zero overlap.

    Parameters
    ----------
    adatas : list[AnnData]
        Pre-processed AnnData objects (one per dataset, **not** HVG-filtered).
    n_hvg : int
        Number of HVGs to select per dataset for scoring (default 3000).
    min_shared : int
        Minimum number of shared genes required.

    Returns
    -------
    list[str]
        Ordered list of shared gene names.
    """
    n_ds = len(adatas)
    print(f"\n  Finding shared gene set across {n_ds} datasets …")

    # Step 1: common gene names (present in ALL datasets)
    common = set(adatas[0].var_names)
    for ad in adatas[1:]:
        common &= set(ad.var_names)
    print(f"  Common genes across all datasets: {len(common)}")

    if len(common) < min_shared:
        # Relax: genes in at least 80% of datasets
        threshold = max(2, int(n_ds * 0.8))
        counter = Counter()
        for ad in adatas:
            counter.update(ad.var_names)
        common = {g for g, c in counter.items() if c >= threshold}
        # Still need each gene in every dataset we'll use
        for ad in adatas:
            common &= set(ad.var_names)
        print(f"  Relaxed to genes in ≥{threshold}/{n_ds} datasets: {len(common)}")

    if len(common) < min_shared:
        raise RuntimeError(
            f"Could not find {min_shared} common genes across datasets "
            f"(only {len(common)} found). Datasets may mix species."
        )

    common_list = sorted(common)  # deterministic order
    print(f"  Using {len(common_list)} common genes for HVG scoring …")

    # Step 2: Per-dataset HVG ranking on common genes
    freq = Counter()           # how many datasets rank this gene as HVG
    disp_accum = {}            # sum of normalized_dispersion per gene

    for i, ad in enumerate(adatas):
        ad_sub = ad[:, common_list].copy()
        try:
            sc.pp.highly_variable_genes(ad_sub, n_top_genes=min(n_hvg, len(common_list)))
            hvg_mask = ad_sub.var["highly_variable"]
            hvgs_i = set(ad_sub.var_names[hvg_mask])
        except Exception:
            # If HVG calculation fails (e.g. too few genes), treat all as HVG
            hvgs_i = set(common_list)

        freq.update(hvgs_i)

        # accumulate dispersions for tiebreaking
        if "dispersions_norm" in ad_sub.var.columns:
            for g in common_list:
                disp_accum[g] = disp_accum.get(g, 0.0) + float(
                    ad_sub.var.loc[g, "dispersions_norm"] if g in ad_sub.var_names else 0.0
                )

        print(f"    Dataset {i+1}: {len(hvgs_i)} HVGs from {ad_sub.n_vars} common genes")

    # Step 3: rank genes by (frequency, mean dispersion) and pick top N
    target_n = min(n_hvg, len(common_list))  # aim for n_hvg shared genes
    ranked = sorted(
        common_list,
        key=lambda g: (freq.get(g, 0), disp_accum.get(g, 0.0)),
        reverse=True)
    selected = ranked[:target_n]
    print(f"\n  Selected {len(selected)} shared genes")
    print(f"    Top-frequency gene count: {freq[ranked[0]]} / {n_ds} datasets")
    print(f"    Min-frequency gene count: {freq[selected[-1]]} / {n_ds} datasets")

    if len(selected) < min_shared:
        raise RuntimeError(
            f"Could not select {min_shared} shared genes "
            f"(only {len(selected)} found). Check dataset quality."
        )
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Joint dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_joint_adata(shared_genes, adatas, ds_keys, max_cells=MAX_CELLS, seed=SEED):
    """Concatenate all datasets (subsampled) onto the shared gene set.

    Parameters
    ----------
    shared_genes : list[str]
        Ordered list of genes to keep.
    adatas : list[AnnData]
        Pre-processed AnnData objects.
    ds_keys : list[str]
        Dataset names (same length as *adatas*).
    max_cells : int
        Maximum cells per dataset; excess cells are randomly downsampled.

    Returns
    -------
    joint_adata : AnnData
        Concatenated AnnData with ``obs["dataset"]`` column.
    cell_counts : dict[str, int]
        Number of cells from each dataset in the joint object.
    """
    rng = np.random.RandomState(seed)
    blocks = []
    cell_counts = {}

    for ds_key, ad in zip(ds_keys, adatas):
        # Subset to shared genes
        ad_sub = ad[:, shared_genes].copy()

        # Downsample
        n = ad_sub.n_obs
        if n > max_cells:
            idx = rng.choice(n, max_cells, replace=False)
            ad_sub = ad_sub[idx].copy()

        ad_sub.obs["dataset"] = ds_key
        cell_counts[ds_key] = ad_sub.n_obs
        blocks.append(ad_sub)
        print(f"    {ds_key}: {ad_sub.n_obs} cells × {ad_sub.n_vars} genes")

    joint = anndata.concat(blocks, join="inner")
    joint.obs_names_make_unique()
    print(f"\n  Joint dataset: {joint.n_obs} cells × {joint.n_vars} genes")
    return joint, cell_counts


# ──────────────────────────────────────────────────────────────────────────────
# DataSplitter wrapper for joint data
# ──────────────────────────────────────────────────────────────────────────────

def _make_joint_splitter(joint_adata, latent_dim, batch_size=BATCH_SIZE, seed=SEED):
    """Build a DataSplitter-equivalent from the joint AnnData.

    We re-use the existing DataSplitter class directly — it accepts any
    AnnData whose X is already normalised.
    """
    from utils.data import DataSplitter

    # DataSplitter expects adata.obs["cell_type"] for label extraction
    if "cell_type" not in joint_adata.obs.columns:
        # Encode dataset name as pseudo-label
        ds_categories = joint_adata.obs["dataset"].astype("category")
        joint_adata.obs["cell_type"] = ds_categories.cat.codes.values

    splitter = DataSplitter(
        adata=joint_adata,
        latent_dim=latent_dim,
        batch_size=batch_size,
        random_seed=seed,
        verbose=True)
    return splitter


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_joint_model(model_name, joint_adata, device, epochs, seed):
    """Train one model on the joint AnnData and return full latent + results."""
    if model_name not in MODELS:
        print(f"  ⚠ {model_name} not in MODELS registry — skipping")
        return None, None, None

    model_info = MODELS[model_name]
    params = model_info["params"].copy()

    latent_dim = params.get("latent_dim") or params.get("n_topics") or BASE_CONFIG.latent_dim

    splitter = _make_joint_splitter(joint_adata, latent_dim=latent_dim, seed=seed)

    print(f"\n  Training {model_name} on joint dataset …")
    result = train_and_evaluate(
        name=model_name,
        model_cls=model_info["class"],
        params=params,
        splitter=splitter,
        device=device,
        lr=LR,
        verbose_every=50,
        data_type="cluster",           # mixed types → use cluster eval
        extra_fields={"Series": model_info.get("series", ""), "Dataset": "joint"},
        epochs=epochs)

    # Re-extract latent from ALL cells (train+val+test) for the UMAP
    model_obj = result.get("model_obj")
    if model_obj is not None:
        all_loader = splitter.get_all_loader()
        full_latent_dict = model_obj.extract_latent(
            all_loader, device=str(device)
        )
        full_latent = full_latent_dict.get("latent")
    else:
        full_latent = result.get("latent")

    test_latent = result.get("latent")  # test-split only (for metrics)
    return full_latent, test_latent, result


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    set_global_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output dirs
    lat_dir = RESULTS_DIR / "latents"
    csv_dir = RESULTS_DIR / "csv"
    meta_dir = RESULTS_DIR / "meta"
    for d in (lat_dir, csv_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  Joint Multi-Dataset Training Benchmark")
    print(f"  Device: {device} | Epochs: {args.epochs} | MaxCells/DS: {args.max_cells}")
    print(f"{'='*70}")

    # ── Step 1: Load and preprocess all datasets ──────────────────────────
    # Filter by species (or 'all' with gene-name harmonization)
    species = args.species
    if species == "all":
        # Include human + mouse (gene-symbol) datasets; harmonize names
        ds_keys = [k for k, v in DATASET_REGISTRY.items()
                   if v.get("species") in ("human", "mouse")]
    else:
        ds_keys = [k for k, v in DATASET_REGISTRY.items()
                   if v.get("species") == species]
    if not ds_keys:
        raise RuntimeError(f"No datasets found for species='{species}'")
    print(f"\n  Species filter: '{species}' → {len(ds_keys)} datasets")

    adatas_raw = []
    skipped = []
    print("  Loading datasets …")
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    for ds_key in ds_keys:
        info = DATASET_REGISTRY[ds_key]
        p = Path(info["path"])
        if not p.exists():
            print(f"  ⚠ {ds_key}: file not found ({p}) — skipped")
            skipped.append(ds_key)
            continue
        try:
            ad = load_or_preprocess_adata(
                str(p),
                max_cells=args.max_cells * 3,   # load more cells; subsample later
                hvg_top_genes=None,              # keep ALL genes; shared-gene logic selects later
                seed=args.seed,
                cache_dir=str(cache_dir))
            # Harmonize mouse gene names to uppercase (Sox17 → SOX17)
            if species == "all" and info.get("species") == "mouse":
                ad.var_names = pd.Index([g.upper() for g in ad.var_names])
                ad.var_names_make_unique()
                print(f"    ↑ {ds_key}: uppercased gene names (mouse→human orthologs)")
            adatas_raw.append((ds_key, ad))
            print(f"  ✓ {ds_key}: {ad.n_obs} cells × {ad.n_vars} genes")
        except Exception as e:
            print(f"  ✗ {ds_key}: failed to load ({e}) — skipped")
            skipped.append(ds_key)

    if len(adatas_raw) < 3:
        raise RuntimeError(f"Too few datasets loaded ({len(adatas_raw)}). Aborting.")

    valid_keys = [k for k, _ in adatas_raw]
    adatas = [ad for _, ad in adatas_raw]

    # ── Step 2: Find shared gene set ──────────────────────────────────────
    shared_genes = find_shared_genes(adatas, min_shared=args.min_genes)

    # ── Step 3: Build joint AnnData ───────────────────────────────────────
    print("\n  Building joint dataset …")
    joint_adata, cell_counts = build_joint_adata(
        shared_genes, adatas, valid_keys,
        max_cells=args.max_cells, seed=args.seed)

    # Free per-dataset adatas to save memory
    del adatas, adatas_raw
    gc.collect()

    # ── Step 4: Train each model ──────────────────────────────────────────
    selected_models = args.models if args.models else JOINT_MODELS
    result_rows = []

    for model_name in selected_models:
        if model_name not in MODELS:
            print(f"  ⚠ {model_name} not found in registry")
            continue

        try:
            full_latent, test_latent, result = train_joint_model(
                model_name, joint_adata, device, args.epochs, args.seed
            )
        except Exception as e:
            print(f"  ✗ {model_name} training failed: {e}")
            import traceback; traceback.print_exc()
            continue

        if full_latent is None:
            print(f"  ⚠ {model_name}: no latent extracted")
            continue

        # Save latent NPZ with dataset labels -- FULL dataset (all cells)
        ds_labels = np.array(joint_adata.obs["dataset"].values, dtype=str)
        fname = f"{model_name.replace('/', '_')}_joint_{timestamp}.npz"
        np.savez(lat_dir / fname,
                 latent=full_latent,
                 dataset_labels=ds_labels,
                 shared_genes=np.array(shared_genes))
        print(f"  → Saved latent: {fname} ({full_latent.shape})")
        print(f"    {len(ds_labels)} cells, {len(set(ds_labels))} datasets")

        if result:
            row = {k: v for k, v in result.items()
                   if k not in ("latent")}
            row["Model"] = model_name
            row["Dataset"] = "joint"
            result_rows.append(row)

        gc.collect()

    # ── Step 5: Save results ──────────────────────────────────────────────
    if result_rows:
        df = pd.DataFrame(result_rows)
        csv_path = csv_dir / f"results_joint_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Results CSV: {csv_path}")

    meta = {
        "timestamp": timestamp,
        "n_shared_genes": len(shared_genes),
        "shared_genes_sample": shared_genes[:20],
        "datasets": valid_keys,
        "skipped_datasets": skipped,
        "cell_counts": cell_counts,
        "epochs": args.epochs,
        "max_cells_per_ds": args.max_cells,
        "models_trained": selected_models,
    }
    meta_path = meta_dir / f"meta_joint_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta JSON:   {meta_path}")

    print(f"\n{'='*70}")
    print(f"  Joint training complete. Trained on {len(valid_keys)} datasets,")
    print(f"  {len(shared_genes)} shared genes.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models jointly on all 12 datasets and save latents"
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to train (default: all 6 core models)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--max-cells", type=int, default=MAX_CELLS,
                        help=f"Max cells per dataset (default: {MAX_CELLS})")
    parser.add_argument("--min-genes", type=int, default=MIN_SHARED,
                        help=f"Min shared genes (default: {MIN_SHARED})")
    parser.add_argument("--species", default="all",
                        choices=["human", "mouse", "mouse_ensembl", "all"],
                        help="Species filter: human, mouse, all (default: all = human+mouse with name harmonization)")
    parser.add_argument("--device", default=DEVICE,
                        help=f"PyTorch device (default: {DEVICE})")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args)
