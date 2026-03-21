#!/usr/bin/env python
"""
Cross-Dataset Transfer Benchmark (C5)

Train on source dataset, evaluate latent quality on target dataset.
Tests whether learned representations generalize across datasets.

Strategy:
1. Load raw datasets, find common genes
2. Subset to common genes, normalize & HVG on intersection
3. Train model on source, encode target, compute metrics on target latents

Usage:
    python benchmarks/benchmark_transfer.py --source setty --target lung
    python benchmarks/benchmark_transfer.py --source endo --target dentate
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import argparse
import gc
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.config import BASE_CONFIG, ensure_dirs, set_global_seed, result_subdir
from benchmarks.dataset_registry import DATASET_REGISTRY
from utils.data import DataSplitter
from benchmarks.model_registry import MODELS
from benchmarks.metrics_utils import compute_metrics


def standardize_labels(adata, label_key):
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()
    elif "cell_type" not in adata.obs.columns:
        print(f"  WARNING: '{label_key}' not found in obs.")
    return adata


def load_normalize_subset(path, max_cells, common_genes, seed):
    """Load dataset, subsample, subset to common genes, normalize, HVG."""
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    if adata.shape[0] > max_cells:
        np.random.seed(seed)
        idx = np.random.choice(adata.shape[0], max_cells, replace=False)
        adata = adata[idx].copy()
    # Subset to common genes first
    available = [g for g in common_genes if g in adata.var_names]
    adata = adata[:, available].copy()
    # Store raw counts in layer
    if hasattr(adata.X, 'toarray'):
        adata.layers['counts'] = adata.X.toarray().copy()
    else:
        adata.layers['counts'] = adata.X.copy()
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # HVG (use seurat for log-normalized data)
    n_hvg = min(2000, len(available))
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat", subset=False)
        adata = adata[:, adata.var.highly_variable].copy()
    except Exception:
        # Fallback: use cell_ranger flavor
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=False)
            adata = adata[:, adata.var.highly_variable].copy()
        except Exception:
            pass
    return adata


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset transfer benchmark")
    parser.add_argument("--source", type=str, default="setty",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--target", type=str, default="lung",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--models", type=str, nargs="+",
                        default=["DPMM-Base", "Topic-Base", "Pure-AE"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(BASE_CONFIG.device)

    out_dirs = {
        "csv": result_subdir("transfer", "csv"),
    }
    ensure_dirs(*out_dirs.values())
    results = []

    source_info = DATASET_REGISTRY[args.source]
    target_info = DATASET_REGISTRY[args.target]

    print(f"\n{'='*60}")
    print(f"Cross-Dataset Transfer: {args.source} -> {args.target}")
    print(f"{'='*60}")

    # Find common genes from raw data
    print(f"\nFinding common genes...")
    adata_s_raw = sc.read_h5ad(source_info["path"])
    adata_s_raw.var_names_make_unique()
    adata_t_raw = sc.read_h5ad(target_info["path"])
    adata_t_raw.var_names_make_unique()
    common_genes = sorted(set(adata_s_raw.var_names) & set(adata_t_raw.var_names))
    print(f"  Source genes: {adata_s_raw.shape[1]}, Target genes: {adata_t_raw.shape[1]}")
    print(f"  Common genes: {len(common_genes)}")
    del adata_s_raw, adata_t_raw

    if len(common_genes) < 500:
        print("Too few common genes (<500). Skipping.")
        return

    max_cells = BASE_CONFIG.max_cells

    # Load and preprocess both datasets
    print(f"\nLoading source: {args.source}")
    adata_source = load_normalize_subset(source_info["path"], max_cells, common_genes, args.seed)
    adata_source = standardize_labels(adata_source, source_info["label_key"])
    print(f"  Source shape after HVG: {adata_source.shape}")

    print(f"Loading target: {args.target}")
    adata_target = load_normalize_subset(target_info["path"], max_cells, common_genes, args.seed)
    adata_target = standardize_labels(adata_target, target_info["label_key"])
    print(f"  Target shape after HVG: {adata_target.shape}")

    # Align to final common HVGs
    final_genes = sorted(set(adata_source.var_names) & set(adata_target.var_names))
    adata_source = adata_source[:, final_genes].copy()
    adata_target = adata_target[:, final_genes].copy()
    print(f"  Final aligned genes: {len(final_genes)}")

    if len(final_genes) < 100:
        print("Too few aligned HVGs (<100). Skipping.")
        return

    # Create DataSplitter for source (standard train/val/test)
    source_splitter = DataSplitter(
        adata_source, train_size=0.8, val_size=0.1, test_size=0.1,
        batch_size=BASE_CONFIG.batch_size, random_seed=args.seed)

    # For target: create a simple loader with all data for evaluation
    from scipy.sparse import issparse
    X_target = adata_target.X
    if issparse(X_target):
        X_target = X_target.toarray()
    X_target_t = torch.FloatTensor(np.array(X_target))
    target_dataset = TensorDataset(X_target_t, X_target_t)
    target_loader = DataLoader(target_dataset, batch_size=128, shuffle=False)

    # Get target labels
    from sklearn.preprocessing import LabelEncoder
    if "cell_type" in adata_target.obs.columns:
        le = LabelEncoder()
        target_labels = le.fit_transform(adata_target.obs["cell_type"].values)
    else:
        target_labels = np.zeros(adata_target.shape[0], dtype=int)

    input_dim = len(final_genes)

    for model_name in args.models:
        print(f"\n{'_'*50}")
        print(f"Model: {model_name}")
        print(f"{'_'*50}")

        gc.collect()
        torch.cuda.empty_cache()

        model_info = MODELS[model_name]
        model_cls = model_info["class"]
        params = dict(model_info["params"])

        fit_lr = params.pop("fit_lr", BASE_CONFIG.lr)
        fit_epochs = params.pop("fit_epochs", BASE_CONFIG.epochs)
        fit_wd = params.pop("fit_weight_decay", 1e-5)

        try:
            model = model_cls(input_dim=input_dim, **params).to(device)

            print(f"  Training on {args.source} ({fit_epochs} epochs)...")
            start = time.time()
            model.fit(
                train_loader=source_splitter.train_loader,
                val_loader=source_splitter.val_loader,
                epochs=fit_epochs,
                lr=fit_lr,
                weight_decay=fit_wd,
                device=str(device),
                verbose_every=100,
                patience=9999)
            train_time = time.time() - start

            print(f"  Encoding {args.target}...")
            model.eval()
            latents = []
            with torch.no_grad():
                for batch in target_loader:
                    x = batch[0].to(device)
                    z = model.encode(x)
                    latents.append(z.cpu().numpy())

            latents = np.concatenate(latents, axis=0)

            metrics = compute_metrics(
                latents, target_labels,
                data_type=target_info["data_type"])

            res = {
                "Source": args.source,
                "Target": args.target,
                "Model": model_name,
                "Seed": args.seed,
                "Aligned_Genes": len(final_genes),
                "Train_Time_s": round(train_time, 1),
                "Target_NMI": metrics.get("NMI"),
                "Target_ARI": metrics.get("ARI"),
                "Target_ASW": metrics.get("ASW"),
                "Target_DAV": metrics.get("DAV"),
            }
            results.append(res)
            print(f"  NMI={res['Target_NMI']:.4f}, ARI={res['Target_ARI']:.4f}, ASW={res['Target_ASW']:.4f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "Source": args.source, "Target": args.target,
                "Model": model_name, "Seed": args.seed,
                "Error": str(e),
            })

    df = pd.DataFrame(results)
    csv_path = out_dirs["csv"] / f"transfer_{args.source}_to_{args.target}_seed{args.seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
