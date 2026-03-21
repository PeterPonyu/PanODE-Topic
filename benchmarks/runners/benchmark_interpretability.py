#!/usr/bin/env python
"""
Quantitative Interpretability Benchmark (E3)

Computes topic coherence (NPMI) and gene set enrichment breadth for
prior-based models across multiple datasets, providing the quantitative
interpretability metric requested by the reviewer.

Metrics:
1. Topic Coherence (NPMI): Measures co-occurrence of top genes within each
   latent component. Higher = more coherent/interpretable topics.
2. GO Enrichment Breadth: Fraction of latent components that produce
   significant GO terms (adj p < 0.05). Higher = biologically meaningful.
3. Gene Specificity: How specific each gene is to a single topic
   (entropy-based). Lower entropy = more specific/interpretable.

Usage:
    python benchmarks/benchmark_interpretability.py --datasets setty lung endo
    python benchmarks/benchmark_interpretability.py --all
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
from scipy.sparse import issparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.config import BASE_CONFIG, ensure_dirs, set_global_seed, CACHE_DIR, result_subdir
from benchmarks.dataset_registry import DATASET_REGISTRY
from benchmarks.data_utils import load_or_preprocess_adata
from utils.data import DataSplitter
from benchmarks.model_registry import MODELS
from benchmarks.biological_validation.perturbation_analysis import (
    compute_perturbation_importance,
    get_top_genes_per_component,
    run_enrichment)


# ═════════════════════════════════════════════════════════════════════════════
# Topic Coherence (Correlation-based, suitable for sparse scRNA-seq)
# ═════════════════════════════════════════════════════════════════════════════

def compute_topic_coherence(adata, top_genes_per_topic, top_n=20):
    """Compute correlation-based topic coherence for sparse scRNA-seq data.
    
    For each topic, compute the mean pairwise Pearson correlation among
    the top-N genes' expression vectors. Higher coherence = genes in
    each component vary together across cells = biologically coherent.
    
    Also computes NPMI on the above-median binarized expression matrix.
    
    Returns per-topic scores and aggregates.
    """
    X = adata.X
    if issparse(X):
        X = X.toarray()
    
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    
    topic_coherences = []
    topic_npmis = []
    n_cells = X.shape[0]
    
    # For NPMI: binarize at 75th percentile (more selective than median for sparse data)
    gene_q75 = np.percentile(X, 75, axis=0)
    X_bin = (X > gene_q75).astype(float)
    
    for topic_id, gene_list in top_genes_per_topic.items():
        # gene_list may be [(gene_name, score), ...] tuples or plain strings
        raw = gene_list[:top_n]
        genes = [g[0] if isinstance(g, (list, tuple)) else g for g in raw]
        genes = [g for g in genes if g in gene_to_idx]
        if len(genes) < 2:
            topic_coherences.append(0.0)
            topic_npmis.append(0.0)
            continue
        
        idxs = [gene_to_idx[g] for g in genes]
        
        # Correlation-based coherence (on continuous expression)
        gene_expr = X[:, idxs]  # [cells, n_genes]
        # Pearson correlation via corrcoef
        # Each column is a gene vector; corrcoef expects rows=variables
        corr_matrix = np.corrcoef(gene_expr.T)  # [n_genes, n_genes]
        # Handle NaN (constant genes)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # Extract upper triangle (exclude diagonal)
        n_g = len(idxs)
        mask = np.triu(np.ones((n_g, n_g), dtype=bool), k=1)
        pairwise_corrs = corr_matrix[mask]
        topic_coherences.append(float(np.mean(pairwise_corrs)))
        
        # NPMI on binarized data
        gene_vectors = X_bin[:, idxs]
        npmi_pairs = []
        eps = 1e-12
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                p_i = gene_vectors[:, i].sum() / n_cells
                p_j = gene_vectors[:, j].sum() / n_cells
                p_ij = ((gene_vectors[:, i] * gene_vectors[:, j]).sum()) / n_cells
                
                if p_ij < eps or p_i < eps or p_j < eps:
                    npmi_pairs.append(0.0)
                    continue
                
                pmi = np.log(p_ij / (p_i * p_j + eps))
                npmi = pmi / (-np.log(p_ij + eps))
                npmi_pairs.append(npmi)
        
        topic_npmis.append(np.mean(npmi_pairs) if npmi_pairs else 0.0)
    
    return {
        "per_topic_coherence": topic_coherences,
        "mean_coherence": np.mean(topic_coherences) if topic_coherences else 0.0,
        "std_coherence": np.std(topic_coherences) if topic_coherences else 0.0,
        "per_topic_npmi": topic_npmis,
        "mean_npmi": np.mean(topic_npmis) if topic_npmis else 0.0,
        "std_npmi": np.std(topic_npmis) if topic_npmis else 0.0,
    }


def compute_gene_specificity(importance_matrix):
    """Compute entropy-based gene specificity.
    
    For each gene, compute the entropy of its importance distribution
    across topics. Lower entropy = gene is specific to fewer topics.
    
    Returns mean specificity (1 - normalized entropy).
    """
    # importance: [K, G] matrix
    K, G = importance_matrix.shape
    
    # Normalize each gene's importance to a probability distribution
    abs_imp = np.abs(importance_matrix)
    col_sums = abs_imp.sum(axis=0, keepdims=True) + 1e-12
    probs = abs_imp / col_sums
    
    # Compute entropy per gene
    entropies = -np.sum(probs * np.log(probs + 1e-12), axis=0)
    max_entropy = np.log(K)
    
    # Specificity = 1 - normalized_entropy
    specificity = 1.0 - (entropies / max_entropy) if max_entropy > 0 else np.zeros(G)
    
    return {
        "mean_specificity": float(np.mean(specificity)),
        "std_specificity": float(np.std(specificity)),
    }


def compute_enrichment_breadth(model, data_loader, gene_names, device,
                                organism="human", top_n_genes=50):
    """Fraction of latent components with significant GO enrichment."""
    importance, _ = compute_perturbation_importance(model, data_loader, device)
    top_genes = get_top_genes_per_component(importance, gene_names, top_n=top_n_genes)
    
    n_components = len(top_genes)
    n_significant = 0
    
    for comp_id, gene_list in top_genes.items():
        try:
            enr = run_enrichment(gene_list, organism=organism)
            if enr is not None and len(enr) > 0:
                sig = enr[enr["Adjusted P-value"] < 0.05]
                if len(sig) > 0:
                    n_significant += 1
        except Exception:
            pass
    
    breadth = n_significant / max(n_components, 1)
    
    return {
        "enrichment_breadth": breadth,
        "n_significant": n_significant,
        "n_components": n_components,
        "importance": importance,
        "top_genes": top_genes,
    }


def standardize_labels(adata, label_key):
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()
    return adata


def get_organism(ds_name):
    """Get organism for enrichment based on species."""
    ds_info = DATASET_REGISTRY[ds_name]
    species = ds_info.get("species", "human")
    if "mouse" in species:
        return "mouse"
    return "human"


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative interpretability benchmark (E3)")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["setty", "lung", "endo"],
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--all", action="store_true",
                        help="Run on all 12 datasets")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["DPMM-Base", "Topic-Base",
                                 "DPMM-Contrastive", "Topic-Contrastive", "Pure-AE"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n-genes", type=int, default=30,
                        help="Top N genes per component for coherence")
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip GO enrichment (faster)")
    args = parser.parse_args()

    if args.all:
        args.datasets = list(DATASET_REGISTRY.keys())

    set_global_seed(args.seed)
    device = torch.device(BASE_CONFIG.device)

    out_dirs = {
        "csv": result_subdir("interpretability", "csv"),
        "cache": CACHE_DIR,
    }
    ensure_dirs(*out_dirs.values())

    all_results = []

    for ds_name in args.datasets:
        ds_info = DATASET_REGISTRY[ds_name]
        organism = get_organism(ds_name)
        print(f"\n{'#'*60}")
        print(f"# DATASET: {ds_name} (organism={organism})")
        print(f"{'#'*60}")

        adata = load_or_preprocess_adata(
            ds_info["path"],
            max_cells=BASE_CONFIG.max_cells,
            hvg_top_genes=BASE_CONFIG.hvg_top_genes,
            seed=args.seed,
            cache_dir=str(out_dirs["cache"]),
            use_cache=True)
        adata = standardize_labels(adata, ds_info["label_key"])

        splitter = DataSplitter(
            adata, train_size=0.7, val_size=0.15, test_size=0.15,
            batch_size=BASE_CONFIG.batch_size, random_seed=args.seed)

        gene_names = list(adata.var_names)
        input_dim = splitter.n_var

        for model_name in args.models:
            print(f"\n--- {model_name} on {ds_name} ---")

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

                # Train
                model.fit(
                    train_loader=splitter.train_loader,
                    val_loader=splitter.val_loader,
                    epochs=fit_epochs, lr=fit_lr,
                    weight_decay=fit_wd,
                    device=str(device),
                    verbose_every=200, patience=9999)

                # Compute perturbation importance
                importance, _ = compute_perturbation_importance(
                    model, splitter.test_loader, device
                )
                top_genes = get_top_genes_per_component(
                    importance, gene_names, top_n=args.top_n_genes
                )

                # 1. Topic Coherence (correlation + NPMI)
                coh = compute_topic_coherence(adata, top_genes, top_n=args.top_n_genes)

                # 2. Gene Specificity
                spec = compute_gene_specificity(importance)

                res = {
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Seed": args.seed,
                    "Mean_Coherence": coh["mean_coherence"],
                    "Std_Coherence": coh["std_coherence"],
                    "Mean_NPMI": coh["mean_npmi"],
                    "Std_NPMI": coh["std_npmi"],
                    "Gene_Specificity": spec["mean_specificity"],
                }

                # 3. GO Enrichment Breadth (optional)
                if not args.skip_enrichment:
                    try:
                        enr_info = compute_enrichment_breadth(
                            model, splitter.test_loader, gene_names, device,
                            organism=organism, top_n_genes=args.top_n_genes)
                        res["Enrichment_Breadth"] = enr_info["enrichment_breadth"]
                        res["Sig_Components"] = enr_info["n_significant"]
                        res["Total_Components"] = enr_info["n_components"]
                    except Exception as e:
                        print(f"  Enrichment failed: {e}")
                        res["Enrichment_Breadth"] = None

                all_results.append(res)
                print(f"  Coherence={res['Mean_Coherence']:.4f}, NPMI={res['Mean_NPMI']:.4f}, "
                      f"Specificity={res['Gene_Specificity']:.4f}")
                if "Enrichment_Breadth" in res and res["Enrichment_Breadth"] is not None:
                    print(f"  Enrichment Breadth={res['Enrichment_Breadth']:.2f} "
                          f"({res['Sig_Components']}/{res['Total_Components']})")

            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({
                    "Dataset": ds_name, "Model": model_name,
                    "Error": str(e),
                })

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = out_dirs["csv"] / f"interpretability_seed{args.seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"INTERPRETABILITY SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    # Print comparison table
    if "Mean_Coherence" in df.columns:
        pivot = df.pivot_table(
            index="Dataset",
            columns="Model",
            values="Mean_Coherence",
            aggfunc="mean")
        print(f"\n--- Mean Coherence by Model × Dataset ---")
        print(pivot.to_string())
        
        pivot_spec = df.pivot_table(
            index="Dataset",
            columns="Model",
            values="Gene_Specificity",
            aggfunc="mean")
        print(f"\n--- Gene Specificity by Model × Dataset ---")
        print(pivot_spec.to_string())


if __name__ == "__main__":
    main()
