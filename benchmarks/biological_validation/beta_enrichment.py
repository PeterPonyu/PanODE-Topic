"""Extract top genes from Topic decoder beta and run GOBP enrichment.

For each Topic model × dataset, extracts the top N genes per topic
from the decoder beta matrix (topic-word distribution), then runs
GO Biological Process enrichment via gseapy Enrichr API.

Outputs per-component enrichment CSVs:
    {model}_{dataset}_beta_enrich_comp{k}.csv

These can be consumed by gen_fig9_subplots.py to add beta-based
enrichment panels alongside perturbation-based enrichment.

Usage:
    python -m benchmarks.biological_validation.beta_enrichment
    python -m benchmarks.biological_validation.beta_enrichment --models Topic-Transformer --datasets setty endo dentate
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.config import BIO_RESULTS_DIR
BIO_RESULTS = BIO_RESULTS_DIR

TOPIC_MODELS = ["Topic-Base", "Topic-Transformer", "Topic-Contrastive"]
DATASETS = ["setty", "endo", "dentate"]


def load_beta(model: str, dataset: str):
    """Load beta matrix and gene names for a Topic model × dataset pair.

    Searches both the root results dir and model-specific subdirectory.
    """
    tag = f"{model}_{dataset}"
    candidates = [
        BIO_RESULTS / model / f"{tag}_beta.npz",
        BIO_RESULTS / f"{tag}_beta.npz",
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            return d["beta"], d["gene_names"]
    return None, None


def top_genes_from_beta(beta, gene_names, top_n=100):
    """Extract top N genes per topic by beta weight.

    Args:
        beta: [K, G] topic-word probability matrix
        gene_names: [G] array of gene symbols
        top_n: number of top genes per topic

    Returns:
        dict mapping topic_id -> list of gene symbols
    """
    K = beta.shape[0]
    result = {}
    for k in range(K):
        idx = np.argsort(beta[k])[::-1][:top_n]
        result[k] = [str(gene_names[i]) for i in idx]
    return result


def run_enrichment(gene_list, gene_sets="GO_Biological_Process_2021",
                   organism="human", top_n=10):
    """Run Enrichr enrichment analysis via gseapy."""
    try:
        import gseapy as gp
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            no_plot=True,
            verbose=False)
        if enr.results is not None and len(enr.results) > 0:
            df = enr.results.sort_values("Adjusted P-value").head(top_n).copy()
            return df
    except Exception as e:
        print(f"    Enrichment failed: {e}")
    return None


def process_model_dataset(model, dataset, top_n_genes=100, gene_sets="GO_Biological_Process_2021",
                           organism="human", force=False):
    """Extract beta top genes, run enrichment, save CSVs."""
    tag = f"{model}_{dataset}"
    beta, gene_names = load_beta(model, dataset)
    if beta is None:
        print(f"  {tag}: beta.npz not found — skipping")
        return False

    K = beta.shape[0]
    print(f"  {tag}: beta shape {beta.shape}, extracting top {top_n_genes} genes per topic")

    # Determine output dir (same as where beta.npz was found)
    out_dir = BIO_RESULTS / model
    if not out_dir.is_dir():
        out_dir = BIO_RESULTS
    # Also check root
    if (BIO_RESULTS / f"{tag}_beta.npz").exists():
        out_dir_root = BIO_RESULTS
    else:
        out_dir_root = BIO_RESULTS / model

    # Use model subdirectory for new outputs
    out_dir = BIO_RESULTS / model
    out_dir.mkdir(parents=True, exist_ok=True)

    top_genes = top_genes_from_beta(beta, gene_names, top_n=top_n_genes)

    n_saved = 0
    for k in range(K):
        csv_path = out_dir / f"{tag}_beta_enrich_comp{k}.csv"
        if csv_path.exists() and not force:
            print(f"    Topic {k}: already exists — skipping")
            n_saved += 1
            continue

        gene_list = top_genes[k]
        enr_df = run_enrichment(gene_list, gene_sets=gene_sets,
                                organism=organism, top_n=10)

        if enr_df is not None and not enr_df.empty:
            enr_df.to_csv(csv_path, index=False)
            top_term = enr_df.iloc[0]["Term"][:55]
            p_val = enr_df.iloc[0]["Adjusted P-value"]
            print(f"    Topic {k}: {len(enr_df)} terms — top: {top_term} (p={p_val:.2e})")
            n_saved += 1
        else:
            print(f"    Topic {k}: no significant enrichment")
        # Rate-limit Enrichr API
        time.sleep(0.5)

    print(f"  {tag}: {n_saved}/{K} topics enriched")
    return True


def main():
    parser = argparse.ArgumentParser(description="Beta-based GOBP enrichment for Topic models")
    parser.add_argument("--models", nargs="+", default=TOPIC_MODELS,
                        help="Topic model variants to process")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Datasets to process")
    parser.add_argument("--top-genes", type=int, default=100,
                        help="Number of top genes per topic from beta")
    parser.add_argument("--gene-sets", default="GO_Biological_Process_2021",
                        help="Enrichr gene set library")
    parser.add_argument("--organism", default="human")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if CSVs already exist")
    args = parser.parse_args()

    print(f"Beta-based GOBP enrichment")
    print(f"  Models: {args.models}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Top genes: {args.top_genes}")
    print(f"  Gene sets: {args.gene_sets}")
    print()

    for model in args.models:
        for dataset in args.datasets:
            process_model_dataset(
                model, dataset,
                top_n_genes=args.top_genes,
                gene_sets=args.gene_sets,
                organism=args.organism,
                force=args.force)

    print("\n✓ Beta enrichment complete.")


if __name__ == "__main__":
    main()
