#!/usr/bin/env python
"""
Latent component perturbation analysis + gene set enrichment.

For each latent component (dim for DPMM, topic for Topic):
  1. Perturb that component by a small delta while holding others fixed
  2. Decode the perturbed latent to gene space
  3. Compute the change in reconstruction -> identifies responsive genes
  4. Run GO/KEGG enrichment on top responsive genes via gseapy

This script produces:
  - Per-component gene importance heatmap
  - Per-component GO/KEGG enrichment dot plots
  - Summary table of top genes and enriched pathways per component

Usage:
    python benchmarks/biological_validation/perturbation_analysis.py \
        --model-path benchmarks/training_dynamics_results/DPMM-Base_setty_model.pt \
        --dataset setty --series dpmm

    python benchmarks/biological_validation/perturbation_analysis.py \
        --model-path benchmarks/training_dynamics_results/Topic-Base_setty_model.pt \
        --dataset setty --series topic \
        --gene-sets GO_Biological_Process_2021

Requires: gseapy, torch (inference only — lightweight GPU usage)
"""

import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.paper_style import apply_style, apply_cli_overrides, add_style_args
from benchmarks.biological_validation import load_model, load_data_with_genes  # shared


# ═══════════════════════════════════════════════════════════════════════════════
# Perturbation analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_perturbation_importance(model, data_loader, device,
                                     delta=0.5, n_samples=500):
    """Perturb each latent dim and measure reconstruction change.

    Returns:
        importance: [K, G] matrix — how much each gene's reconstruction
                    changes when component k is perturbed by +delta
        mean_latent: [K] mean value of each component across cells
    """
    is_topic = hasattr(model, 'n_topics')

    # Collect a subset of latent representations
    all_latent = []
    all_x = []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device).float()
            else:
                x = batch.to(device).float()

            if is_topic:
                z = model.encode(x)
            else:
                z = model.encode(x)

            all_latent.append(z)
            all_x.append(x)

    all_latent = torch.cat(all_latent, dim=0)
    all_x = torch.cat(all_x, dim=0)

    # Subsample for efficiency
    N = all_latent.shape[0]
    if N > n_samples:
        idx = torch.randperm(N)[:n_samples]
        all_latent = all_latent[idx]
        all_x = all_x[idx]

    K = all_latent.shape[1]
    G = all_x.shape[1]

    # Baseline reconstruction
    with torch.no_grad():
        baseline_recon = model.decode(all_latent)  # [N, G]

    # Perturbation for each component
    importance = np.zeros((K, G))

    for k in range(K):
        z_pert = all_latent.clone()
        z_pert[:, k] += delta

        # For Topic models: re-normalize so proportions still sum to 1
        if is_topic:
            z_pert = torch.clamp(z_pert, min=1e-8)
            z_pert = z_pert / z_pert.sum(dim=1, keepdim=True)

        with torch.no_grad():
            recon_pert = model.decode(z_pert)

        # Importance = mean absolute change in reconstruction per gene
        diff = (recon_pert - baseline_recon).abs().mean(dim=0)  # [G]
        importance[k] = diff.cpu().numpy()

    mean_latent = all_latent.mean(dim=0).cpu().numpy()

    return importance, mean_latent


def get_top_genes_per_component(importance, gene_names, top_n=50):
    """Return top responsive genes per component.

    Returns:
        result: dict {component_idx: [(gene_name, importance_score), ...]}
    """
    K = importance.shape[0]
    result = {}
    for k in range(K):
        idx = np.argsort(importance[k])[::-1][:top_n]
        result[k] = [(gene_names[i], float(importance[k, i])) for i in idx]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Gene set enrichment via gseapy
# ═══════════════════════════════════════════════════════════════════════════════

def run_enrichment(gene_list, gene_sets="GO_Biological_Process_2021",
                   organism="human", top_n=10):
    """Run Enrichr enrichment analysis via gseapy.

    Args:
        gene_list: list of gene symbols
        gene_sets: Enrichr library name(s)
        organism: 'human' or 'mouse'
        top_n: number of top terms to return

    Returns:
        DataFrame with enrichment results or None
    """
    try:
        import gseapy as gp
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,  # Don't save to disk by default
            no_plot=True,
            verbose=False)
        if enr.results is not None and len(enr.results) > 0:
            df = enr.results.sort_values("Adjusted P-value").head(top_n).copy()
            return df
    except Exception as e:
        print(f"  Enrichment failed: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_importance_heatmap(importance, gene_names, save_path,
                            title="Gene Importance per Component",
                            top_genes=30, no_title=False, figformat="png",
                            series="dpmm"):
    """Heatmap: rows = components, columns = top genes (union of top per component)."""
    apply_style()

    K = importance.shape[0]
    # Collect top genes (union across components)
    top_idx = set()
    for k in range(K):
        top_idx.update(np.argsort(importance[k])[::-1][:top_genes])
    top_idx = sorted(top_idx, key=lambda i: -importance.max(axis=0)[i])[:top_genes]

    sub = importance[:, top_idx]
    sub_genes = [gene_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(12, len(sub_genes) * 0.45),
                                    max(4, K * 0.6)))
    im = ax.imshow(sub, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(sub_genes)))
    ax.set_xticklabels(sub_genes, rotation=60, ha="right", fontsize=9)
    if series == "topic":
        ylabels = [f"Topic {k+1}" for k in range(K)]
    else:
        ylabels = [f"Dim {k+1}" for k in range(K)]
    ax.set_yticks(range(K))
    ax.set_yticklabels(ylabels, fontsize=11)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Importance")
    if not no_title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

    fig.tight_layout()
    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=mpl.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_enrichment_dotplot(enr_results, component_label, save_path,
                             no_title=False, figformat="png"):
    """Dot plot for enrichment results of one component."""
    apply_style()

    if enr_results is None or enr_results.empty:
        print(f"  No enrichment results for {component_label}")
        return

    df = enr_results.head(10).copy()
    df["neg_log10_p"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-50))
    df["Term_short"] = df["Term"].str[:60]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45)))
    scatter = ax.scatter(
        df["neg_log10_p"], range(len(df)),
        s=df["neg_log10_p"] * 20,
        c=df["neg_log10_p"], cmap="viridis",
        edgecolors="black", linewidths=0.5, alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Term_short"], fontsize=10)
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=13)
    ax.invert_yaxis()

    if not no_title:
        ax.set_title(f"Enrichment — {component_label}", fontsize=14, fontweight="bold")

    plt.colorbar(scatter, ax=ax, shrink=0.6, label="-log10(p)")
    fig.tight_layout()
    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=mpl.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_enrichment_summary(all_enr, save_path, title="Top Enriched Terms per Component",
                             no_title=False, figformat="png", series="dpmm"):
    """Combined dot plot: all components side by side."""
    apply_style()

    # Collect top 3 terms per component
    rows = []
    for k, enr_df in all_enr.items():
        if enr_df is None or enr_df.empty:
            continue
        top3 = enr_df.head(3)
        for _, row in top3.iterrows():
            label = f"Topic {k+1}" if series == "topic" else f"Dim {k+1}"
            rows.append({
                "Component": label,
                "Term": row["Term"][:50],
                "neg_log10_p": -np.log10(max(row["Adjusted P-value"], 1e-50)),
            })

    if not rows:
        print("  No enrichment results to summarize.")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.35)))
    y_labels = [f"{r['Component']}: {r['Term']}" for _, r in df.iterrows()]
    colors = df["neg_log10_p"].values

    scatter = ax.barh(range(len(df)), df["neg_log10_p"],
                      color=mpl.colormaps["viridis"](colors / max(colors.max(), 1)),
                      edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=13)
    ax.invert_yaxis()

    if not no_title:
        ax.set_title(title, fontsize=16, fontweight="bold")

    fig.tight_layout()
    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=mpl.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Latent component perturbation + enrichment")
    parser.add_argument("--model-path", required=True,
                        help="Path to saved model .pt")
    parser.add_argument("--dataset", required=True, choices=["setty", "lung", "endo", "dentate"])
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--delta", type=float, default=0.5,
                        help="Perturbation magnitude")
    parser.add_argument("--top-genes", type=int, default=50,
                        help="Number of top genes per component for enrichment")
    parser.add_argument("--gene-sets", type=str, default="GO_Biological_Process_2021",
                        help="Enrichr gene set library")
    parser.add_argument("--organism", type=str, default="human",
                        choices=["human", "mouse"])
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of cells to use for perturbation")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip gseapy enrichment (just compute gene importance)")
    add_style_args(parser)
    args = parser.parse_args()

    apply_style()
    apply_cli_overrides(args)

    out_root = Path(args.output_dir) if args.output_dir else \
        ROOT / "benchmarks" / "biological_validation" / "results"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, model_name = load_model(args.model_path, device)

    # Per-model sub-directory keeps outputs organized
    out_dir = out_root / model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data with gene names
    print(f"Loading dataset {args.dataset}...")
    splitter, gene_names = load_data_with_genes(args.dataset, seed=args.seed)
    print(f"  {len(gene_names)} genes, {splitter.n_obs} cells")

    # Perturbation analysis
    print("Running perturbation analysis...")
    importance, mean_latent = compute_perturbation_importance(
        model, splitter.test_loader, device,
        delta=args.delta, n_samples=args.n_samples)
    K = importance.shape[0]
    print(f"  {K} components × {importance.shape[1]} genes")

    tag = f"{model_name}_{args.dataset}"

    # Save raw importance matrix
    np.savez(out_dir / f"{tag}_importance.npz",
             importance=importance,
             mean_latent=mean_latent,
             gene_names=np.array(gene_names))

    # Plot importance heatmap
    plot_importance_heatmap(
        importance, gene_names,
        out_dir / f"{tag}_importance_heatmap",
        title=f"Gene Importance per Component — {model_name}",
        top_genes=30, series=args.series,
        no_title=getattr(args, 'no_title', False),
        figformat=args.fig_format)

    # Top genes per component
    top_genes = get_top_genes_per_component(importance, gene_names, top_n=args.top_genes)

    # Save top genes as JSON
    top_genes_json = {str(k): [(g, float(s)) for g, s in v]
                      for k, v in top_genes.items()}
    with open(out_dir / f"{tag}_top_genes.json", "w") as f:
        json.dump(top_genes_json, f, indent=2)
    print(f"  Top genes saved: {out_dir / f'{tag}_top_genes.json'}")

    # Enrichment analysis
    if not args.skip_enrichment:
        print(f"\nRunning enrichment ({args.gene_sets})...")
        all_enr = {}
        comp_label_fn = (lambda k: f"Topic {k+1}") if args.series == "topic" \
                        else (lambda k: f"Dim {k+1}")

        for k in range(K):
            gene_list = [g for g, _ in top_genes[k]]
            label = comp_label_fn(k)
            print(f"  Component {k+1}/{K}: {label} ({len(gene_list)} genes)")

            enr_df = run_enrichment(
                gene_list, gene_sets=args.gene_sets,
                organism=args.organism, top_n=10)
            all_enr[k] = enr_df

            if enr_df is not None and not enr_df.empty:
                print(f"    Top term: {enr_df.iloc[0]['Term'][:60]} "
                      f"(p={enr_df.iloc[0]['Adjusted P-value']:.2e})")

                # Per-component dot plot
                plot_enrichment_dotplot(
                    enr_df, label,
                    out_dir / f"{tag}_enrichment_{k}",
                    no_title=getattr(args, 'no_title', False),
                    figformat=args.fig_format)

        # Summary enrichment across all components
        plot_enrichment_summary(
            all_enr,
            out_dir / f"{tag}_enrichment_summary",
            title=f"Enrichment Summary — {model_name}",
            series=args.series,
            no_title=getattr(args, 'no_title', False),
            figformat=args.fig_format)

        # Save enrichment tables
        for k, enr_df in all_enr.items():
            if enr_df is not None:
                csv_path = out_dir / f"{tag}_enrichment_comp{k}.csv"
                enr_df.to_csv(csv_path, index=False)
        print(f"  Enrichment CSVs saved to {out_dir}")

    print("\n✓ Perturbation analysis complete.")


if __name__ == "__main__":
    main()
