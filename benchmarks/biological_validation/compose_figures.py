#!/usr/bin/env python
"""
Compose multi-panel biological validation figures for paper blueprints.

Reads pre-computed data (.npz, .csv) from biological_validation/results/
and creates composite publication-quality figures:

  Fig.7-composite: 3-row layout
    Row A: Component UMAP grid (cell-type ref + K component heatmaps)
    Row B: Perturbation importance heatmap (component × top genes)
    Row C: Enrichment summary bar chart (top 3 terms per component)

  Fig.7-enrichment-grid: K-panel enrichment dot-plot grid

Usage:
    python benchmarks/biological_validation/compose_figures.py \
        --model DPMM-Base --dataset setty --series dpmm

    python benchmarks/biological_validation/compose_figures.py \
        --model Topic-Base --dataset setty --series topic

No GPU or model loading required — works entirely from saved results.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.paper_style import apply_style, apply_cli_overrides, add_style_args


def _load_latent_data(results_dir, tag):
    """Load pre-computed latent data from .npz."""
    path = results_dir / f"{tag}_latent_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Latent data not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["latent"], data.get("components"), data.get("labels")


def _load_importance(results_dir, tag):
    """Load pre-computed importance matrix from .npz."""
    path = results_dir / f"{tag}_importance.npz"
    if not path.exists():
        raise FileNotFoundError(f"Importance data not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["importance"], data.get("gene_names")


def _load_enrichment_csvs(results_dir, tag, K):
    """Load enrichment CSV files for all components."""
    all_enr = {}
    for k in range(K):
        csv_path = results_dir / f"{tag}_enrichment_comp{k}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                all_enr[k] = df
    return all_enr


def _compute_umap(latent):
    """Compute UMAP embedding from latent data.

    Falls back to PCA + t-SNE if umap-learn is not available or broken.
    """
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=15, min_dist=0.5, random_state=42)
        return reducer.fit_transform(latent)
    except (ImportError, Exception):
        print("  [fallback] umap-learn unavailable, using PCA + t-SNE")
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        # PCA to 30 dims first for efficiency
        n_pca = min(30, latent.shape[1])
        pca = PCA(n_components=n_pca, random_state=42)
        latent_pca = pca.fit_transform(latent)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     init="pca", learning_rate="auto")
        return tsne.fit_transform(latent_pca)


# ═══════════════════════════════════════════════════════════════════════════════
# Composite Figure: UMAP + Importance + Enrichment (3-row)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_composite_figure(latent, components, labels, importance, gene_names,
                          all_enr, series, model_name, dataset,
                          save_path, figformat="png", max_components=10,
                          top_genes=25):
    """Create compact 2-row composite biological validation figure.

    Row A: UMAP grid (cell-type + K component panels)
    Row B: [Gene importance heatmap | Enrichment summary]  (side-by-side)
    """
    apply_style()

    from sklearn.preprocessing import LabelEncoder

    K = min(components.shape[1], max_components)

    # ── Compute UMAP / t-SNE ──
    print("  Computing 2D embedding...")
    umap_xy = _compute_umap(latent)

    # ── Encode labels ──
    if labels is not None and len(labels) > 0:
        if isinstance(labels[0], (str, np.str_)):
            le = LabelEncoder()
            labels_enc = le.fit_transform(labels)
            label_names = le.classes_
        else:
            labels_enc = np.array(labels, dtype=int)
            label_names = None
    else:
        labels_enc = np.zeros(len(latent), dtype=int)
        label_names = None

    # ── Layout ──
    umap_cols = min(6, K + 1)
    umap_rows = ((K + 1) + umap_cols - 1) // umap_cols

    cell_w, cell_h = 2.6, 2.4
    bottom_h = max(3.0, K * 0.35)

    total_w = cell_w * umap_cols
    total_h = cell_h * umap_rows + bottom_h + 1.0

    fig = plt.figure(figsize=(total_w, total_h))

    # GridSpec: 2 major rows
    gs_main = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[cell_h * umap_rows, bottom_h],
        hspace=0.30)

    # ── Row A: UMAP grid ──
    gs_umap = gridspec.GridSpecFromSubplotSpec(
        umap_rows, umap_cols, subplot_spec=gs_main[0],
        hspace=0.35, wspace=0.35)

    n_cls = len(np.unique(labels_enc))
    cmap_ct = mpl.colormaps.get_cmap("tab20" if n_cls <= 20 else "nipy_spectral")
    comp_prefix = "Topic" if series == "topic" else "Dim"

    # Cell-type reference
    ax0 = fig.add_subplot(gs_umap[0, 0])
    ax0.scatter(umap_xy[:, 0], umap_xy[:, 1], c=labels_enc,
                cmap=cmap_ct, s=4, alpha=0.7, edgecolors="none", rasterized=True)
    ax0.set_title("Cell Types", fontsize=9, fontweight="bold")
    ax0.set_xlabel("UMAP-1", fontsize=7); ax0.set_ylabel("UMAP-2", fontsize=7)
    ax0.tick_params(labelsize=6)
    ax0.text(-0.15, 1.08, "(a)", transform=ax0.transAxes,
             fontsize=11, fontweight="bold", va="top")

    for k in range(K):
        idx = k + 1
        r, c = idx // umap_cols, idx % umap_cols
        ax = fig.add_subplot(gs_umap[r, c])
        vals = components[:, k]
        sc_p = ax.scatter(umap_xy[:, 0], umap_xy[:, 1], c=vals,
                          cmap="viridis", s=4, alpha=0.7,
                          edgecolors="none", rasterized=True)
        ax.set_title(f"{comp_prefix} {k+1}", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=5)
        plt.colorbar(sc_p, ax=ax, shrink=0.5, pad=0.02, aspect=15)

    for idx in range(K + 1, umap_rows * umap_cols):
        r, c = idx // umap_cols, idx % umap_cols
        fig.add_subplot(gs_umap[r, c]).axis("off")

    # ── Row B: Importance (left) + Enrichment (right) side-by-side ──
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[1],
        wspace=0.30, width_ratios=[1.2, 1.0])

    # -- B-left: Importance heatmap --
    ax_heat = fig.add_subplot(gs_bottom[0, 0])
    top_idx = set()
    for k in range(K):
        top_idx.update(np.argsort(importance[k])[::-1][:top_genes])
    top_idx = sorted(top_idx, key=lambda i: -importance.max(axis=0)[i])[:top_genes]

    sub_imp = importance[:K, :][:, top_idx]
    sub_genes = [str(gene_names[i]) for i in top_idx]

    im = ax_heat.imshow(sub_imp, aspect="auto", cmap="YlOrRd",
                         interpolation="nearest")
    ax_heat.set_xticks(range(len(sub_genes)))
    ax_heat.set_xticklabels(sub_genes, rotation=55, ha="right", fontsize=5.5)
    ylabels = [f"{comp_prefix} {k+1}" for k in range(K)]
    ax_heat.set_yticks(range(K))
    ax_heat.set_yticklabels(ylabels, fontsize=7)
    plt.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.02, label="Importance")
    ax_heat.set_title("Gene Importance", fontsize=10, fontweight="bold", pad=6)
    ax_heat.text(-0.08, 1.08, "(b)", transform=ax_heat.transAxes,
                 fontsize=11, fontweight="bold", va="top")

    # -- B-right: Enrichment summary --
    ax_enr = fig.add_subplot(gs_bottom[0, 1])
    rows_data = []
    for k in sorted(all_enr.keys()):
        if k >= K:
            continue
        enr_df = all_enr[k]
        if enr_df is None or enr_df.empty:
            continue
        top3 = enr_df.head(3)
        for _, row in top3.iterrows():
            pval = row.get("Adjusted P-value", row.get("P-value", 1.0))
            term = str(row.get("Term", ""))[:45]
            rows_data.append({
                "Component": f"{comp_prefix} {k+1}",
                "Term": term,
                "neg_log10_p": -np.log10(max(float(pval), 1e-50)),
            })

    if rows_data:
        df = pd.DataFrame(rows_data)
        y_labels = [f"{r['Component']}: {r['Term']}" for _, r in df.iterrows()]
        colors = df["neg_log10_p"].values
        norm_c = colors / max(colors.max(), 1)

        ax_enr.barh(range(len(df)), df["neg_log10_p"],
                     color=mpl.colormaps["viridis"](norm_c),
                     edgecolor="white", linewidth=0.3)
        ax_enr.set_yticks(range(len(df)))
        ax_enr.set_yticklabels(y_labels, fontsize=5.5)
        ax_enr.set_xlabel(u"-log\u2081\u2080(adj. P)", fontsize=8)
        ax_enr.invert_yaxis()
        ax_enr.set_title("GO Enrichment (Top 3/comp.)", fontsize=10,
                          fontweight="bold", pad=6)
    else:
        ax_enr.text(0.5, 0.5, "No enrichment results",
                     ha="center", va="center", fontsize=10,
                     transform=ax_enr.transAxes)
        ax_enr.axis("off")
    ax_enr.text(-0.08, 1.08, "(c)", transform=ax_enr.transAxes,
                fontsize=11, fontweight="bold", va="top")

    # ── Save ──
    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  \u2713 Saved composite figure: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Enrichment grid: all components' dot-plots in one figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_enrichment_grid(all_enr, series, model_name, save_path,
                          figformat="png", max_components=10, top_terms=8):
    """Grid of enrichment dot-plots (one panel per component)."""
    apply_style()

    K = min(max(all_enr.keys()) + 1 if all_enr else 0, max_components)
    if K == 0:
        print("  No enrichment data to plot.")
        return

    comp_prefix = "Topic" if series == "topic" else "Dim"

    cols = min(3, K)
    rows_grid = (K + cols - 1) // cols
    cell_w, cell_h = 5.0, 3.5

    fig, axes = plt.subplots(rows_grid, cols,
                              figsize=(cell_w * cols, cell_h * rows_grid))
    if rows_grid == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows_grid == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for k in range(K):
        r, c = k // cols, k % cols
        ax = axes[r, c]

        if k in all_enr and all_enr[k] is not None and not all_enr[k].empty:
            df = all_enr[k].head(top_terms).copy()
            pval_col = "Adjusted P-value" if "Adjusted P-value" in df.columns else "P-value"
            df["neg_log10_p"] = -np.log10(df[pval_col].clip(lower=1e-50).astype(float))
            df["Term_short"] = df["Term"].astype(str).str[:45]

            colors = df["neg_log10_p"].values
            scatter = ax.barh(range(len(df)), df["neg_log10_p"],
                              color=mpl.colormaps["viridis"](colors / max(colors.max(), 1)),
                              edgecolor="white", linewidth=0.3)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["Term_short"], fontsize=7)
            ax.set_xlabel("-log₁₀(p)", fontsize=8)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                     fontsize=10, transform=ax.transAxes, color="gray")

        ax.set_title(f"{comp_prefix} {k+1}", fontsize=11, fontweight="bold")

    # Hide unused panels
    for idx in range(K, rows_grid * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    fig.suptitle(f"GO Enrichment per Component — {model_name}",
                  fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = Path(save_path).with_suffix(f".{figformat}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved enrichment grid: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compose multi-panel biological validation figures"
    )
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., DPMM-Base, Topic-Base)")
    parser.add_argument("--dataset", required=True,
                        choices=["setty", "lung", "endo", "dentate"])
    parser.add_argument("--series", required=True,
                        choices=["dpmm", "topic"])
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing .npz and .csv results")
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--top-genes", type=int, default=25,
                        help="Top genes to show in heatmap")
    add_style_args(parser)
    args = parser.parse_args()

    apply_style()
    apply_cli_overrides(args)

    results_root = Path(args.results_dir) if args.results_dir else \
        ROOT / "benchmarks" / "biological_validation" / "results"

    # Support both per-model sub-directory (new) and flat layout (legacy)
    model_dir = results_root / args.model.replace("/", "_")
    results_dir = model_dir if model_dir.is_dir() else results_root

    tag = f"{args.model}_{args.dataset}"

    print(f"Loading pre-computed data for {tag}...")

    # Load all data
    latent_data = np.load(results_dir / f"{tag}_latent_data.npz",
                           allow_pickle=True)
    latent = latent_data["latent"]
    components = latent_data.get("components")
    if components is None:
        components = latent  # fallback
    labels = latent_data.get("labels")

    imp_data = np.load(results_dir / f"{tag}_importance.npz",
                        allow_pickle=True)
    importance = imp_data["importance"]
    gene_names = imp_data.get("gene_names", np.array([]))

    K = min(importance.shape[0], args.max_components)
    all_enr = _load_enrichment_csvs(results_dir, tag, K)

    print(f"  {K} components, {len(gene_names)} genes, "
          f"{len(all_enr)} enrichment tables")

    # ── Generate composite figure ──
    out_composite = results_dir / f"{tag}_composite"
    plot_composite_figure(
        latent, components, labels, importance, gene_names,
        all_enr, args.series, args.model, args.dataset,
        out_composite,
        figformat=args.fig_format,
        max_components=args.max_components,
        top_genes=args.top_genes)

    # ── Generate enrichment grid ──
    out_enr_grid = results_dir / f"{tag}_enrichment_grid"
    plot_enrichment_grid(
        all_enr, args.series, args.model,
        out_enr_grid,
        figformat=args.fig_format,
        max_components=args.max_components)

    print(f"\n✓ All composite figures saved to {results_dir}")


if __name__ == "__main__":
    main()
