#!/usr/bin/env python3
"""
Extra Dataset Pre-processing Pipeline
--------------------------------------
Prepares cancer and perturbation datasets for benchmarking by:
1. Loading raw h5ad (no obs annotations)
2. Standard preprocessing: normalize_total → log1p → HVG selection
3. Computing Leiden clusters as pseudo-ground-truth labels
4. Saving an enriched h5ad with 'cell_type' obs column

Usage:
    python scripts/prep_extra_datasets.py [--datasets melanoma tnbc_brain ...]
    python scripts/prep_extra_datasets.py --all
    python scripts/prep_extra_datasets.py --check-only
"""

import argparse, os, sys, gc, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

# ── Output directory for enriched h5ad files ─────────────────────────────────
PREP_OUT = Path(os.environ.get(
    "PREP_EXTRA_OUTDIR",
    "/home/zeyufu/Desktop/datasets/extra_preprocessed"))
PREP_OUT.mkdir(parents=True, exist_ok=True)

# ── Catalog of extra datasets with preprocessing config ──────────────────────
EXTRA_CATALOG = {
    # ── Labeled developmental ─────────────────────────────────────────────────
    "irall": {
        "src":         "/home/zeyufu/Desktop/datasets/IRALL.h5ad",
        "label_key":   "cell_type",      # 12 well-annotated classes
        "leiden_res":  None,             # No Leiden needed — labels exist
        "data_type":   "cluster",
        "species":     "human",
        "domain":      "disease",
        "desc":        "IR-ALL leukemia (cluster, 41k cells, 12 cell types)",
        "note":        "Uses existing 'cell_type' annotation",
    },
    "wtko": {
        "src":         "/home/zeyufu/Desktop/datasets/wtko0312.h5ad",
        "label_key":   "leiden",          # Pre-computed leiden
        "leiden_res":  None,
        "data_type":   "cluster",
        "species":     "mouse",
        "domain":      "perturbation",
        "desc":        "WT vs KO (cluster, 10k cells, leiden pseudo-labels)",
        "note":        "Uses existing Leiden clusters from original analysis",
    },
    # ── Cancer datasets (raw, need Leiden) ───────────────────────────────────
    "melanoma": {
        "src":         "/home/zeyufu/Desktop/datasets/CancerDatasets2/GSE120575_melanomaHmCancer.h5ad",
        "label_key":   "cell_type",      # Will be created from Leiden
        "leiden_res":  0.5,
        "data_type":   "cluster",
        "species":     "human",
        "domain":      "cancer",
        "desc":        "Melanoma tumor microenvironment (cluster, 16k cells, Leiden pseudo-labels)",
        "note":        "Leiden-based pseudo-labels from scRNA-seq tumor microenvironment",
    },
    "tnbc_brain": {
        "src":         "/home/zeyufu/Desktop/datasets/CancerDatasets/GSE143423_tnbc_CancerBrainHm.h5ad",
        "label_key":   "cell_type",
        "leiden_res":  0.4,
        "data_type":   "cluster",
        "species":     "human",
        "domain":      "cancer",
        "desc":        "Triple-negative breast cancer brain metastasis (7k cells, Leiden pseudo-labels)",
        "note":        "Brain metastatic microenvironment from TNBC",
    },
    "lbm_brain": {
        "src":         "/home/zeyufu/Desktop/datasets/CancerDatasets/GSE143423_lbm_CancerBrainHm.h5ad",
        "label_key":   "cell_type",
        "leiden_res":  0.4,
        "data_type":   "cluster",
        "species":     "human",
        "domain":      "cancer",
        "desc":        "Lung brain metastasis microenvironment (12k cells, Leiden pseudo-labels)",
        "note":        "Brain metastatic microenvironment from lung cancer",
    },
    "hepatoblastoma": {
        "src":         "/home/zeyufu/Desktop/datasets/CancerDatasets2/GSE283205_hepatoblastomaCancer.h5ad",
        "label_key":   "cell_type",
        "leiden_res":  0.5,
        "data_type":   "cluster",
        "species":     "human",
        "domain":      "cancer",
        "desc":        "Hepatoblastoma pediatric liver cancer (16k cells, Leiden pseudo-labels)",
        "note":        "Pediatric liver cancer tumor microenvironment",
    },
    "bc_ec": {
        "src":         "/home/zeyufu/Desktop/datasets/CancerDatasets/GSE155109_bcECHmCancer.h5ad",
        "label_key":   "cell_type",
        "leiden_res":  0.4,
        "data_type":   "cluster",
        "species":     "human",
        "domain":      "cancer",
        "desc":        "Breast cancer endothelial cells (8k cells, Leiden pseudo-labels)",
        "note":        "Tumor endothelial cell heterogeneity in breast cancer",
    },
    "bcc": {
        "src":         "/home/zeyufu/Desktop/datasets/CancerDatasets/GSE123813_bccHmCancer.h5ad",
        "label_key":   "cell_type",
        "leiden_res":  0.8,
        "data_type":   "cluster",
        "species":     "human",
        "domain":      "cancer",
        "desc":        "Basal cell carcinoma immune TME (53k cells, Leiden pseudo-labels)",
        "note":        "Tumor-infiltrating immune cell diversity in BCC",
    },
}


def preprocess_dataset(key, info, max_cells=3000, hvg_genes=3000, seed=42):
    """Load, preprocess, compute Leiden (if needed), and save enriched h5ad."""
    import anndata as ad

    src = Path(info["src"])
    dst = PREP_OUT / f"{key}_prepped.h5ad"

    if dst.exists():
        print(f"  [{key}] Already exists: {dst.name} — skipping (use --force to redo)")
        return True

    print(f"\n{'='*60}")
    print(f"  Processing: {key}")
    print(f"  Source:     {src.name}")
    print(f"  Desc:       {info['desc']}")
    print(f"{'='*60}")

    # 1. Load
    print("  Loading...")
    adata = sc.read_h5ad(src)
    print(f"  Shape: {adata.shape} | obs_cols: {list(adata.obs.columns)}")

    # 2. Obs names unique
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    # 3. Subsample if needed
    if adata.n_obs > max_cells * 4:  # Only hard-subsample very large datasets
        print(f"  Subsampling {adata.n_obs:,} → {max_cells * 4:,} for Leiden precompute")
        sc.pp.subsample(adata, n_obs=max_cells * 4, random_state=seed)

    # 4. Preprocessing (only if labels need to be computed)
    needs_leiden = (info["leiden_res"] is not None)
    if needs_leiden and info["label_key"] not in adata.obs.columns:
        print("  Normalizing + log1p + HVG...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(hvg_genes, adata.n_vars - 1))
        adata_hvg = adata[:, adata.var.highly_variable].copy()

        print(f"  PCA + neighbors + Leiden (res={info['leiden_res']})...")
        sc.pp.scale(adata_hvg, max_value=10)
        sc.tl.pca(adata_hvg, n_comps=50, random_state=seed)
        sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=40, random_state=seed)
        sc.tl.leiden(adata_hvg, resolution=info["leiden_res"], random_state=seed)

        n_clusters = adata_hvg.obs["leiden"].nunique()
        print(f"  Leiden: {n_clusters} clusters")

        # Copy Leiden back to original adata
        adata.obs["cell_type"] = adata_hvg.obs["leiden"].values
        del adata_hvg
        gc.collect()
    elif info["label_key"] in adata.obs.columns:
        print(f"  Using existing labels: '{info['label_key']}' ({adata.obs[info['label_key']].nunique()} classes)")
        if info["label_key"] != "cell_type":
            adata.obs["cell_type"] = adata.obs[info["label_key"]].copy()
    else:
        print(f"  WARNING: No label key found and leiden_res=None → skipping label creation")

    # 5. Save (save only the raw counts if possible for benchmark preproc)
    print(f"  Saving → {dst}")
    adata.write_h5ad(dst, compression="gzip")
    n_classes = adata.obs["cell_type"].nunique() if "cell_type" in adata.obs else "N/A"
    print(f"  Done: {adata.n_obs:,} cells | {n_classes} cell_type classes")
    return True


def check_status():
    """Print quick prep status for all datasets."""
    print(f"\n{'Dataset':<20} {'Source cells':>12} {'Prep file':>12} {'Status'}")
    print("-" * 70)
    import anndata as ad
    for key, info in EXTRA_CATALOG.items():
        dst = PREP_OUT / f"{key}_prepped.h5ad"
        try:
            n_src = ad.read_h5ad(info["src"], backed="r").n_obs
        except Exception:
            n_src = "ERR"
        status = "DONE" if dst.exists() else "PENDING"
        n_classes = "N/A"
        if dst.exists():
            try:
                h = ad.read_h5ad(dst, backed="r")
                n_classes = f"{h.obs.get('cell_type', h.obs.iloc[:, 0]).nunique()} types" if h.obs.shape[1] > 0 else "0 types"
                h.file.close()
            except Exception:
                n_classes = "ERR"
        print(f"  {key:<20} {str(n_src):>12} {dst.name if dst.exists() else '---':>16}  [{status}] {n_classes}")


def main():
    parser = argparse.ArgumentParser(description="Prep extra datasets for benchmark")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset keys to process (default: all)")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--check-only", action="store_true", help="Print status only")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if file exists")
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--hvg-genes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.check_only:
        check_status()
        return

    targets = list(EXTRA_CATALOG.keys()) if (args.all or not args.datasets) else args.datasets
    invalid = [k for k in targets if k not in EXTRA_CATALOG]
    if invalid:
        print(f"Unknown datasets: {invalid}")
        sys.exit(1)

    if args.force:
        for k in targets:
            dst = PREP_OUT / f"{k}_prepped.h5ad"
            if dst.exists():
                dst.unlink()
                print(f"  Removed {dst}")

    print(f"\nPreparing {len(targets)} datasets: {targets}")
    for key in targets:
        try:
            preprocess_dataset(key, EXTRA_CATALOG[key],
                               max_cells=args.max_cells,
                               hvg_genes=args.hvg_genes,
                               seed=args.seed)
        except Exception as e:
            print(f"  [{key}] FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\n\nFinal status:")
    check_status()


if __name__ == "__main__":
    main()
