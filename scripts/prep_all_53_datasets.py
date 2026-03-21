#!/usr/bin/env python3
"""
Full 53-Dataset Pre-processing Pipeline
-----------------------------------------
Prepares ALL datasets needed for the 53-dataset benchmark refresh.

Datasets are split into three tiers:
  1. **Core 16** (already have labels in SCRNA_16_DATASETS) — no prep needed
  2. **Extra 8** (irall, wtko + 6 cancer) — handled by prep_extra_datasets.py
  3. **New 29** (10 unregistered development + 22 unregistered cancer) — handled here

For the new datasets, this script:
  1. Loads raw h5ad (no obs annotations or only 'batch')
  2. Standard preprocessing: normalize_total → log1p → HVG selection
  3. Computes Leiden clusters as pseudo-ground-truth labels
  4. Saves an enriched h5ad with 'cell_type' obs column

Usage:
    python scripts/prep_all_53_datasets.py --all
    python scripts/prep_all_53_datasets.py --datasets scc lung_adre
    python scripts/prep_all_53_datasets.py --check-only
"""

import argparse, os, sys, gc, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc

# ── Output directory ─────────────────────────────────────────────────────────
PREP_OUT = Path("/home/zeyufu/Desktop/datasets/extra_preprocessed")
PREP_OUT.mkdir(parents=True, exist_ok=True)

DATASETS_ROOT = Path("/home/zeyufu/Desktop/datasets")

# ── Catalog of NEW datasets to prep (29 datasets) ───────────────────────────
# These are datasets NOT currently in SCRNA_16_DATASETS or EXTRA_DATASET_REGISTRY
NEW_CATALOG = {
    # ═══════════════════════════════════════════════════════════════════════════
    # 10 Unregistered Development Datasets (no labels, need Leiden)
    # ═══════════════════════════════════════════════════════════════════════════
    "hesc_hspc_cd8": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets/GSE148215_hESCHSPCD8Hm.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "development",
        "desc": "hESC HSPC CD8 differentiation (cluster, ~10k cells, Leiden pseudo-labels)",
    },
    "lsk_batch": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets/GSE165844_LSKMmBatch.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse LSK batch effects (cluster, ~20k cells, Leiden pseudo-labels)",
    },
    "hsc_aged": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets/GSE226131_HSCMmAged.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse aged HSC (cluster, ~16k cells, Leiden pseudo-labels)",
    },
    "bm_niche": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets/GSE253355_bmNicheHm.h5ad"),
        "leiden_res": 0.6,
        "data_type": "cluster",
        "species": "human",
        "domain": "development",
        "desc": "Human BM niche (cluster, ~100k cells, Leiden pseudo-labels)",
    },
    "lps_mm": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets2/GSE115571_LPSMmDev.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse LPS development (cluster, ~20k cells, Leiden pseudo-labels)",
    },
    "progastin": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets2/GSE145929_ProgastinMmDev.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse progastrin development (cluster, ~61k cells, Leiden pseudo-labels)",
    },
    "urine": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets2/GSE145929_UrineMmDev.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse urine development (cluster, ~29k cells, Leiden pseudo-labels)",
    },
    "astrocytes_sci": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets2/GSE189070_astrocytesSCIMmDev.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse astrocytes SCI development (cluster, ~103k cells, Leiden pseudo-labels)",
    },
    "ad_hm": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets2/GSE213740_ADHm.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "development",
        "desc": "Human Alzheimer's disease (cluster, ~24k cells, Leiden pseudo-labels)",
    },
    "blood_stroke": {
        "src": str(DATASETS_ROOT / "DevelopmentDatasets2/GSE225948_bloodMmStrokeDev.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse blood stroke development (cluster, ~52k cells, Leiden pseudo-labels)",
    },
    # ═══════════════════════════════════════════════════════════════════════════
    # 10 CancerDatasets (unregistered, no labels)
    # ═══════════════════════════════════════════════════════════════════════════
    "scc": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE123813_sccHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Squamous cell carcinoma TME (cluster, ~26k cells, Leiden pseudo-labels)",
    },
    "lung_adre": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE123902_LungAdreHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Lung adenocarcinoma (cluster, ~43k cells, Leiden pseudo-labels)",
    },
    "aml_pbmc": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE132509_acutelymluekPBMCHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "AML PBMC (cluster, ~39k cells, Leiden pseudo-labels)",
    },
    "bm_all": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE148218_bmALLHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "BM ALL leukemia (cluster, ~51k cells, Leiden pseudo-labels)",
    },
    "bc_stroma": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE155109_bcStromaHmCancer.h5ad"),
        "leiden_res": 0.4,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer stroma (cluster, ~18k cells, Leiden pseudo-labels)",
    },
    "gastric": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE183904_GastricHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Gastric cancer TME (cluster, ~62k cells, Leiden pseudo-labels)",
    },
    "tcell_cancer": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE222002_TcellsHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Pan-cancer T cells (cluster, ~78k cells, Leiden pseudo-labels)",
    },
    "nk_lymphoma": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE222369_NKsLymphomaHmCancer.h5ad"),
        "leiden_res": 0.6,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "NK cells in lymphoma (cluster, ~139k cells, Leiden pseudo-labels)",
    },
    "breast_cancer": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE225600_breast_CancerHm.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer atlas (cluster, ~82k cells, Leiden pseudo-labels)",
    },
    "bcell_all": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE235787_bcellsALLHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "B cells in ALL (cluster, ~113k cells, Leiden pseudo-labels)",
    },
    "breast_metastasis": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE262288_breastMetasisHmCancer.h5ad"),
        "leiden_res": 0.6,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer metastasis (cluster, ~158k cells, Leiden pseudo-labels)",
    },
    "tcell_liver": {
        "src": str(DATASETS_ROOT / "CancerDatasets/GSE98638_TcellLiverHmCancer.h5ad"),
        "leiden_res": 0.4,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "T cells in liver cancer (cluster, ~5k cells, Leiden pseudo-labels)",
    },
    # ═══════════════════════════════════════════════════════════════════════════
    # 12 CancerDatasets2 (unregistered, no labels) -- minus hepatoblastoma
    #   already in EXTRA_DATASET_REGISTRY
    # ═══════════════════════════════════════════════════════════════════════════
    "mcc_pbmc": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE117988_MCCPBMCCancer.h5ad"),
        "leiden_res": 0.4,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "MCC PBMC (cluster, ~13k cells, Leiden pseudo-labels)",
    },
    "mcc_tumor": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE117988_MCCTumorCancer.h5ad"),
        "leiden_res": 0.4,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "MCC tumor (cluster, ~7k cells, Leiden pseudo-labels)",
    },
    # melanoma already in EXTRA_DATASET_REGISTRY
    "mm_cancer": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE124310_MMHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Multiple myeloma (cluster, ~28k cells, Leiden pseudo-labels)",
    },
    "liver_cancer": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE138709_LiverCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Liver cancer (cluster, ~34k cells, Leiden pseudo-labels)",
    },
    "ca_cancer": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE149655_CAHmCancer.h5ad"),
        "leiden_res": 0.4,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Cancer-associated TME (cluster, ~13k cells, Leiden pseudo-labels)",
    },
    "stomach_cancer": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE163558_stomachHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Stomach cancer (cluster, ~55k cells, Leiden pseudo-labels)",
    },
    "breast_hm": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE168181_BreastHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer (cluster, ~24k cells, Leiden pseudo-labels)",
    },
    "lung_adre2": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE189357_lungAdreHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Lung adenocarcinoma 2 (cluster, ~46k cells, Leiden pseudo-labels)",
    },
    "liver_colon_metastasis": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE225857_liverColonMetasisHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Liver-colon metastasis (cluster, ~22k cells, Leiden pseudo-labels)",
    },
    "breast_hm2": {
        "src": str(DATASETS_ROOT / "CancerDatasets2/GSE228499_breastHmCancer.h5ad"),
        "leiden_res": 0.5,
        "data_type": "cluster",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer 2 (cluster, ~32k cells, Leiden pseudo-labels)",
    },
    # hepatoblastoma already in EXTRA_DATASET_REGISTRY
}


def preprocess_dataset(key, info, max_cells=3000, hvg_genes=3000, seed=42):
    """Load, preprocess, compute Leiden (if needed), and save enriched h5ad."""
    src = Path(info["src"])
    dst = PREP_OUT / f"{key}_prepped.h5ad"

    if dst.exists():
        print(f"  [{key}] Already exists: {dst.name} -- skipping (use --force to redo)")
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

    # 3. Subsample if needed (cap at 12k for Leiden)
    if adata.n_obs > max_cells * 4:
        print(f"  Subsampling {adata.n_obs:,} -> {max_cells * 4:,} for Leiden precompute")
        sc.pp.subsample(adata, n_obs=max_cells * 4, random_state=seed)

    # 4. Preprocessing + Leiden
    leiden_res = info.get("leiden_res", 0.5)
    if "cell_type" not in adata.obs.columns:
        print("  Normalizing + log1p + HVG...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(hvg_genes, adata.n_vars - 1))
        adata_hvg = adata[:, adata.var.highly_variable].copy()

        print(f"  PCA + neighbors + Leiden (res={leiden_res})...")
        sc.pp.scale(adata_hvg, max_value=10)
        sc.tl.pca(adata_hvg, n_comps=50, random_state=seed)
        sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=40, random_state=seed)
        sc.tl.leiden(adata_hvg, resolution=leiden_res, random_state=seed)

        n_clusters = adata_hvg.obs["leiden"].nunique()
        print(f"  Leiden: {n_clusters} clusters")

        # Copy Leiden back to original adata
        adata.obs["cell_type"] = adata_hvg.obs["leiden"].values
        del adata_hvg
        gc.collect()
    else:
        print(f"  Using existing 'cell_type' ({adata.obs['cell_type'].nunique()} classes)")

    # 5. Save
    print(f"  Saving -> {dst}")
    adata.write_h5ad(dst, compression="gzip")
    n_classes = adata.obs["cell_type"].nunique() if "cell_type" in adata.obs else "N/A"
    print(f"  Done: {adata.n_obs:,} cells | {n_classes} cell_type classes")
    del adata
    gc.collect()
    return True


def check_status():
    """Print quick prep status for all datasets."""
    import anndata as ad
    print(f"\n{'Dataset':<25} {'Source cells':>12} {'Prep file':>30} {'Status':>8}")
    print("-" * 85)
    for key, info in NEW_CATALOG.items():
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
                n_classes = f"{h.obs['cell_type'].nunique()} types" if "cell_type" in h.obs else "0"
                h.file.close()
            except Exception:
                n_classes = "ERR"
        print(f"  {key:<25} {str(n_src):>12} {dst.name if dst.exists() else '---':>30}  [{status}] {n_classes}")


def main():
    parser = argparse.ArgumentParser(description="Prep all 53 datasets for benchmark")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset keys to process (default: all)")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--check-only", action="store_true", help="Print status only")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    parser.add_argument("--max-cells", type=int, default=3000)
    parser.add_argument("--hvg-genes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.check_only:
        check_status()
        return

    targets = list(NEW_CATALOG.keys()) if (args.all or not args.datasets) else args.datasets
    invalid = [k for k in targets if k not in NEW_CATALOG]
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
            preprocess_dataset(key, NEW_CATALOG[key],
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
