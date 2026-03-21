"""Shared dataset registry for benchmark scripts.

Centralizes dataset paths, label keys, and metadata so multiple benchmark
entry scripts do not duplicate the same large registry block.
"""

from pathlib import Path


DATASETS_ROOT = Path("/home/zeyufu/Desktop/datasets")

DATASET_REGISTRY = {
    "setty": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets" / "setty.h5ad"),
        "data_type": "trajectory",
        "label_key": "clusters",
        "species": "human",
        "desc": "Hematopoiesis (trajectory, ~5780 cells)",
    },
    "lung": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets2" / "GSE130148_LungHmDev.h5ad"),
        "data_type": "cluster",
        "label_key": "celltype",
        "species": "human",
        "desc": "Lung development (cluster, ~10360 cells, 13 types)",
    },
    "endo": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets" / "endo.h5ad"),
        "data_type": "mixed",
        "label_key": "clusters",
        "species": "mouse",
        "desc": "Pancreatic endocrinogenesis (mixed, ~2531 cells, 7 types + pseudotime)",
    },
    "dentate": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets" / "dentate.h5ad"),
        "data_type": "cluster",
        "label_key": "Clusters",
        "species": "mouse",
        "desc": "Mouse dentate gyrus development (cluster, ~18k cells)",
    },
    "hemato": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets" / "hemato.h5ad"),
        "data_type": "trajectory",
        "label_key": "Cell type",
        "species": "mouse",
        "desc": "Hematopoietic differentiation compendium (trajectory, ~22k cells)",
    },
    "pansci_muscle": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets2" / "GSE247719_PanSci_05_Muscle_adata.h5ad"),
        "data_type": "mixed",
        "label_key": "Main_cell_type",
        "species": "mouse_ensembl",
        "desc": "PanSci muscle atlas subset (mixed, large-scale)",
    },
    "blood_aged": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets" / "GSE120505_bloodAged.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "desc": "Aged blood (cluster, ~14.6k cells, Leiden labels)",
    },
    "hesc": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets" / "hESC_GSE144024.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "desc": "hESC differentiation (cluster, ~9.5k cells, Leiden labels)",
    },
    "retina": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets2" / "GSE165784_RetinaHmDev.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "desc": "Human retina development (cluster, ~11.5k cells, Leiden labels)",
    },
    "teeth": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets2" / "GSE275119_TeethMmDev.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "desc": "Mouse teeth development (cluster, ~6.8k cells, Leiden labels)",
    },
    "pituitary": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets2" / "GSE142653pitHmDev.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "desc": "Human pituitary development (cluster, ~5.2k cells, Leiden labels)",
    },
    "pansci_tcell": {
        "path": str(DATASETS_ROOT / "DevelopmentDatasets2" / "GSE247719_PanSci_T_cell_adata.h5ad"),
        "data_type": "cluster",
        "label_key": "Main_cell_type",
        "species": "mouse_ensembl",
        "desc": "PanSci T-cell atlas (cluster, ~958k cells, 16 types)",
    },
}


# ── Extra / sample datasets (run after prep_extra_datasets.py) ───────────────
EXTRA_DATASETS_ROOT = Path("/home/zeyufu/Desktop/datasets/extra_preprocessed")

EXTRA_DATASET_REGISTRY = {
    # -- Labeled developmental / disease -----------------------------------
    "irall": {
        "path": str(EXTRA_DATASETS_ROOT / "irall_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "disease",
        "desc": "IR-ALL leukemia (cluster, ~41k cells, 12 annotated cell types)",
    },
    "wtko": {
        "path": str(EXTRA_DATASETS_ROOT / "wtko_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "perturbation",
        "desc": "WT vs KO perturbation (cluster, ~10k cells, leiden pseudo-labels)",
    },
    # -- Cancer TME datasets (leiden pseudo-labels) ------------------------
    "melanoma": {
        "path": str(EXTRA_DATASETS_ROOT / "melanoma_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Melanoma tumor microenvironment (cluster, ~16k cells, leiden-based)",
    },
    "tnbc_brain": {
        "path": str(EXTRA_DATASETS_ROOT / "tnbc_brain_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "TNBC brain metastasis (cluster, ~7k cells, leiden-based)",
    },
    "lbm_brain": {
        "path": str(EXTRA_DATASETS_ROOT / "lbm_brain_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Lung brain metastasis (cluster, ~12k cells, leiden-based)",
    },
    "hepatoblastoma": {
        "path": str(EXTRA_DATASETS_ROOT / "hepatoblastoma_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Hepatoblastoma pediatric liver cancer (cluster, ~16k cells, leiden-based)",
    },
    "bc_ec": {
        "path": str(EXTRA_DATASETS_ROOT / "bc_ec_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer endothelial cells (cluster, ~8k cells, leiden-based)",
    },
    "bcc": {
        "path": str(EXTRA_DATASETS_ROOT / "bcc_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Basal cell carcinoma tumor microenvironment (cluster, ~53k cells, leiden-based)",
    },
}


# ── Expanded datasets (run after prep_all_53_datasets.py) ────────────────────

EXPANDED_DATASET_REGISTRY = {
    # -- Unregistered development datasets (Leiden pseudo-labels) ---------------
    "hesc_hspc_cd8": {
        "path": str(EXTRA_DATASETS_ROOT / "hesc_hspc_cd8_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "development",
        "desc": "hESC HSPC CD8 diff (cluster, ~10k cells, leiden-based)",
    },
    "lsk_batch": {
        "path": str(EXTRA_DATASETS_ROOT / "lsk_batch_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse LSK batch effects (cluster, ~20k cells, leiden-based)",
    },
    "hsc_aged": {
        "path": str(EXTRA_DATASETS_ROOT / "hsc_aged_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse aged HSC (cluster, ~16k cells, leiden-based)",
    },
    "bm_niche": {
        "path": str(EXTRA_DATASETS_ROOT / "bm_niche_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "development",
        "desc": "Human BM niche (cluster, ~100k cells, leiden-based)",
    },
    "lps_mm": {
        "path": str(EXTRA_DATASETS_ROOT / "lps_mm_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse LPS development (cluster, ~20k cells, leiden-based)",
    },
    "progastin": {
        "path": str(EXTRA_DATASETS_ROOT / "progastin_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse progastrin development (cluster, ~61k cells, leiden-based)",
    },
    "urine": {
        "path": str(EXTRA_DATASETS_ROOT / "urine_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse urine development (cluster, ~29k cells, leiden-based)",
    },
    "astrocytes_sci": {
        "path": str(EXTRA_DATASETS_ROOT / "astrocytes_sci_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse astrocytes SCI (cluster, ~103k cells, leiden-based)",
    },
    "ad_hm": {
        "path": str(EXTRA_DATASETS_ROOT / "ad_hm_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "development",
        "desc": "Human Alzheimer's disease (cluster, ~24k cells, leiden-based)",
    },
    "blood_stroke": {
        "path": str(EXTRA_DATASETS_ROOT / "blood_stroke_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "mouse",
        "domain": "development",
        "desc": "Mouse blood stroke (cluster, ~52k cells, leiden-based)",
    },
    # -- Cancer datasets (CancerDatasets, leiden pseudo-labels) -----------------
    "scc": {
        "path": str(EXTRA_DATASETS_ROOT / "scc_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Squamous cell carcinoma TME (cluster, ~26k cells, leiden-based)",
    },
    "lung_adre": {
        "path": str(EXTRA_DATASETS_ROOT / "lung_adre_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Lung adenocarcinoma (cluster, ~43k cells, leiden-based)",
    },
    "aml_pbmc": {
        "path": str(EXTRA_DATASETS_ROOT / "aml_pbmc_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "AML PBMC (cluster, ~39k cells, leiden-based)",
    },
    "bm_all": {
        "path": str(EXTRA_DATASETS_ROOT / "bm_all_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "BM ALL leukemia (cluster, ~51k cells, leiden-based)",
    },
    "bc_stroma": {
        "path": str(EXTRA_DATASETS_ROOT / "bc_stroma_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer stroma (cluster, ~18k cells, leiden-based)",
    },
    "gastric": {
        "path": str(EXTRA_DATASETS_ROOT / "gastric_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Gastric cancer TME (cluster, ~62k cells, leiden-based)",
    },
    "tcell_cancer": {
        "path": str(EXTRA_DATASETS_ROOT / "tcell_cancer_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Pan-cancer T cells (cluster, ~78k cells, leiden-based)",
    },
    "nk_lymphoma": {
        "path": str(EXTRA_DATASETS_ROOT / "nk_lymphoma_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "NK cells in lymphoma (cluster, ~139k cells, leiden-based)",
    },
    "breast_cancer": {
        "path": str(EXTRA_DATASETS_ROOT / "breast_cancer_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer atlas (cluster, ~82k cells, leiden-based)",
    },
    "bcell_all": {
        "path": str(EXTRA_DATASETS_ROOT / "bcell_all_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "B cells in ALL (cluster, ~113k cells, leiden-based)",
    },
    "breast_metastasis": {
        "path": str(EXTRA_DATASETS_ROOT / "breast_metastasis_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer metastasis (cluster, ~158k cells, leiden-based)",
    },
    "tcell_liver": {
        "path": str(EXTRA_DATASETS_ROOT / "tcell_liver_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "T cells in liver cancer (cluster, ~5k cells, leiden-based)",
    },
    # -- Cancer datasets (CancerDatasets2, leiden pseudo-labels) ----------------
    "mcc_pbmc": {
        "path": str(EXTRA_DATASETS_ROOT / "mcc_pbmc_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "MCC PBMC (cluster, ~13k cells, leiden-based)",
    },
    "mcc_tumor": {
        "path": str(EXTRA_DATASETS_ROOT / "mcc_tumor_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "MCC tumor (cluster, ~7k cells, leiden-based)",
    },
    "mm_cancer": {
        "path": str(EXTRA_DATASETS_ROOT / "mm_cancer_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Multiple myeloma (cluster, ~28k cells, leiden-based)",
    },
    "liver_cancer": {
        "path": str(EXTRA_DATASETS_ROOT / "liver_cancer_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Liver cancer (cluster, ~34k cells, leiden-based)",
    },
    "ca_cancer": {
        "path": str(EXTRA_DATASETS_ROOT / "ca_cancer_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Cancer-associated TME (cluster, ~13k cells, leiden-based)",
    },
    "stomach_cancer": {
        "path": str(EXTRA_DATASETS_ROOT / "stomach_cancer_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Stomach cancer (cluster, ~55k cells, leiden-based)",
    },
    "breast_hm": {
        "path": str(EXTRA_DATASETS_ROOT / "breast_hm_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer (cluster, ~24k cells, leiden-based)",
    },
    "lung_adre2": {
        "path": str(EXTRA_DATASETS_ROOT / "lung_adre2_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Lung adenocarcinoma 2 (cluster, ~46k cells, leiden-based)",
    },
    "liver_colon_metastasis": {
        "path": str(EXTRA_DATASETS_ROOT / "liver_colon_metastasis_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Liver-colon metastasis (cluster, ~22k cells, leiden-based)",
    },
    "breast_hm2": {
        "path": str(EXTRA_DATASETS_ROOT / "breast_hm2_prepped.h5ad"),
        "data_type": "cluster",
        "label_key": "cell_type",
        "species": "human",
        "domain": "cancer",
        "desc": "Breast cancer 2 (cluster, ~32k cells, leiden-based)",
    },
}

# Combined registry (main + extra + expanded)
ALL_DATASET_REGISTRY = {**DATASET_REGISTRY, **EXTRA_DATASET_REGISTRY, **EXPANDED_DATASET_REGISTRY}

# Groupings for targeted experiments
_CANCER_ORIGINAL = ["melanoma", "tnbc_brain", "lbm_brain", "hepatoblastoma", "bc_ec", "bcc"]
_CANCER_EXPANDED = [
    "scc", "lung_adre", "aml_pbmc", "bm_all", "bc_stroma", "gastric",
    "tcell_cancer", "nk_lymphoma", "breast_cancer", "bcell_all",
    "breast_metastasis", "tcell_liver", "mcc_pbmc", "mcc_tumor",
    "mm_cancer", "liver_cancer", "ca_cancer", "stomach_cancer",
    "breast_hm", "lung_adre2", "liver_colon_metastasis", "breast_hm2",
]
_DEV_EXPANDED = [
    "hesc_hspc_cd8", "lsk_batch", "hsc_aged", "bm_niche", "lps_mm",
    "progastin", "urine", "astrocytes_sci", "ad_hm", "blood_stroke",
]

DATASET_GROUPS = {
    "core":            list(DATASET_REGISTRY.keys()),                  # original 12
    "disease":         ["irall"],
    "perturbation":    ["wtko"],
    "cancer":          _CANCER_ORIGINAL,
    "cancer_expanded": _CANCER_ORIGINAL + _CANCER_EXPANDED,
    "dev_expanded":    _DEV_EXPANDED,
    "extra":           list(EXTRA_DATASET_REGISTRY.keys()),
    "expanded":        list(EXPANDED_DATASET_REGISTRY.keys()),
    "sample_cancer":   ["melanoma", "tnbc_brain", "lbm_brain", "bc_ec"],  # small/fast
    "all":             list(ALL_DATASET_REGISTRY.keys()),
}


def resolve_datasets(dataset_keys=None, registry=None):
    """Resolve dataset selection list. Returns ordered keys."""
    if registry is None:
        registry = DATASET_REGISTRY
    if not dataset_keys:
        return list(registry.keys())
    # Allow group names
    expanded = []
    for k in dataset_keys:
        if k in DATASET_GROUPS:
            expanded.extend(DATASET_GROUPS[k])
        elif k in registry:
            expanded.append(k)
        else:
            print(f"  WARNING: unknown dataset key/group '{k}' — skipping")
    return list(dict.fromkeys(expanded))  # deduplicate preserving order
