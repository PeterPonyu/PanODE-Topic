"""Shared constants, style helpers, and truly generic drawing primitives.

This module provides ONLY:
  1. Path constants and model-list constants
  2. Metric-list constants used by multiple figures
  3. Style application (``apply_clean_style``)
  4. Generic low-level drawing primitives (``compute_umap``)
  5. Diagnostic helpers (``check_text_overlaps``)

Figure-specific drawing logic (boxplots, sweep trends, scatter+hulls, etc.)
lives inside each figure module so that modifications to one figure NEVER
affect another.

Data loading functions live in ``data_loaders.py``.
"""

import sys
import numpy as np
import matplotlib as mpl
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Re-export from paper_style so figure modules only need to import from here.
from utils.paper_style import (          # noqa: F401  — re-exports
    apply_style, get_model_order, get_color, MODEL_COLORS,
    MODEL_SHORT_NAMES, get_cell_cmap, sort_df_by_model_order,
    CORE_METRICS, RCPARAMS, METRIC_DIRECTION)

# Re-export key path constants from data_loaders (single source of truth).
from .data_loaders import (              # noqa: F401
    RESULTS_DIR, BIO_RESULTS, DYNAMICS_DIR, OUTPUT_DIR,
    CROSS_DATASETS, NEW_CROSS_DATASETS,
    load_scalability_csv, load_latent_dim_csv, load_warmup_csv,
    load_transfer_csv, load_interpretability_csv, load_external_csv)

# ═══════════════════════════════════════════════════════════════════════════════
# Model-list & dataset constants
# ═══════════════════════════════════════════════════════════════════════════════

PRIOR_MODELS_DPMM = ["DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive"]
PRIOR_MODELS_TOPIC = ["Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive"]

REPRESENTATIVE_DATASETS = ["setty", "endo", "dentate"]

# ═══════════════════════════════════════════════════════════════════════════════
# Sweep-parameter display labels (shared by Fig 3 & Fig 4)
# ═══════════════════════════════════════════════════════════════════════════════

SWEEP_LABELS = {
    "warmup_ratio": "Warmup Ratio",
    "kl_weight":    "KL Weight",
    "latent_dim":   "Latent Dim",
    "encoder_size": "Encoder Size",
    "dropout_rate": "Dropout Rate",
    "lr":           "Learning Rate",
    "epochs":       "Epochs",
    "batch_size":   "Batch Size",
    "weight_decay": "Weight Decay",
    "hvg_top_genes":"HVG Top Genes",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Metric lists (used by Fig 2 and potentially others)
# ═══════════════════════════════════════════════════════════════════════════════

ALL_BOXPLOT_METRICS_CORE = [
    ("NMI",                   "NMI \u2191",            True),
    ("ARI",                   "ARI \u2191",            True),
    ("ASW",                   "ASW \u2191",            True),
    ("DAV",                   "DAV \u2193",            False),
    ("DRE_umap_overall_quality", "DRE UMAP \u2191",   True),
    ("LSE_overall_quality",   "LSE Overall \u2191",    True),
]

# Topic models use a simplex latent space where LSE / DRE / DREX / LSEX
# metrics are not meaningful (inherently lower).  Use a reduced core set.
ALL_BOXPLOT_METRICS_CORE_TOPIC = [
    ("NMI",                   "NMI \u2191",            True),
    ("ARI",                   "ARI \u2191",            True),
    ("ASW",                   "ASW \u2191",            True),
    ("DAV",                   "DAV \u2193",            False),
]

ALL_BOXPLOT_METRICS_EXT = [
    # ── Classical clustering ──
    ("COR",                         "Corr \u2191",              True),
    ("CAL",                         "Cal\u2013H \u2191",        True),
    # ── DRE UMAP projection quality ──
    ("DRE_umap_distance_correlation","DRE UMAP DistCorr \u2191", True),
    # ── DRE tSNE projection quality ──
    ("DRE_tsne_distance_correlation","DRE tSNE DistCorr \u2191", True),
    ("DRE_tsne_overall_quality",    "DRE tSNE Overall \u2191",   True),
    # ── LSE latent-space structure ──
    ("LSE_manifold_dimensionality", "LSE ManDim \u2191",         True),
    ("LSE_spectral_decay_rate",     "LSE SpDecay \u2191",        True),
    ("LSE_participation_ratio",     "LSE PartRat \u2191",        True),
    ("LSE_anisotropy_score",        "LSE Aniso \u2193",          False),
    ("LSE_trajectory_directionality","LSE TrajDir \u2191",       True),
    ("LSE_noise_resilience",        "LSE NoiseR \u2191",         True),
    ("LSE_core_quality",            "LSE Core \u2191",           True),
    # ── DREX extended DR quality ──
    ("DREX_trustworthiness",        "DREX Trust \u2191",         True),
    ("DREX_continuity",             "DREX Cont \u2191",          True),
    ("DREX_distance_spearman",      "DREX Spear \u2191",         True),
    ("DREX_distance_pearson",       "DREX Pearson \u2191",       True),
    ("DREX_local_scale_quality",    "DREX LocScale \u2191",      True),
    ("DREX_neighborhood_symmetry",  "DREX NbrSym \u2191",        True),
    ("DREX_overall_quality",        "DREX Overall \u2191",       True),
    # ── LSEX extended latent structure ──
    ("LSEX_two_hop_connectivity",   "LSEX 2Hop \u2191",         True),
    ("LSEX_radial_concentration",   "LSEX RadConc \u2191",      True),
    ("LSEX_local_curvature",        "LSEX LocCurv \u2191",      True),
    ("LSEX_entropy_stability",      "LSEX Entropy \u2191",      True),
    ("LSEX_overall_quality",        "LSEX Overall \u2191",      True),
]

# Topic series: only classical clustering (COR, CAL) — LSE/DRE not meaningful
ALL_BOXPLOT_METRICS_EXT_TOPIC = [
    ("COR",                         "Corr \u2191",              True),
    ("CAL",                         "Cal\u2013H \u2191",        True),
]


def get_core_metrics(series=None):
    """Return the core metric list (always Topic series)."""
    return ALL_BOXPLOT_METRICS_CORE_TOPIC


def get_ext_metrics(series=None):
    """Return the extended metric list (always Topic series)."""
    return ALL_BOXPLOT_METRICS_EXT_TOPIC

# ═══════════════════════════════════════════════════════════════════════════════
# Style helpers
# ═══════════════════════════════════════════════════════════════════════════════

def apply_clean_style(font_size=None):
    """Apply publication style: Arial font, no bold titles.

    All font sizes are clamped to >=6 pt to meet journal requirements.
    The font size hierarchy follows the constants defined in
    ``subplot_style`` (FONTSIZE_TITLE, FONTSIZE_LABEL, etc.).

    Parameters
    ----------
    font_size : float or None
        If given, override the base font size. Minimum 6 pt.
    """
    from .subplot_style import (
        FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, FONTSIZE_LEGEND)
    apply_style()
    base = max(font_size or 8, 6.0)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica",
                            "Liberation Sans"],
        "axes.titleweight": "normal",
        "figure.titleweight": "normal",
        "font.size": base,
        "axes.titlesize": max(FONTSIZE_TITLE, 6),
        "axes.labelsize": max(FONTSIZE_LABEL, 6),
        "xtick.labelsize": max(FONTSIZE_TICK, 6),
        "ytick.labelsize": max(FONTSIZE_TICK, 6),
        "legend.fontsize": max(FONTSIZE_LEGEND, 6),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic helpers
# ═══════════════════════════════════════════════════════════════════════════════

def clip_extreme_outliers(data_arrays, iqr_mult=6.0):
    """Clip extreme outliers across a list of 1-D arrays using Tukey fences.

    Computes Q1/Q3/IQR over the pooled data, then clips every value to
    [Q1 − iqr_mult*IQR, Q3 + iqr_mult*IQR].  Uses a generous multiplier
    (default 6×) to preserve real spread while removing computational
    artifacts (e.g. CAL = 83 M when all others are < 120 K).

    Parameters
    ----------
    data_arrays : list[np.ndarray]
        Per-model arrays of metric values.
    iqr_mult : float
        IQR multiplier for the fence (default 6×).

    Returns
    -------
    list[np.ndarray]
        Same structure, with extreme values clipped.
    """
    pooled = np.concatenate([a for a in data_arrays if len(a) > 0])
    if len(pooled) < 4:
        return data_arrays
    q1, q3 = np.percentile(pooled, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-12:
        return data_arrays
    lo = q1 - iqr_mult * iqr
    hi = q3 + iqr_mult * iqr
    return [np.clip(a, lo, hi) for a in data_arrays]

def check_text_overlaps(fig, label="", verbose=True, overlap_tol_px=2):
    """Run the full 19-pass Visual Conflict Detector on *fig*.

    This function delegates to ``detect_all_conflicts`` from
    ``visual_conflict_detector.py`` which provides comprehensive
    detection of text overlaps, artist truncation, cross-panel
    spillover, legend conflicts, colorbar issues, and more.

    Backwards-compatible: accepts the same arguments as the old
    text-only checker, so all existing ``gen_fig*.py`` callers work
    unchanged.

    Parameters
    ----------
    overlap_tol_px : int
        Pixels to shrink each bounding box on every side before the
        pairwise overlap test.

    Returns
    -------
    issues : list[dict]
        Each dict has ``type``, ``severity``, ``detail``, ``elements``.
    """
    from .visual_conflict_detector import detect_all_conflicts
    return detect_all_conflicts(
        fig,
        label=label,
        verbose=verbose,
        text_overlap_tol_px=float(overlap_tol_px))


# ═══════════════════════════════════════════════════════════════════════════════
# Generic drawing primitives (used by >=2 figure modules)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_umap(latent, n_neighbors=15, min_dist=0.5):
    """Compute 2-D UMAP embedding from a latent matrix."""
    try:
        from umap import UMAP
        return UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                    random_state=42).fit_transform(latent.astype(np.float32))
    except ImportError:
        import scanpy as sc
        adata_tmp = sc.AnnData(latent.astype(np.float32))
        sc.pp.neighbors(adata_tmp, use_rep="X", n_neighbors=n_neighbors)
        sc.tl.umap(adata_tmp, min_dist=min_dist)
        return adata_tmp.obsm["X_umap"]
