"""eval_lib.metrics.battery — Unified metric battery and experiment constants.

Provides a single-call ``compute_metrics()`` that runs all 6 metric suites
(clustering, DRE-UMAP, DRE-tSNE, LSE, DREX, LSEX) and returns a flat dict
whose keys match ``METRIC_COLUMNS``.

Also provides:
- ``compute_latent_diagnostics()`` — collapse / redundancy diagnostics
- ``convergence_diagnostics()``    — training convergence from loss history
- ``DataSplitter``                 — index-based train/val/test splitter
- ``METRIC_COLUMNS``               — canonical column order (38 columns)
- ``METRIC_GROUPS``                 — metric grouping for visualisation

All functions are self-contained — no project-specific imports.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score)
from sklearn.model_selection import train_test_split

from .dre import evaluate_dimensionality_reduction
from .drex import evaluate_extended_dimensionality_reduction
from .lse import evaluate_single_cell_latent_space
from .lsex import evaluate_extended_latent_space


# ═══════════════════════════════════════════════════════════════════════════════
# Index-based data splitter
# ═══════════════════════════════════════════════════════════════════════════════

class DataSplitter:
    """Index-based train / val / test splitter (70 / 15 / 15).

    Produces integer index arrays — not PyTorch DataLoaders.
    Projects should build loaders from these indices.

    Attributes
    ----------
    train_idx, val_idx, test_idx : np.ndarray
        Integer arrays of sample indices.
    train_val_idx : np.ndarray
        Combined train + val indices.
    """

    def __init__(self, n_samples: int, test_size: float = 0.15,
                 val_size: float = 0.15, random_state: int = 42):
        self.n_samples = n_samples
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.train_val_size = 1 - test_size
        self.val_size_relative = val_size / self.train_val_size
        self._create_splits()

    def _create_splits(self):
        indices = np.arange(self.n_samples)
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_size,
            random_state=self.random_state, shuffle=True)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=self.val_size_relative,
            random_state=self.random_state, shuffle=True)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.train_val_idx = train_val_idx

    def get_validation_fraction(self) -> float:
        """Fraction of train+val reserved for validation."""
        return self.val_size_relative


# ═══════════════════════════════════════════════════════════════════════════════
# Metric column definitions (canonical order — 38 columns)
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_COLUMNS = [
    # Clustering
    "NMI", "ARI", "ASW", "DAV", "CAL", "COR",
    # DRE UMAP
    "DRE_umap_distance_correlation", "DRE_umap_Q_local",
    "DRE_umap_Q_global", "DRE_umap_K_max", "DRE_umap_overall_quality",
    # DRE t-SNE
    "DRE_tsne_distance_correlation", "DRE_tsne_Q_local",
    "DRE_tsne_Q_global", "DRE_tsne_K_max", "DRE_tsne_overall_quality",
    # LSE intrinsic
    "LSE_manifold_dimensionality", "LSE_spectral_decay_rate",
    "LSE_participation_ratio", "LSE_anisotropy_score",
    "LSE_trajectory_directionality", "LSE_noise_resilience",
    "LSE_core_quality", "LSE_overall_quality",
    # DREX
    "DREX_trustworthiness", "DREX_continuity", "DREX_distance_spearman",
    "DREX_distance_pearson", "DREX_local_scale_quality",
    "DREX_neighborhood_symmetry", "DREX_overall_quality",
    # LSEX
    "LSEX_two_hop_connectivity", "LSEX_radial_concentration",
    "LSEX_local_curvature", "LSEX_entropy_stability", "LSEX_overall_quality",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Metric group definitions (for visualisation)
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_GROUPS = {
    "clustering": {
        "metrics": ["NMI", "ARI", "ASW", "DAV", "CAL", "COR"],
        "display": {
            "NMI": "NMI", "ARI": "ARI", "ASW": "ASW",
            "DAV": "DAV", "CAL": "CAL", "COR": "COR",
        },
        "ncols": 6,
    },
    "dre_umap": {
        "metrics": [
            "DRE_umap_distance_correlation", "DRE_umap_Q_local",
            "DRE_umap_Q_global", "DRE_umap_K_max",
            "DRE_umap_overall_quality",
        ],
        "display": {
            "DRE_umap_distance_correlation": "Dist. corr.",
            "DRE_umap_Q_local":              "Q_local",
            "DRE_umap_Q_global":             "Q_global",
            "DRE_umap_K_max":                "K_max",
            "DRE_umap_overall_quality":      "Overall (umap)",
        },
        "ncols": 5,
    },
    "dre_tsne": {
        "metrics": [
            "DRE_tsne_distance_correlation", "DRE_tsne_Q_local",
            "DRE_tsne_Q_global", "DRE_tsne_K_max",
            "DRE_tsne_overall_quality",
        ],
        "display": {
            "DRE_tsne_distance_correlation": "Dist. corr.",
            "DRE_tsne_Q_local":              "Q_local",
            "DRE_tsne_Q_global":             "Q_global",
            "DRE_tsne_K_max":                "K_max",
            "DRE_tsne_overall_quality":      "Overall (tsne)",
        },
        "ncols": 5,
    },
    "lse_intrinsic": {
        "metrics": [
            "LSE_manifold_dimensionality", "LSE_spectral_decay_rate",
            "LSE_participation_ratio", "LSE_anisotropy_score",
            "LSE_core_quality", "LSE_trajectory_directionality",
            "LSE_noise_resilience", "LSE_overall_quality",
        ],
        "display": {
            "LSE_manifold_dimensionality":   "Man. dim.",
            "LSE_spectral_decay_rate":       "Spec. decay",
            "LSE_participation_ratio":       "Part. ratio",
            "LSE_anisotropy_score":          "Anisotropy",
            "LSE_core_quality":              "Core qual.",
            "LSE_trajectory_directionality": "Traj. dir.",
            "LSE_noise_resilience":          "Noise res.",
            "LSE_overall_quality":           "Overall (intrin)",
        },
        "ncols": 8,
    },
    "drex": {
        "metrics": [
            "DREX_trustworthiness", "DREX_continuity",
            "DREX_distance_spearman", "DREX_distance_pearson",
            "DREX_local_scale_quality", "DREX_neighborhood_symmetry",
            "DREX_overall_quality",
        ],
        "display": {
            "DREX_trustworthiness":       "Trust.",
            "DREX_continuity":            "Contin.",
            "DREX_distance_spearman":     "Dist. Spear.",
            "DREX_distance_pearson":      "Dist. Pear.",
            "DREX_local_scale_quality":   "Local scale",
            "DREX_neighborhood_symmetry": "Neigh. sym.",
            "DREX_overall_quality":       "Overall (drex)",
        },
        "ncols": 7,
    },
    "lsex": {
        "metrics": [
            "LSEX_two_hop_connectivity", "LSEX_radial_concentration",
            "LSEX_local_curvature", "LSEX_entropy_stability",
            "LSEX_overall_quality",
        ],
        "display": {
            "LSEX_two_hop_connectivity":  "2-hop conn.",
            "LSEX_radial_concentration":  "Radial conc.",
            "LSEX_local_curvature":       "Local curv.",
            "LSEX_entropy_stability":     "Entr. stab.",
            "LSEX_overall_quality":       "Overall (lsex)",
        },
        "ncols": 5,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Publication metric groups — K_max excluded (diagnostic, not comparable)
# ═══════════════════════════════════════════════════════════════════════════════

PUBLICATION_METRIC_GROUPS = {
    "clustering": {
        "metrics": ["NMI", "ARI", "ASW", "DAV", "CAL", "COR"],
        "display": {
            "NMI": "NMI", "ARI": "ARI", "ASW": "ASW",
            "DAV": "DAV", "CAL": "CAL", "COR": "COR",
        },
        "ncols": 6,
    },
    "dre_umap": {
        "metrics": [
            "DRE_umap_distance_correlation", "DRE_umap_Q_local",
            "DRE_umap_Q_global", "DRE_umap_overall_quality",
        ],
        "display": {
            "DRE_umap_distance_correlation": "DC (umap)",
            "DRE_umap_Q_local":              "QL (umap)",
            "DRE_umap_Q_global":             "QG (umap)",
            "DRE_umap_overall_quality":      "OV (umap)",
        },
        "ncols": 4,
    },
    "dre_tsne": {
        "metrics": [
            "DRE_tsne_distance_correlation", "DRE_tsne_Q_local",
            "DRE_tsne_Q_global", "DRE_tsne_overall_quality",
        ],
        "display": {
            "DRE_tsne_distance_correlation": "DC (tsne)",
            "DRE_tsne_Q_local":              "QL (tsne)",
            "DRE_tsne_Q_global":             "QG (tsne)",
            "DRE_tsne_overall_quality":      "OV (tsne)",
        },
        "ncols": 4,
    },
    "lse_intrinsic": {
        "metrics": [
            "LSE_manifold_dimensionality", "LSE_spectral_decay_rate",
            "LSE_participation_ratio", "LSE_anisotropy_score",
            "LSE_core_quality", "LSE_trajectory_directionality",
            "LSE_noise_resilience", "LSE_overall_quality",
        ],
        "display": {
            "LSE_manifold_dimensionality":   "Man. dim.",
            "LSE_spectral_decay_rate":       "Spec. decay",
            "LSE_participation_ratio":       "Part. ratio",
            "LSE_anisotropy_score":          "Anisotropy",
            "LSE_core_quality":              "Core qual.",
            "LSE_trajectory_directionality": "Traj. dir.",
            "LSE_noise_resilience":          "Noise res.",
            "LSE_overall_quality":           "Overall (intrin)",
        },
        "ncols": 8,
    },
    "drex": {
        "metrics": [
            "DREX_trustworthiness", "DREX_continuity",
            "DREX_distance_spearman", "DREX_distance_pearson",
            "DREX_local_scale_quality", "DREX_neighborhood_symmetry",
            "DREX_overall_quality",
        ],
        "display": {
            "DREX_trustworthiness":       "Trust.",
            "DREX_continuity":            "Contin.",
            "DREX_distance_spearman":     "Dist. Spear.",
            "DREX_distance_pearson":      "Dist. Pear.",
            "DREX_local_scale_quality":   "Local scale",
            "DREX_neighborhood_symmetry": "Neigh. sym.",
            "DREX_overall_quality":       "OV (drex)",
        },
        "ncols": 7,
    },
    "lsex": {
        "metrics": [
            "LSEX_two_hop_connectivity", "LSEX_radial_concentration",
            "LSEX_local_curvature", "LSEX_entropy_stability",
            "LSEX_overall_quality",
        ],
        "display": {
            "LSEX_two_hop_connectivity":  "2-hop conn.",
            "LSEX_radial_concentration":  "Radial conc.",
            "LSEX_local_curvature":       "Local curv.",
            "LSEX_entropy_stability":     "Entr. stab.",
            "LSEX_overall_quality":       "OV (lsex)",
        },
        "ncols": 5,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

_DRE_KEYS = ('distance_correlation', 'Q_local', 'Q_global',
             'K_max', 'overall_quality')

_LSE_KEYS = ('manifold_dimensionality', 'spectral_decay_rate',
             'participation_ratio', 'anisotropy_score',
             'trajectory_directionality', 'noise_resilience',
             'core_quality', 'overall_quality')

_DREX_KEYS = ('trustworthiness', 'continuity', 'distance_spearman',
              'distance_pearson', 'local_scale_quality',
              'neighborhood_symmetry', 'overall_quality')

_LSEX_KEYS = ('two_hop_connectivity', 'radial_concentration',
              'local_curvature', 'entropy_stability', 'overall_quality')

_LSEX_RAW_MAP = {
    'two_hop_connectivity': 'two_hop_connectivity',
    'radial_concentration': 'radial_concentration_quality',
    'local_curvature':      'local_curvature_linearity',
    'entropy_stability':    'neighbor_entropy_stability',
    'overall_quality':      'overall_quality',
}


def _safe_dre(metrics, prefix, source, target, k):
    try:
        dre = evaluate_dimensionality_reduction(source, target, k=k, verbose=False)
        for key in _DRE_KEYS:
            metrics[f'{prefix}_{key}'] = dre.get(key, np.nan)
    except Exception:
        for key in _DRE_KEYS:
            metrics[f'{prefix}_{key}'] = np.nan


def _safe_lse(metrics, latent, data_type):
    try:
        lse = evaluate_single_cell_latent_space(latent, data_type=data_type,
                                                verbose=False)
        for key in _LSE_KEYS:
            metrics[f'LSE_{key}'] = lse.get(key, np.nan)
    except Exception:
        for key in _LSE_KEYS:
            metrics[f'LSE_{key}'] = np.nan


def _safe_drex(metrics, x_high, x_low, k):
    try:
        drex = evaluate_extended_dimensionality_reduction(x_high, x_low,
                                                         n_neighbors=k)
        for key in _DREX_KEYS:
            metrics[f'DREX_{key}'] = drex.get(key, np.nan)
    except Exception:
        for key in _DREX_KEYS:
            metrics[f'DREX_{key}'] = np.nan


def _safe_lsex(metrics, latent, k):
    try:
        lsex = evaluate_extended_latent_space(latent, n_neighbors=k)
        for key, raw_key in _LSEX_RAW_MAP.items():
            metrics[f'LSEX_{key}'] = lsex.get(raw_key, np.nan)
    except Exception:
        for key in _LSEX_KEYS:
            metrics[f'LSEX_{key}'] = np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Public API: compute_metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(latent: np.ndarray, labels, *,
                    data_type: str = "cluster", dre_k: int = 15) -> dict:
    """Run the full 6-suite metric battery on a latent embedding.

    Parameters
    ----------
    latent : np.ndarray, shape (n_cells, latent_dim)
    labels : array-like
        Ground-truth cluster / cell-type labels.
    data_type : str
        ``"cluster"`` | ``"trajectory"`` | ``"mixed"`` — affects LSE.
    dre_k : int
        Neighbourhood size for DRE / DREX / LSEX.

    Returns
    -------
    dict
        Keys match ``METRIC_COLUMNS`` (NaN for any that fail).
    """
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, random_state=42,
                  n_init=10).fit_predict(latent)

    metrics: dict = {
        'NMI': normalized_mutual_info_score(labels, pred),
        'ARI': adjusted_rand_score(labels, pred),
    }

    try:
        metrics['ASW'] = (silhouette_score(latent, pred)
                          if len(np.unique(pred)) > 1 else np.nan)
    except Exception:
        metrics['ASW'] = np.nan

    try:
        metrics['DAV'] = davies_bouldin_score(latent, pred)
    except Exception:
        metrics['DAV'] = np.nan

    try:
        metrics['CAL'] = calinski_harabasz_score(latent, pred)
    except Exception:
        metrics['CAL'] = np.nan

    try:
        acorr = np.abs(np.corrcoef(latent.T))
        cor_val = acorr.sum(axis=1).mean() - 1
        metrics['COR'] = cor_val if np.isfinite(cor_val) else np.nan
    except Exception:
        metrics['COR'] = np.nan

    # 2-D projections for DR quality evaluation
    umap_2d = tsne_2d = None
    try:
        adata_eval = sc.AnnData(latent.astype(np.float32))
        sc.pp.neighbors(adata_eval, use_rep='X')
        sc.tl.umap(adata_eval)
        umap_2d = adata_eval.obsm['X_umap']
        sc.tl.tsne(adata_eval, use_rep='X')
        tsne_2d = adata_eval.obsm['X_tsne']
    except Exception:
        pass

    if umap_2d is not None:
        _safe_dre(metrics, 'DRE_umap', latent, umap_2d, dre_k)
    else:
        for k in _DRE_KEYS:
            metrics[f'DRE_umap_{k}'] = np.nan

    if tsne_2d is not None:
        _safe_dre(metrics, 'DRE_tsne', latent, tsne_2d, dre_k)
    else:
        for k in _DRE_KEYS:
            metrics[f'DRE_tsne_{k}'] = np.nan

    _safe_lse(metrics, latent, data_type)

    if umap_2d is not None:
        _safe_drex(metrics, latent, umap_2d, dre_k)
    else:
        for k in _DREX_KEYS:
            metrics[f'DREX_{k}'] = np.nan

    _safe_lsex(metrics, latent, dre_k)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Latent diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_latent_diagnostics(latent: np.ndarray,
                               max_samples: int = 2000) -> dict:
    """Collapse / redundancy diagnostics for a latent embedding."""
    if latent is None or len(latent) == 0:
        return {k: np.nan for k in (
            'latent_mean_norm', 'latent_std_mean', 'latent_std_min',
            'latent_std_max', 'latent_var_mean', 'latent_var_min',
            'latent_var_max', 'latent_near_zero_dims',
            'latent_pairwise_dist_mean', 'latent_pairwise_dist_std')}

    z = np.asarray(latent)
    std = z.std(axis=0)
    var = z.var(axis=0)

    n = z.shape[0]
    z_sub = z[np.random.choice(n, size=min(n, max_samples), replace=False)]

    try:
        dists = pdist(z_sub, metric='euclidean')
        dist_mean = float(np.mean(dists)) if dists.size else np.nan
        dist_std = float(np.std(dists)) if dists.size else np.nan
    except Exception:
        dist_mean = dist_std = np.nan

    return {
        'latent_mean_norm': float(np.linalg.norm(z.mean(axis=0))),
        'latent_std_mean': float(std.mean()),
        'latent_std_min': float(std.min()),
        'latent_std_max': float(std.max()),
        'latent_var_mean': float(var.mean()),
        'latent_var_min': float(var.min()),
        'latent_var_max': float(var.max()),
        'latent_near_zero_dims': int((std < 1e-3).sum()),
        'latent_pairwise_dist_mean': dist_mean,
        'latent_pairwise_dist_std': dist_std,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def convergence_diagnostics(history: dict, window: int = 50) -> dict:
    """Assess training convergence from loss history.

    Returns
    -------
    dict
        ``recon_final``, ``recon_rel_change_pct``, ``converged`` (bool), etc.
        Empty dict if insufficient history.
    """
    train_loss = history.get('train_loss', [])
    if len(train_loss) < window + 1:
        return {}

    total_final = float(np.mean(train_loss[-window // 5:]))
    recon = history.get('recon_loss', history.get('train_recon', train_loss))
    recon_final = float(np.mean(recon[-window // 5:]))
    recon_prev = float(np.mean(recon[-(window + window // 5):-window]))
    recon_pct = ((recon_final - recon_prev) /
                 max(abs(recon_prev), 1e-8)) * 100

    result = {
        'recon_final': round(recon_final, 6),
        'recon_prev_window': round(recon_prev, 6),
        'recon_rel_change_pct': round(recon_pct, 3),
        'total_loss_final': round(total_final, 6),
        'converged': abs(recon_pct) < 1.0,
        'window': window,
    }

    kl = history.get('kl_loss', history.get('train_kl'))
    if kl is not None and len(kl) >= window + 1:
        kl_final = float(np.mean(kl[-window // 5:]))
        kl_prev = float(np.mean(kl[-(window + window // 5):-window]))
        kl_pct = ((kl_final - kl_prev) / max(abs(kl_prev), 1e-8)) * 100
        result.update(kl_final=round(kl_final, 6),
                      kl_prev_window=round(kl_prev, 6),
                      kl_rel_change_pct=round(kl_pct, 3))

    return result
