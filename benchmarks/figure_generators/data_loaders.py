"""Centralised data-loading helpers for paper figure generation.

Every CSV / NPZ / JSON loading function lives here.  Figure modules import
only the loaders they need — no drawing logic, no matplotlib.
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Paths  (single source of truth — imported from benchmarks.config)
# ═══════════════════════════════════════════════════════════════════════════════

from benchmarks.config import (
    DEFAULT_OUTPUT_DIR as RESULTS_DIR,
    BIO_RESULTS_DIR as BIO_RESULTS,
    DYNAMICS_DIR,
    PAPER_FIGURES_DIR as OUTPUT_DIR)

ROOT = Path(__file__).resolve().parent.parent.parent

# Legacy timestamp constants (loaders now auto-discover latest)
BASE_TS = {"dpmm": "20260211_232144", "topic": "20260211_232852"}
SENS_TS = "20260212_161246"
TRAIN_TS = "20260212_154957"
PREPROC_TS = "20260212_185159"
CROSS_TS = "20260213_123600"
CROSS_DATASETS = {
    "setty": "20260212_152000", "lung": "20260212_152000",
    "endo": "20260212_152000", "dentate": "20260213_121910",
    "hemato": "20260213_122440", "pansci_muscle": "20260213_123015",
}
NEW_CROSS_DATASETS = [
    "blood_aged", "hesc", "retina", "teeth", "pituitary", "pansci_tcell",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def safe_model_name(name):
    """Filesystem-safe model name."""
    return str(name).replace("/", "_").replace(" ", "_")


def parse_sweep_value(model_name):
    """Extract numeric sweep value from a model name like 'Topic-Base(kl=0.1)'."""
    m = re.search(r"=([^\)]+)\)", str(model_name))
    if not m:
        return np.nan
    txt = m.group(1)
    try:
        return float(txt)
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# CSV loaders — auto-discover latest timestamp
# ═══════════════════════════════════════════════════════════════════════════════

def load_base_csv(series="topic"):
    """Load latest base-ablation CSV for *series*."""
    csv_dir = RESULTS_DIR / "base" / "csv" / series
    cand = sorted(csv_dir.glob("results_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if not cand:
        ts = BASE_TS[series]
        fallback = csv_dir / f"results_setty_3000c_ep600_lr1e-3_{ts}.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                f"No base CSV found in {csv_dir} — run base benchmark first")
        return pd.read_csv(fallback)
    best = [p for p in cand if "with_gse" not in p.name]
    return pd.read_csv(best[0] if best else cand[0])


def load_sensitivity_csv(series="topic"):
    csv_dir = RESULTS_DIR / "sensitivity" / "csv" / series
    cand = sorted(csv_dir.glob("results_sensitivity_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if cand:
        return pd.read_csv(cand[0])
    fallback = csv_dir / f"results_sensitivity_{SENS_TS}.csv"
    if not fallback.exists():
        raise FileNotFoundError(
            f"No sensitivity CSV found in {csv_dir} — run sensitivity sweep first")
    return pd.read_csv(fallback)


def load_training_csv(series="topic"):
    csv_dir = RESULTS_DIR / "training" / "csv" / series
    cand = sorted(csv_dir.glob("results_training_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if cand:
        return pd.read_csv(cand[0])
    fallback = csv_dir / f"results_training_{TRAIN_TS}.csv"
    if not fallback.exists():
        raise FileNotFoundError(
            f"No training CSV found in {csv_dir} — run training sweep first")
    return pd.read_csv(fallback)


def load_preprocessing_csv(series="topic"):
    """Load preprocessing sweep CSV, merging all available files.

    Multiple preprocessing runs may cover different subsets of
    datasets.  This merges all available CSVs, keeping the latest
    entry per (Dataset, Sweep, SweepVal) combination.
    """
    csv_dir = RESULTS_DIR / "preprocessing" / "csv" / series
    cand = sorted(csv_dir.glob("results_preprocessing_*.csv"),
                  key=lambda x: x.name)  # chronological order
    if not cand:
        return pd.read_csv(csv_dir / f"results_preprocessing_{PREPROC_TS}.csv")
    frames = []
    for p in cand:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        return pd.read_csv(cand[-1])
    merged = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in ["Dataset", "Sweep", "SweepVal"] if c in merged.columns]
    if dedup_cols:
        merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    return merged


def load_crossdata_combined(prefer_multiseed=True):
    """Load cross-dataset combined CSV, preferring the 5-seed version.

    Look-up order:
      1. ``results_combined_5seed.csv``  (720 rows, seeds {0,1,2,3,42})
      2. ``results_combined_3seed.csv``  (912 rows, seeds {0,1,42})
      3. ``results_combined_multiseed.csv`` (288 rows, seeds {0,42})
      4. Any ``results_combined_*.csv`` merged together

    When ``prefer_multiseed`` is True (default), the function returns the
    full multi-seed DataFrame with a ``seed`` column.  When False, it
    collapses to per-(Model, Dataset) means for backward compatibility.
    """
    csv_dir = RESULTS_DIR / "crossdata" / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError("No crossdata CSV dir found.")

    for fname in ["results_combined_5seed.csv",
                  "results_combined_3seed.csv",
                  "results_combined_multiseed.csv"]:
        p = csv_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            if not prefer_multiseed and "seed" in df.columns:
                num_cols = df.select_dtypes(include="number").columns
                group_cols = ["Model", "Dataset"]
                group_cols = [c for c in group_cols if c in df.columns]
                if group_cols:
                    df = df.groupby(group_cols, as_index=False)[num_cols.tolist()].mean()
            return df

    # Fallback: merge all combined CSVs
    cand = sorted(csv_dir.glob("results_combined_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if not cand:
        raise FileNotFoundError("No crossdata combined CSV found.")
    frames = [pd.read_csv(f) for f in cand]
    merged = pd.concat(frames, ignore_index=True)
    if "Model" in merged.columns and "Dataset" in merged.columns:
        merged = merged.drop_duplicates(subset=["Model", "Dataset"], keep="first")
    return merged


def load_crossdata_per_dataset():
    """Load multi-seed crossdata CSV split by dataset, preserving all seeds.

    Returns a dict ``{dataset_name: DataFrame}`` where each DataFrame
    has one row per (Model, seed) pair — typically 5 rows per model when
    the 5-seed CSV is available.  This provides seed-level variance for
    boxplots (n = 5 × 12 = 60 points per model across all datasets).

    Falls back to the legacy per-file auto-discovery only when no combined
    multi-seed CSV exists.
    """
    # ── Preferred: use the 5-seed combined CSV ──
    try:
        combined = load_crossdata_combined(prefer_multiseed=True)
        if "Dataset" in combined.columns:
            dfs = {}
            for ds, grp in combined.groupby("Dataset"):
                dfs[str(ds)] = grp.reset_index(drop=True)
            if dfs:
                return dfs
    except FileNotFoundError:
        pass

    # ── Fallback: legacy per-file auto-discovery (dedup by Model) ──
    dfs = {}
    csv_dir = RESULTS_DIR / "crossdata" / "csv"
    if not csv_dir.exists():
        return dfs
    files_by_ds = {}
    for f in csv_dir.glob("results_*_*.csv"):
        name = f.stem
        parts = name.split("_")
        if len(parts) < 3 or parts[1] == "combined":
            continue
        if (len(parts) >= 4
                and parts[-2].isdigit() and parts[-1].isdigit()
                and len(parts[-2]) == 8 and len(parts[-1]) == 6):
            ds = "_".join(parts[1:-2])
        else:
            ds = "_".join(parts[1:-1])
        files_by_ds.setdefault(ds, []).append(f)
    for ds, file_list in files_by_ds.items():
        file_list = sorted(file_list)
        frames = [pd.read_csv(f) for f in file_list]
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["Model"], keep="last")
        dfs[ds] = merged
    return dfs


def load_pairwise_wilcoxon():
    """Load pairwise Wilcoxon signed-rank test CSV.

    Returns
    -------
    pd.DataFrame or None
        Columns: Structured, Pure, Metric, N_pairs, Mean_diff, Wins,
        Losses, Ties, W_stat, p_value, Significant_005, Cliffs_delta,
        Effect_size.
    """
    stat_dir = RESULTS_DIR / "statistical_exports"
    for fname in ["pairwise_wilcoxon.csv", "pairwise_wilcoxon_2seed.csv"]:
        p = stat_dir / fname
        if p.exists():
            return pd.read_csv(p)
    return None


def load_pairwise_wilcoxon_external():
    """Load external-model pairwise Wilcoxon signed-rank test CSV.

    Returns
    -------
    pd.DataFrame or None
        Same schema as :func:`load_pairwise_wilcoxon` but with
        ``Structured`` = Best-DPMM or Best-Topic, ``Pure`` = external model.
    """
    stat_dir = RESULTS_DIR / "statistical_exports"
    p = stat_dir / "pairwise_wilcoxon_external.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Latent / NPZ loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_cross_latent(model_name, dataset):
    """Load latest crossdata latent npz for one model on one dataset."""
    safe = safe_model_name(model_name)
    lat_dir = RESULTS_DIR / "crossdata" / "latents" / dataset
    if not lat_dir.exists():
        return None
    files = sorted(lat_dir.glob(f"{safe}_{dataset}_*.npz"), reverse=True)
    if not files:
        return None
    data = np.load(files[0])
    return data.get("latent")


def load_joint_latent(model_name):
    """Load the latest joint-training latent npz for *model_name*.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or (None, None)
        ``(latent, dataset_labels)`` where *latent* is shape (N, D) and
        *dataset_labels* is a string array of length N giving the source
        dataset for each cell.  Returns ``(None, None)`` when no file exists.
    """
    safe = safe_model_name(model_name)
    lat_dir = RESULTS_DIR / "joint" / "latents"
    if not lat_dir.exists():
        return None, None
    # Files are named  {safe}_joint_{timestamp}.npz
    files = sorted(lat_dir.glob(f"{safe}_joint_*.npz"), reverse=True)
    if not files:
        return None, None
    data = np.load(files[0], allow_pickle=True)
    latent = data.get("latent")
    labels = data.get("dataset_labels")
    if labels is not None:
        labels = np.array(labels, dtype=str)
    return latent, labels


def load_sweep_latents(sweep_type, series="topic", model_names=None, n_select=4,
                       dataset_filter=None):
    """Load latent .npz files for sweep models.

    Parameters
    ----------
    dataset_filter : str or None
        If given, only load latents whose filename contains this dataset tag.
    """
    if sweep_type not in {"training", "sensitivity", "preprocessing"}:
        return []
    lat_dir = RESULTS_DIR / sweep_type / "latents" / series
    if not lat_dir.exists():
        return []
    loaded = []
    all_files = sorted(lat_dir.glob("*.npz"),
                       key=lambda p: p.name, reverse=True)

    # Apply per-dataset filter
    if dataset_filter:
        all_files = [f for f in all_files if f"_{dataset_filter}_" in f.name]

    if model_names:
        for m in model_names:
            safe = safe_model_name(m)
            candidates = [f for f in all_files if f.name.startswith(safe)]
            if not candidates:
                continue
            data = np.load(candidates[0])
            latent = data.get("latent")
            if latent is not None:
                loaded.append((m, latent))
        if loaded:
            loaded = sorted(loaded, key=lambda x: parse_sweep_value(x[0]))
            return loaded
    for f in all_files:
        data = np.load(f)
        latent = data.get("latent")
        if latent is None:
            continue
        name = f.stem
        name = re.sub(r"_\d{8}_\d{6}$", "", name)
        name = re.sub(r"_(sensitivity|training|preprocessing)$", "", name)
        name = re.sub(
            r"_(setty|lung|endo|dentate|hemato|pansci_muscle"
            r"|blood_aged|hesc|retina|teeth|pituitary|pansci_tcell)$",
            "", name)
        loaded.append((name, latent))
    if len(loaded) > n_select:
        idx = np.linspace(0, len(loaded) - 1, n_select, dtype=int)
        loaded = [loaded[i] for i in idx]
    return sorted(loaded, key=lambda x: parse_sweep_value(x[0]))


def load_dynamics_history(model_name, dataset="setty"):
    """Load training dynamics history JSON for a model.

    Searches in order:
      1. DYNAMICS_DIR/{model_name}/  (new per-model layout)
      2. DYNAMICS_DIR/              (legacy flat layout)
      3. RESULTS_DIR/models/        (fallback)
    """
    pattern = f"{model_name}_{dataset}*_history.json"

    # New per-model subdirectory
    model_dir = DYNAMICS_DIR / model_name.replace("/", "_")
    if model_dir.is_dir():
        candidates = sorted(model_dir.glob(pattern), reverse=True)
        if candidates:
            with open(candidates[0]) as f:
                return json.load(f)

    # Legacy flat directory
    candidates = sorted(DYNAMICS_DIR.glob(pattern), reverse=True)
    if candidates:
        with open(candidates[0]) as f:
            return json.load(f)

    # Fallback to benchmark models dir
    for d in [RESULTS_DIR / "models"]:
        candidates = sorted(d.glob(f"{model_name}*_history.json"),
                            reverse=True)
        if candidates:
            with open(candidates[0]) as f:
                return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# New experiment loaders (reviewer revisions)
# ═══════════════════════════════════════════════════════════════════════════════

def load_scalability_csv():
    """Load scalability experiment CSV (runtime/memory vs. cell count)."""
    csv_dir = RESULTS_DIR / "scalability" / "csv"
    cand = sorted(csv_dir.glob("scalability_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if cand:
        return pd.read_csv(cand[0])
    return None


def load_latent_dim_csv():
    """Load latent dimension sweep CSV."""
    csv_dir = RESULTS_DIR / "latent_dim" / "csv"
    cand = sorted(csv_dir.glob("latent_dim_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if cand:
        return pd.read_csv(cand[0])
    return None


def load_warmup_csv():
    """Load warmup ablation CSV."""
    csv_dir = RESULTS_DIR / "warmup" / "csv"
    cand = sorted(csv_dir.glob("warmup_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if cand:
        return pd.read_csv(cand[0])
    return None


def load_transfer_csv():
    """Load cross-dataset transfer CSV(s) and merge them."""
    csv_dir = RESULTS_DIR / "transfer" / "csv"
    cand = sorted(csv_dir.glob("transfer_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if not cand:
        return None
    frames = [pd.read_csv(f) for f in cand]
    return pd.concat(frames, ignore_index=True)


def load_interpretability_csv():
    """Load interpretability / gene specificity index CSV."""
    csv_dir = RESULTS_DIR / "interpretability" / "csv"
    cand = sorted(csv_dir.glob("interpretability_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if cand:
        return pd.read_csv(cand[0])
    return None


def load_external_csv():
    """Load external model benchmark combined CSV."""
    csv_dir = RESULTS_DIR / "external" / "csv"
    cand = sorted(csv_dir.glob("results_combined_*.csv"),
                  key=lambda x: x.name, reverse=True)
    if cand:
        return pd.read_csv(cand[0])
    return None
