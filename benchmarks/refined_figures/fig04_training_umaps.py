"""Refined Figure 4 — Training / Sweep UMAPs (multi-panel composed figure).

For each sweep parameter × representative dataset, produces KMeans-colored
UMAP embedding plots showing how latent geometry evolves across sweep values.

Data source: benchmarks/benchmark_results/sensitivity/csv/{series}/
             benchmarks/benchmark_results/training/csv/{series}/
             benchmarks/benchmark_results/preprocessing/csv/{series}/
             + latent NPZ files

Usage:
    python -m benchmarks.refined_figures.fig04_training_umaps --series dpmm
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion)
from benchmarks.figure_generators.common import (
    compute_umap, REPRESENTATIVE_DATASETS)
from benchmarks.figure_generators.data_loaders import (
    load_sensitivity_csv, load_training_csv, load_preprocessing_csv,
    load_sweep_latents, parse_sweep_value)

DPI = 300

_SENSITIVITY_PARAMS = {
    "dpmm":  ["warmup_ratio", "latent_dim", "encoder_size", "dropout_rate"],
    "topic": ["kl_weight",    "latent_dim", "encoder_size", "dropout_rate"],
}


def _key_params_by_source(series):
    return {
        "sensitivity": _SENSITIVITY_PARAMS.get(series,
                                               _SENSITIVITY_PARAMS["dpmm"]),
        "training":    ["lr", "epochs"],
        "preprocessing": ["hvg_top_genes"],
    }


def _draw_kmeans_umap(ax, latent, title, n_clusters=8):
    """Draw a KMeans-colored UMAP on given axes."""
    if latent is None or len(latent) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
        return
    if len(latent) > 5000:
        idx = np.random.RandomState(0).choice(len(latent), 5000, replace=False)
        latent = latent[idx]
    emb = compute_umap(latent)
    k = min(n_clusters, len(latent))
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(latent)
    cmap = plt.colormaps["tab10"]
    colors = [cmap(l / max(k - 1, 1)) for l in labels]
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=1.5, alpha=0.55,
               rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=2, loc="left", fontweight="normal")
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)


def generate(series, out_dir):
    """Generate refined Figure 4."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    params_by_source = _key_params_by_source(series)
    loaders = {
        "sensitivity":    load_sensitivity_csv,
        "training":       load_training_csv,
        "preprocessing":  load_preprocessing_csv,
    }

    # Collect (source, param, sweep_values) tuples that have latent data
    panels = []
    for source, params in params_by_source.items():
        try:
            df = loaders[source](series)
            if "Series" in df.columns:
                df = df[df["Series"] == series]
        except Exception:
            continue
        for param in params:
            if "Sweep" in df.columns:
                sub = df[df["Sweep"] == param]
            else:
                continue
            if "SweepVal" not in sub.columns or len(sub) < 2:
                continue
            vals = sorted(sub["SweepVal"].dropna().unique(),
                          key=lambda x: float(x) if pd.notna(pd.to_numeric(x, errors="coerce")) else 0)
            # Only keep first 4 sweep values for space
            vals = vals[:4]
            panels.append((source, param, vals))

    if not panels:
        print("    No latent data for sweep UMAPs — skipping Fig 4")
        return

    ds = REPRESENTATIVE_DATASETS[0] if REPRESENTATIVE_DATASETS else "setty"
    n_rows = len(panels)
    n_cols = max(len(p[2]) for p in panels)
    figw, figh = 17.0, 3.5 * n_rows + 0.5

    fig = plt.figure(figsize=(figw, figh))
    root = bind_figure_region(fig, (0.04, 0.03, 0.97, 0.96))
    row_regions = root.split_rows(n_rows, gap=0.03)

    for r_idx, (source, param, vals) in enumerate(panels):
        col_regions = row_regions[r_idx].split_cols(n_cols, gap=0.02)
        # Load all sweep latents for this source+series, filtered by dataset
        latent_tuples = load_sweep_latents(source, series,
                                           dataset_filter=ds)
        # Index by sweep-value substring for matching
        latent_by_val = {}
        for mname, arr in latent_tuples:
            sv = parse_sweep_value(mname)
            latent_by_val[str(sv)] = arr
            # Also store raw model name for substring matching
            latent_by_val[mname] = arr
        for c_idx in range(n_cols):
            ax = col_regions[c_idx].add_axes(fig)
            style_axes(ax, kind="umap")
            if c_idx < len(vals):
                sv = vals[c_idx]
                # Try matching by sweep value or substring
                latent = latent_by_val.get(str(sv))
                if latent is None:
                    for mname, arr in latent_tuples:
                        if str(sv) in mname:
                            latent = arr
                            break
                title = f"{param}={sv}"
                _draw_kmeans_umap(ax, latent, title)
            else:
                ax.axis("off")
            if r_idx == 0 and c_idx == 0:
                add_panel_label(ax, "a")

    out_path = out_dir / f"Fig4_training_{series}.png"
    save_with_vcd(fig, out_path, dpi=DPI, close=True)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "refined_figures" / "output" / args.series)
    generate(args.series, out)
