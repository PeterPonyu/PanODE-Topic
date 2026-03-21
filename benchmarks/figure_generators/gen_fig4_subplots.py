"""Generate individual subplot PNGs for Figure 4 (Training / Sweep UMAPs).

For each sweep parameter × representative dataset, produces KMeans-colored
UMAP embedding plots showing how latent geometry evolves across sweep values.

Output: benchmarks/paper_figures/{series}/subplots/fig4/

Usage:
    python -m benchmarks.figure_generators.gen_fig4_subplots --series dpmm
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

from benchmarks.figure_generators.subplot_style import (
    apply_subplot_style, save_subplot, build_manifest,
    FIGSIZE_4COL, LINE_WIDTH_SPINE, SCATTER_SIZE_UMAP, SUBPLOT_DPI,
    FONTSIZE_TITLE, FONTSIZE_CAPTION)
from benchmarks.figure_generators.common import (
    compute_umap, REPRESENTATIVE_DATASETS
)
from benchmarks.figure_generators.data_loaders import (
    load_sensitivity_csv, load_training_csv, load_preprocessing_csv,
    load_sweep_latents, parse_sweep_value)


# Series-specific sensitivity params (kl_weight = Topic, warmup_ratio = DPMM)
_SENSITIVITY_PARAMS = {
    "dpmm":  ["warmup_ratio", "latent_dim", "encoder_size", "dropout_rate"],
    "topic": ["kl_weight",    "latent_dim", "encoder_size", "dropout_rate"],
}

def _key_params_by_source(series):
    """Return sweep parameter dict keyed by source, series-aware for sensitivity."""
    return {
        "sensitivity":    _SENSITIVITY_PARAMS.get(series, _SENSITIVITY_PARAMS["dpmm"]),
        "training":       ["lr", "epochs", "batch_size", "weight_decay"],
        "preprocessing":  ["hvg_top_genes"],
    }


def _format_sweep_value(name):
    """Extract and format the sweep value from a model name.

    e.g. 'DPMM-Base(lr=0.0001)' → 'lr = 1×10⁻⁴'
         'DPMM-Base(epochs=200)' → 'epochs = 200'
    """
    import re
    m = re.search(r'\(([^=]+)=([^\)]+)\)', str(name))
    if not m:
        return str(name)
    param, val_str = m.group(1), m.group(2)
    try:
        val = float(val_str)
        # Use scientific notation for very small or very large numbers
        if val != 0 and (abs(val) < 0.01 or abs(val) >= 10000):
            exp = int(np.floor(np.log10(abs(val))))
            coeff = val / (10 ** exp)
            if abs(coeff - round(coeff)) < 1e-9:
                coeff = int(round(coeff))
            if coeff == 1:
                return f"{param} = 10$^{{{exp}}}$"
            return f"{param} = {coeff}×10$^{{{exp}}}$"
        # Integer-like values
        if val == int(val):
            return f"{param} = {int(val)}"
        return f"{param} = {val}"
    except (ValueError, TypeError):
        return f"{param} = {val_str}"


def gen_sweep_umap(name, latent, out_path):
    """Generate one KMeans-colored UMAP subplot PNG."""
    n_pts = len(latent)
    nn = min(15, max(5, n_pts // 6))
    emb = compute_umap(latent, n_neighbors=nn)
    n_cl = int(max(2, min(10, n_pts // 20)))
    labels = KMeans(n_clusters=n_cl, random_state=42,
                    n_init=10).fit_predict(latent)

    fig = plt.figure(figsize=FIGSIZE_4COL)

    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))

    ax = layout.add_axes(fig)

    style_axes(ax)
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10",
               s=SCATTER_SIZE_UMAP, alpha=0.75, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    # Show formatted hyperparameter value as title
    title = _format_sweep_value(name)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=2,
                 loc="left", fontweight="normal")
    for sp in ax.spines.values():
        sp.set_linewidth(LINE_WIDTH_SPINE)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def generate(out_dir):
    """Generate all subplot PNGs for Figure 4."""
    series = "topic"
    print(f"\n  Figure 4 subplots ({series})")
    sub_dir = out_dir / "fig4"
    sub_dir.mkdir(parents=True, exist_ok=True)
    apply_subplot_style()

    rep_datasets = list(REPRESENTATIVE_DATASETS)
    loaders = {
        "sensitivity": load_sensitivity_csv,
        "training": load_training_csv,
        "preprocessing": load_preprocessing_csv,
    }

    param_manifest = {}
    key_params = _key_params_by_source(series)
    for source, params in key_params.items():
        try:
            df = loaders[source](series)
            if "Series" in df.columns:
                df = df[df["Series"] == series].copy()
        except Exception:
            continue
        if "Sweep" not in df.columns:
            continue
        for param in params:
            sub = df[df["Sweep"] == param].copy()
            if sub.empty or "SweepVal" not in sub.columns:
                continue
            tmp = sub.copy()
            tmp["_sv"] = pd.to_numeric(tmp["SweepVal"], errors="coerce")
            tmp = (tmp.sort_values("_sv") if tmp["_sv"].notna().any()
                   else tmp.sort_values("SweepVal"))
            model_names = list(dict.fromkeys(tmp["Model"].dropna().tolist()))
            if len(model_names) > 4:
                idx = np.linspace(0, len(model_names) - 1, 4, dtype=int)
                model_names = [model_names[i] for i in idx]

            ds_plots = {}
            for ds in rep_datasets:
                latents = load_sweep_latents(source, series,
                                             model_names=model_names,
                                             n_select=4,
                                             dataset_filter=ds)
                if not latents:
                    continue
                snap_files = []
                for k, (name, latent) in enumerate(latents):
                    safe_p = param.replace("/", "_")
                    safe_n = (name.replace("/", "_").replace("(", "")
                              .replace(")", "").replace("=", "_"))
                    fname = f"{safe_p}_{ds}_{k}_{safe_n}.png"
                    gen_sweep_umap(name, latent, sub_dir / fname)
                    snap_files.append({"file": fname, "label": name})
                ds_plots[ds] = snap_files

            if ds_plots:
                param_manifest[param] = {
                    "source": source,
                    "datasets": ds_plots,
                }

    manifest = build_manifest(sub_dir, {
        "params": param_manifest,
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 4 subplots")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(out)
