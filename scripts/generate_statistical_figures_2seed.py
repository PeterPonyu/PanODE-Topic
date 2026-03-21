#!/usr/bin/env python3
"""
Generate publication-ready statistical figures from 2-seed data — series-separated only.

All outputs are per-series: statistical/dpmm/ and statistical/topic/.
No combined or cross-series figures (two separate articles).

1. Wilcoxon + Cliff's delta heatmap (2-seed) — per series
2. Individual variant ranking (2-seed) with error bars — per series
3. Per-metric decomposition (NMI, ARI, ASW) — per series
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

from _stat_figure_common import (
    setup_fonts,
    SERIES_STRUCTURED,
    SERIES_MODELS,
    color_dpmm,
    color_topic)

setup_fonts()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from benchmarks.config import STATISTICAL_EXPORTS_DIR, PAPER_FIGURES_DIR
except Exception:
    STATISTICAL_EXPORTS_DIR = ROOT / "benchmarks" / "benchmark_results" / "statistical_exports"
    PAPER_FIGURES_DIR = ROOT / "benchmarks" / "paper_figures"
EXPORT_DIR = STATISTICAL_EXPORTS_DIR
STAT_BASE = PAPER_FIGURES_DIR / "statistical"


# ══════════════════════════════════════════════════════════════════════════════
# 1. WILCOXON HEATMAP (2-seed) — per series
# ══════════════════════════════════════════════════════════════════════════════
wilcoxon_df = pd.read_csv(EXPORT_DIR / "pairwise_wilcoxon_2seed.csv")
core = wilcoxon_df[wilcoxon_df["Metric"].isin(["NMI", "ARI", "ASW", "DAV"])].copy()
core["Pair"] = core["Struct"] + "\nvs\n" + core["Pure"]

for series_name, struct_set in SERIES_STRUCTURED.items():
    core_s = core[core["Struct"].isin(struct_set)].copy()
    if core_s.empty:
        continue
    pivot_delta = core_s.pivot_table(index="Pair", columns="Metric", values="delta", aggfunc="first")
    pivot_delta = pivot_delta[[c for c in ["NMI", "ARI", "ASW", "DAV"] if c in pivot_delta.columns]]
    pivot_p = core_s.pivot_table(index="Pair", columns="Metric", values="p", aggfunc="first")
    pivot_p = pivot_p[[c for c in ["NMI", "ARI", "ASW", "DAV"] if c in pivot_p.columns]]

    out_dir = STAT_BASE / series_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, max(3, len(pivot_delta) * 1.0 + 1)))
    im = ax.imshow(pivot_delta.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot_delta.columns)))
    ax.set_xticklabels(pivot_delta.columns, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(pivot_delta.index)))
    ax.set_yticklabels(pivot_delta.index, fontsize=8)
    for i in range(len(pivot_delta.index)):
        for j in range(len(pivot_delta.columns)):
            val = pivot_delta.values[i, j]
            pval = pivot_p.values[i, j] if i < pivot_p.shape[0] and j < pivot_p.shape[1] else 1.0
            if pval < 0.001:
                star = "***"
            elif pval < 0.01:
                star = "**"
            elif pval < 0.05:
                star = "*"
            else:
                star = ""
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:+.2f}{star}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cliff's δ (positive = structured better)")
    series_title = "DPMM" if series_name == "dpmm" else "Topic"
    ax.set_title(
        f"{series_title} Prior vs. Pure Ablation (2-Seed)\n"
        "Cliff's δ: * p<0.05, ** p<0.01, *** p<0.001",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "wilcoxon_cliffs_delta_heatmap_2seed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {series_name}/wilcoxon_cliffs_delta_heatmap_2seed.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL VARIANT RANKING (2-seed) with error bars — per series
# ══════════════════════════════════════════════════════════════════════════════
ranking_df = pd.read_csv(EXPORT_DIR / "individual_variant_ranking_2seed.csv")
for series_name, model_set in SERIES_MODELS.items():
    sub = ranking_df[ranking_df["Model"].isin(model_set)].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("Score", ascending=False).reset_index(drop=True)
    out_dir = STAT_BASE / series_name
    out_dir.mkdir(parents=True, exist_ok=True)
    color_fn = color_dpmm if series_name == "dpmm" else color_topic
    colors = [color_fn(m) for m in sub["Model"]]
    score_sd = sub["Score_SD"].values if "Score_SD" in sub.columns else np.zeros(len(sub))

    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.45)))
    ax.barh(
        range(len(sub)), sub["Score"], xerr=score_sd,
        color=colors, edgecolor="gray", linewidth=0.5, capsize=3,
        error_kw={"linewidth": 1.2, "capthick": 1.2})
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Model"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Composite Score = (NMI + ARI + ASW) / 3", fontsize=10)
    series_title = "DPMM" if series_name == "dpmm" else "Topic"
    ax.set_title(f"{series_title} — Individual Model Ranking (2-Seed Mean ± SD)", fontsize=12, fontweight="bold")
    for i, (_, row) in enumerate(sub.iterrows()):
        sd = row.get("Score_SD", 0)
        ax.text(row["Score"] + sd + 0.005, i, f"{row['Score']:.4f}±{sd:.4f}", va="center", fontsize=8)
    if series_name == "dpmm":
        ax.legend(handles=[
            Patch(facecolor="#E6550D", label="DPMM"),
            Patch(facecolor="#9ECAE1", label="Pure-AE"),
        ], loc="lower right", fontsize=9, framealpha=0.9)
    else:
        ax.legend(handles=[
            Patch(facecolor="#756BB1", label="Topic"),
            Patch(facecolor="#C7E9C0", label="Pure-VAE"),
        ], loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, (sub["Score"].values + score_sd).max() * 1.18)
    fig.tight_layout()
    fig.savefig(out_dir / "individual_variant_ranking_2seed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {series_name}/individual_variant_ranking_2seed.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PER-METRIC DECOMPOSITION (2-seed) — per series
# ══════════════════════════════════════════════════════════════════════════════
for series_name, model_set in SERIES_MODELS.items():
    sub = ranking_df[ranking_df["Model"].isin(model_set)].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("Score", ascending=False).reset_index(drop=True)
    out_dir = STAT_BASE / series_name
    out_dir.mkdir(parents=True, exist_ok=True)
    color_fn = color_dpmm if series_name == "dpmm" else color_topic

    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, len(sub) * 0.4)), sharey=True)
    metric_labels = ["NMI", "ARI", "ASW"]
    for ax, metric in zip(axes, metric_labels):
        if metric not in sub.columns:
            continue
        vals = sub[metric].values
        colors_m = [color_fn(m) for m in sub["Model"]]
        ax.barh(range(len(sub)), vals, color=colors_m, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(range(len(sub)))
        if metric == "NMI":
            ax.set_yticklabels(sub["Model"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(metric, fontsize=10)
        ax.set_title(f"({chr(65 + metric_labels.index(metric))}) {metric}", fontsize=11, fontweight="bold")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
    series_title = "DPMM" if series_name == "dpmm" else "Topic"
    axes[-1].legend(handles=[
        Patch(facecolor="#E6550D" if series_name == "dpmm" else "#756BB1", label=series_title),
        Patch(facecolor="#9ECAE1" if series_name == "dpmm" else "#C7E9C0", label="Pure"),
    ], loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{series_title} — Per-Metric Ranking (2-Seed Mean)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "per_metric_decomposition_2seed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {series_name}/per_metric_decomposition_2seed.png")

print("\nAll 2-seed statistical figures (series-separated) in:", STAT_BASE)
