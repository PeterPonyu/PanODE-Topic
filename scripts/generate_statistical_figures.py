#!/usr/bin/env python3
"""
Generate statistical analysis figures for the manuscripts — series-separated only.

All outputs are per-series: statistical/dpmm/ and statistical/topic/.
No combined or cross-series figures (two separate articles).

1. Wilcoxon + Cliff's delta heatmap (per series)
2. Individual variant ranking (per series)
3. kNN downstream classification (per series)
4. Runtime analysis (per series)
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
# 1. WILCOXON HEATMAP — per series only (no combined)
# ══════════════════════════════════════════════════════════════════════════════
wilcoxon_df = pd.read_csv(EXPORT_DIR / "pairwise_wilcoxon.csv")
core = wilcoxon_df[wilcoxon_df["Metric"].isin(["NMI", "ARI", "ASW", "DAV"])].copy()
core["Pair"] = core["Structured"] + "\nvs\n" + core["Pure"]

for series_name, series_models in SERIES_STRUCTURED.items():
    core_s = core[core["Structured"].isin(series_models)].copy()
    if core_s.empty:
        continue
    core_s["Pair"] = core_s["Structured"] + "\nvs\n" + core_s["Pure"]
    piv_s = core_s.pivot_table(
        index="Pair", columns="Metric", values="Cliffs_delta", aggfunc="first"
    )
    piv_s = piv_s[[c for c in ["NMI", "ARI", "ASW", "DAV"] if c in piv_s.columns]]

    out_dir = STAT_BASE / series_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_s, ax_s = plt.subplots(figsize=(7, max(3, len(piv_s) * 1.0 + 1)))
    im_s = ax_s.imshow(piv_s.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_s.set_xticks(range(len(piv_s.columns)))
    ax_s.set_xticklabels(piv_s.columns, fontsize=11, fontweight="bold")
    ax_s.set_yticks(range(len(piv_s.index)))
    ax_s.set_yticklabels(piv_s.index, fontsize=8)
    for i2 in range(len(piv_s.index)):
        for j2 in range(len(piv_s.columns)):
            val2 = piv_s.values[i2, j2]
            row2 = core_s[
                (core_s["Pair"] == piv_s.index[i2])
                & (core_s["Metric"] == piv_s.columns[j2])
            ]
            sig2 = row2["Significant_005"].values[0] if len(row2) > 0 else "No"
            star2 = "*" if sig2 == "Yes" else ""
            color2 = "white" if abs(val2) > 0.5 else "black"
            ax_s.text(j2, i2, f"{val2:.2f}{star2}", ha="center", va="center",
                      fontsize=9, color=color2, fontweight="bold")
    fig_s.colorbar(im_s, ax=ax_s, shrink=0.8,
                   label="Cliff's δ (positive = structured better)")
    series_title = "DPMM" if series_name == "dpmm" else "Topic"
    ax_s.set_title(
        f"{series_title} Prior vs. Pure Ablation\n(Cliff's δ with * = p < 0.05)",
        fontsize=12, fontweight="bold")
    fig_s.tight_layout()
    fig_s.savefig(out_dir / "wilcoxon_cliffs_delta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig_s)
    print(f"Saved {series_name}/wilcoxon_cliffs_delta_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL VARIANT RANKING — per series only
# ══════════════════════════════════════════════════════════════════════════════
ranking_df = pd.read_csv(EXPORT_DIR / "individual_variant_ranking.csv")
for series_name, model_set in SERIES_MODELS.items():
    if "Series" in ranking_df.columns:
        sub = ranking_df[ranking_df["Series"] == series_name].copy()
    else:
        sub = ranking_df[ranking_df["Model"].isin(model_set)].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("Score", ascending=False).reset_index(drop=True)
    out_dir = STAT_BASE / series_name
    out_dir.mkdir(parents=True, exist_ok=True)
    color_fn = color_dpmm if series_name == "dpmm" else color_topic
    colors = [color_fn(m) for m in sub["Model"]]

    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.45)))
    ax.barh(range(len(sub)), sub["Score"], color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Model"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Composite Score = (NMI + ARI + ASW) / 3", fontsize=10)
    series_title = "DPMM" if series_name == "dpmm" else "Topic"
    ax.set_title(f"{series_title} — Individual Model Ranking (12-Dataset Mean)", fontsize=12, fontweight="bold")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["Score"] + 0.003, i, f"{row['Score']:.4f}", va="center", fontsize=8)
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
    ax.set_xlim(0, max(sub["Score"]) * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "individual_variant_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {series_name}/individual_variant_ranking.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. KNN DOWNSTREAM CLASSIFICATION — per series only
# ══════════════════════════════════════════════════════════════════════════════
knn_summary = pd.read_csv(EXPORT_DIR / "knn_summary_by_model.csv")
for series_name, model_set in SERIES_MODELS.items():
    sub = knn_summary[knn_summary["Model"].isin(model_set)].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("Mean_Acc", ascending=True).reset_index(drop=True)
    out_dir = STAT_BASE / series_name
    out_dir.mkdir(parents=True, exist_ok=True)
    color_fn = color_dpmm if series_name == "dpmm" else color_topic
    knn_colors = [color_fn(m) for m in sub["Model"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(sub) * 0.4)))
    ax = axes[0]
    ax.barh(range(len(sub)), sub["Mean_Acc"], color=knn_colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Model"], fontsize=9)
    ax.set_xlabel("Mean kNN Accuracy", fontsize=10)
    ax.set_title("(A) kNN Classification Accuracy", fontsize=11, fontweight="bold")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["Mean_Acc"] + 0.005, i, f"{row['Mean_Acc']:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.55, 0.87)
    ax = axes[1]
    ax.barh(range(len(sub)), sub["Mean_F1"], color=knn_colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Model"], fontsize=9)
    ax.set_xlabel("Mean kNN F1-macro", fontsize=10)
    ax.set_title("(B) kNN Classification F1-macro", fontsize=11, fontweight="bold")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["Mean_F1"] + 0.005, i, f"{row['Mean_F1']:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.30, 0.68)
    series_title = "DPMM" if series_name == "dpmm" else "Topic"
    axes[1].legend(handles=[
        Patch(facecolor="#E6550D" if series_name == "dpmm" else "#756BB1", label=series_title),
        Patch(facecolor="#9ECAE1" if series_name == "dpmm" else "#C7E9C0", label="Pure"),
    ], loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "knn_downstream_classification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {series_name}/knn_downstream_classification.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUNTIME ANALYSIS — per series only
# ══════════════════════════════════════════════════════════════════════════════
runtime_df = pd.read_csv(EXPORT_DIR / "runtime_analysis.csv")
for series_name, model_set in SERIES_MODELS.items():
    sub = runtime_df[runtime_df["Model"].isin(model_set)].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("Mean_Time_s", ascending=True).reset_index(drop=True)
    out_dir = STAT_BASE / series_name
    out_dir.mkdir(parents=True, exist_ok=True)
    color_fn = color_dpmm if series_name == "dpmm" else color_topic
    runtime_colors = [color_fn(m) for m in sub["Model"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(3, len(sub) * 0.35)))
    ax = axes[0]
    err = sub["Std_Time_s"] if "Std_Time_s" in sub.columns else np.zeros(len(sub))
    ax.barh(range(len(sub)), sub["Mean_Time_s"], xerr=err, color=runtime_colors,
            edgecolor="gray", linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Model"], fontsize=9)
    ax.set_xlabel("Mean Training Time (s)", fontsize=10)
    ax.set_title("(A) Training Time", fontsize=11, fontweight="bold")
    ax = axes[1]
    ax.barh(range(len(sub)), sub["Mean_PeakGPU_MB"], color=runtime_colors,
            edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Model"], fontsize=9)
    ax.set_xlabel("Peak GPU Memory (MB)", fontsize=10)
    ax.set_title("(B) GPU Memory", fontsize=11, fontweight="bold")
    ax = axes[2]
    params_m = sub["NumParams"] / 1e6
    ax.barh(range(len(sub)), params_m, color=runtime_colors,
            edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Model"], fontsize=9)
    ax.set_xlabel("Parameters (M)", fontsize=10)
    ax.set_title("(C) Model Size", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {series_name}/runtime_analysis.png")

print("\nAll statistical figures (series-separated) in:", STAT_BASE)
