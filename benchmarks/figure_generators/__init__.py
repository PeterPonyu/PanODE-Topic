"""Modular figure generators — one subplot file per paper figure.

Subplot generators (primary workflow):
  gen_fig2_subplots … gen_fig10_subplots → produce individual subplot PNGs.
  Shared style in subplot_style.py.
  CLI dispatcher in generate_subplots.py.

Figure design (no duplication across figures):
  Fig 2   — Base ablation: composed 6-group metric panels (experiment pipeline)
  Fig 3   — All hyperparameter sweeps merged (sensitivity + training + preprocessing)
  Fig 4   — UMAP embedding trends across hyperparameter sweep latent snapshots
  Fig 5   — REMOVED (Cross-model performance trade-offs)
  Fig 6   — Biological validation: gradient-based gene importance heatmaps + Topic β
  Fig 7   — Latent–gene Pearson correlation heatmaps
  Fig 8   — Latent UMAP projections: component intensity + gene expression
  Fig 9   — Biological validation: GO enrichment analysis with improved font consistency
  Fig 10  — External benchmark — Proposed: internal ablation variants in external context
  Fig 11  — External benchmark — Classical: classical external baselines
  Fig 12  — External benchmark — Deep: deep/graph/probabilistic external baselines
"""

# Per-subplot PNG generators for figure composition
from . import (
    gen_fig2_subplots,
    gen_fig3_subplots,
    gen_fig4_subplots,
    gen_fig6_subplots,
    gen_fig7_subplots,
    gen_fig8_subplots,
    gen_fig9_subplots,
    gen_fig10_subplots)

SUBPLOT_GENERATORS = {
    "2": gen_fig2_subplots.generate,
    "3": gen_fig3_subplots.generate,
    "4": gen_fig4_subplots.generate,
    "6": gen_fig6_subplots.generate,
    "7": gen_fig7_subplots.generate,
    "8": gen_fig8_subplots.generate,
    "9": gen_fig9_subplots.generate,
    "10": gen_fig10_subplots.generate,
}
