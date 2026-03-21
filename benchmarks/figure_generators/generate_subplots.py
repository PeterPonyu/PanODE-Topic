#!/usr/bin/env python
"""Generate individual subplot PNGs for figure composition.

Thin CLI dispatcher — actual subplot logic lives in per-figure modules:
    gen_fig2_subplots.py   — Figure 2  (Base Ablation)
    gen_fig3_subplots.py   — Figure 3  (Sensitivity Analysis)
    gen_fig4_subplots.py   — Figure 4  (Training / Sweep UMAPs)
    gen_fig6_subplots.py   — Figure 6  (Biological Validation — Heatmaps)
    gen_fig7_subplots.py   — Figure 7  (Latent–Gene Correlation)
    gen_fig8_subplots.py   — Figure 8  (Latent UMAP Projections)
    gen_fig9_subplots.py   — Figure 9  (GO Enrichment)
    gen_fig10_subplots.py  — Figure 10 (External Model Benchmark — subplot assets)

Figures 10–12 in the paper use the experiment pipeline outputs
(compose_experiment_figures.py) for their final composed panels.

Shared style constants are in subplot_style.py.

Usage:
    python -m benchmarks.figure_generators.generate_subplots \\
        --series dpmm --figures 2 3 4 6 7 8 9 10

Output directory: benchmarks/paper_figures/{series}/subplots/fig{N}/
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.figure_generators.gen_fig2_subplots import generate as gen_fig2
from benchmarks.figure_generators.gen_fig3_subplots import generate as gen_fig3
from benchmarks.figure_generators.gen_fig4_subplots import generate as gen_fig4
from benchmarks.figure_generators.gen_fig6_subplots import generate as gen_fig6
from benchmarks.figure_generators.gen_fig7_subplots import generate as gen_fig7
from benchmarks.figure_generators.gen_fig8_subplots import generate as gen_fig8
from benchmarks.figure_generators.gen_fig9_subplots import generate as gen_fig9
from benchmarks.figure_generators.gen_fig10_subplots import generate as gen_fig10


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "2": gen_fig2,
    "3": gen_fig3,
    "4": gen_fig4,
    "6": gen_fig6,
    "7": gen_fig7,
    "8": gen_fig8,
    "9": gen_fig9,
    "10": gen_fig10,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate individual subplot PNGs for Python figure composition")
    parser.add_argument("--figures", nargs="+", default=["all"],
                        help="Which figures: 2 3 4 6 7, or 'all'")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--per-seed", action="store_true",
                        help="Figure 2: show per-seed data points (n≈60) "
                             "instead of seed-averaged (n≈12)")
    args = parser.parse_args()

    series = "topic"
    out_base = (Path(args.output_dir) if args.output_dir
                else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out_base.mkdir(parents=True, exist_ok=True)

    figs = list(GENERATORS.keys()) if "all" in args.figures else args.figures
    for fig_id in figs:
        gen = GENERATORS.get(fig_id)
        if gen is None:
            print(f"Unknown figure: {fig_id}")
            continue
        try:
            if fig_id == "2" and args.per_seed:
                gen(out_base, seed_average=False)
            else:
                gen(out_base)
        except Exception as e:
            import traceback
            print(f"\n  ERROR generating Fig {fig_id} subplots: {e}")
            traceback.print_exc()

    print(f"\n  All subplot PNGs done. Output: {out_base}")


if __name__ == "__main__":
    main()

