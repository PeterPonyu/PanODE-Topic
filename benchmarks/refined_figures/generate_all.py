#!/usr/bin/env python
"""Generate all refined publication figures for one series.

Dispatcher that runs each refined figure generator (Fig 2–10) and writes
composed multi-panel PNGs into ``benchmarks/refined_figures/output/{series}/``.

Usage:
    python -m benchmarks.refined_figures.generate_all --series dpmm
    python -m benchmarks.refined_figures.generate_all --series topic
"""

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.refined_figures.fig01_architecture import generate as gen_fig1
from benchmarks.refined_figures.fig02_base_ablation import generate as gen_fig2
from benchmarks.refined_figures.fig03_sensitivity import generate as gen_fig3
from benchmarks.refined_figures.fig04_training_umaps import generate as gen_fig4
from benchmarks.refined_figures.fig05_crossdataset import generate as gen_fig5
from benchmarks.refined_figures.fig06_biological import generate as gen_fig6
from benchmarks.refined_figures.fig07_correlation import generate as gen_fig7
from benchmarks.refined_figures.fig08_latent_umap import generate as gen_fig8
from benchmarks.refined_figures.fig09_enrichment import generate as gen_fig9
from benchmarks.refined_figures.fig10_external import generate as gen_fig10

GENERATORS = {
    "1":  ("Architecture Overview", gen_fig1),
    "2":  ("Base Ablation",        gen_fig2),
    "3":  ("Sensitivity Analysis", gen_fig3),
    "4":  ("Training UMAPs",       gen_fig4),
    "5":  ("Cross-Dataset",        gen_fig5),
    "6":  ("Biological Heatmaps",  gen_fig6),
    "7":  ("Latent-Gene Corr.",    gen_fig7),
    "8":  ("Latent UMAP Proj.",    gen_fig8),
    "9":  ("GO Enrichment",        gen_fig9),
    "10": ("External Benchmark",   gen_fig10),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate all refined publication figures")
    parser.add_argument("--series", required=True, choices=["dpmm", "topic"])
    parser.add_argument("--figures", nargs="+", default=["all"],
                        help="Figure numbers to generate, or 'all'")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "refined_figures" / "output" / args.series)
    out.mkdir(parents=True, exist_ok=True)

    figs = list(GENERATORS.keys()) if "all" in args.figures else args.figures
    n_ok, n_fail = 0, 0
    for fig_id in figs:
        entry = GENERATORS.get(fig_id)
        if entry is None:
            print(f"  Unknown figure: {fig_id}")
            continue
        label, gen_fn = entry
        try:
            print(f"\n{'='*60}")
            print(f"  Fig {fig_id}: {label}")
            print(f"{'='*60}")
            gen_fn(args.series, out)
            n_ok += 1
        except Exception as e:
            print(f"  ERROR Fig {fig_id}: {e}")
            traceback.print_exc()
            n_fail += 1

    print(f"\n  Done: {n_ok} OK, {n_fail} failed → {out}")


if __name__ == "__main__":
    main()
