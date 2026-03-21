#!/bin/bash
# Full benchmark refresh training pipeline
# Runs all training steps sequentially: Topic -> Full Comp -> External -> Merge -> Stats -> Bio -> Figures
#
# The DPMM ablation is assumed to already be running or completed as a parallel job.
#
# Usage: bash scripts/run_full_benchmark_refresh.sh

PYTHON=/home/zeyufu/miniconda3/envs/dl/bin/python
PROJECT_ROOT=/home/zeyufu/Desktop/PanODE-LAB
cd "$PROJECT_ROOT"

echo "============================================================"
echo " FULL BENCHMARK REFRESH PIPELINE"
echo " $(date)"
echo "============================================================"

# ── Step 1: Topic ablation training (new datasets, 1000 epochs) ───────────────
echo ""
echo "=== Step 1/8: Topic ablation training (ablation_topic_full) ==="
$PYTHON -m experiments.run_experiment --preset ablation_topic_full --epochs 1000 || {
    echo "ERROR: Topic ablation failed"; exit 1;
}

# ── Step 2: Full 12-model comparison (new datasets, 1000 epochs) ──────────────
echo ""
echo "=== Step 2/8: Full 12-model comparison (full_comparison_all) ==="
$PYTHON -m experiments.run_experiment --preset full_comparison_all --epochs 1000 || {
    echo "ERROR: Full comparison failed"; exit 1;
}

# ── Step 3: External benchmark (all models, all datasets) ─────────────────────
echo ""
echo "=== Step 3/8: External benchmark (all groups, all datasets) ==="
$PYTHON -m experiments.run_external_benchmark --all-datasets --name external_full || {
    echo "WARNING: External benchmark had errors (some models may have failed)"
}

# ── Step 4: Combine experiment CSVs into cross-dataset format ─────────────────
echo ""
echo "=== Step 4/8: Combine CSVs for statistical analysis ==="
# Combine DPMM ablation results
$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/ablation_dpmm_full/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_dpmm_full.csv

# Combine Topic ablation results
$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/ablation_topic_full/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_topic_full.csv

# Combine full comparison results
$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/full_comparison_all/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_full.csv

# ── Step 5: Merge internal + external ─────────────────────────────────────────
echo ""
echo "=== Step 5/8: Merge internal + external results ==="
$PYTHON -m experiments.merge_and_visualize \
    --internal-name full_comparison_all \
    --external-name external_full \
    --merged-name full_vs_external_all \
    --grouped --top-n 12 2>&1 || echo "  (merge_and_visualize encountered issues)"

# ── Step 6: Statistical analysis ──────────────────────────────────────────────
echo ""
echo "=== Step 6/8: Statistical analysis ==="
$PYTHON scripts/statistical_analysis.py 2>&1 || echo "  (statistical_analysis.py may need path updates)"

# ── Step 7: Statistical + paper figures ───────────────────────────────────────
echo ""
echo "=== Step 7/8: Generate statistical figures ==="
$PYTHON scripts/generate_statistical_figures.py 2>&1 || echo "  (generate_statistical_figures.py may need updates)"

# Visualize ablation results
echo ""
echo "  Visualizing DPMM and Topic ablations..."
$PYTHON -m experiments.visualize_experiment --preset ablation_dpmm_full --per-group 2>&1 || true
$PYTHON -m experiments.visualize_experiment --preset ablation_topic_full --per-group 2>&1 || true

# ── Step 8: Biological validation ─────────────────────────────────────────────
echo ""
echo "=== Step 8/8: Biological validation ==="
if [ -f benchmarks/biological_validation/run_all_bio_validation.sh ]; then
    bash benchmarks/biological_validation/run_all_bio_validation.sh 2>&1 || echo "  (bio validation may need checkpoint path updates)"
else
    echo "  Skipping: run_all_bio_validation.sh not found"
fi

echo ""
echo "============================================================"
echo " FULL BENCHMARK REFRESH COMPLETE"
echo " Started: see timestamps above"
echo " Finished: $(date)"
echo "============================================================"
echo ""
echo "Results summary:"
echo "  DPMM ablation:      $(ls experiments/results/ablation_dpmm_full/tables/*.csv 2>/dev/null | wc -l) datasets"
echo "  Topic ablation:     $(ls experiments/results/ablation_topic_full/tables/*.csv 2>/dev/null | wc -l) datasets"
echo "  Full comparison:    $(ls experiments/results/full_comparison_all/tables/*.csv 2>/dev/null | wc -l) datasets"
echo "  External baseline:  $(ls experiments/results/external_full/tables/*.csv 2>/dev/null | wc -l) datasets"
echo "  Merged figures:     $(ls experiments/results/full_vs_external_all/figures/*.pdf 2>/dev/null | wc -l) PDFs"
echo "  Statistical exports: $(ls benchmarks/benchmark_results/statistical_exports/*.csv 2>/dev/null | wc -l) files"
