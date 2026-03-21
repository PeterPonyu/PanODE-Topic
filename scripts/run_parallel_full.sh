#!/bin/bash
# Parallel full benchmark execution: maximize GPU utilization
# Splits full_comparison_all into 4 parallel streams (non-overlapping datasets)
# Splits external into 5 parallel model-group streams (each to its own dir)
# Then merges external group results into a unified external_full directory
#
# Usage: bash scripts/run_parallel_full.sh

set -u
PYTHON=/home/zeyufu/miniconda3/envs/dl/bin/python
cd /home/zeyufu/Desktop/PanODE-LAB
LOGDIR=scripts/parallel_logs
mkdir -p "$LOGDIR"

echo "============================================================"
echo " PARALLEL BENCHMARK EXECUTION"
echo " $(date)"
echo " GPU streams: 4 × full_comparison + 5 × external groups"
echo "============================================================"

# ── Remaining datasets for full_comparison_all ────────────────────────────────
# (18 already done with resume support; these are the 38 remaining)
# Split into 4 groups of ~10 non-overlapping datasets

FC_A="melanoma tnbc_brain lbm_brain hepatoblastoma bc_ec bcc hesc_hspc_cd8 lsk_batch hsc_aged bm_niche"
FC_B="lps_mm progastin urine astrocytes_sci ad_hm blood_stroke scc lung_adre aml_pbmc bm_all"
FC_C="bc_stroma gastric tcell_cancer nk_lymphoma breast_cancer bcell_all breast_metastasis tcell_liver mcc_pbmc"
FC_D="mcc_tumor mm_cancer liver_cancer ca_cancer stomach_cancer breast_hm lung_adre2 liver_colon_metastasis breast_hm2"

echo ""
echo "=== Launching 4 parallel full_comparison_all streams ==="

$PYTHON -m experiments.run_experiment --preset full_comparison_all --epochs 1000 --datasets $FC_A \
    > "$LOGDIR/fc_stream_A.log" 2>&1 &
PID_FC_A=$!
echo "  Stream A (PID $PID_FC_A): 10 datasets"

$PYTHON -m experiments.run_experiment --preset full_comparison_all --epochs 1000 --datasets $FC_B \
    > "$LOGDIR/fc_stream_B.log" 2>&1 &
PID_FC_B=$!
echo "  Stream B (PID $PID_FC_B): 10 datasets"

$PYTHON -m experiments.run_experiment --preset full_comparison_all --epochs 1000 --datasets $FC_C \
    > "$LOGDIR/fc_stream_C.log" 2>&1 &
PID_FC_C=$!
echo "  Stream C (PID $PID_FC_C): 9 datasets"

$PYTHON -m experiments.run_experiment --preset full_comparison_all --epochs 1000 --datasets $FC_D \
    > "$LOGDIR/fc_stream_D.log" 2>&1 &
PID_FC_D=$!
echo "  Stream D (PID $PID_FC_D): 9 datasets"

# ── External benchmark: 5 parallel model-group streams, each to own dir ──────
# Using --all-groups writes to external/{group}/ which avoids CSV collisions.
# We use --all-datasets to cover all 56 datasets.
# Each group gets its own output directory, then we merge later.

echo ""
echo "=== Launching 5 parallel external benchmark group streams ==="

$PYTHON -m experiments.run_external_benchmark --all-datasets --group generative \
    > "$LOGDIR/ext_generative.log" 2>&1 &
PID_EXT_GEN=$!
echo "  Generative (PID $PID_EXT_GEN): 8 models × 56 datasets -> external/generative/"

$PYTHON -m experiments.run_external_benchmark --all-datasets --group gaussian_geometric \
    > "$LOGDIR/ext_gaussian.log" 2>&1 &
PID_EXT_GAU=$!
echo "  Gaussian Geometric (PID $PID_EXT_GAU): 5 models × 56 datasets -> external/gaussian_geometric/"

$PYTHON -m experiments.run_external_benchmark --all-datasets --group disentanglement \
    > "$LOGDIR/ext_disentangle.log" 2>&1 &
PID_EXT_DIS=$!
echo "  Disentanglement (PID $PID_EXT_DIS): 4 models × 56 datasets -> external/disentanglement/"

$PYTHON -m experiments.run_external_benchmark --all-datasets --group graph_contrastive \
    > "$LOGDIR/ext_graphcontr.log" 2>&1 &
PID_EXT_GRA=$!
echo "  Graph & Contrastive (PID $PID_EXT_GRA): 3 models × 56 datasets -> external/graph_contrastive/"

$PYTHON -m experiments.run_external_benchmark --all-datasets --group scvi_family \
    > "$LOGDIR/ext_scvi.log" 2>&1 &
PID_EXT_SCV=$!
echo "  scVI Family (PID $PID_EXT_SCV): 3 models × 56 datasets -> external/scvi_family/"

echo ""
echo "=== All 9 streams launched ==="
echo "  Full Comparison PIDs: $PID_FC_A $PID_FC_B $PID_FC_C $PID_FC_D"
echo "  External PIDs: $PID_EXT_GEN $PID_EXT_GAU $PID_EXT_DIS $PID_EXT_GRA $PID_EXT_SCV"
echo ""
echo "Monitoring progress... (check logs in $LOGDIR)"
echo ""

# ── Wait for all full comparison to finish ────────────────────────────────────
echo "Waiting for full_comparison_all streams..."
wait $PID_FC_A $PID_FC_B $PID_FC_C $PID_FC_D
FC_DONE=$(ls experiments/results/full_comparison_all/tables/*_df.csv 2>/dev/null | wc -l)
echo "[$(date '+%H:%M')] Full Comparison DONE: ${FC_DONE}/56 datasets"

# ── Wait for all external streams to finish ───────────────────────────────────
echo "Waiting for external benchmark streams..."
wait $PID_EXT_GEN $PID_EXT_GAU $PID_EXT_DIS $PID_EXT_GRA $PID_EXT_SCV
echo "[$(date '+%H:%M')] All external groups complete"

# ── Merge external group CSVs into unified external_full ──────────────────────
echo ""
echo "=== Merging external group results into external_full ==="

$PYTHON - << 'MERGE_SCRIPT'
import pandas as pd
from pathlib import Path

results_root = Path("experiments/results")
ext_full_dir = results_root / "external_full" / "tables"
ext_full_dir.mkdir(parents=True, exist_ok=True)

groups = ["generative", "gaussian_geometric", "disentanglement", "graph_contrastive", "scvi_family"]
all_datasets = set()

# Discover all dataset keys across all groups
for group in groups:
    group_dir = results_root / "external" / group / "tables"
    if group_dir.exists():
        for csv in group_dir.glob("*_df.csv"):
            ds = csv.stem.replace("_df", "")
            all_datasets.add(ds)

print(f"Found {len(all_datasets)} datasets across external groups")

# For each dataset, merge rows from all group CSVs
for ds in sorted(all_datasets):
    frames = []
    # First, check if external_full already has a "legacy" CSV (pre-copied 16)
    existing = ext_full_dir / f"{ds}_df.csv"
    if existing.exists():
        legacy = pd.read_csv(existing)
        # Only use legacy if it has real model results (not from group merge)
        if len(legacy) > 0:
            frames.append(legacy)

    for group in groups:
        csv_path = results_root / "external" / group / "tables" / f"{ds}_df.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        # Deduplicate by model name (keep last = group-specific result)
        model_col = "method" if "method" in combined.columns else "Model"
        combined = combined.drop_duplicates(subset=[model_col], keep="last")
        combined.to_csv(ext_full_dir / f"{ds}_df.csv", index=False)
        print(f"  {ds}: {len(combined)} models")

print(f"\nMerged external results: {ext_full_dir}")
print(f"  Total datasets: {len(all_datasets)}")
MERGE_SCRIPT

EXT_DONE=$(ls experiments/results/external_full/tables/*_df.csv 2>/dev/null | wc -l)
echo ""
echo "============================================================"
echo " ALL PARALLEL STREAMS COMPLETE"
echo " $(date)"
echo "============================================================"
echo "Results:"
echo "  Full Comparison: $(ls experiments/results/full_comparison_all/tables/*_df.csv 2>/dev/null | wc -l)/56 datasets"
echo "  External (merged): ${EXT_DONE}/56 datasets"
echo ""
echo "Logs: $LOGDIR/"

# ── Post-processing: combine, merge, stats, figures ───────────────────────────
echo ""
echo "=== Step 4: Combine CSVs for statistical analysis ==="
$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/ablation_dpmm_full/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_dpmm_full.csv

$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/ablation_topic_full/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_topic_full.csv

$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/full_comparison_all/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_full.csv

echo ""
echo "=== Step 5: Merge internal + external results ==="
$PYTHON -m experiments.merge_and_visualize \
    --internal-name full_comparison_all \
    --external-name external_full \
    --merged-name full_vs_external_all \
    --grouped --top-n 12 2>&1 || echo "  (merge_and_visualize encountered issues)"

# Create combined CSV from merged results (for statistical_analysis.py)
$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/full_vs_external_all/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_zz_all_models.csv
echo "  -> results_combined_zz_all_models.csv (named zz_ to sort last for stats)"

echo ""
echo "=== Step 6: Statistical analysis ==="
$PYTHON scripts/statistical_analysis.py 2>&1 || echo "  (statistical_analysis.py encountered issues)"

echo ""
echo "=== Step 7: Generate statistical figures ==="
$PYTHON scripts/generate_statistical_figures.py 2>&1 || echo "  (figure generation encountered issues)"

echo "  Visualizing DPMM and Topic ablations..."
$PYTHON -m experiments.visualize_experiment --preset ablation_dpmm_full --per-group 2>&1 || true
$PYTHON -m experiments.visualize_experiment --preset ablation_topic_full --per-group 2>&1 || true

echo ""
echo "=== Step 8: Biological validation ==="
if [ -f benchmarks/biological_validation/run_all_bio_validation.sh ]; then
    bash benchmarks/biological_validation/run_all_bio_validation.sh 2>&1 || echo "  (bio validation encountered issues)"
else
    echo "  Skipping: run_all_bio_validation.sh not found"
fi

echo ""
echo "============================================================"
echo " FULL PIPELINE COMPLETE"
echo " $(date)"
echo "============================================================"
echo "Final Results:"
echo "  DPMM ablation:       $(ls experiments/results/ablation_dpmm_full/tables/*.csv 2>/dev/null | wc -l)/56"
echo "  Topic ablation:      $(ls experiments/results/ablation_topic_full/tables/*.csv 2>/dev/null | wc -l)/56"
echo "  Full comparison:     $(ls experiments/results/full_comparison_all/tables/*.csv 2>/dev/null | wc -l)/56"
echo "  External (merged):   $(ls experiments/results/external_full/tables/*.csv 2>/dev/null | wc -l)/56"
echo "  Merged (int+ext):    $(ls experiments/results/full_vs_external_all/tables/*.csv 2>/dev/null | wc -l) datasets"
echo "  Statistical exports: $(ls benchmarks/benchmark_results/statistical_exports/*.csv 2>/dev/null | wc -l) files"
