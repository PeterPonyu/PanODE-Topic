#!/bin/bash
# Watchdog + resume + post-processing: keeps parallel training alive until complete.
# Auto-restarts dead FC and external streams. Runs merge + stats after all 56/56.
#
# Usage: nohup bash scripts/watchdog_resume.sh > scripts/parallel_logs/watchdog.log 2>&1 &

set -u
PYTHON=/home/zeyufu/miniconda3/envs/dl/bin/python
cd /home/zeyufu/Desktop/PanODE-LAB
LOGDIR=scripts/parallel_logs
mkdir -p "$LOGDIR"

# ── FC dataset lists (same as run_parallel_full.sh) ──────────────────────────
FC_A="melanoma tnbc_brain lbm_brain hepatoblastoma bc_ec bcc hesc_hspc_cd8 lsk_batch hsc_aged bm_niche"
FC_B="lps_mm progastin urine astrocytes_sci ad_hm blood_stroke scc lung_adre aml_pbmc bm_all"
FC_C="bc_stroma gastric tcell_cancer nk_lymphoma breast_cancer bcell_all breast_metastasis tcell_liver mcc_pbmc"
FC_D="mcc_tumor mm_cancer liver_cancer ca_cancer stomach_cancer breast_hm lung_adre2 liver_colon_metastasis breast_hm2"

# ── Helper: count CSVs in a directory ─────────────────────────────────────────
count_csvs() {
    ls "$1"/*_df.csv 2>/dev/null | wc -l
}

# ── Helper: check if a process matching pattern exists ────────────────────────
proc_alive() {
    pgrep -f "$1" > /dev/null 2>&1
}

# ── Helper: launch an FC stream if it has remaining datasets ──────────────────
launch_fc() {
    local label=$1 datasets=$2 logfile=$3
    # Check if any dataset in the list still needs processing
    local needs_work=false
    for ds in $datasets; do
        if [ ! -f "experiments/results/full_comparison_all/tables/${ds}_df.csv" ]; then
            needs_work=true
            break
        fi
    done
    if $needs_work; then
        if ! pgrep -f "full_comparison_all.*--datasets.*$(echo $datasets | awk '{print $1}')" > /dev/null 2>&1; then
            echo "  [LAUNCH] FC stream $label"
            nohup $PYTHON -m experiments.run_experiment --preset full_comparison_all --epochs 1000 --datasets $datasets \
                >> "$logfile" 2>&1 &
        fi
    fi
}

# ── Helper: launch an external group if incomplete ────────────────────────────
launch_ext() {
    local group=$1 skip_models="${2:-}"
    local dir="experiments/results/external/$group/tables"
    local n=$(count_csvs "$dir" 2>/dev/null || echo 0)
    if [ "$n" -lt 56 ]; then
        if ! pgrep -f "run_external_benchmark.*--group $group" > /dev/null 2>&1; then
            echo "  [LAUNCH] External $group ($n/56 done)"
            local skip_arg=""
            [ -n "$skip_models" ] && skip_arg="--skip $skip_models"
            nohup $PYTHON -m experiments.run_external_benchmark --all-datasets --group $group $skip_arg \
                >> "$LOGDIR/ext_${group}.log" 2>&1 &
        fi
    fi
}

echo "============================================================"
echo " WATCHDOG + RESUME PIPELINE"
echo " $(date)"
echo "============================================================"

# ── Main watchdog loop ────────────────────────────────────────────────────────
MAX_RETRIES=500
retry=0
while [ $retry -lt $MAX_RETRIES ]; do
    retry=$((retry + 1))

    FC_N=$(count_csvs "experiments/results/full_comparison_all/tables")
    EXT_GEN=$(count_csvs "experiments/results/external/generative/tables" 2>/dev/null || echo 0)
    EXT_GAU=$(count_csvs "experiments/results/external/gaussian_geometric/tables" 2>/dev/null || echo 0)
    EXT_GRA=$(count_csvs "experiments/results/external/graph_contrastive/tables" 2>/dev/null || echo 0)
    # Already complete:
    EXT_DIS=$(count_csvs "experiments/results/external/disentanglement/tables" 2>/dev/null || echo 0)
    EXT_SCV=$(count_csvs "experiments/results/external/scvi_family/tables" 2>/dev/null || echo 0)

    NPROCS=$(ps aux | grep -E "(run_experiment|run_external)" | grep -v grep | wc -l)
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "N/A")

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iteration $retry"
    echo "  FC: $FC_N/56 | Ext: gen=$EXT_GEN gauss=$EXT_GAU graph=$EXT_GRA disent=$EXT_DIS scvi=$EXT_SCV"
    echo "  Procs: $NPROCS | GPU: $GPU_INFO"

    # Check if ALL complete
    if [ "$FC_N" -ge 56 ] && [ "$EXT_GEN" -ge 56 ] && [ "$EXT_GAU" -ge 56 ] && [ "$EXT_GRA" -ge 56 ] && [ "$EXT_DIS" -ge 56 ] && [ "$EXT_SCV" -ge 56 ]; then
        echo ""
        echo "  *** ALL STREAMS COMPLETE ***"
        break
    fi

    # ── Restart any dead FC streams ───────────────────────────────────────
    if [ "$FC_N" -lt 56 ]; then
        launch_fc "A" "$FC_A" "$LOGDIR/fc_stream_A.log"
        launch_fc "B" "$FC_B" "$LOGDIR/fc_stream_B.log"
        launch_fc "C" "$FC_C" "$LOGDIR/fc_stream_C.log"
        launch_fc "D" "$FC_D" "$LOGDIR/fc_stream_D.log"
    fi

    # ── Restart any dead external streams ─────────────────────────────────
    launch_ext "generative" ""
    launch_ext "gaussian_geometric" ""
    launch_ext "graph_contrastive" "scGCC"
    # disentanglement and scvi_family already 56/56, skip

    # Wait 5 minutes before next check
    sleep 300
done

echo ""
echo "============================================================"
echo " ALL TRAINING COMPLETE - Starting post-processing"
echo " $(date)"
echo "============================================================"

# ── PHASE 2: Merge external groups into external_full ─────────────────────────
echo ""
echo "=== Merging external group CSVs into external_full ==="

$PYTHON - << 'MERGE_SCRIPT'
import pandas as pd
from pathlib import Path

results_root = Path("experiments/results")
ext_full_dir = results_root / "external_full" / "tables"
ext_full_dir.mkdir(parents=True, exist_ok=True)

groups = ["generative", "gaussian_geometric", "disentanglement", "graph_contrastive", "scvi_family"]
all_datasets = set()

for group in groups:
    group_dir = results_root / "external" / group / "tables"
    if group_dir.exists():
        for csv in group_dir.glob("*_df.csv"):
            ds = csv.stem.replace("_df", "")
            all_datasets.add(ds)

print(f"Found {len(all_datasets)} datasets across external groups")

for ds in sorted(all_datasets):
    frames = []
    for group in groups:
        csv_path = results_root / "external" / group / "tables" / f"{ds}_df.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        model_col = "method" if "method" in combined.columns else "Model"
        combined = combined.drop_duplicates(subset=[model_col], keep="last")
        combined.to_csv(ext_full_dir / f"{ds}_df.csv", index=False)
        print(f"  {ds}: {len(combined)} models")

print(f"\nMerged external results: {ext_full_dir}")
print(f"  Total datasets: {len(all_datasets)}")
MERGE_SCRIPT

EXT_MERGED=$(count_csvs "experiments/results/external_full/tables")
echo "  External merged: $EXT_MERGED datasets"

# ── PHASE 3: Combine CSVs for statistical analysis ───────────────────────────
echo ""
echo "=== Combining CSVs for statistical analysis ==="

$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/ablation_dpmm_full/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_dpmm_full.csv

$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/ablation_topic_full/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_topic_full.csv

$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/full_comparison_all/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_full.csv

# ── PHASE 4: Merge internal + external ────────────────────────────────────────
echo ""
echo "=== Merge internal + external results ==="
$PYTHON -m experiments.merge_and_visualize \
    --internal-name full_comparison_all \
    --external-name external_full \
    --merged-name full_vs_external_all \
    --grouped --top-n 12 2>&1 || echo "  (merge_and_visualize encountered issues)"

# Create combined CSV from merged (named zz_ to sort last for stats loading)
$PYTHON scripts/combine_experiment_csvs.py \
    --input experiments/results/full_vs_external_all/tables \
    --output benchmarks/benchmark_results/crossdata/csv/results_combined_zz_all_models.csv

# ── PHASE 5: Statistical analysis ─────────────────────────────────────────────
echo ""
echo "=== Statistical analysis ==="
$PYTHON scripts/statistical_analysis.py 2>&1 || echo "  (statistical_analysis.py encountered issues)"

# ── PHASE 6: Generate figures ─────────────────────────────────────────────────
echo ""
echo "=== Generate statistical figures ==="
$PYTHON scripts/generate_statistical_figures.py 2>&1 || echo "  (figure generation encountered issues)"

echo ""
echo "  Visualizing DPMM and Topic ablations..."
$PYTHON -m experiments.visualize_experiment --preset ablation_dpmm_full --per-group 2>&1 || true
$PYTHON -m experiments.visualize_experiment --preset ablation_topic_full --per-group 2>&1 || true

# ── PHASE 7: Biological validation ────────────────────────────────────────────
echo ""
echo "=== Biological validation ==="
if [ -f benchmarks/biological_validation/run_all_bio_validation.sh ]; then
    bash benchmarks/biological_validation/run_all_bio_validation.sh 2>&1 || echo "  (bio validation encountered issues)"
else
    echo "  Skipping: run_all_bio_validation.sh not found"
fi

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " FULL PIPELINE COMPLETE"
echo " $(date)"
echo "============================================================"
echo "Final Results:"
echo "  DPMM ablation:       $(count_csvs experiments/results/ablation_dpmm_full/tables)/56"
echo "  Topic ablation:      $(count_csvs experiments/results/ablation_topic_full/tables)/56"
echo "  Full comparison:     $(count_csvs experiments/results/full_comparison_all/tables)/56"
echo "  External (merged):   $(count_csvs experiments/results/external_full/tables)/56"
echo "  Merged (int+ext):    $(ls experiments/results/full_vs_external_all/tables/*_df.csv 2>/dev/null | wc -l) datasets"
echo "  Statistical exports: $(ls benchmarks/benchmark_results/statistical_exports/*.csv 2>/dev/null | wc -l) files"
