"""PanODE benchmarking suite.

Submodules
----------
config           – ``BenchmarkConfig`` dataclass and helpers
dataset_registry – 12-dataset registry
model_registry   – 12 model configurations, series groups, ablation steps
metrics_utils    – ``compute_metrics``, ``compute_latent_diagnostics``, etc.
data_utils       – ``load_data``, ``load_or_preprocess_adata``
train_utils      – ``train_and_evaluate``, param factories, I/O helpers
run_manifest     – JSONL run manifest (``append_run``, ``list_runs``, ``compare_runs``)

Entry-point scripts (in benchmarks/runners/)
---------------------------------------------
benchmark_base, benchmark_crossdata, benchmark_sensitivity,
benchmark_training, benchmark_preprocessing

One-off scripts (in scripts/)
-----------------------------
training_dynamics, compute_gse_offline, generate_statistical_figures,
generate_nextjs_figures, statistical_analysis
"""
