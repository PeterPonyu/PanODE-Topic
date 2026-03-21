"""eval_lib — Portable evaluation toolkit.

Reusable, project-agnostic modules for latent-space evaluation,
dimensionality-reduction quality assessment, publication-ready
figure generation, external model benchmarking, and cross-experiment
merging.  Copy this folder into any project to reuse.

Sub-packages
------------
metrics/
    dre          Dimensionality Reduction Evaluator (UMAP/tSNE quality)
    drex         Extended DR metrics (trustworthiness, continuity, …)
    lse          Latent Space Evaluator (spectral decay, anisotropy, …)
    lsex         Extended latent space metrics (2-hop, curvature, …)
    battery      Unified metric battery: compute_metrics(), METRIC_COLUMNS,
                 METRIC_GROUPS, DataSplitter, diagnostics

viz/
    rea          RigorousExperimentalAnalyzer — statistical tables + figures
    loss         Training-loss curve visualisation

experiment/
    config       ExperimentConfig dataclass
    merge        MergedExperimentConfig — combine results across experiments
    templates/   Copy-to-project experiment runner templates

baselines/
    registry     Registry of 12 external baseline models + default HPs
    models/      Model wrapper implementations (BaseModel interface)
"""
