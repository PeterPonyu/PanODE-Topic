# PanODE-Topic

Flow-matching-refined Dirichlet-prior variational autoencoders for interpretable and structurally balanced single-cell representation learning.

## Project Structure

```
PanODE-Topic/
├── models/                        # Core model implementations
│   ├── encoders.py                # Encoder architectures (MLP, Transformer, Hybrid, GAT)
│   ├── shared_modules.py          # Reusable layers, TopicDecoder, kNN graph utils
│   ├── topic_flow_matching.py     # Flow matching mixin (SinusoidalTimeEmb, LatentFlowField)
│   ├── topic_base.py              # Topic-Base (logistic-normal topic VAE)
│   ├── topic_transformer.py       # Topic-Transformer (self-attention encoder)
│   ├── topic_contrastive.py       # Topic-Contrastive (MoCo contrastive)
│   ├── topic_fm_base.py           # Topic-FM-Base (+ flow matching)
│   ├── topic_fm_transformer.py    # Topic-FM-Transformer (+ flow matching)
│   ├── topic_fm_contrastive.py    # Topic-FM-Contrastive (+ flow matching)
│   ├── topic_fm_gat.py            # Topic-FM-GAT (GAT encoder + flow matching)
│   └── pure_vae.py                # Prior-free VAE baselines
│
├── eval_lib/                      # Evaluation & experiment library
│   ├── metrics/                   # DRE, LSE, DREX, LSEX metric batteries
│   ├── viz/                       # RigorousExperimentalAnalyzer, publication figures
│   ├── experiment/                # Experiment config, merge, templates
│   └── baselines/                 # 21+ external baseline model wrappers
│
├── utils/                         # Shared utilities
│   ├── base_model.py              # Unified model interface (fit / extract_latent)
│   ├── mixins.py                  # PriorMixin, ReconstructionLossMixin
│   ├── data.py                    # Data loading and preprocessing
│   └── viz.py                     # Visualisation helpers
│
├── benchmarks/                    # Benchmarking infrastructure
│   ├── config.py                  # Centralised benchmark configuration
│   ├── model_registry.py          # Model factory & registration
│   ├── dataset_registry.py        # 56-dataset catalogue
│   ├── figure_generators/         # Subplot generation for composed figures
│   ├── runners/                   # Benchmark runner modules
│   └── biological_validation/     # GO enrichment, perturbation analysis
│
├── refined_figures/               # Publication figure generators (Fig 1, 3-9)
│   ├── fig01_architecture.py      # Architecture diagram (4 FM variants)
│   ├── fig03_sensitivity.py       # Hyperparameter sensitivity
│   ├── fig04_training_umaps.py    # UMAP sweep evolution
│   ├── fig05_crossdataset.py      # Cross-dataset scatter plots
│   ├── fig06_biological.py        # Gene importance & decoder beta
│   ├── fig07_correlation.py       # Latent-gene correlation
│   ├── fig08_latent_umap.py       # Component UMAP projections
│   ├── fig09_enrichment.py        # GO enrichment dot plots
│   └── generate_all.py            # Dispatcher
│
├── scripts/                       # Pipeline scripts
│   ├── regenerate_figures.py      # Experiment-pipeline figure generation
│   ├── refresh_figures.sh         # Full 5-step figure refresh pipeline
│   └── generate_latex_tables.py   # Automated table generation
│
├── experiments/                   # Experiment runners & results
│   ├── run_external_benchmark.py  # External baseline comparison
│   └── merge_and_visualize.py     # Result merging & visualization
│
├── article/topic/                 # LaTeX manuscript (MDPI template)
│   ├── main_mdpi.tex              # Main article source
│   ├── tables/                    # Auto-generated table .tex files
│   └── Definitions/               # MDPI class & style files
│
└── src/visualization/             # Layout & style engine
    ├── direct_layout.py           # LayoutRegion-based figure composition
    └── style.py                   # Publication styles, VCD integration
```

## Model Variants

| Model | Encoder | Prior | Flow Matching | Description |
|-------|---------|-------|---------------|-------------|
| Topic-FM-Base | MLP | Dirichlet | Yes | Feedforward + FM refinement |
| Topic-FM-Transformer | Self-attention | Dirichlet | Yes | Cell-as-token + FM |
| Topic-FM-Contrastive | MLP + MoCo | Dirichlet | Yes | Contrastive + FM |
| Topic-FM-GAT | GAT (kNN) | Dirichlet | Yes | Graph attention + FM |
| Pure-VAE | MLP | Gaussian | No | Prior-free baseline |
| Pure-Transformer-VAE | Self-attention | Gaussian | No | Baseline |
| Pure-Contrastive-VAE | MLP + MoCo | Gaussian | No | Baseline |

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-3 | All models |
| KL weight | 0.01 | Topic & VAE series |
| Latent dim (n_topics) | 10 | All topic models |
| Batch size | 128 | All models |
| Epochs | 1000 | Canonical refresh |
| FM warmup | 50 epochs | Flow matching activation delay |
| FM weight | 0.1 | Flow loss coefficient |
| Gradient clipping | 10.0 | All models |

## Evaluation

- **Clustering**: NMI, ARI, ASW, DAV, CAL, COR
- **DRE**: Dimensionality reduction evaluation (UMAP/t-SNE structure preservation)
- **LSE**: Latent space evaluation (intrinsic properties)
- **DREX/LSEX**: Extended metric variants
- **Composite**: (NMI + ARI + ASW) / 3

## Data Format

Input: AnnData (`.h5ad`) with raw counts in `adata.X`, labels in `adata.obs['cell_type']`.
Preprocessing: top 3000 HVGs, library-size normalization, log(1+x), subsample to 3000 cells.
