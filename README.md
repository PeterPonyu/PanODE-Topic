# PanODE-Topic

Topic-flow-matching models for interpretable single-cell representation learning.

The current repo focus is the `Topic-FM` family:
- `Topic-FM-Base`
- `Topic-FM-Transformer`
- `Topic-FM-Contrastive`
- `Topic-FM-GAT` (optional, when graph dependencies are installed)

Matched baselines remain in scope through the `Pure-VAE` family.

## Active Layout

```text
PanODE-Topic/
├── models/                   # Topic-FM, Topic, and Pure-VAE implementations
├── benchmarks/               # Active internal benchmarking + biological validation
├── eval_lib/                 # External baseline wrappers and evaluation utilities
├── refined_figures/          # Publication-quality figure generators
├── experiments/              # External comparison pipeline and merged outputs
├── article/                  # Manuscript sources
├── utils/                    # Shared training/data/viz helpers
├── src/                      # Figure layout / visualization helpers
└── vcd/                      # Visual consistency diagnostics
```

## Active Benchmark Models

| Model | Encoder | Prior | Flow Matching |
|-------|---------|-------|---------------|
| `Topic-FM-Base` | MLP | Dirichlet / logistic-normal | Yes |
| `Topic-FM-Transformer` | Self-attention | Dirichlet / logistic-normal | Yes |
| `Topic-FM-Contrastive` | MLP + MoCo | Dirichlet / logistic-normal | Yes |
| `Topic-FM-GAT` | GAT over kNN | Dirichlet / logistic-normal | Yes |
| `Pure-VAE` | MLP | Gaussian | No |
| `Pure-Transformer-VAE` | Self-attention | Gaussian | No |
| `Pure-Contrastive-VAE` | MLP + MoCo | Gaussian | No |

## Current Defaults

| Parameter | Value |
|-----------|-------|
| Learning rate | `1e-3` |
| Batch size | `128` |
| Topics / latent dim | `10` |
| Epochs | `1000` |
| KL weight | `0.01` for Topic-FM |
| Flow warmup | `50` epochs |
| Flow weight | `0.1` |
| HVGs | `3000` |
| Max cells | `3000` |

## Running

```bash
python benchmarks/runners/benchmark_base.py --series topic
python benchmarks/runners/benchmark_base.py --models Topic-FM-Transformer Pure-VAE
python benchmarks/runners/benchmark_crossdata.py --datasets setty lung endo
```

## Notes

- Dataset files are expected under the path configured in `PANODE_DATASETS_ROOT` (or passed directly to benchmark commands).
- Large generated outputs under `benchmarks/`, `experiments/`, and `refined_figures/output/` are treated as regenerable local artefacts.
- The maintained benchmark registry targets Topic-FM plus Pure-VAE only.
