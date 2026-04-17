# PanODE-Topic

Topic-flow-matching models for interpretable single-cell representation learning.

The repository currently focuses on the `Topic-FM` family:
- `Topic-FM-Base`
- `Topic-FM-Transformer`
- `Topic-FM-Contrastive`
- `Topic-FM-GAT` (optional, when graph dependencies are installed)

Matched baselines are included through the `Pure-VAE` family.

## Project Layout

```text
PanODE-Topic/
├── models/                   # Topic-FM, Topic, and Pure-VAE implementations
├── benchmarks/               # Training and evaluation runners
├── eval_lib/                 # Baseline wrappers and evaluation utilities
├── utils/                    # Shared training, data, and visualization helpers
├── src/                      # Visualization helpers
└── vcd/                      # Visual consistency diagnostics
```

## Model Families

| Model | Encoder | Prior | Flow Matching |
|-------|---------|-------|---------------|
| `Topic-FM-Base` | MLP | Dirichlet / logistic-normal | Yes |
| `Topic-FM-Transformer` | Self-attention | Dirichlet / logistic-normal | Yes |
| `Topic-FM-Contrastive` | MLP + MoCo | Dirichlet / logistic-normal | Yes |
| `Topic-FM-GAT` | GAT over kNN | Dirichlet / logistic-normal | Yes |
| `Pure-VAE` | MLP | Gaussian | No |
| `Pure-Transformer-VAE` | Self-attention | Gaussian | No |
| `Pure-Contrastive-VAE` | MLP + MoCo | Gaussian | No |

## Default Settings

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

- Configure dataset locations through your local benchmark setup before running the benchmark scripts.
- The maintained benchmark registry targets the Topic-FM and Pure-VAE families.
