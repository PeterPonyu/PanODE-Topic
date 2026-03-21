# Experiment Runner Templates

Project-agnostic experiment runner templates that use `eval_lib` as
the portable evaluation layer.  Copy the template files into any new
project's `experiments/` directory and fill in the clearly marked
**`# ═══ PROJECT-SPECIFIC`** sections.

## Architecture

```
eval_lib/                              ← portable, never modified
  metrics/
    dre.py, drex.py, lse.py, lsex.py   ← metric evaluators
    battery.py                          ← compute_metrics(), METRIC_COLUMNS, METRIC_GROUPS
  viz/
    rea.py                              ← statistical analysis + figures
    loss.py                             ← loss curve plotting
  experiment/
    config.py                           ← ExperimentConfig dataclass
    merge.py                            ← cross-experiment merging
    templates/                          ← THIS DIRECTORY
  baselines/
    registry.py                         ← 15 external baseline models (12 core + 3 scVI)
    models/                             ← model wrappers

experiments/                           ← per-project, generated from templates
  experiment_config.py                 ← your PRESETS + dataset catalogue
  run_experiment.py                    ← from templates/run_experiment.py
  run_external_benchmark.py            ← from templates/run_external_benchmark.py
  visualize_experiment.py              ← from templates/visualize_experiment.py
```

## Files

| Template | Purpose | Project-specific sections |
|----------|---------|--------------------------|
| `run_experiment.py` | Train internal models, compute metrics → CSV | `load_data()`, `standardize_labels()`, `create_data_splitter()`, PRESETS import, HP override map |
| `run_external_benchmark.py` | Train external baselines with identical metrics | Dataset catalogue, data loading imports, default HP constants |
| `visualize_experiment.py` | CSV → REA → publication-quality figures | PRESETS import, `DEFAULT_PALETTE` |

**Note:** `ExperimentConfig` and `METRIC_COLUMNS` are imported directly from
`eval_lib` — they are NOT defined in the templates.  Projects create their own
`experiment_config.py` with model imports, dataset catalogues, and PRESETS.

## What to customise when switching projects

### 1. Create `experiments/experiment_config.py`
- **Model imports**: Your model classes
- **Dataset catalogue**: Paths, label keys, data types
- **Model builder functions**: Return `{name: {"class": Cls, "params": {...}}}`
- **PRESETS dict**: Register experiment presets as `ExperimentConfig` instances

### 2. `run_experiment.py` (copy from template)
- **`load_data()`**: Data loading / preprocessing pipeline
- **`standardize_labels()`**: Extend `CANDIDATE_KEYS` for your label columns
- **`create_data_splitter()`**: Your DataSplitter producing PyTorch loaders
- **`_MODEL_HP_MAP`**: CLI flags → model param keys

### 3. `run_external_benchmark.py` (copy from template)
- **Imports**: Point to your `load_data`, `create_data_splitter`, PRESETS
- **Default HP constants**: Match your internal experiment settings

### 4. `visualize_experiment.py` (copy from template)
- **PRESETS import**: Same as experiment_config.py
- **DEFAULT_PALETTE**: Cosmetic choice (optional)

## Model interface contract

```python
class MyModel(nn.Module):
    def __init__(self, input_dim: int, **params): ...
    def fit(self, train_loader, val_loader, epochs, lr, device,
            patience, verbose, verbose_every, weight_decay=0) -> dict: ...
    def extract_latent(self, loader, device) -> {"latent": np.ndarray}: ...
```

`fit()` returns dict with at minimum `train_loss` (list of per-epoch floats).
Optional keys: `val_loss`, `train_recon_loss`, `val_recon_loss`, `kl_loss`.

## CSV output format

Both runners produce identical CSV schemas:

- **Per-dataset tables** (`tables/{dataset}_df.csv`): one row per method,
  columns = `method` + 38 metric columns (`METRIC_COLUMNS`)
- **Training series** (`series/{dataset}_dfs.csv`): one row per
  (method, epoch), columns = `epoch`, `hue`, `train_loss`, `val_loss`, etc.

These are consumed by `visualize_experiment.py` and
`eval_lib.viz.rea.RigorousExperimentalAnalyzer`.
