#!/usr/bin/env python
"""
Consolidated Benchmark — All Model Variants (No ODE)

Models (12 variants across 2 architecture families):
  Pure baselines (no clustering prior, no strategy):
    - Pure-AE / Pure-VAE
  Standalone strategy ablations (no clustering prior):
    - Pure-Transformer-AE / Pure-Contrastive-AE
    - Pure-Transformer-VAE / Pure-Contrastive-VAE
  DPMM series (AE backbone + DPMM clustering):
    - DPMM-Base / DPMM-Transformer / DPMM-Contrastive
  Topic / LDA series (VAE backbone + Dirichlet prior):
    - Topic-Base / Topic-Transformer / Topic-Contrastive

Usage:
  python benchmarks/benchmark_base.py --epochs 200 --no-early-stopping
  python benchmarks/benchmark_base.py --series dpmm --override-epochs 400
  python benchmarks/benchmark_base.py --models Pure-AE DPMM-Base
"""

import sys
import os
import argparse
import json
import gc
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import warnings
import numpy as np
import pandas as pd
import torch
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Imports from extracted modules ────────────────────────────────────────────
from benchmarks.config import (
    BASE_CONFIG, DEFAULT_OUTPUT_DIR, ensure_dirs, set_global_seed)
from benchmarks.dataset_registry import DATASET_REGISTRY
from benchmarks.model_registry import (
    MODELS, SERIES_GROUPS, ABLATION_STEPS, is_cuda_oom, paper_group)
from benchmarks.metrics_utils import convergence_diagnostics
from benchmarks.data_utils import load_or_preprocess_adata
from benchmarks.run_manifest import append_run
from benchmarks.train_utils import (
    train_and_evaluate, print_convergence,
    select_models, apply_model_overrides)

from utils.data import DataSplitter
from utils.viz import plot_umap_grid, plot_all_metrics_barplot

# ═══════════════════════════════════════════════════════════════════════════════
# Module-level configuration (overridden by CLI via apply_cli_overrides)
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH = str(BASE_CONFIG.data_path)
EPOCHS = BASE_CONFIG.epochs
LATENT_DIM = BASE_CONFIG.latent_dim
LR = BASE_CONFIG.lr
BATCH_SIZE = BASE_CONFIG.batch_size
DEVICE = BASE_CONFIG.device
HVG_TOP_GENES = BASE_CONFIG.hvg_top_genes
MAX_CELLS = BASE_CONFIG.max_cells
PATIENCE = BASE_CONFIG.patience
EARLY_STOPPING = BASE_CONFIG.early_stopping
VERBOSE_EVERY = BASE_CONFIG.verbose_every
DATA_TYPE = BASE_CONFIG.data_type
DRE_K = BASE_CONFIG.dre_k
SEED = BASE_CONFIG.seed

# Output directories
BASE_DIR = DEFAULT_OUTPUT_DIR / "base"
ensure_dirs(BASE_DIR)

DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"

CSV_DIR = BASE_DIR / "csv"
PLOT_DIR = BASE_DIR / "plots"
META_DIR = BASE_DIR / "meta"

SERIES_DIRS = {
    'dpmm': {
        'csv': CSV_DIR / 'dpmm',
        'plots': PLOT_DIR / 'dpmm',
        'meta': META_DIR / 'dpmm',
    },
    'topic': {
        'csv': CSV_DIR / 'topic',
        'plots': PLOT_DIR / 'topic',
        'meta': META_DIR / 'topic',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Training wrapper (delegates to train_and_evaluate)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_model(name, cfg, splitter, device, lr, epochs, patience):
    """Train a single model variant via the shared training loop."""
    return train_and_evaluate(
        name=name,
        model_cls=cfg['class'],
        params=cfg['params'],
        splitter=splitter,
        device=device,
        lr=lr,
        epochs=epochs if epochs is not None else EPOCHS,
        verbose_every=VERBOSE_EVERY,
        data_type=DATA_TYPE,
        patience=patience,
        dre_k=DRE_K,
        extra_fields={'Series': cfg.get('series', '')})


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark base models (no ODE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    g = p.add_argument_group('Data')
    g.add_argument("--data-path", type=str, default=str(DATA_PATH))
    g.add_argument("--max-cells", type=int, default=MAX_CELLS)
    g.add_argument("--hvg-top-genes", type=int, default=HVG_TOP_GENES)
    g.add_argument("--data-type", type=str, default=DATA_TYPE,
                   choices=["trajectory", "scRNA-seq", "clustering"])

    # Model
    g = p.add_argument_group('Model')
    g.add_argument("--latent-dim", type=int, default=LATENT_DIM)

    # Training
    g = p.add_argument_group('Training')
    g.add_argument("--epochs", type=int, default=EPOCHS)
    g.add_argument("--lr", type=float, default=LR)
    g.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    g.add_argument("--patience", type=int, default=PATIENCE)
    g.add_argument("--no-early-stopping", action="store_true")
    g.add_argument("--verbose-every", type=int, default=VERBOSE_EVERY)

    # Output
    g = p.add_argument_group('Output')
    g.add_argument("--results-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    g.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--no-plots", action="store_true")

    # Misc
    g = p.add_argument_group('Misc')
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--dre-k", type=int, default=DRE_K)

    # Batch / series
    g = p.add_argument_group('Batch Runs')
    g.add_argument("--epoch-series", type=str, default="",
                   help="Comma-separated epoch list, e.g. 100,200,300")
    g.add_argument("--lr-series", type=str, default="",
                   help="Comma-separated LR list, e.g. 1e-2,1e-3,1e-4")
    g.add_argument("--series", type=str, default="all",
                   help="all, dpmm, topic, pure, pure-ae, pure-vae "
                        "or comma-separated")
    g.add_argument("--models", nargs="+", default=None,
                   help="Explicit model names (overrides --series)")
    g.add_argument("--override-epochs", type=int, default=None)
    g.add_argument("--override-wd", type=float, default=None)
    g.add_argument("--override-dropout", type=float, default=None)
    g.add_argument("--override-kl-weight", type=float, default=None)

    return p.parse_args()


def apply_cli_overrides(args):
    """Apply CLI arguments to module-level globals."""
    global DATA_PATH, EPOCHS, LATENT_DIM, LR, BATCH_SIZE, MAX_CELLS
    global HVG_TOP_GENES, SEED, DATA_TYPE, DRE_K, VERBOSE_EVERY
    global PATIENCE, EARLY_STOPPING
    global BASE_DIR, DEFAULT_CACHE_DIR, CSV_DIR, PLOT_DIR, META_DIR
    global SERIES_DIRS

    DATA_PATH = args.data_path
    EPOCHS = args.epochs
    LATENT_DIM = args.latent_dim
    LR = args.lr
    BATCH_SIZE = args.batch_size
    MAX_CELLS = args.max_cells if args.max_cells > 0 else None
    HVG_TOP_GENES = args.hvg_top_genes
    SEED = args.seed
    DATA_TYPE = args.data_type
    DRE_K = args.dre_k
    VERBOSE_EVERY = args.verbose_every

    EARLY_STOPPING = not args.no_early_stopping
    PATIENCE = args.patience if EARLY_STOPPING else None

    BASE_DIR = Path(args.results_dir) / "base"
    DEFAULT_CACHE_DIR = Path(args.cache_dir)
    ensure_dirs(BASE_DIR, DEFAULT_CACHE_DIR)

    CSV_DIR = BASE_DIR / "csv"
    PLOT_DIR = BASE_DIR / "plots"
    META_DIR = BASE_DIR / "meta"
    ensure_dirs(CSV_DIR, PLOT_DIR, META_DIR)

    SERIES_DIRS = {
        'dpmm': {
            'csv': CSV_DIR / 'dpmm',
            'plots': PLOT_DIR / 'dpmm',
            'meta': META_DIR / 'dpmm',
        },
        'topic': {
            'csv': CSV_DIR / 'topic',
            'plots': PLOT_DIR / 'topic',
            'meta': META_DIR / 'topic',
        },
    }
    ensure_dirs(
        SERIES_DIRS['dpmm']['csv'], SERIES_DIRS['dpmm']['plots'],
        SERIES_DIRS['dpmm']['meta'],
        SERIES_DIRS['topic']['csv'], SERIES_DIRS['topic']['plots'],
        SERIES_DIRS['topic']['meta'])


# ═══════════════════════════════════════════════════════════════════════════════
# Epoch / LR sweep runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark_epoch(splitter, adata, epoch, lr, args):
    """Run benchmark for a specific epoch count and learning rate.

    Returns ``(df, tag, generated_plots, latents)``.
    """
    global EPOCHS, LR
    EPOCHS = epoch
    LR = lr

    results, latents = [], {}
    device = torch.device(DEVICE)

    selected_models = select_models(args)
    apply_model_overrides(selected_models, args)

    es_str = f"Patience={PATIENCE}" if PATIENCE is not None else "Disabled"
    print(f"\n{'='*60}\nTRAINING {len(selected_models)} MODELS\n"
          f"  Epochs: {epoch}, LR: {lr:.0e}, Early Stop: {es_str}\n{'='*60}")

    for name, cfg in selected_models.items():
        r = _train_model(name, cfg, splitter, device, lr, epoch, PATIENCE)
        if r.get('latent') is not None:
            latents[name] = r.pop('latent')
        results.append(r)
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lr_str = f"{lr:.0e}".replace('+', '').replace('-0', '-')
    data_name = Path(DATA_PATH).stem
    n_cells = int(adata.n_obs)
    tag = f"{data_name}_{n_cells}c_ep{epoch}_lr{lr_str}_{timestamp}"

    # ── Save per-series CSVs (group by paper: pure-ae→dpmm, pure-vae→topic) ──
    df['_paper'] = df['Series'].map(lambda s: paper_group(s))
    for sk in ('dpmm', 'topic'):
        sdf = df[df['_paper'] == sk].drop(columns=['_paper'])
        if sdf.empty:
            continue
        csv_path = SERIES_DIRS[sk]['csv'] / f"results_{tag}.csv"
        sdf.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    df.drop(columns=['_paper'], inplace=True, errors='ignore')

    # ── Per-model metadata ──
    model_configs = {}
    epochs_map = {}
    for name, cfg in selected_models.items():
        params = dict(cfg['params'])
        fit_lr = params.pop('fit_lr', lr)
        row = df[df['Model'] == name]
        ep_trained = (int(row['EpochsTrained'].values[0])
                      if not row.empty and 'EpochsTrained' in row.columns
                      else epoch)
        epochs_map[name] = ep_trained
        model_configs[name] = {
            'series': cfg['series'],
            'class': cfg['class'].__name__,
            'lr': fit_lr,
            'epochs_requested': epoch,
            'epochs_trained': ep_trained,
            'params': params,
        }

    meta = {
        "timestamp": timestamp,
        "data_name": data_name,
        "data_path": DATA_PATH,
        "n_cells": n_cells,
        "n_genes": int(adata.n_vars),
        "hvg_top_genes": HVG_TOP_GENES,
        "max_cells": MAX_CELLS,
        "epochs_requested": epoch,
        "epochs_trained": epochs_map,
        "latent_dim": LATENT_DIM,
        "lr": lr,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "data_type": DATA_TYPE,
        "dre_k": DRE_K,
        "early_stopping": EARLY_STOPPING,
        "patience": PATIENCE,
        "verbose_every": VERBOSE_EVERY,
        "models": list(selected_models.keys()),
        "model_configs": model_configs,
        "metrics": [c for c in df.columns if c not in ("latent", "history")],
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "cache_used": not args.no_cache,
    }

    active_papers = {paper_group(cfg['series']) for cfg in selected_models.values()}
    for sk in ('dpmm', 'topic'):
        if sk not in active_papers:
            continue
        meta_path = SERIES_DIRS[sk]['meta'] / f"run_{tag}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({**meta, "series": sk}, f, indent=2)

    # ── Save model states + histories ──
    models_dir = Path(args.results_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if not isinstance(r, dict):
            continue
        m_name = r.get('Model', '')
        if r.get('model_obj') is not None:
            try:
                torch.save({
                    "state_dict": r['model_obj'].state_dict(),
                    "config": {
                        "model_name": m_name,
                        "input_dim": splitter.n_var,
                        "params": dict(MODELS[m_name]["params"]),
                    },
                }, models_dir / f"{m_name}_{tag}.pt")
            except Exception:
                pass
        if r.get('history') is not None:
            try:
                with open(models_dir / f"{m_name}_{tag}_history.json", "w") as hf:
                    json.dump(r['history'], hf)
            except Exception:
                pass

    # ── Append to run manifest ──
    for sk in ('dpmm', 'topic'):
        if sk not in active_papers:
            continue
        csv_path = SERIES_DIRS[sk]['csv'] / f"results_{tag}.csv"
        append_run(
            results_dir=Path(args.results_dir),
            script="benchmark_base",
            tag=tag,
            series=sk,
            dataset=data_name,
            n_cells=n_cells,
            epochs=epoch,
            lr=lr,
            seed=SEED,
            csv_path=str(csv_path),
            models=list(selected_models.keys()))

    # ── Plots ──
    generated_plots = []
    if getattr(args, 'no_plots', False):
        print("  Plots disabled via --no-plots")
        return df, tag, generated_plots, latents

    display_names = {'dpmm': 'DPMM', 'topic': 'Topic'}
    for sk in ('dpmm', 'topic'):
        if sk not in active_papers:
            continue
        sdf = df[df['Series'].map(lambda s: paper_group(s)) == sk]
        if sdf.empty:
            continue
        sl = {k: v for k, v in latents.items()
              if paper_group(MODELS[k]['series']) == sk}
        if sl:
            umap_path = SERIES_DIRS[sk]['plots'] / f"umap_{tag}.png"
            try:
                plot_umap_grid(
                    sl, splitter.labels_test,
                    f"{display_names[sk]} UMAP (Test Set) - {epoch} epochs",
                    str(umap_path))
                generated_plots.append(str(umap_path))
                print(f"  UMAP plot saved: {umap_path}")
            except Exception as e:
                print(f"  UMAP plotting failed: {e}")

        metrics_path = SERIES_DIRS[sk]['plots'] / f"all_metrics_{tag}.png"
        try:
            plot_all_metrics_barplot(
                sdf, str(metrics_path),
                title=f"{display_names[sk]} - All Metrics ({epoch} epochs)")
            generated_plots.append(str(metrics_path))
            print(f"  Metrics plot saved: {metrics_path}")
        except Exception as e:
            print(f"  Metrics plotting failed: {e}")

    return df, tag, generated_plots, latents


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    apply_cli_overrides(args)

    es_str = f"Patience={PATIENCE}" if PATIENCE is not None else "Disabled"
    print("=" * 60)
    print("CONSOLIDATED BENCHMARK (No ODE)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")
    print(f"Max cells: {MAX_CELLS if MAX_CELLS else 'No limit'}")
    print(f"HVG genes: {HVG_TOP_GENES}")
    print(f"Epochs: {EPOCHS}, LR: {LR:.0e}")
    print(f"Early Stopping: {es_str}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Models: {list(MODELS.keys())}")

    set_global_seed(SEED)

    adata = load_or_preprocess_adata(
        DATA_PATH, max_cells=MAX_CELLS, hvg_top_genes=HVG_TOP_GENES,
        seed=SEED, cache_dir=DEFAULT_CACHE_DIR,
        use_cache=not args.no_cache)

    # Standardize label column → 'cell_type' for DataSplitter compatibility
    label_key = 'clusters' if 'clusters' in adata.obs.columns else 'cell_type'
    if label_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_key].copy()
        print(f"  Labels: {len(np.unique(adata.obs['cell_type'].values))} "
              f"types (from '{label_key}' → 'cell_type')")
    else:
        print("  WARNING: No label column found — DataSplitter will use KMeans pseudo-labels")

    splitter = DataSplitter(
        adata=adata, layer='counts', train_size=0.7, val_size=0.15,
        test_size=0.15, batch_size=BATCH_SIZE, latent_dim=LATENT_DIM,
        random_seed=SEED, verbose=True)

    epoch_series = [int(e) for e in args.epoch_series.split(',') if e.strip()]
    lr_series = [float(v) for v in args.lr_series.split(',') if v.strip()]
    if not epoch_series:
        epoch_series = [EPOCHS]
    if not lr_series:
        lr_series = [LR]

    all_plots = []
    total = len(epoch_series) * len(lr_series)
    idx = 0

    for lr in lr_series:
        for ep in epoch_series:
            idx += 1
            print(f"\n{'#'*60}\n# RUN {idx}/{total}: "
                  f"Epochs={ep}, LR={lr:.0e}\n{'#'*60}")
            df, tag, plots, latents = run_benchmark_epoch(
                splitter, adata, ep, lr, args)
            all_plots.extend(plots)

    if total > 1:
        print(f"\n{'='*60}\nAll {total} runs completed.  "
              f"Epoch: {epoch_series}  "
              f"LR: {[f'{v:.0e}' for v in lr_series]}")
        if all_plots:
            print(f"Generated {len(all_plots)} plot files.")
        print(f"{'='*60}")
        return

    # ── Single-run comprehensive summary ──
    _print_summary(df, all_plots)


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary(df, generated_plots):
    """Print the full summary table block for a single-run benchmark."""
    print("\n" + "=" * 80)
    print("SUMMARY (sorted by model order)")
    print("=" * 80)

    _print_table(
        df, 'CLUSTERING METRICS',
        cols=[('NMI', '.4f'), ('ARI', '.4f'), ('ASW', '.4f'),
              ('DAV', '.4f'), ('CAL', '.1f')],
        extra_header=f"{'EpTrained':>10} {'Time':>8}",
        extra_fn=lambda r: (f"{int(r.get('EpochsTrained', 0)):>10d} "
                            f"{r.get('Time_s', 0):>7.1f}s"))

    if 'recon_final' in df.columns:
        _print_table(
            df, 'CONVERGENCE DIAGNOSTICS',
            cols=[('recon_final', '.4f', 'ReconFinal'),
                  ('recon_rel_change_pct', '.2f', 'Recon_Δ%'),
                  ('total_loss_final', '.4f', 'TotalLoss')],
            extra_header=f"{'Converged':>10}",
            extra_fn=lambda r: (f"{'✓':>10}" if r.get('converged')
                                else f"{'✗':>10}"))

    if 'latent_std_mean' in df.columns:
        _print_table(
            df, 'LATENT DIAGNOSTICS',
            cols=[('latent_std_mean', '.4f', 'StdMean'),
                  ('latent_std_min', '.4f', 'StdMin'),
                  ('latent_near_zero_dims', 'd', 'DeadDims'),
                  ('latent_pairwise_dist_mean', '.4f', 'DistMean')])

    for title, prefix in [
        ('DRE (Embedding Quality)', 'DRE'),
        ('DRE (UMAP)', 'DRE_umap'),
        ('DRE (t-SNE)', 'DRE_tsne'),
    ]:
        _print_table(
            df, title,
            cols=[(f'{prefix}_distance_correlation', '.4f', 'DistCorr'),
                  (f'{prefix}_Q_local', '.4f', 'Q_local'),
                  (f'{prefix}_Q_global', '.4f', 'Q_global'),
                  (f'{prefix}_K_max', '.1f', 'K_max'),
                  (f'{prefix}_overall_quality', '.4f', 'Overall')])

    _print_table(
        df, 'LSE (Structure)',
        cols=[('LSE_manifold_dimensionality', '.4f', 'ManifDim'),
              ('LSE_spectral_decay_rate', '.4f', 'SpectDecay'),
              ('LSE_participation_ratio', '.4f', 'Particip'),
              ('LSE_anisotropy_score', '.4f', 'Anisotropy')])

    _print_table(
        df, 'LSE (Trajectory & Quality)',
        cols=[('LSE_trajectory_directionality', '.4f', 'TrajDir'),
              ('LSE_noise_resilience', '.4f', 'NoiseRes'),
              ('LSE_core_quality', '.4f', 'CoreQual'),
              ('LSE_overall_quality', '.4f', 'Overall')])

    _print_ablation_summary(df)

    print("\n" + "=" * 80)
    print(f"Results saved to {BASE_DIR}/")
    result_series = set(df['Series'].unique()) if 'Series' in df.columns else set()
    for sk in ('dpmm', 'topic'):
        if sk in result_series:
            print(f"  {sk.upper()}: csv/  meta/  plots/")
    if generated_plots:
        print(f"\nGenerated {len(generated_plots)} plots:")
        for p in generated_plots:
            print(f"  ✓ {p}")
    print("=" * 80)


def _print_table(df, title, cols, extra_header="", extra_fn=None):
    """Generic table printer."""
    header_parts = [f"{'Series':<6} {'Model':<22}"]
    for spec in cols:
        name = spec[2] if len(spec) > 2 else spec[0]
        header_parts.append(f"{name:>10}")
    if extra_header:
        header_parts.append(extra_header)
    header = " ".join(header_parts)

    print(f"\n{'='*40} {title} {'='*40}")
    print(header)
    print("-" * len(header))

    for _, r in df.iterrows():
        if 'Error' in r and pd.notna(r.get('Error')):
            print(f"{str(r.get('Series',''))[:6]:<6} {r['Model']:<22} "
                  f"ERROR: {str(r.get('Error',''))[:40]}")
            continue
        parts = [f"{str(r.get('Series',''))[:6]:<6} {r['Model']:<22}"]
        for spec in cols:
            key, fmt = spec[0], spec[1]
            val = r.get(key, np.nan)
            if pd.notna(val):
                parts.append(f"{int(val):>10d}" if fmt == 'd'
                             else f"{val:>10{fmt}}")
            else:
                parts.append(f"{'N/A':>10}")
        if extra_fn:
            parts.append(extra_fn(r))
        print(" ".join(parts))


def _print_ablation_summary(df):
    """Print ablation deltas within DPMM and Topic series."""
    ab_metrics = [('NMI', '.4f'), ('ARI', '.4f'),
                  ('LSE_overall_quality', '.4f'),
                  ('DRE_umap_overall_quality', '.4f')]

    for sk, steps in ABLATION_STEPS.items():
        print(f"\n{'='*40} ABLATION SUMMARY ({sk.upper()}) {'='*40}")
        hdr = f"{'Step':<16} {'Model':<22} "
        for key, _ in ab_metrics:
            hdr += f"{key:>10} "
        hdr += f"{'ΔNMI':>10} {'ΔARI':>10}"
        print(hdr)
        print("-" * len(hdr))

        prev_row = None
        for model_name, step_label in steps:
            row = df[df['Model'] == model_name]
            if row.empty:
                print(f"{step_label:<16} {model_name:<22} MISSING")
                prev_row = None
                continue
            row = row.iloc[0]
            parts = [f"{step_label:<16} {model_name:<22}"]
            for key, fmt in ab_metrics:
                val = row.get(key, np.nan)
                parts.append(f"{val:>10{fmt}}" if pd.notna(val)
                             else f"{'N/A':>10}")
            if prev_row is not None:
                dnmi = row.get('NMI', np.nan) - prev_row.get('NMI', np.nan)
                dari = row.get('ARI', np.nan) - prev_row.get('ARI', np.nan)
                parts.append(f"{dnmi:>10.4f}" if pd.notna(dnmi)
                             else f"{'N/A':>10}")
                parts.append(f"{dari:>10.4f}" if pd.notna(dari)
                             else f"{'N/A':>10}")
            else:
                parts.extend([f"{'-':>10}", f"{'-':>10}"])
            print(" ".join(parts))
            prev_row = row


if __name__ == '__main__':
    main()
