"""Shared training utilities for all benchmark scripts.

Centralises the duplicated training loop, model parameter factories,
convergence diagnostics printing, output directory setup, latent saving,
and common CLI argument definitions so each benchmark script only contains
its own sweep logic and ``main()``.

Functions:
    train_and_evaluate   -- unified train → extract → metrics pipeline
    make_topic_params    -- default Topic-Base model params
    make_dpmm_params     -- default DPMM-Base model params
    print_convergence    -- pretty-print convergence diagnostics
    setup_series_dirs    -- create csv/plots/meta/latents dir trees
    save_latents         -- persist latent .npz files per-series
    add_common_cli_args  -- add --verbose-every / --seed / --no-plots etc.
"""

import gc
import time
import numpy as np
import torch

from benchmarks.config import BASE_CONFIG, ensure_dirs
from benchmarks.model_registry import is_cuda_oom
from benchmarks.metrics_utils import (
    compute_metrics,
    compute_latent_diagnostics,
    convergence_diagnostics as _convergence_diagnostics)


# ═══════════════════════════════════════════════════════════════════════════════
# Default model parameter factories (previously duplicated 3×)
# ═══════════════════════════════════════════════════════════════════════════════

def make_topic_params(latent_dim=None, kl_weight=0.01,
                      encoder_hidden=128, dropout=0.0):
    """Return default Topic-Base architecture params.

    Used by benchmark_sensitivity, benchmark_training, and
    benchmark_preprocessing.
    """
    if latent_dim is None:
        latent_dim = BASE_CONFIG.latent_dim
    return {
        "n_topics": latent_dim,
        "encoder_hidden": encoder_hidden,
        "kl_weight": kl_weight,
        "encoder_drop": dropout,
    }


def make_dpmm_params(latent_dim=None, warmup_ratio=0.9,
                     encoder_dims=None, decoder_dims=None, dropout=0.15):
    """Return default DPMM-Base architecture params.

    Used by benchmark_sensitivity, benchmark_training, and
    benchmark_preprocessing.
    """
    if latent_dim is None:
        latent_dim = BASE_CONFIG.latent_dim
    if encoder_dims is None:
        encoder_dims = [256, 128]
    if decoder_dims is None:
        decoder_dims = [128, 256]
    return {
        "latent_dim": latent_dim,
        "encoder_dims": list(encoder_dims),
        "decoder_dims": list(decoder_dims),
        "dpmm_warmup_ratio": warmup_ratio,
        "dropout_rate": dropout,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Unified training → evaluation loop (previously duplicated 5×)
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    name, model_cls, params, splitter, device,
    lr=1e-3, epochs=600, weight_decay=1e-5,
    verbose_every=50, data_type="trajectory",
    extra_fields=None, patience=None, dre_k=15):
    """Train a single model variant, compute metrics, return result dict.

    This is the single canonical training loop used by all benchmark
    entry-point scripts.

    Parameters
    ----------
    name : str
        Human-readable model name (e.g. ``"Topic(kl=0.01)"``).
    model_cls : type
        Model class (e.g. ``TopicODEModel``).
    params : dict
        Architecture params passed to ``model_cls(input_dim=..., **params)``.
    splitter : DataSplitter
        Provides train/val/test loaders and labels.
    device : torch.device
        CUDA or CPU.
    lr, epochs, weight_decay, verbose_every, data_type
        Training hyper-parameters.
    extra_fields : dict, optional
        Additional key-value pairs merged into the result dict.
    patience : int or None
        Early stopping patience.  ``None`` disables early stopping
        (patience=9999 is passed to ``model.fit``).
    dre_k : int
        Number of neighbours for DRE / DREX / LSEX evaluation.

    Returns
    -------
    dict
        Contains ``'Model'``, all metrics, diagnostics, convergence info,
        ``'latent'`` (np.ndarray or None), and efficiency stats.
    """
    gc.collect()
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")

    try:
        params = dict(params)  # copy to avoid mutation
        # Pop fit-specific keys that model.__init__ doesn't accept
        fit_lr = params.pop("fit_lr", lr)
        fit_wd = params.pop("fit_weight_decay", weight_decay)
        fit_epochs = params.pop("fit_epochs", epochs)

        model = model_cls(input_dim=splitter.n_var, **params)
        model = model.to(device)

        fit_patience = patience if patience is not None else 9999

        history = model.fit(
            train_loader=splitter.train_loader,
            val_loader=splitter.val_loader,
            epochs=fit_epochs,
            lr=fit_lr,
            device=str(device),
            patience=fit_patience,
            verbose=1,
            verbose_every=verbose_every,
            weight_decay=fit_wd)

        epochs_trained = len(history.get("train_loss", [])) or fit_epochs
        elapsed = time.time() - start
        sec_per_epoch = elapsed / max(epochs_trained, 1)

        peak_gpu_mb = 0.0
        if device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Convergence diagnostics
        conv_diag = _convergence_diagnostics(history, window=50)
        print_convergence(conv_diag)

        # Extract latent representations
        latent_dict = model.extract_latent(
            splitter.test_loader, device=str(device))
        latent = latent_dict["latent"]

        # Compute metrics
        metrics = compute_metrics(
            latent, splitter.labels_test,
            data_type=data_type, dre_k=dre_k)
        diagnostics = compute_latent_diagnostics(latent)
        n_params = sum(p.numel() for p in model.parameters())

        print(
            f"  NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f} "
            f"ASW={metrics['ASW']:.4f} | {elapsed:.1f}s "
            f"({sec_per_epoch:.2f}s/ep, GPU {peak_gpu_mb:.0f}MB, "
            f"params {n_params:,})"
        )

        result = {
            "Model": name,
            "Time_s": elapsed,
            "SecPerEpoch": sec_per_epoch,
            "PeakGPU_MB": peak_gpu_mb,
            "NumParams": n_params,
            "LR": fit_lr,
            "Epochs": fit_epochs,
            "EpochsTrained": epochs_trained,
            "latent": latent,
            "history": history,
            "model_obj": model,
        }
        result.update(metrics)
        result.update(diagnostics)
        if conv_diag:
            result.update(conv_diag)
        if extra_fields:
            result.update(extra_fields)
        return result

    except Exception as exc:
        if device.type == "cuda" and is_cuda_oom(exc):
            print("CUDA OOM → retry CPU…")
            torch.cuda.empty_cache()
            gc.collect()
            return train_and_evaluate(
                name, model_cls, params, splitter, torch.device("cpu"),
                lr=lr, epochs=epochs, weight_decay=weight_decay,
                verbose_every=verbose_every, data_type=data_type,
                extra_fields=extra_fields, patience=patience,
                dre_k=dre_k)
        elapsed = time.time() - start
        print(f"ERROR: {str(exc)[:120]}")
        import traceback
        traceback.print_exc()
        result = {
            "Model": name,
            "Time_s": elapsed,
            "Error": str(exc)[:200],
            "latent": None,
            "NMI": 0, "ARI": 0,
            "Epochs": epochs,
            "EpochsTrained": 0,
        }
        if extra_fields:
            result.update(extra_fields)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence printing (previously duplicated 5×)
# ═══════════════════════════════════════════════════════════════════════════════

def print_convergence(conv_diag):
    """Pretty-print convergence diagnostics dict (if available)."""
    if not conv_diag:
        return
    rpct = conv_diag.get("recon_rel_change_pct", float("nan"))
    flag = "✓ CONVERGED" if abs(rpct) < 1.0 else "✗ NOT converged"
    print(f"  Convergence: Δ%={rpct:+.2f}%  "
          f"(last {conv_diag['window']} ep)  {flag}")


# ═══════════════════════════════════════════════════════════════════════════════
# Output directory helpers (previously duplicated 4×)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_series_dirs(base_dir, include_latents=True):
    """Create csv/plots/meta[/latents] sub-dirs for topic & dpmm.

    Returns
    -------
    dict : ``{series_key: {"csv": Path, "plots": Path, "meta": Path, ...}}``
    """
    sdirs = {}
    for sk in ("topic", "dpmm"):
        d = {
            "csv":   base_dir / "csv"   / sk,
            "plots": base_dir / "plots" / sk,
            "meta":  base_dir / "meta"  / sk,
        }
        if include_latents:
            d["latents"] = base_dir / "latents" / sk
        ensure_dirs(*d.values())
        sdirs[sk] = d
    return sdirs


def save_latents(latents, sdirs, tag, ds_keys=None, variants=None):
    """Persist latent .npz files organised by paper group (dpmm / topic).

    *latents* is a dict mapping ``"ds_key::model_name"`` → np.ndarray.
    """
    from benchmarks.model_registry import MODELS, paper_group as _pg

    for lat_key, lat_arr in latents.items():
        ds_key, model_key = lat_key.split("::", 1) if "::" in lat_key else ("", lat_key)
        safe = model_key.replace("/", "_").replace(" ", "_")

        # Determine paper group for this model
        if model_key in MODELS:
            pg = _pg(MODELS[model_key]['series'])
        else:
            # Fallback to name-based detection
            is_topic = "topic" in model_key.lower()
            pg = "topic" if is_topic else "dpmm"

        if variants is not None:
            # Check if model is in the variants list
            matched = False
            for v in variants:
                if v.get("name") == model_key:
                    pg = _pg(v.get("series", pg))
                    matched = True
                    break
            if not matched:
                continue

        lat_dir = sdirs.get(pg, {}).get("latents")
        if lat_dir is None:
            continue
        fname = f"{safe}_{ds_key}_{tag}.npz" if ds_key else f"{safe}_{tag}.npz"
        np.savez(lat_dir / fname, latent=lat_arr)
        if ds_keys and len(ds_keys) == 1:
            np.savez(lat_dir / f"{safe}_{tag}.npz", latent=lat_arr)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI helpers (previously duplicated 4×)
# ═══════════════════════════════════════════════════════════════════════════════

def add_common_cli_args(parser):
    """Add the standard --verbose-every / --seed / --no-plots / --datasets args."""
    from benchmarks.dataset_registry import DATASET_REGISTRY

    parser.add_argument("--verbose-every", type=int, default=BASE_CONFIG.verbose_every)
    parser.add_argument("--seed", type=int, default=BASE_CONFIG.seed)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--datasets", nargs="+", default=["setty"],
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to run (default: setty only).")


# ═══════════════════════════════════════════════════════════════════════════════
# Model selection & override helpers (shared by benchmark_base, crossdata)
# ═══════════════════════════════════════════════════════════════════════════════

def select_models(args, models_dict=None):
    """Filter *models_dict* based on ``--series`` / ``--models`` CLI flags.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``series`` (str) and optionally ``models`` (list[str]).
    models_dict : dict or None
        Model registry to filter.  Defaults to ``MODELS`` from
        ``model_registry``.

    Returns
    -------
    dict
        Ordered subset of *models_dict* matching the selection.
    """
    from benchmarks.model_registry import MODELS, SERIES_GROUPS
    if models_dict is None:
        models_dict = MODELS

    if getattr(args, "models", None):
        names = [n.strip() if isinstance(n, str) else n for n in args.models]
        ordered = [n for n in names if n in models_dict]
        selected = {k: models_dict[k] for k in ordered}
        missing = set(names) - set(selected)
        if missing:
            print(f"  WARNING: unknown model names ignored: {missing}")
        return selected

    series = getattr(args, "series", "all")
    if series == "all":
        return dict(models_dict)

    allowed = []
    for s in series.split(","):
        s = s.strip()
        if s in SERIES_GROUPS:
            for name in SERIES_GROUPS[s]:
                if name not in allowed:
                    allowed.append(name)
        else:
            for k, v in models_dict.items():
                if v.get("series") == s and k not in allowed:
                    allowed.append(k)
    return {k: models_dict[k] for k in allowed if k in models_dict}


def apply_model_overrides(selected_models, args):
    """Mutate *selected_models* params according to CLI override flags.

    Recognises ``--override-epochs``, ``--override-wd``,
    ``--override-dropout``, ``--override-kl-weight`` on *args*.
    """
    if getattr(args, "override_epochs", None) is not None:
        for v in selected_models.values():
            v["params"]["fit_epochs"] = args.override_epochs
    if getattr(args, "override_wd", None) is not None:
        for v in selected_models.values():
            v["params"]["fit_weight_decay"] = args.override_wd
    if getattr(args, "override_dropout", None) is not None:
        for v in selected_models.values():
            p = v["params"]
            for dkey in ("dropout_rate", "dropout", "encoder_drop"):
                if dkey in p:
                    p[dkey] = args.override_dropout
    if getattr(args, "override_kl_weight", None) is not None:
        for v in selected_models.values():
            if "kl_weight" in v["params"]:
                v["params"]["kl_weight"] = args.override_kl_weight
    return selected_models
