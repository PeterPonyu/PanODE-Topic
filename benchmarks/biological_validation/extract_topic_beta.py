"""Extract Topic decoder beta (topic-word distribution) from saved checkpoints.

For each Topic model checkpoint, saves a ``{model}_{dataset}_beta.npz``
file containing:
  - ``beta``       : numpy array [n_topics, n_genes] — softmax topic-word probs
  - ``gene_names`` : list of HVG gene names

Usage
-----
    python -m benchmarks.biological_validation.extract_topic_beta

This runs automatically on all Topic-* checkpoints found in
``benchmarks/benchmark_results/models/``.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.config import BIO_RESULTS_DIR, MODELS_DIR
BIO_RESULTS = BIO_RESULTS_DIR


def extract_all():
    """Iterate over Topic-* checkpoints, extract beta, and save."""
    from benchmarks.biological_validation import load_model

    if not MODELS_DIR.is_dir():
        print(f"  Models directory not found: {MODELS_DIR}")
        return

    topic_ckpts = sorted(MODELS_DIR.glob("Topic-*.pt"))
    if not topic_ckpts:
        print("  No Topic model checkpoints found.")
        return

    BIO_RESULTS.mkdir(parents=True, exist_ok=True)

    for ckpt_path in topic_ckpts:
        stem = ckpt_path.stem  # e.g. "Topic-Base_dentate_bio" or "Topic-Base_setty_3000c_ep600_lr1e-3_20260224_123849"
        # Parse model name and dataset from checkpoint filename
        # Format 1: {ModelName}_{dataset}_bio.pt
        # Format 2: {ModelName}_{dataset}_{params}_{timestamp}.pt (setty-style)
        parts = stem.rsplit("_bio", 1)
        if len(parts) == 2 and parts[1] == "":
            # Format 1: ends with "_bio"
            prefix = parts[0]  # "Topic-Base_dentate"
            tokens = prefix.split("_")
            dataset = tokens[-1]
            model_name_raw = "_".join(tokens[:-1])  # "Topic-Base"
        else:
            # Format 2: parameterized filename — model is first hyphenated
            # token, dataset is the second underscore-delimited token
            tokens = stem.split("_")
            model_name_raw = tokens[0]  # "Topic-Base"
            dataset = tokens[1] if len(tokens) > 1 else "unknown"  # "setty"

        tag = f"{model_name_raw}_{dataset}"
        out_path = BIO_RESULTS / f"{tag}_beta.npz"

        try:
            model, _ = load_model(str(ckpt_path), device="cpu")
        except Exception as e:
            print(f"  Failed to load {stem}: {e}")
            continue

        # Extract beta via model method (all Topic models expose this)
        if hasattr(model, "get_topic_word_distribution"):
            beta = model.get_topic_word_distribution().cpu().numpy()
        elif hasattr(model, "ae") and hasattr(model.ae, "decoder"):
            beta = model.ae.decoder.beta.detach().cpu().numpy()
        else:
            print(f"  {stem}: no beta found — not a Topic model?")
            continue

        # Try to get gene names from a matching importance file,
        # otherwise load the training dataset to get them
        gene_names = None
        imp_path = BIO_RESULTS / f"{tag}_importance.npz"
        if imp_path.exists():
            gene_names = np.load(imp_path, allow_pickle=True).get("gene_names")

        if gene_names is None or len(gene_names) != beta.shape[1]:
            # Load training dataset to get gene names
            try:
                from benchmarks.biological_validation import load_data_with_genes
                _, gn = load_data_with_genes(dataset, hvg=beta.shape[1])
                gene_names = np.array(gn)
            except Exception as e2:
                print(f"    Could not load gene names for {dataset}: {e2}")
                gene_names = np.array([f"gene_{i}" for i in range(beta.shape[1])])

        save_kw = {"beta": beta}
        if gene_names is not None:
            save_kw["gene_names"] = gene_names
        np.savez(out_path, **save_kw)
        print(f"  Saved {out_path.name}  beta shape {beta.shape}")

    print("  Done.")


if __name__ == "__main__":
    extract_all()
