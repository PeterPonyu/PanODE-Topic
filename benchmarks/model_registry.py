"""Canonical model registry for the active Topic-FM benchmark suite.

The repository is now centered on the flow-matching topic models used in the
current article, plus their matched Pure-VAE baselines.

Exports
-------
MODELS
    Ordered mapping of benchmark model name to class/params/series metadata.
SERIES_GROUPS
    CLI-friendly series aliases used by runner ``--series`` flags.
ABLATION_STEPS
    Ordered labels for topic and pure-vae ablation summaries.
paper_group
    Map a model or series to the output group (currently ``"topic"``).
is_cuda_oom
    Lightweight helper for CUDA out-of-memory retry logic.
"""

from models.pure_vae import (
    PureVAEContrastiveModel,
    PureVAEModel,
    PureVAETransformerModel,
)
from models.topic_fm_base import TopicFMModel
from models.topic_fm_contrastive import TopicFMContrastiveModel
from models.topic_fm_transformer import TopicFMTransformerModel

try:
    from models.topic_fm_gat import TopicFMGATModel
except ImportError:
    TopicFMGATModel = None


MODELS = {
    "Pure-VAE": {
        "class": PureVAEModel,
        "params": {
            "latent_dim": 10,
            "encoder_hidden": 128,
            "decoder_dims": [128, 256],
            "encoder_drop": 0.1,
            "kl_weight": 1.0,
            "fit_lr": 1e-3,
            "fit_weight_decay": 1e-3,
            "fit_epochs": 1000,
        },
        "series": "pure-vae",
    },
    "Pure-Transformer-VAE": {
        "class": PureVAETransformerModel,
        "params": {
            "latent_dim": 10,
            "d_model": 128,
            "decoder_dims": [128, 256],
            "dropout": 0.1,
            "kl_weight": 1.0,
            "nhead": 4,
            "num_encoder_layers": 2,
            "fit_lr": 1e-3,
            "fit_weight_decay": 1e-3,
            "fit_epochs": 1000,
        },
        "series": "pure-vae",
    },
    "Pure-Contrastive-VAE": {
        "class": PureVAEContrastiveModel,
        "params": {
            "latent_dim": 10,
            "encoder_hidden": 128,
            "decoder_dims": [128, 256],
            "encoder_drop": 0.1,
            "kl_weight": 1.0,
            "moco_weight": 0.1,
            "fit_lr": 1e-3,
            "fit_weight_decay": 1e-3,
            "fit_epochs": 1000,
        },
        "series": "pure-vae",
    },
    "Topic-FM-Base": {
        "class": TopicFMModel,
        "params": {
            "n_topics": 10,
            "encoder_hidden": 128,
            "encoder_drop": 0.0,
            "kl_weight": 0.01,
            "flow_weight": 0.1,
            "flow_warmup_epochs": 50,
            "flow_integration_steps": 16,
            "fit_lr": 1e-3,
            "fit_weight_decay": 1e-3,
            "fit_epochs": 1000,
        },
        "series": "topic",
    },
    "Topic-FM-Transformer": {
        "class": TopicFMTransformerModel,
        "params": {
            "n_topics": 10,
            "d_model": 128,
            "dropout": 0.0,
            "kl_weight": 0.01,
            "nhead": 4,
            "num_encoder_layers": 2,
            "flow_weight": 0.1,
            "flow_warmup_epochs": 50,
            "flow_integration_steps": 16,
            "fit_lr": 1e-3,
            "fit_weight_decay": 1e-3,
            "fit_epochs": 1000,
        },
        "series": "topic",
    },
    "Topic-FM-Contrastive": {
        "class": TopicFMContrastiveModel,
        "params": {
            "n_topics": 10,
            "encoder_hidden": 128,
            "encoder_drop": 0.0,
            "kl_weight": 0.01,
            "moco_weight": 0.1,
            "flow_weight": 0.1,
            "flow_warmup_epochs": 50,
            "flow_integration_steps": 16,
            "fit_lr": 1e-3,
            "fit_weight_decay": 1e-3,
            "fit_epochs": 1000,
        },
        "series": "topic",
    },
}

if TopicFMGATModel is not None:
    MODELS["Topic-FM-GAT"] = {
        "class": TopicFMGATModel,
        "params": {
            "n_topics": 10,
            "hidden_dim": 128,
            "nhead": 4,
            "num_gat_layers": 2,
            "dropout": 0.0,
            "knn_k": 15,
            "knn_pca_dim": 50,
            "kl_weight": 0.01,
            "flow_weight": 0.1,
            "flow_warmup_epochs": 50,
            "flow_integration_steps": 16,
            "fit_lr": 1e-3,
            "fit_weight_decay": 1e-3,
            "fit_epochs": 1000,
        },
        "series": "topic",
    }


def is_cuda_oom(exception) -> bool:
    """Return True if *exception* looks like a CUDA OOM failure."""
    msg = str(exception).lower()
    return "out of memory" in msg or ("cuda" in msg and "alloc" in msg)


ABLATION_STEPS = {
    "topic": [
        ("Topic-FM-Base", "Base Topic-FM"),
        ("Topic-FM-Transformer", "+Transformer"),
        ("Topic-FM-Contrastive", "+Contrastive"),
    ],
    "pure-vae": [
        ("Pure-VAE", "Base VAE"),
        ("Pure-Transformer-VAE", "+Transformer"),
        ("Pure-Contrastive-VAE", "+Contrastive"),
    ],
}

if "Topic-FM-GAT" in MODELS:
    ABLATION_STEPS["topic"].append(("Topic-FM-GAT", "+GAT"))


SERIES_GROUPS = {
    "topic": {k for k, v in MODELS.items() if v["series"] == "topic"},
    "topic-fm": {k for k, v in MODELS.items() if v["series"] == "topic"},
    "pure": {k for k in MODELS if k.startswith("Pure-")},
    "pure-vae": {k for k, v in MODELS.items() if v["series"] == "pure-vae"},
}


SERIES_TO_PAPER = {
    "topic": "topic",
    "topic-fm": "topic",
    "pure-vae": "topic",
    # Legacy aliases kept so older scripts degrade more gracefully.
    "dpmm": "topic",
    "pure-ae": "topic",
}


def paper_group(series_or_model: str) -> str:
    """Return the output group for a series key or model name."""
    if series_or_model in SERIES_TO_PAPER:
        return SERIES_TO_PAPER[series_or_model]
    if series_or_model in MODELS:
        return SERIES_TO_PAPER.get(MODELS[series_or_model]["series"], "topic")
    return "topic"
