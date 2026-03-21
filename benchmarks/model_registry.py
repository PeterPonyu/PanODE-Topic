"""Centralised model registry for all benchmark scripts.

Defines the canonical MODELS dict (12 variants across 4 architecture
families: DPMM, Topic, Pure-VAE, Pure-AE), series grouping constants,
ablation step definitions, and utility helpers.

Architecture Families:
- DPMM: AE + Dirichlet Process Mixture Model clustering
- Topic: VAE + Dirichlet/logistic-normal prior (simplex)
- Pure-VAE: Standard VAE with Gaussian N(0,I) prior (independent)
- Pure-AE: Deterministic autoencoder without any prior (independent)

Exports
-------
MODELS            : dict  — name → {class, params, series}
SERIES_GROUPS     : dict  — group tag → set of model names
ABLATION_STEPS    : dict  — series → [(model_name, step_label), …]
is_cuda_oom       : fn    — detect CUDA out-of-memory exceptions
"""

from models.dpmm_base import DPMMODEModel
from models.topic_base import TopicODEModel
from models.dpmm_contrastive import DPMMODEContrastiveModel
from models.topic_contrastive import TopicODEContrastiveModel
from models.dpmm_transformer import DPMMODETransformerModel
from models.topic_transformer import TopicODETransformerModel

# Independent Pure baselines (NOT derived from DPMM or Topic classes)
from models.pure_vae import PureVAEModel, PureVAETransformerModel, PureVAEContrastiveModel
from models.pure_ae import PureAEModel, PureAETransformerModel, PureAEContrastiveModel


# ═══════════════════════════════════════════════════════════════════════════════
# 12 model configurations — tuned parameters from sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    # ── Pure-AE baselines (independent, no prior, no KL) ───────────────────
    'Pure-AE': {
        'class': PureAEModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dropout_rate': 0.2,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'pure-ae',
    },
    'Pure-Transformer-AE': {
        'class': PureAETransformerModel,
        'params': {
            'latent_dim': 10,
            'd_model': 128,
            'decoder_dims': [128, 256],
            'dropout_rate': 0.2,
            'nhead': 4,
            'num_encoder_layers': 2,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'pure-ae',
    },
    'Pure-Contrastive-AE': {
        'class': PureAEContrastiveModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dropout_rate': 0.2,
            'moco_weight': 0.1,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'pure-ae',
    },

    # ── Pure-VAE baselines (Gaussian N(0,I) prior — independent from Topic) ─
    'Pure-VAE': {
        'class': PureVAEModel,
        'params': {
            'latent_dim': 10,
            'encoder_hidden': 128,
            'decoder_dims': [128, 256],
            'encoder_drop': 0.1,
            'kl_weight': 1.0,
            'fit_lr': 1e-3,
            'fit_weight_decay': 1e-3,
            'fit_epochs': 1000,
        },
        'series': 'pure-vae',
    },
    'Pure-Transformer-VAE': {
        'class': PureVAETransformerModel,
        'params': {
            'latent_dim': 10,
            'd_model': 128,
            'decoder_dims': [128, 256],
            'dropout': 0.1,
            'kl_weight': 1.0,
            'nhead': 4,
            'num_encoder_layers': 2,
            'fit_lr': 1e-3,
            'fit_weight_decay': 1e-3,
            'fit_epochs': 1000,
        },
        'series': 'pure-vae',
    },
    'Pure-Contrastive-VAE': {
        'class': PureVAEContrastiveModel,
        'params': {
            'latent_dim': 10,
            'encoder_hidden': 128,
            'decoder_dims': [128, 256],
            'encoder_drop': 0.1,
            'kl_weight': 1.0,
            'moco_weight': 0.1,
            'fit_lr': 1e-3,
            'fit_weight_decay': 1e-3,
            'fit_epochs': 1000,
        },
        'series': 'pure-vae',
    },

    # ── DPMM series (AE backbone + DPMM clustering prior) ────────────────
    'DPMM-Base': {
        'class': DPMMODEModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dpmm_warmup_ratio': 0.9,
            'dropout_rate': 0.2,               # from sensitivity: dropout=0.2 best
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,             # from sensitivity: wd=0 best for DPMM
            'fit_epochs': 1000,                # unified to 1000 epochs
        },
        'series': 'dpmm',
    },
    'DPMM-Transformer': {
        'class': DPMMODETransformerModel,
        'params': {
            'latent_dim': 10,
            'd_model': 128,
            'decoder_dims': [128, 256],
            'dpmm_warmup_ratio': 0.9,
            'dropout_rate': 0.2,
            'nhead': 4,
            'num_encoder_layers': 2,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'dpmm',
    },
    'DPMM-Contrastive': {
        'class': DPMMODEContrastiveModel,
        'params': {
            'latent_dim': 10,
            'encoder_dims': [256, 128],
            'decoder_dims': [128, 256],
            'dpmm_warmup_ratio': 0.9,
            'dropout_rate': 0.2,
            'moco_weight': 0.1,
            'fit_lr': 1e-3,
            'fit_weight_decay': 0,
            'fit_epochs': 1000,
        },
        'series': 'dpmm',
    },

    # ── Topic / LDA series (VAE backbone + Dirichlet prior) ──────────────
    'Topic-Base': {
        'class': TopicODEModel,
        'params': {
            'n_topics': 10,
            'encoder_hidden': 128,
            'kl_weight': 0.01,                 # light Dirichlet KL
            'encoder_drop': 0.0,               # from sensitivity: no dropout for Topic
            'fit_lr': 1e-3,
            'fit_weight_decay': 1e-3,          # from sensitivity: wd=1e-3 best for Topic
            'fit_epochs': 1000,                # from sensitivity: 1000 optimal for Topic
        },
        'series': 'topic',
    },
    'Topic-Transformer': {
        'class': TopicODETransformerModel,
        'params': {
            'n_topics': 10,
            'd_model': 128,
            'kl_weight': 0.01,
            'dropout': 0.0,
            'nhead': 4,
            'num_encoder_layers': 2,
            'fit_lr': 1e-3,
            'fit_weight_decay': 1e-3,
            'fit_epochs': 1000,
        },
        'series': 'topic',
    },
    'Topic-Contrastive': {
        'class': TopicODEContrastiveModel,
        'params': {
            'n_topics': 10,
            'encoder_hidden': 128,
            'kl_weight': 0.01,
            'encoder_drop': 0.0,
            'moco_weight': 0.1,
            'fit_lr': 1e-3,
            'fit_weight_decay': 1e-3,
            'fit_epochs': 1000,
        },
        'series': 'topic',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA OOM detection
# ═══════════════════════════════════════════════════════════════════════════════

def is_cuda_oom(exception) -> bool:
    """Return True if *exception* is a CUDA out-of-memory error."""
    msg = str(exception).lower()
    return "out of memory" in msg or "cuda" in msg and "alloc" in msg


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation step ordering (used by summary printers)
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_STEPS = {
    'dpmm': [
        ('DPMM-Base',           'Base AE+DPMM'),
        ('DPMM-Transformer',    '+Transformer'),
        ('DPMM-Contrastive',    '+Contrastive'),
    ],
    'topic': [
        ('Topic-Base',            'Base VAE+Topic'),
        ('Topic-Transformer',     '+Transformer'),
        ('Topic-Contrastive',     '+Contrastive'),
    ],
    'pure-ae': [
        ('Pure-AE',             'Base AE'),
        ('Pure-Transformer-AE', '+Transformer'),
        ('Pure-Contrastive-AE', '+Contrastive'),
    ],
    'pure-vae': [
        ('Pure-VAE',              'Base VAE'),
        ('Pure-Transformer-VAE',  '+Transformer'),
        ('Pure-Contrastive-VAE',  '+Contrastive'),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Series grouping for --series CLI flag
# ═══════════════════════════════════════════════════════════════════════════════

SERIES_GROUPS = {
    'dpmm':     {k for k, v in MODELS.items() if v['series'] == 'dpmm'},
    'topic':    {k for k, v in MODELS.items() if v['series'] == 'topic'},
    'pure':     {k for k in MODELS if k.startswith('Pure-')},
    'pure-ae':  {k for k, v in MODELS.items() if v['series'] == 'pure-ae'},
    'pure-vae': {k for k, v in MODELS.items() if v['series'] == 'pure-vae'},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Paper grouping — map each series to its output / comparison group
# Pure-AE is compared against DPMM, Pure-VAE against Topic
# ═══════════════════════════════════════════════════════════════════════════════

SERIES_TO_PAPER = {
    'dpmm':     'dpmm',
    'topic':    'topic',
    'pure-ae':  'dpmm',
    'pure-vae': 'topic',
}


def paper_group(series_or_model: str) -> str:
    """Return the paper output group ('dpmm' or 'topic') for a series or model name."""
    if series_or_model in SERIES_TO_PAPER:
        return SERIES_TO_PAPER[series_or_model]
    # Lookup by model name
    if series_or_model in MODELS:
        return SERIES_TO_PAPER.get(MODELS[series_or_model]['series'], 'dpmm')
    return 'dpmm'
