"""Model modules for Topic, Topic-FM, and Pure-VAE series.

Architecture Families:
- Topic: VAE + Dirichlet/logistic-normal prior (simplex constraint)
- Topic-FM: Topic + flow matching refinement in R^K pre-softmax space
- Pure-VAE: Standard VAE with Gaussian N(0,I) prior (scVI-style)

Variant Options:
- Base: MLP encoder
- Contrastive: MLP + MoCo contrastive learning
- Transformer: Multi-head projection encoder with O(batch_size) attention
- GAT: Graph Attention Network encoder over k-NN graph

Shared Modules:
- shared_modules.py: Common utilities (MLP, decoders, graph)
- encoders.py: Unified encoder implementations (MLP, Transformer, Hybrid, GAT)
- topic_flow_matching.py: Flow matching mixin (SinusoidalTimeEmbedding, LatentFlowField)
"""

# Shared modules
from .shared_modules import (
    weight_init,
    MLP,
    ResidualMLP,
    InformationBottleneck,
    MLPDecoder,
    TopicDecoder,
    SubgraphDataset,
    precompute_knn_graph,
    reparameterize,
    log_to_simplex)

# Encoder modules
from .encoders import (
    MultiHeadProjectionEncoder,
    MultiHeadProjectionLogisticNormalEncoder,
    HybridMLPAttentionEncoder,
    HybridMLPAttentionLogisticNormalEncoder,
    MLPEncoder,
    MLPLogisticNormalEncoder,
    create_encoder,
    create_topic_encoder)

# GAT encoder (optional — requires torch_geometric)
try:
    from .encoders import GATLogisticNormalEncoder
except ImportError:
    pass

# Flow matching components
from .topic_flow_matching import (
    SinusoidalTimeEmbedding,
    LatentFlowField,
    TopicFlowMatchingMixin)

# Topic Models (base autoencoders)
from .topic_base import TopicAutoEncoder
from .topic_contrastive import TopicContrastiveAutoEncoder
from .topic_transformer import TopicTransformerAutoEncoder

# Topic-FM Models (flow matching variants)
from .topic_fm_base import TopicFMModel
from .topic_fm_transformer import TopicFMTransformerModel
from .topic_fm_contrastive import TopicFMContrastiveModel
try:
    from .topic_fm_gat import TopicFMGATModel
except ImportError:
    pass

# Pure-VAE Models (Gaussian prior — Topic ablation baseline)
from .pure_vae import PureVAEModel, PureVAETransformerModel, PureVAEContrastiveModel

__all__ = [
    # Shared utilities
    'weight_init', 'MLP', 'ResidualMLP', 'InformationBottleneck',
    'MLPDecoder', 'TopicDecoder', 'SubgraphDataset', 'precompute_knn_graph',
    'reparameterize', 'log_to_simplex',
    # Encoders
    'MultiHeadProjectionEncoder', 'MultiHeadProjectionLogisticNormalEncoder',
    'HybridMLPAttentionEncoder', 'HybridMLPAttentionLogisticNormalEncoder',
    'MLPEncoder', 'MLPLogisticNormalEncoder', 'GATLogisticNormalEncoder',
    'create_encoder', 'create_topic_encoder',
    # Flow matching
    'SinusoidalTimeEmbedding', 'LatentFlowField', 'TopicFlowMatchingMixin',
    # Topic autoencoders
    'TopicAutoEncoder', 'TopicContrastiveAutoEncoder', 'TopicTransformerAutoEncoder',
    # Topic-FM models
    'TopicFMModel', 'TopicFMTransformerModel', 'TopicFMContrastiveModel', 'TopicFMGATModel',
    # Pure-VAE Models (Gaussian prior)
    'PureVAEModel', 'PureVAETransformerModel', 'PureVAEContrastiveModel',
]
