"""
LDAODEContrastive: Topic model with momentum contrastive learning and ODE.

Core components:
- Momentum Contrastive (MoCo) learning for topic representation
- InfoNCE loss for self-supervised contrastive pre-training
- Data augmentation strategies for single-cell count data

Single-phase training:
- Phase 1: Train VAE-based topic model + contrastive learning (fit method)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from torch.distributions import Normal
import math
import numpy as np
try:
    from .base_model import BaseModel
    from .mixins import PriorMixin, ReconstructionLossMixin
except ImportError:
    from utils.base_model import BaseModel
    from utils.mixins import PriorMixin, ReconstructionLossMixin


def weight_init(m):
    """Xavier normal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


class _Act(nn.Module):
    """Activation helper"""
    def __init__(self, name: str):
        super().__init__()
        if name == "relu":
            self.fn = nn.ReLU()
        elif name == "mish":
            self.fn = nn.Mish()
        elif name == "elu":
            self.fn = nn.ELU()
        elif name == "sigmoid":
            self.fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        return self.fn(x)


class MLP(nn.Module):
    """Flexible MLP with normalization and dropout"""
    def __init__(
        self,
        features: list,
        hid_act: str = "relu",
        out_act: Optional[str] = None,
        norm: Optional[str] = None,
        drop: float = 0.0):
        super().__init__()
        layers = []
        for i in range(1, len(features)):
            layers.append(nn.Linear(features[i-1], features[i]))
            
            if i < len(features) - 1:
                if norm == "bn":
                    layers.append(nn.BatchNorm1d(features[i]))
                elif norm == "ln":
                    layers.append(nn.LayerNorm(features[i]))
                layers.append(_Act(hid_act))
                if drop > 0:
                    layers.append(nn.Dropout(drop))
            else:
                if out_act is not None:
                    layers.append(_Act(out_act))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DataAugmentation(nn.Module):
    """Enhanced data augmentation strategies for single-cell count data
    
    Integrates augmentation from:
    - scSimGCL: Feature dropout + noise
    - scHSC: Multi-view augmentation
    - scGPCL: Heterogeneous dropout
    """
    def __init__(
        self,
        noise_prob: float = 0.2,
        noise_std: float = 0.1,
        mask_prob: float = 0.1,
        dropout_prob: float = 0.05,
        feature_dropout: float = 0.2,  # From scSimGCL
    ):
        super().__init__()
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.dropout_prob = dropout_prob
        self.feature_dropout = feature_dropout
    
    def forward(self, x: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        """Apply random augmentation for count data with scSimGCL-style dropout"""
        batch_size, n_features = x.shape
        device = x.device
        x_aug = x.clone()
        
        # scSimGCL-style feature dropout (Bernoulli mask)
        feature_mask = None
        if self.feature_dropout > 0:
            feature_mask = torch.bernoulli(
                torch.ones(batch_size, n_features, device=device) * (1 - self.feature_dropout)
            )
            x_aug = x_aug * feature_mask
        
        # Gaussian noise (applied to log-transformed data)
        if self.noise_prob > 0 and torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise
        
        # Feature masking (dropout)
        if self.mask_prob > 0:
            mask = torch.rand(batch_size, n_features, device=device) > self.mask_prob
            x_aug = x_aug * mask.float()
        
        # Random dropout of features
        if self.dropout_prob > 0:
            dropout = torch.rand(batch_size, n_features, device=device) > self.dropout_prob
            x_aug = x_aug * dropout.float()
        
        if return_mask and feature_mask is not None:
            return x_aug, feature_mask
        return x_aug


class MomentumContrast(nn.Module):
    """Enhanced Momentum Contrast (MoCo) module with multi-level contrastive learning
    
    Integrates contrastive strategies from:
    - MoCo: Momentum encoder + memory queue
    - scAGCL: Symmetric contrastive loss + projection
    - scGPCL: Instance-level + Prototype-level contrastive
    - scHSC: Hard sample weighting
    """
    def __init__(
        self,
        n_topics: int,
        embedding_dim: int = 64,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.2,
        device: torch.device = torch.device("cuda"),
        use_prototype: bool = True,  # From scGPCL
        n_prototypes: int = 10):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.embedding_dim = embedding_dim
        self.n_topics = n_topics
        self.use_prototype = use_prototype
        self.n_prototypes = n_prototypes
        
        # Query projector (scAGCL-style 2-layer MLP with BatchNorm)
        self.query_projector = nn.Sequential(
            nn.Linear(n_topics, n_topics),
            nn.BatchNorm1d(n_topics),
            nn.ReLU(),
            nn.Linear(n_topics, embedding_dim)
        )
        
        # Key projector (momentum-updated copy)
        self.key_projector = nn.Sequential(
            nn.Linear(n_topics, n_topics),
            nn.BatchNorm1d(n_topics),
            nn.ReLU(),
            nn.Linear(n_topics, embedding_dim)
        )
        
        # Initialize key projector with query projector weights
        for param_q, param_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Initialize queue
        self.register_buffer("queue", torch.randn(embedding_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Prototype parameters (from scGPCL)
        if use_prototype:
            self.prototypes = nn.Parameter(torch.randn(n_prototypes, embedding_dim))
            nn.init.xavier_uniform_(self.prototypes)
        
        self.apply(weight_init)
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update key projector with momentum"""
        for param_q, param_k in zip(self.query_projector.parameters(), self.key_projector.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            part1_size = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:part1_size].T
            part2_size = batch_size - part1_size
            self.queue[:, :part2_size] = keys[part1_size:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, log_theta_query: torch.Tensor, log_theta_key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive logits and labels for topic distributions
        
        Args:
            log_theta_query: Query log-topic vectors [batch, n_topics]
            log_theta_key: Key log-topic vectors [batch, n_topics]
        
        Returns:
            logits: Contrastive logits [batch, 1 + queue_size]
            labels: Target labels (zeros for positive pairs)
        """
        # Project queries
        q = self.query_projector(log_theta_query)
        q = F.normalize(q, dim=1)
        
        # Project keys with momentum update
        with torch.no_grad():
            self._momentum_update()
            k = self.key_projector(log_theta_key)
            k = F.normalize(k, dim=1)
        
        # Positive logits: [batch, 1]
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        
        # Negative logits: [batch, queue_size]
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        
        # Logits: [batch, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positive pairs have index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels
    
    def symmetric_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """scAGCL-style symmetric contrastive loss"""
        h1 = self.query_projector(z1)
        h2 = self.query_projector(z2)
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        
        # Compute similarity
        sim_11 = torch.mm(h1, h1.T) / self.temperature
        sim_12 = torch.mm(h1, h2.T) / self.temperature
        sim_22 = torch.mm(h2, h2.T) / self.temperature
        
        # Symmetric loss (scAGCL)
        pos_sim = torch.diag(sim_12)
        
        # Loss from view 1
        neg_sim_1 = torch.cat([sim_11, sim_12], dim=1)
        neg_sim_1 = neg_sim_1 - torch.diag(torch.diag(sim_11)).repeat(1, 2)[:, :neg_sim_1.size(1)]
        l1 = -pos_sim + torch.logsumexp(neg_sim_1, dim=1)
        
        # Loss from view 2
        neg_sim_2 = torch.cat([sim_22, sim_12.T], dim=1)
        neg_sim_2 = neg_sim_2 - torch.diag(torch.diag(sim_22)).repeat(1, 2)[:, :neg_sim_2.size(1)]
        l2 = -pos_sim + torch.logsumexp(neg_sim_2, dim=1)
        
        return (l1.mean() + l2.mean()) / 2
    
    def prototype_contrastive_loss(self, z: torch.Tensor, cluster_assignments: Optional[torch.Tensor] = None) -> torch.Tensor:
        """scGPCL-style prototype contrastive loss"""
        if not self.use_prototype:
            return torch.tensor(0.0, device=z.device)
        
        h = self.query_projector(z)
        h = F.normalize(h, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        
        # Similarity to prototypes
        sim = torch.mm(h, prototypes.T) / self.temperature
        
        if cluster_assignments is not None:
            # Supervised prototype loss
            pos_mask = F.one_hot(cluster_assignments, num_classes=self.n_prototypes).float()
            pos_sim = (sim * pos_mask).sum(dim=1)
            loss = -pos_sim + torch.logsumexp(sim, dim=1)
        else:
            # Self-supervised: assign to nearest prototype
            assignments = sim.argmax(dim=1)
            pos_mask = F.one_hot(assignments, num_classes=self.n_prototypes).float()
            pos_sim = (sim * pos_mask).sum(dim=1)
            loss = -pos_sim + torch.logsumexp(sim, dim=1)
        
        return loss.mean()
    
    def topic_aware_contrastive_loss(self, theta_q: torch.Tensor, theta_k: torch.Tensor,
                                      temperature: float = 0.5) -> torch.Tensor:
        """Topic-aware contrastive loss that respects topic structure.
        
        Samples with similar topic distributions should be pulled together.
        Uses Jensen-Shannon divergence for topic similarity.
        """
        batch_size = theta_q.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=theta_q.device)
        
        # Convert to probability (in case log-space)
        if theta_q.min() < 0:  # log-space
            theta_q_prob = F.softmax(theta_q, dim=1)
            theta_k_prob = F.softmax(theta_k, dim=1)
        else:
            theta_q_prob = theta_q
            theta_k_prob = theta_k
        
        # Compute pairwise JS divergence as distance
        m = 0.5 * (theta_q_prob.unsqueeze(1) + theta_k_prob.unsqueeze(0))  # [B, B, K]
        
        kl_pm = torch.sum(theta_q_prob.unsqueeze(1) * (torch.log(theta_q_prob.unsqueeze(1) + 1e-10) - torch.log(m + 1e-10)), dim=-1)
        kl_qm = torch.sum(theta_k_prob.unsqueeze(0) * (torch.log(theta_k_prob.unsqueeze(0) + 1e-10) - torch.log(m + 1e-10)), dim=-1)
        js_dist = 0.5 * (kl_pm + kl_qm)  # [B, B]
        
        # Convert JS distance to similarity (closer = more similar)
        topic_sim = torch.exp(-js_dist / temperature)
        
        # Positive pairs: diagonal (same sample augmented)
        pos_sim = torch.diag(topic_sim)
        
        # Negative pairs: off-diagonal
        mask = torch.eye(batch_size, device=theta_q.device).bool()
        neg_sim = topic_sim.masked_fill(mask, 0).sum(dim=1) / (batch_size - 1)
        
        # Contrastive loss: maximize pos_sim, minimize neg_sim
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-10)).mean()
        
        return loss


class LogisticNormalEncoder(nn.Module):
    """Encoder for Logistic-Normal distribution."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, 
                 norm: str = "bn", drop: float = 0.1, var_eps: float = 1e-4):
        super().__init__()
        self.encoder = MLP([input_dim, hidden_dim, hidden_dim], hid_act="relu", norm=norm, drop=drop)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.var_head = nn.Linear(hidden_dim, output_dim)
        self.var_eps = var_eps
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, var) for logistic-normal distribution."""
        x_input = x_norm if x_norm is not None else torch.log1p(x)
        h = self.encoder(x_input)
        mu = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + self.var_eps
        return mu, var


class TopicDecoder(nn.Module):
    """Decoder: theta -> word distribution."""
    def __init__(self, n_topics: int, n_words: int):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.beta_logit = nn.Parameter(torch.randn(n_topics, n_words))
        self.apply(weight_init)
    
    @property
    def beta(self) -> torch.Tensor:
        """Topic-word distribution [n_topics, n_words]"""
        return F.softmax(self.beta_logit, dim=1)
    
    @property
    def log_beta(self) -> torch.Tensor:
        """Log topic-word distribution [n_topics, n_words]"""
        return F.log_softmax(self.beta_logit, dim=1)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Decode theta to word distribution."""
        log_theta = torch.log(theta + 1e-10)
        log_beta = self.log_beta.T.unsqueeze(0)
        log_prob = torch.logsumexp(log_theta.unsqueeze(1) + log_beta, dim=2)
        return torch.exp(log_prob)


class InformationBottleneck(nn.Module):
    """Compress and reconstruct topic logits."""
    def __init__(self, latent_dim: int, bottleneck_dim: int, drop: float = 0.1):
        super().__init__()
        hidden_dim = max(latent_dim // 2, bottleneck_dim)
        self.compress = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, bottleneck_dim))
        self.expand = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, latent_dim))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_bottleneck = self.compress(z)
        z_reconstructed = self.expand(z_bottleneck)
        return z_bottleneck, z_reconstructed


class TopicContrastiveAutoEncoder(nn.Module):
    """Topic model autoencoder with enhanced MoCo contrastive learning and optional ODE.
    
    Integrates contrastive learning from:
    - MoCo: Momentum encoder + memory queue
    - scAGCL: Symmetric contrastive loss + projection
    - scGPCL: Instance-level + Prototype-level contrastive
    - scSimGCL: Feature dropout augmentation
    """
    def __init__(
        self,
        n_words: int,
        n_topics: int,
        encoder_hidden: int = 128,
        encoder_norm: str = "bn",
        encoder_drop: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        # Contrastive learning params
        use_moco: bool = True,
        moco_embedding_dim: int = 64,
        moco_queue_size: int = 4096,
        moco_momentum: float = 0.999,
        moco_temperature: float = 0.2,
        use_prototype: bool = True,  # From scGPCL
        n_prototypes: int = 10,
        # Augmentation params
        aug_noise_prob: float = 0.2,
        aug_noise_std: float = 0.1,
        aug_mask_prob: float = 0.1,
        aug_feature_dropout: float = 0.2,  # From scSimGCL
        device: torch.device = torch.device("cuda")):
        super().__init__()
        self.n_words = n_words
        self.n_topics = n_topics
        self.use_moco = use_moco
        self.use_prototype = use_prototype
        self.device = device
        self.use_bottleneck = use_bottleneck
        
        # Encoder
        self.encoder = LogisticNormalEncoder(
            n_words, n_topics, encoder_hidden, encoder_norm, encoder_drop
        )
        
        # Decoder
        self.decoder = TopicDecoder(n_topics, n_words)

        if use_bottleneck:
            if bottleneck_dim is None:
                bottleneck_dim = max(n_topics // 2, 8)
            self.bottleneck = InformationBottleneck(n_topics, bottleneck_dim, drop=encoder_drop)
        
        # Contrastive learning components
        if use_moco:
            self.augmentation = DataAugmentation(
                noise_prob=aug_noise_prob,
                noise_std=aug_noise_std,
                mask_prob=aug_mask_prob,
                feature_dropout=aug_feature_dropout,  # scSimGCL-style
            )
            self.moco = MomentumContrast(
                n_topics=n_topics,
                embedding_dim=moco_embedding_dim,
                queue_size=moco_queue_size,
                momentum=moco_momentum,
                temperature=moco_temperature,
                device=device,
                use_prototype=use_prototype,  # scGPCL-style
                n_prototypes=n_prototypes)
        
        self.apply(weight_init)
    
    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for logistic-normal"""
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def log_to_simplex(self, log_theta: torch.Tensor) -> torch.Tensor:
        """Convert log-space to probability simplex via softmax"""
        return F.softmax(log_theta, dim=-1)
    
    def forward(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """VAE forward pass with optional MoCo (no ODE during training)"""
        return self._forward_normal(x, x_norm)
    
    def _forward_normal(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Standard VAE forward pass with enhanced multi-level contrastive learning"""
        # Encode
        mu, var = self.encoder(x, x_norm)
        log_theta = self.reparameterize(mu, var)
        theta = self.log_to_simplex(log_theta)
        x_recon = self.decoder(theta)

        result = {
            "x": x,
            "x_recon": x_recon,
            "mu": mu,
            "var": var,
            "log_theta": log_theta,
            "theta": theta,
        }

        if self.use_bottleneck:
            log_theta_le, log_theta_ld = self.bottleneck(log_theta)
            theta_ld = self.log_to_simplex(log_theta_ld)
            x_recon_bottleneck = self.decoder(theta_ld)
            result.update({
                "log_theta_le": log_theta_le,
                "log_theta_bottleneck": log_theta_ld,
                "x_recon_bottleneck": x_recon_bottleneck,
            })
        
        # Enhanced multi-level contrastive learning
        if self.use_moco and self.training:
            x_input = x_norm if x_norm is not None else torch.log1p(x)
            
            # scSimGCL-style augmentation
            x_aug_q = self.augmentation(x_input)
            x_aug_k = self.augmentation(x_input)
            
            # Get augmented representations
            mu_q, var_q = self.encoder(torch.exp(x_aug_q) - 1, x_aug_q)
            log_theta_q = self.reparameterize(mu_q, var_q)
            
            mu_k, var_k = self.encoder(torch.exp(x_aug_k) - 1, x_aug_k)
            log_theta_k = self.reparameterize(mu_k, var_k)
            
            # MoCo contrastive logits
            moco_logits, moco_labels = self.moco(log_theta_q, log_theta_k)
            result["moco_logits"] = moco_logits
            result["moco_labels"] = moco_labels
            
            # Store augmented representations for scAGCL symmetric loss and scGPCL prototype loss
            result["log_theta_q"] = log_theta_q
            result["log_theta_k"] = log_theta_k
        
        return result
    

class TopicODEContrastiveModel(PriorMixin, ReconstructionLossMixin, BaseModel):
    """LDAODEContrastive: Topic model with MoCo and optional ODE."""
    def __init__(
        self,
        input_dim: int,
        n_topics: int = 10,
        latent_dim: Optional[int] = None,
        encoder_hidden: int = 128,
        encoder_norm: str = "bn",
        encoder_drop: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        sparsity_strength: float = 10.0,  # Tighter prior (variance~0.9) to prevent collapse
        cell_topic_prior: Optional[torch.Tensor] = None,
        topic_word_prior: Optional[torch.Tensor] = None,
        kl_weight: float = 0.01,  # Fixed KL weight (no annealing)
        reconstruction_loss: str = "multinomial",
        model_name: str = "LDAODEContrastive",
        use_raw_counts: bool = True,
        use_moco: bool = True,
        moco_weight: float = 0.5,
        moco_embedding_dim: int = 64,
        moco_queue_size: int = 4096,
        moco_momentum: float = 0.999,
        moco_temperature: float = 0.2,
        # Augmentation parameters
        aug_noise_prob: float = 0.2,
        aug_noise_std: float = 0.1,
        aug_mask_prob: float = 0.1):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim or n_topics,
            hidden_dims=[encoder_hidden],
            model_name=model_name,
            use_raw_counts=use_raw_counts)
        
        self.n_words = input_dim
        self.n_topics = n_topics
        self.kl_weight = kl_weight
        self.reconstruction_loss = reconstruction_loss
        
        # Contrastive learning
        self.use_moco = use_moco
        self.moco_weight = moco_weight
        self.moco_loss_fn = nn.CrossEntropyLoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize autoencoder
        self.ae = TopicContrastiveAutoEncoder(
            n_words=input_dim,
            n_topics=n_topics,
            encoder_hidden=encoder_hidden,
            encoder_norm=encoder_norm,
            encoder_drop=encoder_drop,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            ode_hidden=ode_hidden,
            use_moco=use_moco,
            moco_embedding_dim=moco_embedding_dim,
            moco_queue_size=moco_queue_size,
            moco_momentum=moco_momentum,
            moco_temperature=moco_temperature,
            aug_noise_prob=aug_noise_prob,
            aug_noise_std=aug_noise_std,
            aug_mask_prob=aug_mask_prob,
            device=self.device)
        
        # Priors
        if cell_topic_prior is None:
            cell_topic_prior = torch.ones(n_topics) / n_topics * sparsity_strength
        if topic_word_prior is None:
            topic_word_prior = torch.ones(input_dim) / n_topics * sparsity_strength
        
        prior_mu_topics, prior_sigma_topics = self._dirichlet_to_logistic_normal(cell_topic_prior)
        self.register_buffer("prior_mu_topics", prior_mu_topics)
        self.register_buffer("prior_sigma_topics", prior_sigma_topics)
        
        prior_mu_words, prior_sigma_words = self._dirichlet_to_logistic_normal(topic_word_prior)
        self.register_buffer("prior_mu_words", prior_mu_words)
        self.register_buffer("prior_sigma_words", prior_sigma_words)
    
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to topic distribution (theta)."""
        x_norm = kwargs.get('x_norm', None)
        mu, var = self.ae.encoder(x, x_norm)
        log_theta = mu
        theta = self.ae.log_to_simplex(log_theta)
        return theta
    
    def encode_log_theta(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to log_theta (before softmax)."""
        x_norm = kwargs.get('x_norm', None)
        mu, var = self.ae.encoder(x, x_norm)
        return mu
    
    def extract_latent(self, data_loader, device='cuda', return_reconstructions: bool = False,
                       return_log_theta: bool = False):
        """Extract latent representations."""
        self.eval()
        self.to(device)
        latents, recons = [], []
        
        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                
                if return_log_theta:
                    z = self.encode_log_theta(x, **batch_kwargs)
                else:
                    z = self.encode(x, **batch_kwargs)
                
                latents.append(z.cpu().numpy())
                
                if return_reconstructions:
                    theta = z if not return_log_theta else self.ae.log_to_simplex(z)
                    recons.append(self.decode(theta).cpu().numpy())
        
        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result
    
    def decode(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode theta to word distribution."""
        return self.ae.decoder(theta)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        x_norm = kwargs.get('x_norm', None)
        return self.ae(x, x_norm)
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss + topic-preserving contrastive loss (Phase 1)
        
        CRITICAL: For topic models, MoCo queue-based contrastive learning DESTROYS 
        topic structure by treating all negative samples equally. Instead we use:
        - Reconstruction loss (multinomial/KL) - PRIMARY driver of learning
        - KL divergence (prior regularization)
        - Symmetric contrastive loss (preserves augmentation invariance without queue)
        - Topic prototype loss (encourages distinct topic clusters)
        
        The key insight is that contrastive learning for topic models should:
        1. Pull together augmented views of the SAME sample
        2. NOT push apart all other samples (destroys topic structure)
        """
        # Reconstruction loss - most important for topic learning
        if self.reconstruction_loss == "multinomial":
            recon_main = self._multinomial_nll(outputs["x"], outputs["x_recon"])
            if "x_recon_bottleneck" in outputs:
                recon_bottleneck = self._multinomial_nll(outputs["x"], outputs["x_recon_bottleneck"])
                recon_loss = 0.5 * (recon_main + recon_bottleneck)
            else:
                recon_loss = recon_main
        elif self.reconstruction_loss == "kl":
            recon_main = self._kl_divergence(outputs["x"], outputs["x_recon"])
            if "x_recon_bottleneck" in outputs:
                recon_bottleneck = self._kl_divergence(outputs["x"], outputs["x_recon_bottleneck"])
                recon_loss = 0.5 * (recon_main + recon_bottleneck)
            else:
                recon_loss = recon_main
        else:
            raise ValueError(f"Unknown reconstruction_loss: {self.reconstruction_loss}")
        
        # KL divergence - regularization
        kl_loss = self._kl_logistic_normal(
            outputs["mu"],
            outputs["var"],
            self.prior_mu_topics,
            self.prior_sigma_topics ** 2)
        
        # MoCo InfoNCE loss: queue-based contrastive learning
        moco_loss = torch.tensor(0.0, device=outputs["x"].device)
        if self.use_moco and "log_theta_q" in outputs and "log_theta_k" in outputs:
            log_theta_q = outputs["log_theta_q"]
            log_theta_k = outputs["log_theta_k"]
            logits, labels = self.ae.moco(log_theta_q, log_theta_k)
            moco_loss = F.cross_entropy(logits, labels)
        
        # Symmetric contrastive loss: only pulls together augmented views of same sample
        # Does NOT use negative queue, preserving topic structure
        symmetric_loss = torch.tensor(0.0, device=outputs["x"].device)
        if self.use_moco and "log_theta_q" in outputs and "log_theta_k" in outputs:
            log_theta_q = outputs["log_theta_q"]
            log_theta_k = outputs["log_theta_k"]
            
            # Simple augmentation consistency loss (MSE between augmented views)
            # This pulls together augmented views without pushing apart other samples
            symmetric_loss = F.mse_loss(log_theta_q, log_theta_k)
        
        # Topic prototype contrastive loss: encourages distinct topic clusters
        prototype_loss = torch.tensor(0.0, device=outputs["x"].device)
        if self.use_moco and "log_theta_q" in outputs:
            prototype_loss = self.ae.moco.prototype_contrastive_loss(outputs["log_theta_q"])
        
        # Topic coherence: encourage each sample to have dominant topic(s)
        topic_sparsity_loss = torch.tensor(0.0, device=outputs["x"].device)
        if "theta" in outputs:
            theta = outputs["theta"]
            # Encourage sparsity: high entropy = bad (uniform), low entropy = good (peaked)
            per_sample_entropy = -torch.sum(theta * torch.log(theta + 1e-10), dim=1)
            # We want to MINIMIZE entropy (encourage peaked distributions)
            topic_sparsity_loss = per_sample_entropy.mean()
        
        # Topic diversity: ensure all topics are used across the batch
        topic_diversity_bonus = torch.tensor(0.0, device=outputs["x"].device)
        if "theta" in outputs:
            theta = outputs["theta"]
            topic_means = theta.mean(dim=0)  # Average topic usage across batch
            # Maximize entropy of topic usage across batch (encourages all topics used)
            topic_diversity_bonus = torch.sum(topic_means * torch.log(topic_means + 1e-10))
        
        # Final loss: reconstruction-focused with contrastive losses
        # MoCo InfoNCE provides queue-based contrastive learning
        total_loss = (recon_loss + 
                      self.kl_weight * kl_loss + 
                      0.1 * moco_loss +           # MoCo InfoNCE loss (re-enabled)
                      0.05 * symmetric_loss +     # Augmentation consistency
                      0.05 * prototype_loss +     # Soft topic clustering
                      0.01 * topic_sparsity_loss +  # Encourage peaked topic distributions
                      0.01 * topic_diversity_bonus)  # Encourage diverse topic usage
        
        loss_dict = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "moco_loss": moco_loss,  # MoCo InfoNCE loss
            "symmetric_loss": symmetric_loss,
            "prototype_loss": prototype_loss,
            "topic_sparsity_loss": topic_sparsity_loss,
        }
        if "x_recon_bottleneck" in outputs:
            loss_dict["recon_bottleneck"] = recon_bottleneck
        
        return loss_dict

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1000,
        lr: float = 0.001,
        device: str = "cuda",
        save_path: Optional[str] = None,
        patience: int = 50,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 1e-3,
        **kwargs):
        """Phase 1: Train VAE-based LDA + MoCo
        
        Args:
            weight_decay: AdamW weight decay (L2 regularization).
        """
        self.to(device)
        self.device = torch.device(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses, recon_losses, kl_losses, moco_losses = [], [], [], []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            self.train()
            epoch_loss, epoch_recon, epoch_kl, epoch_moco = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                loss_dict = self.compute_loss(out, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]
                
                if not torch.isfinite(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_kl += loss_dict["kl_loss"].item()
                if "moco_loss" in loss_dict:
                    epoch_moco += loss_dict["moco_loss"].item()
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            avg_moco = epoch_moco / n_batches
            
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
            moco_losses.append(avg_moco)

            do_print = (verbose >= 1) and (((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs))

            if do_print:
                print(f"Epoch {epoch+1:3d}/{epochs} [Phase1-LDA+MoCo] | "
                      f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | MoCo: {avg_moco:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return {
            "train_loss": train_losses,
            "recon_loss": recon_losses,
            "kl_loss": kl_losses,
            "moco_loss": moco_losses,
        }


    def get_topic_word_distribution(self) -> torch.Tensor:
        """Get topic-word distribution [n_topics, n_words]."""
        return self.ae.decoder.beta.detach()
    
    def get_top_words_per_topic(self, vocab: list, n_words: int = 10) -> Dict[int, list]:
        """Get top words per topic."""
        beta = self.get_topic_word_distribution().cpu().numpy()
        result = {}
        for topic_id in range(self.n_topics):
            top_idx = np.argsort(beta[topic_id])[::-1][:n_words]
            result[topic_id] = [(vocab[idx], beta[topic_id, idx]) for idx in top_idx]
        return result


def create_topic_ode_contrastive_model(
    n_words: int,
    n_topics: int = 10,
    use_moco: bool = True,
    **kwargs
) -> TopicODEContrastiveModel:
    """
    Create TopicODEContrastive (LDAODEContrastive) model.
    
    Args:
        n_words: Vocabulary size (input dimension)
        n_topics: Number of topics (latent dimension)
        use_moco: Enable MoCo contrastive learning (default: True)
    
    Critical Parameters:
        encoder_hidden: Encoder hidden dimension (64-256)
        moco_weight: Contrastive loss weight (default: 0.5)
        moco_temperature: Contrastive temperature (default: 0.2)
        kl_weight: KL divergence weight (0.1-2.0)
    """
    return TopicODEContrastiveModel(
        input_dim=n_words,
        n_topics=n_topics,
        use_moco=use_moco,
        **kwargs
    )
