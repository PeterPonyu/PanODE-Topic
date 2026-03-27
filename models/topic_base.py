"""
Topic Model — VAE with logistic-normal prior.

Provides the base TopicAutoEncoder (encoder + decoder) and TopicODEModel
used as the foundation for the Topic-FM family. Single-phase training
with optional flow matching (in the FM variants).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from torch.distributions import Normal, Dirichlet, MultivariateNormal
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
        if m.bias is not None:
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


class InformationBottleneck(nn.Module):
    """Compress and reconstruct topic logits."""
    def __init__(self, latent_dim: int, bottleneck_dim: int, norm: str = "bn", drop: float = 0.1):
        super().__init__()
        self.compress = MLP(
            [latent_dim, latent_dim // 2, bottleneck_dim],
            hid_act="relu",
            norm=norm,
            drop=drop)
        self.expand = MLP(
            [bottleneck_dim, latent_dim // 2, latent_dim],
            hid_act="relu",
            norm=norm,
            drop=drop)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_bottleneck = self.compress(z)
        z_reconstructed = self.expand(z_bottleneck)
        return z_bottleneck, z_reconstructed


class LogisticNormalEncoder(nn.Module):
    """Encoder for Logistic-Normal distribution."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, 
                 norm: str = "bn", drop: float = 0.1, var_eps: float = 1e-4):
        super().__init__()
        # Two-layer encoder
        self.encoder = MLP([input_dim, hidden_dim, hidden_dim], hid_act="relu", norm=norm, drop=drop)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        # Variance head: outputs unconstrained values, softplus ensures positivity
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
        # Topic-word distribution (learned parameter)
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


class TopicAutoEncoder(nn.Module):
    """Topic model autoencoder with optional ODE components."""
    def __init__(
        self,
        n_words: int,
        n_topics: int,
        encoder_hidden: int = 128,
        encoder_norm: str = "bn",
        encoder_drop: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None):
        super().__init__()
        self.n_words = n_words
        self.n_topics = n_topics
        self.use_bottleneck = use_bottleneck
        
        # Encoder: words -> log-topic distribution (logistic-normal)
        self.encoder = LogisticNormalEncoder(
            n_words, n_topics, encoder_hidden, encoder_norm, encoder_drop
        )
        
        # Decoder: topic distribution -> word distribution
        self.decoder = TopicDecoder(n_topics, n_words)

        if use_bottleneck:
            if bottleneck_dim is None:
                bottleneck_dim = max(n_topics // 2, 8)
            self.bottleneck = InformationBottleneck(n_topics, bottleneck_dim, encoder_norm, encoder_drop)
        
        self.apply(weight_init)
    
    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for logistic-normal
        """
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def log_to_simplex(self, log_theta: torch.Tensor) -> torch.Tensor:
        """Convert log-space to probability simplex via softmax"""
        return F.softmax(log_theta, dim=-1)
    
    def forward(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Standard VAE forward pass (no ODE during training)
        
        ODE dynamics are trained separately in Phase 2.
        """
        return self._forward_normal(x, x_norm)
    
    def _forward_normal(self, x: torch.Tensor, x_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Standard VAE forward pass"""
        # Encode to log-topic distribution parameters
        mu, var = self.encoder(x, x_norm)
        
        # Sample log-theta using reparameterization trick
        log_theta = self.reparameterize(mu, var)
        
        # Convert to simplex (topic distribution)
        theta = self.log_to_simplex(log_theta)
        
        # Decode to word distribution
        x_recon = self.decoder(theta)

        result = {
            "x": x,
            "x_recon": x_recon,
            "mu": mu,
            "var": var,  # Return variance, not logvar
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

        return result
    

class TopicODEModel(PriorMixin, ReconstructionLossMixin, BaseModel):
    """Topic model with DPMM-like training."""
    def __init__(
        self,
        input_dim: int,  # n_words (vocabulary size)
        n_topics: int = 10,
        latent_dim: Optional[int] = None,  # Kept for BaseModel compatibility
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
        model_name: str = "LDAODE",
        use_raw_counts: bool = True,  # LDA requires raw counts for multinomial likelihood
):
        # LDA requires raw counts
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim or n_topics,
            hidden_dims=[encoder_hidden],
            model_name=model_name,
            use_raw_counts=use_raw_counts,  # Use raw counts as primary input
        )
        
        self.n_words = input_dim
        self.n_topics = n_topics
        self.kl_weight = kl_weight
        self.reconstruction_loss = reconstruction_loss
        
        # Track training phases
        
        # Initialize autoencoder
        self.ae = TopicAutoEncoder(
            n_words=input_dim,
            n_topics=n_topics,
            encoder_hidden=encoder_hidden,
            encoder_norm=encoder_norm,
            encoder_drop=encoder_drop,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim)
        
        # Priors (logistic-normal approximation of Dirichlet)
        if cell_topic_prior is None:
            cell_topic_prior = torch.ones(n_topics) / n_topics * sparsity_strength
        if topic_word_prior is None:
            topic_word_prior = torch.ones(input_dim) / n_topics * sparsity_strength
        
        # Convert Dirichlet priors to logistic-normal
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
        """
        Forward pass
        """
        # Extract normalized data if provided
        x_norm = kwargs.get('x_norm', None)
        return self.ae(x, x_norm)
    
    def compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        kl_weight: Optional[float] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss = Reconstruction + KL divergence (Phase 1 only)
        
        Uses free bits KL to prevent posterior collapse.
        """
        # Use provided kl_weight or default
        current_kl_weight = kl_weight if kl_weight is not None else self.kl_weight
        
        # Reconstruction loss
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
        
        # KL divergence with free bits to prevent posterior collapse
        # Each dimension must contribute at least 0.1 nats
        kl_loss = self._kl_logistic_normal_free_bits(
            outputs["mu"], 
            outputs["var"],
            self.prior_mu_topics,
            self.prior_sigma_topics ** 2,
            free_bits=0.1
        )
        
        # Total ELBO loss
        total_loss = recon_loss + current_kl_weight * kl_loss
        
        loss_dict = {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
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
        kl_weight: Optional[float] = None,
        weight_decay: float = 1e-3,
        **kwargs):
        """
        Phase 1: Train VAE-based LDA with fixed KL weight.
        
        Args:
            kl_weight: Fixed KL weight. If None, uses self.kl_weight from __init__.
            weight_decay: AdamW weight decay (L2 regularization).
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Use provided kl_weight or fall back to self.kl_weight (set at __init__)
        current_kl_weight = kl_weight if kl_weight is not None else self.kl_weight
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        recon_losses = []
        kl_losses = []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            self.train()
            
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                # Use fixed KL weight
                loss_dict = self.compute_loss(out, kl_weight=current_kl_weight, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]
                
                # Check for non-finite losses
                if not torch.isfinite(loss):
                    if verbose >= 2:
                        print(f"Warning: Non-finite loss at epoch {epoch+1}, skipping batch")
                    continue
                
                loss.backward()
                # Match scVI's gradient clipping (ClippedAdam uses clip_norm=10.0)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_kl += loss_dict["kl_loss"].item()
                
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            
            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)

            # Verbose logging with verbose_every
            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )

            if do_print:
                print(f"Epoch {epoch+1:3d}/{epochs} [Phase1-LDA] | "
                      f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} (β={current_kl_weight:.2f})")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
                    if verbose >= 2:
                        print(f"  Saved checkpoint to {save_path}")
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
        }


    def get_topic_word_distribution(self) -> torch.Tensor:
        """Get topic-word distribution [n_topics, n_words]."""
        return self.ae.decoder.beta.detach()
    
    def get_top_words_per_topic(self, vocab: list, n_words: int = 10) -> Dict[int, list]:
        """Get top words per topic. Returns {topic_id: [(word, prob), ...]}."""
        beta = self.get_topic_word_distribution().cpu().numpy()
        result = {}
        for topic_id in range(self.n_topics):
            top_idx = np.argsort(beta[topic_id])[::-1][:n_words]
            result[topic_id] = [(vocab[idx], beta[topic_id, idx]) for idx in top_idx]
        return result


def create_topic_ode_model(
    n_words: int,
    n_topics: int = 10,
    **kwargs
) -> TopicODEModel:
    """
    Create TopicODE model.
    
    Args:
        n_words: Vocabulary size (input dimension)
        n_topics: Number of topics (latent dimension)
    
    Critical Parameters:
        encoder_hidden: Encoder hidden dimension (64-256)
        sparsity_strength: Prior sparsity (default: 1.0)
        kl_weight: KL divergence weight (0.1-2.0)
    """
    # Strip ODE-related kwargs for backward compatibility
    for _k in ('use_latent_dynamics', 'ae_reg', 'ode_reg', 'ode_epochs',
               'ode_lr', 'ode_consistency_weight', 'ode_recon_weight', 'use_ode'):
        kwargs.pop(_k, None)
    return TopicODEModel(
        input_dim=n_words,
        n_topics=n_topics,
        **kwargs
    )