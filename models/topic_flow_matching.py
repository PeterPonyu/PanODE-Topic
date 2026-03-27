"""
Shared Flow Matching Module for Topic Models

Provides conditional flow matching (CFM) with optimal transport (OT) interpolation
in pre-softmax R^K space (log_theta), NOT on the simplex.

Components:
- SinusoidalTimeEmbedding: Continuous time encoding
- LatentFlowField: Velocity MLP v(z_t, t)
- TopicFlowMatchingMixin: Mixin providing flow loss, smoothing, sampling,
  adaptive flow weight scheduling, and EMA model support
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for continuous time t in [0, 1]."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch_size] or [batch_size, 1] in [0, 1]

        Returns:
            embeddings: [batch_size, embed_dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)

        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Handle odd embed_dim
        if self.embed_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding


class LatentFlowField(nn.Module):
    """Velocity field MLP: [z_t, t_emb] -> v(z_t, t)

    Uses LayerNorm + Mish activation built with nn.Sequential directly
    (not the shared MLP class).

    Args:
        latent_dim: Dimension of the latent space (number of topics K)
        time_embed_dim: Dimension of sinusoidal time embedding
        hidden_dims: Hidden layer dimensions for the velocity MLP
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int,
        time_embed_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.latent_dim = latent_dim

        # Build velocity MLP: input is [z_t, t_emb], output is v(z_t, t)
        input_dim = latent_dim + time_embed_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Mish(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Interpolated latent [batch_size, latent_dim]
            t: Time [batch_size] or scalar

        Returns:
            v: Predicted velocity [batch_size, latent_dim]
        """
        t_emb = self.time_embedding(t)
        x = torch.cat([z_t, t_emb], dim=-1)
        return self.net(x)


class TopicFlowMatchingMixin:
    """Mixin class providing flow matching functionality for topic models.

    Operates in pre-softmax R^K space (log_theta), NOT on the simplex.
    Uses optimal transport (OT) interpolation for conditional flow matching.

    Includes:
    - Adaptive flow weight scheduling (linear ramp after warmup)
    - EMA (Exponential Moving Average) model support for evaluation
    - Minibatch OT assignment for flow matching

    Must be mixed into a class that is also an nn.Module (or subclass thereof)
    so that self.flow_field is properly registered.
    """

    def _init_flow_matching(
        self,
        n_topics: int,
        flow_weight: float = 0.1,
        flow_noise_scale: float = 0.5,
        flow_hidden_dims: Optional[List[int]] = None,
        flow_time_dim: int = 32,
        flow_dropout: float = 0.05,
        flow_warmup_epochs: int = 50,
        flow_ramp_epochs: int = 20,
        flow_integration_steps: int = 16,
        flow_t0: float = 0.8,
        flow_smoothing: bool = True,
        flow_detach_target: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        use_minibatch_ot: bool = True,
    ):
        """Initialize flow matching components and store hyperparameters.

        Args:
            n_topics: Latent dimension (number of topics K)
            flow_weight: Target weight for flow matching loss in total loss
            flow_noise_scale: Standard deviation of source noise distribution
            flow_hidden_dims: Hidden dims for velocity MLP (default [128, 128])
            flow_time_dim: Sinusoidal time embedding dimension
            flow_dropout: Dropout in velocity MLP
            flow_warmup_epochs: Number of epochs before activating flow loss
            flow_ramp_epochs: Epochs over which flow weight ramps from 0 to target
            flow_integration_steps: Number of Euler integration steps
            flow_t0: Starting time for smoothing integration (t0 -> 1.0)
            flow_smoothing: Whether to apply flow smoothing at inference
            flow_detach_target: Whether to detach target from computation graph
            use_ema: Whether to maintain EMA copy of model weights
            ema_decay: EMA momentum coefficient (higher = slower update)
            use_minibatch_ot: Whether to use minibatch OT assignment
        """
        if flow_hidden_dims is None:
            flow_hidden_dims = [128, 128]

        # Velocity field network (registered as submodule via nn.Module)
        self.flow_field = LatentFlowField(
            latent_dim=n_topics,
            time_embed_dim=flow_time_dim,
            hidden_dims=flow_hidden_dims,
            dropout=flow_dropout,
        )

        # Store all hyperparameters
        self.flow_weight = flow_weight
        self.flow_noise_scale = flow_noise_scale
        self.flow_warmup_epochs = flow_warmup_epochs
        self.flow_ramp_epochs = flow_ramp_epochs
        self.flow_integration_steps = flow_integration_steps
        self.flow_t0 = flow_t0
        self.flow_smoothing = flow_smoothing
        self.flow_detach_target = flow_detach_target
        self._flow_current_epoch = 0
        self._flow_n_topics = n_topics
        self.use_minibatch_ot = use_minibatch_ot

        # EMA state
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self._ema_shadow = None  # Lazily initialized on first update
        self._ema_initialized = False

    def _flow_active(self) -> bool:
        """Check if flow matching is active (past warmup and weight > 0)."""
        return (
            self._flow_current_epoch >= self.flow_warmup_epochs
            and self.flow_weight > 0
        )

    def _current_flow_weight(self) -> float:
        """Compute current flow weight with linear ramp after warmup.

        Returns 0 before warmup, then linearly ramps from 0 to self.flow_weight
        over flow_ramp_epochs after the warmup period ends.
        """
        if not self._flow_active():
            return 0.0
        epochs_since_warmup = self._flow_current_epoch - self.flow_warmup_epochs
        if self.flow_ramp_epochs <= 0:
            return self.flow_weight
        ramp = min(1.0, (epochs_since_warmup + 1) / self.flow_ramp_epochs)
        return self.flow_weight * ramp

    # ── EMA support ──────────────────────────────────────────────────────

    def _ema_init(self):
        """Initialize EMA shadow parameters (lazy, called on first update)."""
        if not self.use_ema or self._ema_initialized:
            return
        self._ema_shadow = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._ema_shadow[name] = param.data.clone()
        self._ema_initialized = True

    @torch.no_grad()
    def _ema_update(self):
        """Update EMA shadow parameters after each optimizer step."""
        if not self.use_ema:
            return
        if not self._ema_initialized:
            self._ema_init()
            return
        decay = self.ema_decay
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_shadow:
                self._ema_shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def _ema_swap(self):
        """Swap model parameters with EMA shadow (for evaluation)."""
        if not self.use_ema or not self._ema_initialized:
            return
        self._ema_backup = {}
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_shadow:
                self._ema_backup[name] = param.data.clone()
                param.data.copy_(self._ema_shadow[name])

    def _ema_restore(self):
        """Restore original parameters after EMA evaluation."""
        if not self.use_ema or not hasattr(self, '_ema_backup'):
            return
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_backup:
                param.data.copy_(self._ema_backup[name])
        del self._ema_backup

    @staticmethod
    def _minibatch_ot_assignment(z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Compute minibatch optimal transport assignment via cost matrix.

        Finds a permutation of z_source that minimizes total squared distance
        to z_target. Uses greedy assignment (fast approximation) rather than
        full Hungarian algorithm for speed.

        Args:
            z_source: Source samples [batch_size, K]
            z_target: Target samples [batch_size, K]

        Returns:
            z_source_permuted: Reordered source [batch_size, K]
        """
        with torch.no_grad():
            # Cost matrix: pairwise squared L2 distances [B, B]
            cost = torch.cdist(z_source, z_target, p=2).pow(2)
            # Greedy assignment: for each target, pick closest remaining source
            batch_size = z_source.shape[0]
            perm = torch.zeros(batch_size, dtype=torch.long, device=z_source.device)
            used = torch.zeros(batch_size, dtype=torch.bool, device=z_source.device)
            for i in range(batch_size):
                # Mask already-used sources with inf
                col = cost[:, i].clone()
                col[used] = float('inf')
                j = col.argmin()
                perm[i] = j
                used[j] = True
        return z_source[perm]

    def _sample_flow_targets(self, log_theta: torch.Tensor):
        """Sample OT interpolation targets in R^K space.

        Constructs interpolated points z_t = (1-t)*z_source + t*z_target
        and corresponding target velocities v_target = z_target - z_source.

        When use_minibatch_ot is True, applies minibatch OT assignment to
        pair source noise samples with targets, reducing crossing paths.

        Args:
            log_theta: Pre-softmax topic logits [batch_size, K]

        Returns:
            z_t: Interpolated points [batch_size, K]
            velocity_target: Target velocity [batch_size, K]
            t: Sampled time values [batch_size]
        """
        batch_size = log_theta.shape[0]
        device = log_theta.device

        # Target: log_theta (optionally detached from computation graph)
        z_target = log_theta.detach() if self.flow_detach_target else log_theta

        # Source: random noise scaled by noise_scale
        z_source = torch.randn_like(z_target) * self.flow_noise_scale

        # Minibatch OT: reorder source to minimize transport cost
        if self.use_minibatch_ot and batch_size > 1:
            z_source = self._minibatch_ot_assignment(z_source, z_target)

        # Random time in [0, 1]
        t = torch.rand(batch_size, device=device)

        # OT interpolation: z_t = (1 - t) * z_source + t * z_target
        t_expand = t.unsqueeze(-1)  # [batch_size, 1]
        z_t = (1 - t_expand) * z_source + t_expand * z_target

        # Target velocity: z_target - z_source (constant-speed OT path)
        velocity_target = z_target - z_source

        return z_t, velocity_target, t

    def compute_flow_loss(self, log_theta: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss (MSE between predicted and target velocity).

        Args:
            log_theta: Pre-softmax topic logits [batch_size, K]

        Returns:
            loss: MSE flow matching loss (scalar)
        """
        z_t, velocity_target, t = self._sample_flow_targets(log_theta)
        v_pred = self.flow_field(z_t, t)
        loss = F.mse_loss(v_pred, velocity_target)
        return loss

    @torch.no_grad()
    def smooth_latent(
        self,
        log_theta: torch.Tensor,
        t0: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Smooth latent representations via Euler integration from t0 to 1.0.

        Uses the learned velocity field to refine encoder outputs by
        integrating the flow from t0 (near the data) to t=1.0.

        Args:
            log_theta: Pre-softmax topic logits [batch_size, K]
            t0: Starting time (default: self.flow_t0)
            steps: Number of Euler integration steps (default: self.flow_integration_steps)

        Returns:
            z: Smoothed latent [batch_size, K]
        """
        if t0 is None:
            t0 = self.flow_t0
        if steps is None:
            steps = self.flow_integration_steps

        dt = (1.0 - t0) / steps
        z = log_theta.clone()

        for i in range(steps):
            t_current = t0 + i * dt
            t_batch = torch.full((z.shape[0],), t_current, device=z.device)
            v = self.flow_field(z, t_batch)
            z = z + v * dt

        return z

    @torch.no_grad()
    def sample_latent_prior(
        self,
        n_samples: int,
        device: torch.device,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample from learned prior via Euler integration from 0 to 1.

        Starts from noise and integrates the velocity field forward
        to generate samples from the learned latent distribution.

        Args:
            n_samples: Number of samples to generate
            device: Target device
            steps: Number of Euler integration steps (default: self.flow_integration_steps)

        Returns:
            z: Sampled latent [n_samples, K]
        """
        if steps is None:
            steps = self.flow_integration_steps

        # Start from noise
        z = torch.randn(n_samples, self._flow_n_topics, device=device) * self.flow_noise_scale

        dt = 1.0 / steps

        for i in range(steps):
            t_current = i * dt
            t_batch = torch.full((n_samples,), t_current, device=device)
            v = self.flow_field(z, t_batch)
            z = z + v * dt

        return z
