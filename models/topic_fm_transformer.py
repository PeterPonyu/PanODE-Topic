"""
TopicFMTransformer: Flow Matching variant of TopicODETransformerModel

Identical base behavior to TopicODETransformerModel (Transformer VAE with
logistic-normal prior) plus conditional flow matching in pre-softmax R^K
space for latent smoothing and structured prior learning.

Single-phase training:
- Phase 1: Train Transformer VAE-based topic model with optional flow matching loss
  (flow matching activates after flow_warmup_epochs)

Uses shared modules from:
- topic_transformer.py: TopicTransformerAutoEncoder
- topic_flow_matching.py: TopicFlowMatchingMixin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Literal
import numpy as np

try:
    from .base_model import BaseModel
    from .mixins import PriorMixin, ReconstructionLossMixin
    from .topic_transformer import TopicTransformerAutoEncoder
    from .topic_flow_matching import TopicFlowMatchingMixin
except ImportError:
    from utils.base_model import BaseModel
    from utils.mixins import PriorMixin, ReconstructionLossMixin
    from models.topic_transformer import TopicTransformerAutoEncoder
    from models.topic_flow_matching import TopicFlowMatchingMixin


class TopicFMTransformerModel(TopicFlowMatchingMixin, PriorMixin, ReconstructionLossMixin, BaseModel):
    """Topic model with Transformer encoder and flow matching in pre-softmax R^K space.

    Combines a Transformer VAE-based topic model (identical to TopicODETransformerModel)
    with conditional flow matching for latent space regularization and smoothing.

    The flow matching loss activates after a configurable warmup period,
    allowing the VAE to first learn a reasonable latent space before the
    flow field is trained on top of it.

    Args:
        input_dim: Vocabulary size (n_words)
        n_topics: Number of topics (latent dimension K)
        latent_dim: Kept for BaseModel compatibility (defaults to n_topics)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Transformer feedforward dimension
        dropout: Dropout rate
        encoder_type: Encoder architecture ('transformer', 'hybrid', 'mlp')
        use_bottleneck: Whether to use information bottleneck
        bottleneck_dim: Bottleneck dimension (auto-computed if None)
        sparsity_strength: Dirichlet prior concentration parameter
        cell_topic_prior: Custom cell-topic Dirichlet prior
        topic_word_prior: Custom topic-word Dirichlet prior
        kl_weight: Fixed KL divergence weight
        reconstruction_loss: Loss type ("multinomial" or "kl")
        model_name: Model identifier
        use_raw_counts: Whether to use raw counts as primary input
        flow_weight: Weight for flow matching loss
        flow_noise_scale: Scale of source noise distribution
        flow_hidden_dims: Hidden dims for velocity MLP
        flow_time_dim: Sinusoidal time embedding dimension
        flow_dropout: Dropout in velocity MLP
        flow_warmup_epochs: Epochs before activating flow loss
        flow_integration_steps: Euler integration steps
        flow_t0: Starting time for smoothing (t0 -> 1.0)
        flow_smoothing: Whether to apply flow smoothing at inference
        flow_detach_target: Whether to detach flow target from graph
    """

    def __init__(
        self,
        input_dim: int,
        n_topics: int = 10,
        latent_dim: Optional[int] = None,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        encoder_type: Literal['transformer', 'hybrid', 'mlp'] = 'transformer',
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        sparsity_strength: float = 10.0,
        cell_topic_prior: Optional[torch.Tensor] = None,
        topic_word_prior: Optional[torch.Tensor] = None,
        kl_weight: float = 0.01,
        reconstruction_loss: str = "multinomial",
        model_name: str = "TopicFMTransformer",
        use_raw_counts: bool = True,
        # Flow matching parameters
        flow_weight: float = 0.1,
        flow_noise_scale: float = 0.5,
        flow_hidden_dims: Optional[List[int]] = None,
        flow_time_dim: int = 32,
        flow_dropout: float = 0.05,
        flow_warmup_epochs: int = 50,
        flow_integration_steps: int = 16,
        flow_t0: float = 0.8,
        flow_smoothing: bool = True,
        flow_detach_target: bool = False,
    ):
        # Initialize BaseModel
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim or n_topics,
            hidden_dims=[d_model],
            model_name=model_name,
            use_raw_counts=use_raw_counts,
        )

        self.n_words = input_dim
        self.n_topics = n_topics
        self.kl_weight = kl_weight
        self.reconstruction_loss = reconstruction_loss
        self.encoder_type = encoder_type

        # Initialize autoencoder (same as TopicODETransformerModel)
        self.ae = TopicTransformerAutoEncoder(
            n_words=input_dim,
            n_topics=n_topics,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            encoder_type=encoder_type,
        )

        # Priors: logistic-normal approximation of Dirichlet
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

        # Initialize flow matching components
        self._init_flow_matching(
            n_topics=n_topics,
            flow_weight=flow_weight,
            flow_noise_scale=flow_noise_scale,
            flow_hidden_dims=flow_hidden_dims,
            flow_time_dim=flow_time_dim,
            flow_dropout=flow_dropout,
            flow_warmup_epochs=flow_warmup_epochs,
            flow_integration_steps=flow_integration_steps,
            flow_t0=flow_t0,
            flow_smoothing=flow_smoothing,
            flow_detach_target=flow_detach_target,
        )

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to topic distribution (theta on simplex)."""
        x_norm = kwargs.get('x_norm', None)
        mu, var = self.ae.encoder(x, x_norm)
        log_theta = mu
        theta = self.ae.log_to_simplex(log_theta)
        return theta

    def encode_log_theta(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to log_theta (before softmax, in R^K space)."""
        x_norm = kwargs.get('x_norm', None)
        mu, var = self.ae.encoder(x, x_norm)
        return mu

    def decode(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode theta to word distribution."""
        return self.ae.decoder(theta)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the autoencoder."""
        x_norm = kwargs.get('x_norm', None)
        return self.ae(x, x_norm)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        kl_weight: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss + flow matching loss.

        Same as TopicODETransformerModel loss computation but with an additional
        flow matching term that activates after warmup.

        Args:
            outputs: Forward pass output dict
            kl_weight: Override KL weight (uses self.kl_weight if None)

        Returns:
            Dict with total_loss, recon_loss, kl_loss, and flow_loss
        """
        # Use provided kl_weight or default
        current_kl_weight = kl_weight if kl_weight is not None else self.kl_weight

        # Reconstruction loss (same as TopicODETransformerModel)
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

        # KL divergence with free bits (same as TopicFMModel)
        kl_loss = self._kl_logistic_normal_free_bits(
            outputs["mu"],
            outputs["var"],
            self.prior_mu_topics,
            self.prior_sigma_topics ** 2,
            free_bits=0.1,
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

        # Flow matching loss (activates after warmup)
        flow_loss = torch.tensor(0.0, device=total_loss.device)
        if self._flow_active() and "log_theta" in outputs:
            flow_loss = self.compute_flow_loss(outputs["log_theta"])
        loss_dict["flow_loss"] = flow_loss
        loss_dict["total_loss"] = loss_dict["total_loss"] + self.flow_weight * flow_loss

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
        **kwargs,
    ):
        """Train Transformer VAE-based topic model with flow matching.

        Same training loop as TopicODETransformerModel but additionally:
        - Sets self._flow_current_epoch at the start of each epoch
        - Tracks flow_loss in verbose output
        - Flow matching loss activates after flow_warmup_epochs

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (unused, kept for API compatibility)
            epochs: Maximum training epochs
            lr: Learning rate
            device: Computation device
            save_path: Path to save best checkpoint
            patience: Early stopping patience
            verbose: Verbosity level (0=quiet, 1=epoch logs, 2=debug)
            verbose_every: Print frequency for epoch logs
            kl_weight: Override KL weight (uses self.kl_weight if None)
            weight_decay: AdamW weight decay (L2 regularization)
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Use provided kl_weight or fall back to self.kl_weight
        current_kl_weight = kl_weight if kl_weight is not None else self.kl_weight

        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        recon_losses = []
        kl_losses = []
        flow_losses = []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            # Set current epoch for flow warmup tracking
            self._flow_current_epoch = epoch

            self.train()

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_flow = 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)
                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                loss_dict = self.compute_loss(out, kl_weight=current_kl_weight, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]

                # Check for non-finite losses
                if not torch.isfinite(loss):
                    if verbose >= 2:
                        print(f"Warning: Non-finite loss at epoch {epoch+1}, skipping batch")
                    continue

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_kl += loss_dict["kl_loss"].item()
                epoch_flow += loss_dict["flow_loss"].item()

                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            avg_flow = epoch_flow / n_batches

            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
            flow_losses.append(avg_flow)

            # Verbose logging with verbose_every
            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )

            if do_print:
                flow_status = "ON" if self._flow_active() else "OFF"
                print(
                    f"Epoch {epoch+1:3d}/{epochs} [TopicFMTransformer] | "
                    f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | "
                    f"KL: {avg_kl:.4f} (b={current_kl_weight:.2f}) | "
                    f"Flow: {avg_flow:.4f} [{flow_status}]"
                )

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
            "flow_loss": flow_losses,
        }

    def extract_latent(
        self,
        data_loader,
        device='cuda',
        return_reconstructions: bool = False,
        return_log_theta: bool = False,
    ):
        """Extract latent representations with optional flow smoothing.

        If flow_smoothing is enabled and the flow is active, applies the
        learned velocity field to smooth encoder outputs before converting
        to the simplex via softmax.

        Args:
            data_loader: DataLoader providing input data
            device: Computation device
            return_reconstructions: If True, also return word reconstructions
            return_log_theta: If True, return pre-softmax log_theta instead of theta

        Returns:
            Dict with 'latent' (and optionally 'reconstruction')
        """
        self.eval()
        self.to(device)
        latents, recons = [], []

        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)

                # Always get log_theta from encoder
                log_theta = self.encode_log_theta(x, **batch_kwargs)

                # Apply flow smoothing if active
                if self.flow_smoothing and self._flow_active():
                    log_theta = self.smooth_latent(log_theta)

                if return_log_theta:
                    z = log_theta
                else:
                    z = self.ae.log_to_simplex(log_theta)

                latents.append(z.cpu().numpy())

                if return_reconstructions:
                    theta = z if not return_log_theta else self.ae.log_to_simplex(z)
                    recons.append(self.decode(theta).cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result

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
