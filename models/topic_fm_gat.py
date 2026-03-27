"""
Topic FM GAT Model -- Flow Matching variant with Graph Attention Network encoder

Uses a GAT encoder over a precomputed k-NN graph instead of MLP/Transformer.
Otherwise identical to TopicFMModel: VAE with logistic-normal prior + conditional
flow matching in pre-softmax R^K space for latent smoothing.

Single-phase training:
- Phase 1: Build k-NN graph from training data, then train GAT-based topic model
  with optional flow matching loss (activates after flow_warmup_epochs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np

try:
    from .base_model import BaseModel
    from .mixins import PriorMixin, ReconstructionLossMixin
    from .encoders import create_topic_encoder
    from .shared_modules import TopicDecoder, reparameterize, precompute_knn_graph
    from .topic_flow_matching import TopicFlowMatchingMixin
except ImportError:
    from utils.base_model import BaseModel
    from utils.mixins import PriorMixin, ReconstructionLossMixin
    from models.encoders import create_topic_encoder
    from models.shared_modules import TopicDecoder, reparameterize, precompute_knn_graph
    from models.topic_flow_matching import TopicFlowMatchingMixin


class TopicFMGATAutoEncoder(nn.Module):
    """GAT-based topic autoencoder.

    Uses a GATLogisticNormalEncoder (via create_topic_encoder('gat')) over a
    precomputed k-NN graph, paired with a standard TopicDecoder.

    Args:
        n_words: Vocabulary size (input dimension)
        n_topics: Number of topics (latent dimension K)
        hidden_dim: Hidden dimension for GAT layers
        nhead: Number of attention heads in GAT
        num_gat_layers: Number of GATConv layers
        dropout: Dropout rate
        use_bottleneck: Whether to use information bottleneck (placeholder, unused)
        bottleneck_dim: Bottleneck dimension (placeholder, unused)
    """

    def __init__(
        self,
        n_words: int,
        n_topics: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_words = n_words
        self.n_topics = n_topics
        self.use_bottleneck = use_bottleneck

        # GAT encoder: words -> (mu, var) for logistic-normal
        self.encoder = create_topic_encoder(
            encoder_type='gat',
            input_dim=n_words,
            n_topics=n_topics,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_gat_layers,
            dropout=dropout,
        )

        # Decoder: topic distribution -> word distribution
        self.decoder = TopicDecoder(n_topics, n_words)

        # Simplex mapping
        self.log_to_simplex = lambda x: F.softmax(x, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        x_norm: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through GAT encoder and topic decoder.

        Args:
            x: Input word counts [batch_size, n_words]
            x_norm: Normalized input (optional)
            edge_index: k-NN graph edge index [2, n_edges]
            edge_weight: Edge weights [n_edges] (optional)

        Returns:
            Dict with x, x_recon, mu, var, log_theta, theta
        """
        mu, var = self.encoder(x, x_norm, edge_index, edge_weight)

        # Sample log-theta using reparameterization trick
        log_theta = reparameterize(mu, var) if self.training else mu

        # Convert to simplex (topic distribution)
        theta = self.log_to_simplex(log_theta)

        # Decode to word distribution
        x_recon = self.decoder(theta)

        return {
            "x": x,
            "x_recon": x_recon,
            "mu": mu,
            "var": var,
            "log_theta": log_theta,
            "theta": theta,
        }


class TopicFMGATModel(TopicFlowMatchingMixin, PriorMixin, ReconstructionLossMixin, BaseModel):
    """Topic-FM-GAT: GAT encoder + flow matching + Dirichlet prior.

    Combines a GAT-based topic autoencoder with conditional flow matching
    in pre-softmax R^K space. The GAT encoder operates over a precomputed
    k-NN graph built from the training data, enabling message-passing
    between similar cells/documents.

    Training procedure:
    1. Collect all training data and build k-NN graph once
    2. Train VAE with GAT encoder + optional flow matching loss
    3. Flow matching activates after flow_warmup_epochs

    Args:
        input_dim: Vocabulary size (n_words)
        n_topics: Number of topics (latent dimension K)
        hidden_dim: GAT hidden dimension
        nhead: Number of GAT attention heads
        num_gat_layers: Number of GATConv layers
        dropout: Dropout rate
        use_bottleneck: Whether to use information bottleneck
        bottleneck_dim: Bottleneck dimension
        sparsity_strength: Dirichlet prior concentration parameter
        kl_weight: Fixed KL divergence weight
        reconstruction_loss: Loss type ("multinomial" or "kl")
        model_name: Model identifier
        use_raw_counts: Whether to use raw counts as primary input
        knn_k: Number of neighbors for k-NN graph
        knn_pca_dim: PCA dimension reduction before k-NN computation
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
        hidden_dim: int = 128,
        nhead: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.0,
        use_bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        sparsity_strength: float = 10.0,
        cell_topic_prior: Optional[torch.Tensor] = None,
        topic_word_prior: Optional[torch.Tensor] = None,
        kl_weight: float = 0.01,
        reconstruction_loss: str = "multinomial",
        model_name: str = "TopicFMGAT",
        use_raw_counts: bool = True,
        knn_k: int = 10,
        knn_pca_dim: int = 50,
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
            hidden_dims=[hidden_dim],
            model_name=model_name,
            use_raw_counts=use_raw_counts,
        )

        self.n_words = input_dim
        self.n_topics = n_topics
        self.kl_weight = kl_weight
        self.reconstruction_loss = reconstruction_loss
        self.knn_k = knn_k
        self.knn_pca_dim = knn_pca_dim

        # Initialize GAT autoencoder
        self.ae = TopicFMGATAutoEncoder(
            n_words=input_dim,
            n_topics=n_topics,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_gat_layers=num_gat_layers,
            dropout=dropout,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
        )

        # Priors: logistic-normal approximation of Dirichlet (same as TopicFMModel)
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

        # Graph state (built during fit)
        self._edge_index = None
        self._edge_weight = None

    def _build_graph(self, data_tensor: np.ndarray, device: torch.device):
        """Build k-NN graph from data using precompute_knn_graph from shared_modules.

        Args:
            data_tensor: Full training data [n_samples, n_features] as numpy array
            device: Target device for graph tensors
        """
        edge_index, edge_weight = precompute_knn_graph(
            data_tensor, k=self.knn_k, pca_dim=self.knn_pca_dim
        )
        self._edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        self._edge_weight = (
            torch.tensor(edge_weight, dtype=torch.float32, device=device)
            if edge_weight is not None
            else None
        )

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to topic distribution (theta on simplex)."""
        x_norm = kwargs.get('x_norm', None)
        edge_index = kwargs.get('edge_index', getattr(self, '_edge_index', None))
        edge_weight = kwargs.get('edge_weight', getattr(self, '_edge_weight', None))
        mu, var = self.ae.encoder(x, x_norm, edge_index, edge_weight)
        log_theta = mu
        theta = self.ae.log_to_simplex(log_theta)
        return theta

    def encode_log_theta(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to log_theta (before softmax, in R^K space)."""
        x_norm = kwargs.get('x_norm', None)
        edge_index = kwargs.get('edge_index', getattr(self, '_edge_index', None))
        edge_weight = kwargs.get('edge_weight', getattr(self, '_edge_weight', None))
        mu, var = self.ae.encoder(x, x_norm, edge_index, edge_weight)
        return mu

    def decode(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode theta to word distribution."""
        return self.ae.decoder(theta)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the GAT autoencoder.

        Args:
            x: Input word counts [n_samples, n_words]
            **kwargs: Must include or default to edge_index from _build_graph

        Returns:
            Dict with x, x_recon, mu, var, log_theta, theta
        """
        x_norm = kwargs.get('x_norm', None)
        edge_index = kwargs.get('edge_index', getattr(self, '_edge_index', None))
        edge_weight = kwargs.get('edge_weight', getattr(self, '_edge_weight', None))
        return self.ae(x, x_norm=x_norm, edge_index=edge_index, edge_weight=edge_weight)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        kl_weight: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss + flow matching loss.

        Same loss computation as TopicFMModel but applied to GAT encoder outputs.

        Args:
            outputs: Forward pass output dict
            kl_weight: Override KL weight (uses self.kl_weight if None)

        Returns:
            Dict with total_loss, recon_loss, kl_loss, and flow_loss
        """
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

        # KL divergence with free bits
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

        # Flow matching loss (activates after warmup, with linear ramp)
        flow_loss = torch.tensor(0.0, device=total_loss.device)
        current_fw = self._current_flow_weight()
        if current_fw > 0 and "log_theta" in outputs:
            flow_loss = self.compute_flow_loss(outputs["log_theta"])
        loss_dict["flow_loss"] = flow_loss
        loss_dict["total_loss"] = loss_dict["total_loss"] + current_fw * flow_loss

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
        """Train GAT-based topic model with flow matching.

        First collects all training data to build the k-NN graph, then runs
        the standard training loop with GAT encoder + flow matching.

        Note: GAT requires the full dataset graph, so all data from the loader
        is collected upfront and passed through the encoder as a single batch.
        The graph edges connect samples across the entire training set.

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

        # Step 1: Collect all training data to build graph
        all_x = []
        all_x_norm = []
        for batch in train_loader:
            x, batch_kwargs = self._prepare_batch(batch, device)
            all_x.append(x)
            if 'x_norm' in batch_kwargs:
                all_x_norm.append(batch_kwargs['x_norm'])

        all_x = torch.cat(all_x, dim=0)
        all_x_norm = torch.cat(all_x_norm, dim=0) if all_x_norm else None

        # Step 2: Build k-NN graph from training data
        # Use normalized data for graph construction if available, else log1p(raw)
        graph_data = all_x_norm.cpu().numpy() if all_x_norm is not None else torch.log1p(all_x).cpu().numpy()
        self._build_graph(graph_data, device=torch.device(device))

        if verbose >= 1:
            n_nodes = all_x.shape[0]
            n_edges = self._edge_index.shape[1] if self._edge_index is not None else 0
            print(f"[TopicFMGAT] Built k-NN graph: {n_nodes} nodes, {n_edges} edges (k={self.knn_k})")

        # Step 3: Standard training loop (full-batch through GAT)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # LR schedule: linear warmup (10% of epochs) + cosine decay
        warmup_epochs = max(1, epochs // 10)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        target_kl_weight = kl_weight if kl_weight is not None else self.kl_weight
        # KL annealing: ramp KL from 0 to target over warmup_epochs
        kl_anneal_epochs = warmup_epochs

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

            # KL annealing: linearly ramp from 0 to target
            kl_anneal = min(1.0, (epoch + 1) / max(1, kl_anneal_epochs))
            current_kl_weight = target_kl_weight * kl_anneal

            self.train()
            optimizer.zero_grad()

            # Full-batch forward (GAT needs all nodes + graph)
            forward_kwargs = {'edge_index': self._edge_index, 'edge_weight': self._edge_weight}
            if all_x_norm is not None:
                forward_kwargs['x_norm'] = all_x_norm

            out = self.forward(all_x, **forward_kwargs)
            loss_dict = self.compute_loss(out, kl_weight=current_kl_weight)
            loss = loss_dict["total_loss"]

            # Check for non-finite losses
            if not torch.isfinite(loss):
                if verbose >= 2:
                    print(f"Warning: Non-finite loss at epoch {epoch+1}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

            # EMA update after each optimizer step
            self._ema_update()

            avg_loss = loss.item()
            avg_recon = loss_dict["recon_loss"].item()
            avg_kl = loss_dict["kl_loss"].item()
            avg_flow = loss_dict["flow_loss"].item()

            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
            flow_losses.append(avg_flow)

            # Verbose logging
            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )

            if do_print:
                flow_status = "ON" if self._flow_active() else "OFF"
                fw = self._current_flow_weight()
                print(
                    f"Epoch {epoch+1:3d}/{epochs} [TopicFMGAT] | "
                    f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | "
                    f"KL: {avg_kl:.4f} (b={current_kl_weight:.3f}) | "
                    f"Flow: {avg_flow:.4f} (w={fw:.3f}) [{flow_status}]"
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

        # After training, swap to EMA weights for inference
        self._ema_swap()

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

        Collects all data from the loader, rebuilds graph if needed,
        then encodes through the GAT in a single forward pass.

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

        # Collect all data
        all_x = []
        all_x_norm = []
        for batch_data in data_loader:
            x, batch_kwargs = self._prepare_batch(batch_data, device)
            all_x.append(x)
            if 'x_norm' in batch_kwargs:
                all_x_norm.append(batch_kwargs['x_norm'])

        all_x = torch.cat(all_x, dim=0)
        all_x_norm = torch.cat(all_x_norm, dim=0) if all_x_norm else None

        # Build graph if not already built (e.g., loading from checkpoint)
        if self._edge_index is None:
            graph_data = (
                all_x_norm.cpu().numpy()
                if all_x_norm is not None
                else torch.log1p(all_x).cpu().numpy()
            )
            self._build_graph(graph_data, device=torch.device(device))

        with torch.no_grad():
            # Full forward to get log_theta
            edge_index = self._edge_index
            edge_weight = self._edge_weight
            x_norm_kw = all_x_norm if all_x_norm is not None else None

            mu, var = self.ae.encoder(all_x, x_norm_kw, edge_index, edge_weight)
            log_theta = mu

            # Apply flow smoothing if active
            if self.flow_smoothing and self._flow_active():
                log_theta = self.smooth_latent(log_theta)

            if return_log_theta:
                z = log_theta
            else:
                z = self.ae.log_to_simplex(log_theta)

            result = {"latent": z.cpu().numpy()}

            if return_reconstructions:
                theta = z if not return_log_theta else self.ae.log_to_simplex(log_theta)
                recon = self.decode(theta)
                result["reconstruction"] = recon.cpu().numpy()

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
