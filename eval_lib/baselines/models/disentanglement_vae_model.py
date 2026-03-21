"""
Disentanglement VAE: default VAE architecture with optional DIP, TC, Info-VAE (MMD), and beta.

- DIP-VAE (Kim & Mnih 2018): covariance of posterior mean → identity.
- TC: total-correlation penalty (off-diagonal cov).
- Info-VAE (Zhao et al. 2019): MMD between aggregated posterior and prior (no decoder in latent).
- Beta-VAE (Higgins et al.): beta * KL for stronger disentanglement (beta > 1).

All other settings (architecture, lr, epochs) persist default VAE settings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from .base_model import BaseModel


class Encoder(nn.Module):
    """MLP encoder -> mu, logvar (same style as other VAEs)."""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.fc = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """MLP decoder -> reconstruction (MSE)."""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, z):
        return self.fc(z)


def _dip_loss(z_mean: torch.Tensor, lambda_diag: float = 1.0, lambda_off: float = 1.0) -> torch.Tensor:
    """DIP-VAE: regularize Cov(z_mean) toward identity. z_mean: [B, D]."""
    B = z_mean.size(0)
    if B < 2:
        return z_mean.new_zeros(1)
    z_centered = z_mean - z_mean.mean(dim=0)
    cov = (z_centered.t() @ z_centered) / (B - 1)
    diag_diff = cov.diag() - 1
    loss_diag = (diag_diff ** 2).sum()
    D = cov.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=cov.device)
    loss_off = (cov[mask] ** 2).sum()
    return lambda_diag * loss_diag + lambda_off * loss_off


def _tc_penalty(z: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal of Cov(z) to encourage factorized latent. z: [B, D]."""
    B = z.size(0)
    if B < 2:
        return z.new_zeros(1)
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.t() @ z_centered) / (B - 1)
    D = cov.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=cov.device)
    return (cov[mask] ** 2).sum()


def _mmd_prior(z: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """MMD between batch q(z) and prior N(0,I) with RBF kernel (Info-VAE). z: [B, D]."""
    B = z.size(0)
    if B < 2:
        return z.new_zeros(1)
    # Sample from prior
    p = torch.randn_like(z, device=z.device)
    # K(z,z) - 2*K(z,p) + K(p,p); kernel k(a,b) = exp(-||a-b||^2 / (2*sigma^2))
    def k(x, y):
        xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        return (-xy / (2 * sigma ** 2)).exp()
    kzz = k(z, z).mean()
    kzp = k(z, p).mean()
    kpp = k(p, p).mean()
    return kzz - 2 * kzp + kpp


class DisentanglementVAEModel(BaseModel):
    """
    VAE with optional DIP and/or TC regularization. Default architecture matches
    other baseline VAEs (hidden_dims, latent_dim, MSE recon).
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dims: list = None,
        dropout: float = 0.1,
        dip_weight: float = 0.0,
        tc_weight: float = 0.0,
        infovae_mmd_weight: float = 0.0,
        beta: float = 1.0,
        **kwargs):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims or [256, 128],
            model_name="disentanglement_vae")
        self.dip_weight = dip_weight
        self.tc_weight = tc_weight
        self.infovae_mmd_weight = infovae_mmd_weight
        self.beta = beta
        self.encoder = Encoder(input_dim, self.hidden_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, list(reversed(self.hidden_dims)), input_dim, dropout)

    def encode(self, x: torch.Tensor, **kwargs):
        return self.encoder(x)

    def decode(self, z: torch.Tensor, **kwargs):
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
        **kwargs) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {
            "latent": z,
            "mu": mu,
            "logvar": logvar,
            "reconstruction": recon,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        **kwargs) -> Dict[str, torch.Tensor]:
        recon = outputs["reconstruction"]
        mu, logvar = outputs["mu"], outputs["logvar"]
        z = outputs["latent"]

        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        total = recon_loss + self.beta * kl

        if self.dip_weight > 0:
            total = total + self.dip_weight * _dip_loss(mu)
        if self.tc_weight > 0:
            total = total + self.tc_weight * _tc_penalty(z)
        if self.infovae_mmd_weight > 0:
            total = total + self.infovae_mmd_weight * _mmd_prior(z)

        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl,
        }


def create_disentanglement_vae_model(
    input_dim: int,
    latent_dim: int = 10,
    hidden_dims: list = None,
    dropout: float = 0.1,
    dip_weight: float = 0.0,
    tc_weight: float = 0.0,
    infovae_mmd_weight: float = 0.0,
    beta: float = 1.0,
    **kwargs) -> DisentanglementVAEModel:
    """Factory: VAE with optional DIP/TC/Info-VAE (MMD)/beta for disentanglement group."""
    return DisentanglementVAEModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims or [256, 128],
        dropout=dropout,
        dip_weight=dip_weight,
        tc_weight=tc_weight,
        infovae_mmd_weight=infovae_mmd_weight,
        beta=beta)
