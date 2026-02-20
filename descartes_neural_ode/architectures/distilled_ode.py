"""
Volterra-Distilled Neural ODE (T7).

Knowledge distillation: train ODE to match Volterra's Laguerre coefficients
instead of backpropagating through time. Bypasses gradient problem entirely.

Uses Volterra's 89 recovered biological variables as teaching signal.
"""
import torch
import torch.nn as nn
import numpy as np


class DistilledODEFunc(nn.Module):
    """ODE dynamics with biophysical constraints."""

    def __init__(self, latent_dim, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._current_input = None

    def set_input(self, u):
        self._current_input = u

    def forward(self, t, z):
        if self._current_input is not None:
            zu = torch.cat([z, self._current_input], dim=-1)
        else:
            zu = z
        return self.net(zu)


class DistilledODE(nn.Module):
    """Volterra-Distilled Neural ODE for TC circuit replacement.

    Training differs from standard ODE:
    1. Teacher: Volterra model's Laguerre coefficients (89 recovered vars)
    2. Student: ODE latent dimensions
    3. Loss: MSE between ODE latents and Volterra's representations
    4. No gradient through ODE time integration

    The latent space is biophysically constrained:
    - Dimensions 0-59: sigmoid-activated (gating variables, 0-1)
    - Dimensions 60-119: tanh-activated (state variables)
    - Dimensions 120+: unconstrained (synaptic)
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Simple recurrent dynamics (avoid ODE solver overhead for distillation)
        self.dynamics = nn.GRUCell(n_input, latent_dim)

        # Biophysical activation layers
        self.n_gating = min(60, latent_dim)     # Gating variables (sigmoid)
        self.n_state = min(60, latent_dim - self.n_gating)  # State variables (tanh)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_output), nn.Sigmoid(),
        )

    def _apply_biophysical_constraints(self, z):
        """Apply biophysical constraints to latent dimensions."""
        parts = []
        if self.n_gating > 0:
            parts.append(torch.sigmoid(z[:, :self.n_gating]))
        if self.n_state > 0:
            parts.append(torch.tanh(z[:, self.n_gating:self.n_gating + self.n_state]))
        remaining = self.latent_dim - self.n_gating - self.n_state
        if remaining > 0:
            parts.append(z[:, self.n_gating + self.n_state:])
        return torch.cat(parts, dim=-1)

    def forward(self, x, return_latent=True):
        batch, seq_len, _ = x.shape

        z = self.encoder(x[:, 0, :])

        outputs = []
        latents = []

        for t in range(seq_len):
            z_constrained = self._apply_biophysical_constraints(z)
            out = self.decoder(z_constrained)
            outputs.append(out)
            latents.append(z_constrained.detach())

            if t < seq_len - 1:
                z = self.dynamics(x[:, t, :], z)

        y_pred = torch.stack(outputs, dim=1)
        latent_traj = torch.stack(latents, dim=1)

        if return_latent:
            return y_pred, latent_traj
        return y_pred
