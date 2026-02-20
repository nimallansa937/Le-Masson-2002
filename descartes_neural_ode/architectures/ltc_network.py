"""
Liquid Time-Constant Network (T2).

tau(z)*dz/dt = f(z) - z + g(u)
State-dependent time constants that match biological activation/inactivation.
"""
import torch
import torch.nn as nn
import numpy as np


class LTCCell(nn.Module):
    """Single LTC cell: tau(z)*dz/dt = f(z) - z + g(u)"""

    def __init__(self, latent_dim, input_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # f(z): nonlinear dynamics
        self.f_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # tau(z): state-dependent time constants (positive via softplus)
        self.tau_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim), nn.Softplus(),
        )
        # g(u): input coupling (gated)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, latent_dim), nn.Sigmoid(),
        )
        self.input_transform = nn.Sequential(
            nn.Linear(input_dim, latent_dim), nn.Tanh(),
        )

    def forward(self, z, u, dt=1.0):
        """One Euler step: z_{t+1} = z_t + dt/tau * (f(z) - z + gate*input)"""
        tau = self.tau_net(z) + 0.1  # Ensure minimum time constant
        f_z = self.f_net(z)
        gate = self.gate(u)
        inp = self.input_transform(u)

        dz = (f_z - z + gate * inp) / tau
        z_next = z + dt * dz
        return z_next


class LTCModel(nn.Module):
    """Liquid Time-Constant Network for TC circuit replacement.

    Uses direct backpropagation (no adjoint) with Euler integration.
    State-dependent time constants naturally handle multi-timescale dynamics.
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.ltc_cell = LTCCell(latent_dim, n_input, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_output), nn.Sigmoid(),
        )

    def forward(self, x, return_latent=True):
        batch, seq_len, _ = x.shape
        z = self.encoder(x[:, 0, :])

        outputs = []
        latents = []

        for t in range(seq_len):
            out = self.decoder(z)
            outputs.append(out)
            latents.append(z.detach())

            if t < seq_len - 1:
                z = self.ltc_cell(z, x[:, t, :])

        y_pred = torch.stack(outputs, dim=1)
        latent_traj = torch.stack(latents, dim=1)

        if return_latent:
            return y_pred, latent_traj
        return y_pred
