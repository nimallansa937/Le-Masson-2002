"""
GRU-ODE-Bayes (T5).

GRU gating mechanism applied to continuous ODE dynamics.
Reset and update gates control which latent dimensions evolve.
"""
import torch
import torch.nn as nn
import numpy as np


class GRUODECell(nn.Module):
    """GRU-ODE cell: continuous-time GRU dynamics.

    dz/dt = (1 - update_gate) * (reset_gate * candidate - z)
    """

    def __init__(self, latent_dim, input_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Update gate
        self.W_z = nn.Linear(latent_dim + input_dim, latent_dim)
        # Reset gate
        self.W_r = nn.Linear(latent_dim + input_dim, latent_dim)
        # Candidate
        self.W_h = nn.Linear(latent_dim + input_dim, latent_dim)

    def forward(self, z, u, dt=1.0):
        """One Euler step of GRU-ODE.

        Args:
            z: (batch, latent_dim)
            u: (batch, input_dim)
        Returns:
            z_next: (batch, latent_dim)
        """
        zu = torch.cat([z, u], dim=-1)

        update = torch.sigmoid(self.W_z(zu))  # What to update
        reset = torch.sigmoid(self.W_r(zu))   # What to reset

        zu_reset = torch.cat([reset * z, u], dim=-1)
        candidate = torch.tanh(self.W_h(zu_reset))

        # Continuous dynamics: dz/dt = (1 - update) * (candidate - z)
        dz = (1 - update) * (candidate - z)
        z_next = z + dt * dz

        return z_next


class GRUODEModel(nn.Module):
    """GRU-ODE for TC circuit replacement.

    Discrete gates + continuous dynamics may recover ion channel
    gating behavior where gates open/close on different timescales.
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.gru_ode_cell = GRUODECell(latent_dim, n_input, hidden_dim)
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
                z = self.gru_ode_cell(z, x[:, t, :])

        y_pred = torch.stack(outputs, dim=1)
        latent_traj = torch.stack(latents, dim=1)

        if return_latent:
            return y_pred, latent_traj
        return y_pred
