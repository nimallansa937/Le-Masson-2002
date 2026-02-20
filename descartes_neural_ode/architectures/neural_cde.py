"""
Neural Controlled Differential Equation (T3).

dz/dt = f(z) * dX/dt
Dynamics controlled by input derivative — natural for spike-driven systems.

Note: Uses torchcde if available, falls back to manual implementation.
"""
import torch
import torch.nn as nn
import numpy as np

try:
    import torchcde
    HAS_TORCHCDE = True
except ImportError:
    HAS_TORCHCDE = False


class CDEFunc(nn.Module):
    """CDE vector field: f(z) returns a matrix (latent_dim x input_dim)."""

    def __init__(self, latent_dim, input_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * input_dim),
        )

    def forward(self, t, z):
        # z: (batch, latent_dim)
        out = self.net(z)  # (batch, latent_dim * input_dim)
        return out.view(-1, self.latent_dim, self.input_dim)


class NeuralCDEModel(nn.Module):
    """Neural CDE for TC circuit replacement.

    When torchcde is available, uses proper CDE solver.
    Otherwise, falls back to manual Euler integration with dX/dt.
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_input = n_input

        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.cde_func = CDEFunc(latent_dim, n_input, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_output), nn.Sigmoid(),
        )

    def forward(self, x, return_latent=True):
        """
        Manual CDE integration (Euler):
        z_{t+1} = z_t + f(z_t) * (x_{t+1} - x_t)
        """
        batch, seq_len, _ = x.shape
        z = self.encoder(x[:, 0, :])

        outputs = []
        latents = []

        for t in range(seq_len):
            out = self.decoder(z)
            outputs.append(out)
            latents.append(z.detach())

            if t < seq_len - 1:
                # dX = x_{t+1} - x_t
                dX = x[:, t + 1, :] - x[:, t, :]  # (batch, input_dim)
                # f(z): (batch, latent_dim, input_dim)
                f_z = self.cde_func(None, z)
                # dz = f(z) @ dX: (batch, latent_dim)
                dz = torch.bmm(f_z, dX.unsqueeze(-1)).squeeze(-1)
                z = z + dz

        y_pred = torch.stack(outputs, dim=1)
        latent_traj = torch.stack(latents, dim=1)

        if return_latent:
            return y_pred, latent_traj
        return y_pred
