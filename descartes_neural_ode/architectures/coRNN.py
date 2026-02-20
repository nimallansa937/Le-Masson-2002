"""
Coupled Oscillatory RNN — coRNN (T4).

Second-order system: d2z/dt2 + gamma*dz/dt + omega^2*z = f(u)
Built-in oscillatory modes for spindle dynamics (7-10 Hz).
"""
import torch
import torch.nn as nn
import numpy as np


class CoRNNCell(nn.Module):
    """Coupled oscillatory RNN cell.

    Discretized second-order system:
        v_{t+1} = v_t + dt * (-gamma * v_t - omega^2 * tanh(z_t) + W_u * u_t)
        z_{t+1} = z_t + dt * v_t
    where v = dz/dt (velocity), z = position.
    """

    def __init__(self, latent_dim, input_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Learnable damping and frequency
        self.gamma = nn.Parameter(torch.rand(latent_dim) * 0.5 + 0.1)  # Damping
        self.omega = nn.Parameter(torch.rand(latent_dim) * 5.0 + 1.0)  # Angular frequency

        # Input coupling
        self.W_u = nn.Linear(input_dim, latent_dim)

        # Coupling between oscillators
        self.W_z = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z, v, u, dt=1.0):
        """One step of coRNN integration.

        Args:
            z: position (batch, latent_dim)
            v: velocity (batch, latent_dim)
            u: input (batch, input_dim)
        Returns:
            z_next, v_next
        """
        coupled = self.W_z(torch.tanh(z))
        input_force = self.W_u(u)

        dv = -self.gamma * v - self.omega ** 2 * torch.tanh(z) + coupled + input_force
        v_next = v + dt * dv
        z_next = z + dt * v_next

        return z_next, v_next


class CoRNNModel(nn.Module):
    """Coupled Oscillatory RNN for TC circuit replacement.

    Uses second-order dynamics with built-in oscillatory modes.
    The TC-nRt circuit produces 7-10 Hz spindle oscillations.
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.cornn_cell = CoRNNCell(latent_dim, n_input, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.Tanh(),  # Both z and v
            nn.Linear(hidden_dim, n_output), nn.Sigmoid(),
        )

    def forward(self, x, return_latent=True):
        batch, seq_len, _ = x.shape

        z = self.encoder(x[:, 0, :])
        v = torch.zeros_like(z)  # Zero initial velocity

        outputs = []
        latents = []

        for t in range(seq_len):
            # Decode from both position and velocity
            state = torch.cat([z, v], dim=-1)
            out = self.decoder(state)
            outputs.append(out)
            latents.append(z.detach())

            if t < seq_len - 1:
                z, v = self.cornn_cell(z, v, x[:, t, :])

        y_pred = torch.stack(outputs, dim=1)
        latent_traj = torch.stack(latents, dim=1)

        if return_latent:
            return y_pred, latent_traj
        return y_pred
