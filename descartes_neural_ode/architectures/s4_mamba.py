"""
Structured State Space Model — S4/Mamba (T8).

dx/dt = Ax + Bu, y = Cx + Du
With structured A matrix (HiPPO initialization).
Linear recurrence for O(n log n) computation.
"""
import torch
import torch.nn as nn
import numpy as np
import math


def hippo_init(N):
    """HiPPO-LegS initialization for the A matrix.

    Creates a structured state matrix whose eigenvalues
    correspond to a family of orthogonal polynomial bases.
    """
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = n + 1
    return -A


class S4Layer(nn.Module):
    """Single S4 layer: structured state space.

    Discretized: x_{t+1} = A_bar * x_t + B_bar * u_t
                 y_t = C * x_t + D * u_t
    """

    def __init__(self, input_dim, state_dim, output_dim, dt=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt

        # Initialize A with HiPPO
        A_init = hippo_init(state_dim)
        self.A = nn.Parameter(torch.tensor(A_init, dtype=torch.float32))

        # B, C, D matrices
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.zeros(output_dim, input_dim))

    def forward(self, u_seq, x0=None):
        """
        Args:
            u_seq: (batch, seq_len, input_dim)
            x0: (batch, state_dim) initial state

        Returns:
            y_seq: (batch, seq_len, output_dim)
            x_final: (batch, state_dim)
        """
        batch, seq_len, _ = u_seq.shape

        if x0 is None:
            x = torch.zeros(batch, self.state_dim, device=u_seq.device)
        else:
            x = x0

        # Discretize: ZOH
        # A_bar = exp(A * dt) ≈ I + A*dt (first-order approximation)
        A_bar = torch.eye(self.state_dim, device=u_seq.device) + self.A * self.dt
        B_bar = self.B * self.dt

        outputs = []
        for t in range(seq_len):
            u_t = u_seq[:, t, :]  # (batch, input_dim)

            # y_t = C * x + D * u
            y_t = x @ self.C.T + u_t @ self.D.T  # (batch, output_dim)
            outputs.append(y_t)

            # x_{t+1} = A_bar * x + B_bar * u
            x = x @ A_bar.T + u_t @ B_bar.T

        y_seq = torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)
        return y_seq, x


class S4MambaModel(nn.Module):
    """S4/Mamba-style model for TC circuit replacement.

    Uses HiPPO-initialized structured state space.
    Linear recurrence enables efficient long-sequence processing.
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Linear(n_input, hidden_dim)

        # S4 layer
        self.s4 = S4Layer(hidden_dim, latent_dim, hidden_dim, dt=1.0)

        # Output layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_output), nn.Sigmoid(),
        )

        # For latent extraction
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, return_latent=True):
        """
        Args:
            x: (batch, seq_len, n_input)
        Returns:
            y_pred: (batch, seq_len, n_output)
            latents: (batch, seq_len, latent_dim)
        """
        # Project input
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # S4 recurrence
        h_out, _ = self.s4(h)  # (batch, seq_len, hidden_dim)

        # Residual connection
        h_res = h + h_out

        # Decode
        y_pred = self.decoder(h_res)

        if return_latent:
            latents = self.latent_proj(h_res.detach())
            return y_pred, latents

        return y_pred
