"""
Hybrid LSTM->ODE (T6).

LSTM encoder for gradient highways + ODE dynamics between observations.
Combines LSTM's training stability with ODE's continuous-time inductive bias.
"""
import torch
import torch.nn as nn
import numpy as np

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class HybridODEFunc(nn.Module):
    """ODE dynamics for the hybrid model."""

    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t, z):
        return self.net(z)


class HybridLSTMODE(nn.Module):
    """Hybrid LSTM->ODE model for TC circuit replacement.

    Architecture:
    1. LSTM processes input in segments (gradient highways)
    2. Between observations, ODE evolves latent state (continuous dynamics)
    3. LSTM output gates control ODE initialization

    Segment size = 50 steps (matches gradient survival range).
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64,
                 hidden_dim=128, segment_size=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.segment_size = segment_size

        # LSTM encoder — processes segments
        self.lstm = nn.LSTM(
            input_size=n_input, hidden_size=latent_dim,
            num_layers=2, batch_first=True, dropout=0.1,
        )

        # ODE dynamics — evolves between observations
        self.ode_func = HybridODEFunc(latent_dim, hidden_dim) if HAS_TORCHDIFFEQ else None

        # Gating: how much to use LSTM vs ODE
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_output), nn.Sigmoid(),
        )

    def forward(self, x, return_latent=True):
        """
        Process in segments:
        1. LSTM encodes segment -> h_lstm
        2. ODE evolves h_lstm for segment_size steps
        3. Gate combines LSTM output with ODE evolution
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Run full LSTM (gradient highways handle long-range)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, latent_dim)

        # If no ODE available, just use LSTM
        if self.ode_func is None or not HAS_TORCHDIFFEQ:
            y_pred = self.decoder(lstm_out)
            if return_latent:
                return y_pred, lstm_out.detach()
            return y_pred

        # ODE refinement: evolve LSTM states with continuous dynamics
        outputs = []
        latents = []
        z_ode = lstm_out[:, 0, :]  # Initialize ODE state from first LSTM output

        t_span = torch.tensor([0.0, 1.0], device=device)

        for t in range(seq_len):
            # Gate between LSTM and ODE states
            h_lstm = lstm_out[:, t, :]
            combined = torch.cat([h_lstm, z_ode], dim=-1)
            g = self.gate(combined)
            z = g * h_lstm + (1 - g) * z_ode

            out = self.decoder(z)
            outputs.append(out)
            latents.append(z.detach())

            if t < seq_len - 1:
                # ODE step
                z_traj = odeint(
                    self.ode_func, z, t_span,
                    method='midpoint', rtol=1e-2, atol=1e-3,
                )
                z_ode = z_traj[-1]

        y_pred = torch.stack(outputs, dim=1)
        latent_traj = torch.stack(latents, dim=1)

        if return_latent:
            return y_pred, latent_traj
        return y_pred
