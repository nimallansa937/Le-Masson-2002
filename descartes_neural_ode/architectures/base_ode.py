"""
Standard Neural ODE (Known failure baseline — T0).

dz/dt = f([z;u]), solver configurable.
This is the same architecture that failed in A-R3 with 0.012 spike correlation.
Included to seed DreamCoder with failure patterns.
"""
import torch
import torch.nn as nn
import numpy as np

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class ODEFunc(nn.Module):
    """Learned vector field dz/dt = f(z, u)."""

    def __init__(self, latent_dim, input_dim, hidden_dim=128, n_hidden=2):
        super().__init__()
        layers = []
        in_dim = latent_dim + input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        self._current_input = None

    def set_input(self, u):
        self._current_input = u

    def forward(self, t, z):
        if self._current_input is not None:
            zu = torch.cat([z, self._current_input], dim=-1)
        else:
            zu = z
        return self.net(zu)


class TCReplacementNeuralODE(nn.Module):
    """Standard Neural ODE for TC circuit replacement.

    Args:
        n_input: Input dimension (21: 20 retinal + 1 GABA)
        n_output: Output dimension (20 TC neurons)
        latent_dim: ODE state dimension
        hidden_dim: MLP hidden dimension
        solver: ODE solver name
        adjoint: Whether to use adjoint method
    """

    def __init__(self, n_input=21, n_output=20, latent_dim=64,
                 hidden_dim=128, solver='rk4', adjoint=False):
        super().__init__()
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq required: pip install torchdiffeq")

        self.latent_dim = latent_dim
        self.solver = solver
        self.adjoint = adjoint

        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.ode_func = ODEFunc(latent_dim, n_input, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_output), nn.Sigmoid(),
        )

    def forward(self, x, return_latent=True):
        """
        Args:
            x: (batch, seq_len, n_input)
        Returns:
            y_pred: (batch, seq_len, n_output)
            latents: (batch, seq_len, latent_dim) if return_latent
        """
        batch, seq_len, _ = x.shape
        device = x.device
        ode_solver = odeint_adjoint if self.adjoint else odeint

        z = self.encoder(x[:, 0, :])
        t_span = torch.tensor([0.0, 1.0], device=device)

        outputs = []
        latents = []

        for t in range(seq_len):
            out = self.decoder(z)
            outputs.append(out)
            latents.append(z.detach())

            if t < seq_len - 1:
                self.ode_func.set_input(x[:, t, :])
                z_traj = ode_solver(
                    self.ode_func, z, t_span,
                    method=self.solver, rtol=1e-2, atol=1e-3,
                )
                z = z_traj[-1]

        y_pred = torch.stack(outputs, dim=1)
        latent_traj = torch.stack(latents, dim=1)

        if return_latent:
            return y_pred, latent_traj
        return y_pred
