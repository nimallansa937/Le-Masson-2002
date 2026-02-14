"""
Neural ODE Model for thalamic circuit transformation.

Architecture:
  - Encoder: Linear(21 → 64) maps input to latent space
  - ODE function: dz/dt = f(z, u) where f is a Tanh MLP
  - ODE solver: dopri5 (adaptive Runge-Kutta) via torchdiffeq
  - Decoder: Linear(64 → 20, sigmoid) maps latent to output

The ODE function learns a continuous-time dynamical system that maps
retinal input to TC output — the most structurally similar to the
biological circuit (which is itself a set of ODEs).

Latent representation: the ODE trajectory z(t) at each timestep
provides a continuous-time latent representation for comparison
with biological intermediates.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

from rung3.config import (
    NODE_LATENT_DIM, NODE_HIDDEN_DIM, NODE_N_HIDDEN,
    NODE_SOLVER, NODE_RTOL, NODE_ATOL, NODE_ADJOINT,
    INPUT_DIM, OUTPUT_DIM,
)


class ODEFunc(nn.Module):
    """Learned vector field dz/dt = f(z, u).

    Parameters
    ----------
    latent_dim : int
        Dimension of ODE state z.
    input_dim : int
        Dimension of external input u.
    hidden_dim : int
        Hidden layer width.
    n_hidden : int
        Number of hidden layers.
    """

    def __init__(self, latent_dim=NODE_LATENT_DIM,
                 input_dim=INPUT_DIM,
                 hidden_dim=NODE_HIDDEN_DIM,
                 n_hidden=NODE_N_HIDDEN):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Build MLP: [z; u] → dz/dt
        layers = []
        in_dim = latent_dim + input_dim
        for i in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

        # Current input (set externally at each ODE step)
        self._current_input = None

    def set_input(self, u):
        """Set the current external input for the ODE."""
        self._current_input = u

    def forward(self, t, z):
        """Compute dz/dt given state z and stored input u.

        Parameters
        ----------
        t : scalar (unused, but required by torchdiffeq)
        z : (batch, latent_dim)

        Returns
        -------
        dz : (batch, latent_dim)
        """
        if self._current_input is not None:
            zu = torch.cat([z, self._current_input], dim=-1)
        else:
            # Autonomous mode (no external input)
            zu = z
        return self.net(zu)


class ThalamicNeuralODE(nn.Module):
    """Neural ODE for thalamic circuit replacement.

    Shared interface:
      forward(x, return_latent=False)
        x: (batch, seq_len, 21)
        returns: (batch, seq_len, 20)  or  (output, latent_dict)

    Uses a step-by-step ODE integration approach (not solving the full
    trajectory at once) so that external input u(t) can change at each step.
    """

    def __init__(self, input_dim=INPUT_DIM, latent_dim=NODE_LATENT_DIM,
                 hidden_dim=NODE_HIDDEN_DIM, n_hidden=NODE_N_HIDDEN,
                 output_dim=OUTPUT_DIM, solver=NODE_SOLVER,
                 rtol=NODE_RTOL, atol=NODE_ATOL, adjoint=NODE_ADJOINT):
        super().__init__()

        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "torchdiffeq required for Neural ODE: pip install torchdiffeq")

        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint

        # Encoder: input → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ODE function
        self.ode_func = ODEFunc(latent_dim, input_dim, hidden_dim, n_hidden)

        # Decoder: latent → output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        # Integration step size (in abstract time units)
        self.dt_ode = 1.0  # Each bin = 1 time unit

    def forward(self, x, return_latent=False):
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)
        return_latent : bool

        Returns
        -------
        output : (batch, seq_len, output_dim)
        latent_dict : dict (if return_latent)
            'hidden': (batch, seq_len, latent_dim)
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Initialize latent state from first input
        z = self.encoder(x[:, 0, :])  # (batch, latent_dim)

        ode_solver = odeint_adjoint if self.adjoint else odeint

        outputs = []
        latents = [] if return_latent else None

        # Step-by-step integration
        t_span = torch.tensor([0.0, self.dt_ode], device=device)

        for t in range(seq_len):
            # Decode current state
            out = self.decoder(z)
            outputs.append(out)

            if return_latent:
                latents.append(z.detach())

            # Integrate one step with current input
            if t < seq_len - 1:
                self.ode_func.set_input(x[:, t, :])

                # Solve ODE from t to t+dt
                z_traj = ode_solver(
                    self.ode_func, z, t_span,
                    method=self.solver,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                z = z_traj[-1]  # Take endpoint

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)

        if return_latent:
            hidden = torch.stack(latents, dim=1)
            return output, {'hidden': hidden}

        return output
