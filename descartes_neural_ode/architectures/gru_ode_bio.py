"""
GRU-ODE with auxiliary biological projection heads (A-R3b).

The core GRU-ODE dynamics are identical to the A-R3 version.
The addition is linear projection heads that map latent states
to biological variable predictions, with a combined loss function.

Why linear projections? Because we're testing whether biology is
ENCODED in the latent representation, not whether a nonlinear decoder
can reconstruct it. Linear decodability is the standard test for
genuine neural representation (cf. neuroscience probing literature).
"""
import torch
import torch.nn as nn


class BioProjectionHead(nn.Module):
    """
    Linear projection from latent space to biological variable space.

    Single linear layer — no hidden layers, no nonlinearity.
    If biology can't be linearly decoded, it's not genuinely encoded.
    """

    def __init__(self, latent_dim, bio_dim, category_name):
        super().__init__()
        self.category_name = category_name
        self.bio_dim = bio_dim
        self.projection = nn.Linear(latent_dim, bio_dim)

    def forward(self, z_trajectory):
        """
        Args:
            z_trajectory: (batch, time, latent_dim)
        Returns:
            bio_pred: (batch, time, bio_dim)
        """
        return self.projection(z_trajectory)


class GRUODEBio(nn.Module):
    """
    GRU-ODE-Bayes with biological projection heads.

    Forward pass returns BOTH spike predictions AND biological
    variable predictions. The training loop combines both losses.

    Architecture:
      Input → Encoder → z₀
                         ↓
                    GRU-ODE dynamics: dz/dt = (1-update) * (candidate - z)
                         ↓
                    z(t) trajectory
                    /         |          \\
              Spike decoder  Bio head    Bio head
              → ŷ_spikes    → ĝ_tc      → ĝ_syn
    """

    def __init__(self, input_dim=21, output_dim=20, latent_dim=32,
                 hidden_dim=64, bio_dims=None):
        """
        Args:
            input_dim: Input dimension (retinal spikes + GABA parameter)
            output_dim: Output dimension (TC spike predictions)
            latent_dim: GRU-ODE latent state dimension
            hidden_dim: Hidden layer size in GRU-ODE dynamics
            bio_dims: Dict mapping category name to number of bio variables
                      e.g. {'tc_gating': 60, 'nrt_state': 60, 'synaptic': 40}
                      If None, no bio projections (A-R3 baseline mode)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.bio_dims = bio_dims or {}

        # === Core GRU-ODE (identical to A-R3 version) ===
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # GRU-ODE gates
        self.W_z = nn.Linear(latent_dim + input_dim, latent_dim)  # Update gate
        self.W_r = nn.Linear(latent_dim + input_dim, latent_dim)  # Reset gate
        self.W_h = nn.Linear(latent_dim + input_dim, latent_dim)  # Candidate

        # Spike decoder (outputs logits — use BCEWithLogitsLoss)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # === Bio projection heads (NEW for A-R3b) ===
        self.bio_heads = nn.ModuleDict()
        for cat_name, cat_dim in self.bio_dims.items():
            self.bio_heads[cat_name] = BioProjectionHead(
                latent_dim, cat_dim, cat_name
            )

    def gru_ode_step(self, z, u, dt=1.0):
        """
        Single GRU-ODE Euler step.

        dz/dt = (1 - update_gate) * (candidate - z)

        Update gate controls which dims evolve. In A-R3, 25/32 dims
        had update > 0.7 (static). The bio loss should recruit these
        unused dims for biological encoding.
        """
        zu = torch.cat([z, u], dim=-1)
        update = torch.sigmoid(self.W_z(zu))
        reset = torch.sigmoid(self.W_r(zu))
        candidate_input = torch.cat([reset * z, u], dim=-1)
        candidate = torch.tanh(self.W_h(candidate_input))
        dzdt = (1 - update) * (candidate - z)
        z_next = z + dzdt * dt
        return z_next

    def forward(self, x, return_latents=False):
        """
        Forward pass returning spike predictions and bio predictions.

        Args:
            x: (batch, time, input_dim) — input spike trains
            return_latents: if True, also return raw latent trajectory

        Returns:
            spike_pred: (batch, time, output_dim) — logits (no sigmoid)
            bio_preds: dict of category → (batch, time, bio_dim)
            latent_traj: (batch, time, latent_dim) if return_latents=True
        """
        batch, T, _ = x.shape

        z = self.encoder(x[:, 0])

        z_trajectory = []
        for t in range(T):
            z = self.gru_ode_step(z, x[:, t])
            z_trajectory.append(z)

        z_traj = torch.stack(z_trajectory, dim=1)  # (batch, T, latent_dim)

        spike_pred = self.decoder(z_traj)

        bio_preds = {}
        for cat_name, head in self.bio_heads.items():
            bio_preds[cat_name] = head(z_traj)

        if return_latents:
            return spike_pred, bio_preds, z_traj
        return spike_pred, bio_preds
