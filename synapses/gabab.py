"""
GABA_B synapse model using Destexhe et al. 1994 kinetic formalism.

GABA_B is metabotropic -- slower, G-protein mediated.

Two-step kinetic scheme:
  dr/dt = K1 * [T] * (1 - r) - K2 * r     (receptor binding)
  ds/dt = K3 * r - K4 * s                   (G-protein activation)
  G_GABAB(t) = G_max * s^n / (s^n + Kd)    (cooperative activation)

I_GABAB = G_GABAB(t) * (V_post - E_K)
E_K = -110 mV (from Le Masson 2002)

Parameters from Destexhe et al. 1994:
  K1 = 0.09 ms^-1 mM^-1
  K2 = 0.0012 ms^-1
  K3 = 0.18 ms^-1
  K4 = 0.034 ms^-1
  n = 4
  Kd = 100

Control: G_max = 1.4 nS (4% of total GABA conductance)
"""

import numpy as np


class GABABSynapse:
    """Two-step kinetic GABA_B synapse with G-protein cascade."""

    def __init__(self, g_max_nS=1.4, E_rev=-110.0,
                 K1=0.09, K2=0.0012, K3=0.18, K4=0.034,
                 n=4, Kd=100.0,
                 T_max=1.0, T_duration_ms=1.0):
        self.g_max = g_max_nS
        self.E_rev = E_rev
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.n = n
        self.Kd = Kd
        self.T_max = T_max
        self.T_duration = T_duration_ms

        # State variables
        self.r = 0.0   # receptor binding fraction
        self.s = 0.0   # G-protein activation
        self.T = 0.0   # transmitter concentration
        self.T_timer = 0.0

    def activate(self):
        """Trigger transmitter release."""
        self.T = self.T_max
        self.T_timer = self.T_duration

    def step(self, dt_ms):
        """Update synapse state by dt (ms)."""
        if self.T_timer > 0:
            self.T_timer -= dt_ms
            if self.T_timer <= 0:
                self.T = 0.0
                self.T_timer = 0.0

        # Receptor binding
        dr = self.K1 * self.T * (1.0 - self.r) - self.K2 * self.r
        self.r += dr * dt_ms
        self.r = np.clip(self.r, 0.0, 1.0)

        # G-protein activation
        ds = self.K3 * self.r - self.K4 * self.s
        self.s += ds * dt_ms
        self.s = max(self.s, 0.0)

    def _g_eff(self):
        """Effective conductance with cooperative activation."""
        s_n = self.s ** self.n
        return self.g_max * s_n / (s_n + self.Kd)

    def current(self, V_post):
        """Synaptic current in nA (positive = depolarising convention)."""
        g = self._g_eff()
        return -g * (V_post - self.E_rev)

    def conductance(self):
        return self._g_eff()
