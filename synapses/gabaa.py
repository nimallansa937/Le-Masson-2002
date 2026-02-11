"""
GABA_A synapse model using Destexhe et al. 1994 kinetic formalism.

Same kinetic scheme as AMPA with different rate constants:
  alpha = 5.0 ms^-1 mM^-1 (faster binding)
  beta  = 0.18 ms^-1

I_GABAA = G_max * r * (V_post - E_Cl)
E_Cl = -90 mV (from Le Masson 2002)

Control: G_max = 35 nS (96% of total GABA conductance)
"""

import numpy as np


class GABAASynapse:
    """First-order kinetic GABA_A synapse (Destexhe et al. 1994)."""

    def __init__(self, g_max_nS=35.0, E_rev=-90.0, alpha=5.0, beta=0.18,
                 T_max=1.0, T_duration_ms=1.0):
        self.g_max = g_max_nS
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.T_max = T_max
        self.T_duration = T_duration_ms

        self.r = 0.0
        self.T = 0.0
        self.T_timer = 0.0

    def activate(self):
        """Trigger transmitter release (presynaptic spike)."""
        self.T = self.T_max
        self.T_timer = self.T_duration

    def step(self, dt_ms):
        """Update synapse state by dt (ms)."""
        if self.T_timer > 0:
            self.T_timer -= dt_ms
            if self.T_timer <= 0:
                self.T = 0.0
                self.T_timer = 0.0

        dr = self.alpha * self.T * (1.0 - self.r) - self.beta * self.r
        self.r += dr * dt_ms
        self.r = np.clip(self.r, 0.0, 1.0)

    def current(self, V_post):
        """Synaptic current in nA (positive = depolarising convention).

        GABA_A is inhibitory: V_post > E_rev -> current is hyperpolarising
        -> returned as negative (inhibitory).
        """
        return -self.g_max * self.r * (V_post - self.E_rev)

    def conductance(self):
        return self.g_max * self.r
