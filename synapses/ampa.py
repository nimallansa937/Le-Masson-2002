"""
AMPA synapse model using Destexhe et al. 1994 kinetic formalism.

Kinetic scheme:
  dr/dt = alpha * [T] * (1 - r) - beta * r
  I_AMPA = G_max * r * (V_post - E_rev)

Two AMPA connections in circuit:
  Retinal -> TC: G_max = 28 nS
  TC -> nRt:     G_max = 20 nS
"""

import numpy as np


class AMPASynapse:
    """First-order kinetic AMPA synapse (Destexhe et al. 1994)."""

    def __init__(self, g_max_nS=28.0, E_rev=0.0, alpha=1.1, beta=0.19,
                 T_max=1.0, T_duration_ms=1.0):
        """
        Parameters
        ----------
        g_max_nS : float
            Maximal conductance in nanosiemens.
        E_rev : float
            Reversal potential in mV.
        alpha : float
            Forward rate constant (ms^-1 * mM^-1).
        beta : float
            Backward rate constant (ms^-1).
        T_max : float
            Peak transmitter concentration (mM).
        T_duration_ms : float
            Duration of transmitter pulse (ms).
        """
        self.g_max = g_max_nS  # nS
        self.E_rev = E_rev     # mV
        self.alpha = alpha     # ms^-1 mM^-1
        self.beta = beta       # ms^-1
        self.T_max = T_max     # mM
        self.T_duration = T_duration_ms  # ms

        # State
        self.r = 0.0          # fraction of open channels
        self.T = 0.0          # current transmitter concentration
        self.T_timer = 0.0    # remaining duration of T pulse (ms)

    def activate(self):
        """Trigger transmitter release (presynaptic spike)."""
        self.T = self.T_max
        self.T_timer = self.T_duration

    def step(self, dt_ms):
        """Update synapse state by dt (in ms)."""
        # Update transmitter pulse
        if self.T_timer > 0:
            self.T_timer -= dt_ms
            if self.T_timer <= 0:
                self.T = 0.0
                self.T_timer = 0.0

        # Kinetic equation
        dr = self.alpha * self.T * (1.0 - self.r) - self.beta * self.r
        self.r += dr * dt_ms
        self.r = np.clip(self.r, 0.0, 1.0)

    def current(self, V_post):
        """Compute synaptic current in nA.

        Returns current as *negative* value for depolarising input
        (i.e., the sign convention where positive I_ext depolarises).
        I = -g * r * (V - E)  so that when V < E_rev (0mV), current is negative
        -> passed as positive I_ext_nA to neuron.
        """
        # g * r * (V - E) is positive when V < E (outward by ionic convention)
        # We want the *injected* current convention: positive = depolarising
        # I_syn_injected = -g * r * (V_post - E_rev)
        return -self.g_max * self.r * (V_post - self.E_rev)

    def conductance(self):
        """Current conductance in nS."""
        return self.g_max * self.r
