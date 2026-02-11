"""
Thalamocortical (TC) relay neuron model.

Based on Destexhe, Bal, McCormick & Sejnowski 1996 (J Neurophysiol 76:2049-2070)
and McCormick & Huguenard 1992 (J Neurophysiol 68:1384-1400).

Single-compartment Hodgkin-Huxley formalism with:
  I_Na  - fast sodium (spike generation)
  I_K   - delayed rectifier potassium
  I_T   - low-threshold transient Ca2+ (T-type) -- critical for rebound bursts
  I_h   - hyperpolarization-activated cation (HCN) -- terminates spindle epochs
  I_L   - ohmic leak
  I_KL  - potassium leak (sets resting potential)

Target properties (Le Masson 2002):
  AP threshold (rebound burst): -44.4 +/- 0.6 mV (n=11)
  Input resistance: 68.1 +/- 3.1 MOhm (n=19)
"""

import numpy as np


class TCNeuron:
    """Conductance-based single-compartment TC relay neuron."""

    def __init__(self, params=None):
        p = params or {}

        # Membrane capacitance (uF/cm2)
        self.C_m = p.get('C_m', 1.0)

        # Maximal conductances (mS/cm2)
        self.g_Na = p.get('g_Na', 90.0)
        self.g_K = p.get('g_K', 10.0)
        self.g_T = p.get('g_T', 2.2)
        self.g_h = p.get('g_h', 0.02)
        self.g_L = p.get('g_L', 0.02)
        self.g_KL = p.get('g_KL', 0.03)

        # Reversal potentials (mV)
        self.E_Na = p.get('E_Na', 50.0)
        self.E_K = p.get('E_K', -100.0)
        self.E_Ca = p.get('E_Ca', 120.0)
        self.E_h = p.get('E_h', -40.0)
        self.E_L = p.get('E_L', -70.0)
        self.E_KL = p.get('E_KL', -100.0)

        # Membrane area (cm2) -- for converting nS synaptic to mS/cm2
        self.area_cm2 = p.get('area_cm2', 2.9e-4)

        # Temperature correction
        self.temperature = p.get('temperature', 35.0)  # Celsius
        self.q10_Na = 2.5
        self.q10_K = 2.5
        self.q10_T = 3.0
        self.q10_h = 3.0
        self.phi_Na = self.q10_Na ** ((self.temperature - 23.0) / 10.0)
        self.phi_K = self.q10_K ** ((self.temperature - 23.0) / 10.0)
        self.phi_T = self.q10_T ** ((self.temperature - 24.0) / 10.0)
        self.phi_h = self.q10_h ** ((self.temperature - 24.0) / 10.0)

        # Noradrenaline modulation factor (1.0 = no modulation)
        self.na_factor = p.get('na_factor', 1.0)

        # State variables
        V_init = p.get('V_rest', -64.0)
        self.V = V_init
        self.m = self._m_inf(V_init)
        self.h = self._h_inf(V_init)
        self.n = self._n_inf(V_init)
        self.m_T = self._mT_inf(V_init)
        self.h_T = self._hT_inf(V_init)
        self.m_h = self._mh_inf(V_init)

        # Spike detection state
        self.V_prev = V_init
        self.spiked = False
        self.spike_threshold = -20.0  # mV, for detection

    # ------------------------------------------------------------------
    # I_Na gating (Traub & Miles 1991 / McCormick & Huguenard 1992)
    # ------------------------------------------------------------------
    def _alpha_m(self, V):
        v = V + 37.0
        if abs(v) < 1e-6:
            return 0.1 * self.phi_Na
        return 0.1 * v / (1.0 - np.exp(-v / 10.0)) * self.phi_Na

    def _beta_m(self, V):
        return 4.0 * np.exp(-(V + 62.0) / 18.0) * self.phi_Na

    def _m_inf(self, V):
        a = self._alpha_m(V)
        return a / (a + self._beta_m(V))

    def _alpha_h(self, V):
        return 0.07 * np.exp(-(V + 59.0) / 20.0) * self.phi_Na

    def _beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 29.0) / 10.0)) * self.phi_Na

    def _h_inf(self, V):
        a = self._alpha_h(V)
        return a / (a + self._beta_h(V))

    # ------------------------------------------------------------------
    # I_K gating (delayed rectifier)
    # ------------------------------------------------------------------
    def _alpha_n(self, V):
        v = V + 34.0
        if abs(v) < 1e-6:
            return 0.01 * self.phi_K
        return 0.01 * v / (1.0 - np.exp(-v / 10.0)) * self.phi_K

    def _beta_n(self, V):
        return 0.125 * np.exp(-(V + 44.0) / 80.0) * self.phi_K

    def _n_inf(self, V):
        a = self._alpha_n(V)
        return a / (a + self._beta_n(V))

    # ------------------------------------------------------------------
    # I_T gating (low-threshold Ca2+, Destexhe et al. 1996)
    # ------------------------------------------------------------------
    def _mT_inf(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 59.0) / 6.2))

    def _tau_mT(self, V):
        tau = (0.612 + 1.0 / (np.exp(-(V + 132.0) / 16.7)
                                + np.exp((V + 16.8) / 18.2)))
        return tau / self.phi_T

    def _hT_inf(self, V):
        return 1.0 / (1.0 + np.exp((V + 83.0) / 4.0))

    def _tau_hT(self, V):
        if V < -81.0:
            tau = np.exp((V + 467.0) / 66.6)
        else:
            tau = (28.0 + np.exp(-(V + 22.0) / 10.5))
        return tau / self.phi_T

    # ------------------------------------------------------------------
    # I_h gating (HCN, Destexhe et al. 1996)
    # ------------------------------------------------------------------
    def _mh_inf(self, V):
        return 1.0 / (1.0 + np.exp((V + 75.0) / 5.5))

    def _tau_mh(self, V):
        tau = 1.0 / (np.exp(-14.59 - 0.086 * V)
                      + np.exp(-1.87 + 0.0701 * V))
        return tau / self.phi_h

    # ------------------------------------------------------------------
    # Current computations
    # ------------------------------------------------------------------
    def I_Na(self):
        return self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)

    def I_K(self):
        return self.g_K * self.n**4 * (self.V - self.E_K)

    def I_T(self):
        return self.g_T * self.m_T**2 * self.h_T * (self.V - self.E_Ca)

    def I_h(self):
        return self.g_h * self.m_h * (self.V - self.E_h)

    def I_L(self):
        return self.g_L * (self.V - self.E_L)

    def I_KL(self):
        # Noradrenaline reduces KL leak -> depolarises cell
        g_kl_eff = self.g_KL * self.na_factor
        return g_kl_eff * (self.V - self.E_KL)

    # ------------------------------------------------------------------
    # Integration step (forward Euler for gating, can be called by RK4)
    # ------------------------------------------------------------------
    def derivatives(self, V, m, h, n, m_T, h_T, m_h, I_ext_density):
        """Compute dV/dt and gating derivatives.

        Parameters
        ----------
        I_ext_density : float
            External (synaptic) current in mA/cm2.
            Positive = depolarising (convention: current *into* cell).
            Synaptic currents should be passed as negative of the standard
            I = g*(V-E) so that excitatory input depolarises.
        """
        # Ionic currents (positive = outward by convention)
        i_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        i_K = self.g_K * n**4 * (V - self.E_K)
        i_T = self.g_T * m_T**2 * h_T * (V - self.E_Ca)
        i_h = self.g_h * m_h * (V - self.E_h)
        i_L = self.g_L * (V - self.E_L)
        g_kl_eff = self.g_KL * self.na_factor
        i_KL = g_kl_eff * (V - self.E_KL)

        dVdt = (-i_Na - i_K - i_T - i_h - i_L - i_KL + I_ext_density) / self.C_m

        # Gating variable derivatives
        am = self._alpha_m(V)
        bm = self._beta_m(V)
        dm = am * (1 - m) - bm * m

        ah = self._alpha_h(V)
        bh = self._beta_h(V)
        dh = ah * (1 - h) - bh * h

        an = self._alpha_n(V)
        bn = self._beta_n(V)
        dn = an * (1 - n) - bn * n

        mT_inf = self._mT_inf(V)
        tau_mT = self._tau_mT(V)
        dmT = (mT_inf - m_T) / tau_mT

        hT_inf = self._hT_inf(V)
        tau_hT = self._tau_hT(V)
        dhT = (hT_inf - h_T) / tau_hT

        mh_inf = self._mh_inf(V)
        tau_mh = self._tau_mh(V)
        dmh = (mh_inf - m_h) / tau_mh

        return dVdt, dm, dh, dn, dmT, dhT, dmh

    def step(self, dt, I_ext_nA=0.0):
        """Advance neuron by one timestep using RK4.

        Parameters
        ----------
        dt : float
            Timestep in seconds.
        I_ext_nA : float
            External current in nA (positive = depolarising).
            This is the total synaptic current in nA convention.
        """
        dt_ms = dt * 1000.0  # convert to ms for kinetics

        # Convert nA to mA/cm2
        # I (mA/cm2) = I (nA) * 1e-6 / area (cm2)
        I_ext_density = (I_ext_nA * 1e-6) / self.area_cm2

        # Current state
        state = (self.V, self.m, self.h, self.n, self.m_T, self.h_T, self.m_h)

        # RK4
        k1 = self.derivatives(*state, I_ext_density)
        s2 = tuple(s + 0.5 * dt_ms * k for s, k in zip(state, k1))
        k2 = self.derivatives(*s2, I_ext_density)
        s3 = tuple(s + 0.5 * dt_ms * k for s, k in zip(state, k2))
        k3 = self.derivatives(*s3, I_ext_density)
        s4 = tuple(s + dt_ms * k for s, k in zip(state, k3))
        k4 = self.derivatives(*s4, I_ext_density)

        new_state = tuple(
            s + (dt_ms / 6.0) * (a + 2*b + 2*c + d)
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)
        )

        self.V_prev = self.V
        self.V, self.m, self.h, self.n, self.m_T, self.h_T, self.m_h = new_state

        # Clamp gating variables to [0, 1]
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)
        self.m_T = np.clip(self.m_T, 0.0, 1.0)
        self.h_T = np.clip(self.h_T, 0.0, 1.0)
        self.m_h = np.clip(self.m_h, 0.0, 1.0)

        # Spike detection (positive-going threshold crossing)
        self.spiked = (self.V_prev < self.spike_threshold
                       and self.V >= self.spike_threshold)

    def apply_noradrenaline(self, factor):
        """Simulate noradrenaline modulation by reducing KL conductance.

        factor: 0.0 = full NA (no KL), 1.0 = control (full KL)
        """
        self.na_factor = factor

    def input_resistance_MOhm(self):
        """Estimate input resistance at rest (for validation)."""
        # R_in = 1 / (g_total * area)  in MOhm
        g_total = self.g_L + self.g_KL * self.na_factor  # at rest, Na/K gates ~0
        # g_total is in mS/cm2, area in cm2
        # G_total = g_total * area  in mS
        G_total = g_total * self.area_cm2  # mS
        # R = 1/G in kOhm -> * 1000 for MOhm? No: 1 mS -> 1 kOhm = 0.001 MOhm?
        # Actually: 1/mS = 1 kOhm. We want MOhm.
        # R (MOhm) = 1 / (G_total_mS) * 1e3  ... no.
        # G in mS.  1/G in kOhm.  kOhm / 1000 = MOhm? No, kOhm * 1 = kOhm.
        # 1 MOhm = 1000 kOhm. So R(MOhm) = 1/(G_mS) / 1000? No.
        # Let's be careful: G in siemens = g (mS/cm2) * area(cm2) * 1e-3
        G_S = g_total * self.area_cm2 * 1e-3  # Siemens
        R_Ohm = 1.0 / G_S
        return R_Ohm * 1e-6  # MOhm
