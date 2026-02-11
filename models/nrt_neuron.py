"""
Reticular nucleus (nRt/PGN) inhibitory interneuron model.

Based on Destexhe, Bal, McCormick & Sejnowski 1996 (J Neurophysiol 76:2049-2070).

Single-compartment Hodgkin-Huxley formalism with:
  I_Na  - fast sodium
  I_K   - delayed rectifier potassium
  I_Ts  - low-threshold Ca2+ (T-type, slower kinetics than TC I_T)
  I_L   - ohmic leak

Key behaviour: AMPA input from TC triggers burst of spikes in nRt,
whose output drives GABA_A/B inhibition back onto TC.
"""

import numpy as np


class NRtNeuron:
    """Conductance-based single-compartment nRt/PGN neuron."""

    def __init__(self, params=None):
        p = params or {}

        self.C_m = p.get('C_m', 1.0)

        # Maximal conductances (mS/cm2)
        self.g_Na = p.get('g_Na', 100.0)
        self.g_K = p.get('g_K', 10.0)
        self.g_Ts = p.get('g_Ts', 3.0)
        self.g_L = p.get('g_L', 0.05)

        # Reversal potentials (mV)
        self.E_Na = p.get('E_Na', 50.0)
        self.E_K = p.get('E_K', -100.0)
        self.E_Ca = p.get('E_Ca', 120.0)
        self.E_L = p.get('E_L', -77.0)

        # Membrane area (cm2)
        self.area_cm2 = p.get('area_cm2', 1.43e-4)

        # Temperature correction
        self.temperature = p.get('temperature', 35.0)
        self.q10_Na = 2.5
        self.q10_K = 2.5
        self.q10_Ts = 3.0
        self.phi_Na = self.q10_Na ** ((self.temperature - 23.0) / 10.0)
        self.phi_K = self.q10_K ** ((self.temperature - 23.0) / 10.0)
        self.phi_Ts = self.q10_Ts ** ((self.temperature - 24.0) / 10.0)

        # NA modulation factor for excitability
        self.na_excitability = p.get('na_excitability', 1.0)

        # Initialise state
        V_init = p.get('V_rest', -72.0)
        self.V = V_init
        self.m = self._m_inf(V_init)
        self.h = self._h_inf(V_init)
        self.n = self._n_inf(V_init)
        self.m_Ts = self._mTs_inf(V_init)
        self.h_Ts = self._hTs_inf(V_init)

        self.V_prev = V_init
        self.spiked = False
        self.spike_threshold = -20.0

    # ------------------------------------------------------------------
    # I_Na gating (same formalism as TC)
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
    # I_K gating
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
    # I_Ts gating (nRt T-current, slower than TC I_T; Destexhe 1996)
    # ------------------------------------------------------------------
    def _mTs_inf(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 52.0) / 7.4))

    def _tau_mTs(self, V):
        tau = (1.0 + 0.33 / (np.exp((V + 27.0) / 10.0)
                               + np.exp(-(V + 102.0) / 15.0)))
        return tau / self.phi_Ts

    def _hTs_inf(self, V):
        return 1.0 / (1.0 + np.exp((V + 80.0) / 5.0))

    def _tau_hTs(self, V):
        tau = (28.3 + 0.33 / (np.exp((V + 48.0) / 4.0)
                                + np.exp(-(V + 407.0) / 50.0)))
        return tau / self.phi_Ts

    # ------------------------------------------------------------------
    # Derivatives and integration
    # ------------------------------------------------------------------
    def derivatives(self, V, m, h, n, m_Ts, h_Ts, I_ext_density):
        i_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        i_K = self.g_K * n**4 * (V - self.E_K)
        g_Ts_eff = self.g_Ts * self.na_excitability
        i_Ts = g_Ts_eff * m_Ts**2 * h_Ts * (V - self.E_Ca)
        i_L = self.g_L * (V - self.E_L)

        dVdt = (-i_Na - i_K - i_Ts - i_L + I_ext_density) / self.C_m

        am = self._alpha_m(V)
        bm = self._beta_m(V)
        dm = am * (1 - m) - bm * m

        ah = self._alpha_h(V)
        bh = self._beta_h(V)
        dh = ah * (1 - h) - bh * h

        an = self._alpha_n(V)
        bn = self._beta_n(V)
        dn = an * (1 - n) - bn * n

        mTs_inf = self._mTs_inf(V)
        tau_mTs = self._tau_mTs(V)
        dmTs = (mTs_inf - m_Ts) / tau_mTs

        hTs_inf = self._hTs_inf(V)
        tau_hTs = self._tau_hTs(V)
        dhTs = (hTs_inf - h_Ts) / tau_hTs

        return dVdt, dm, dh, dn, dmTs, dhTs

    def step(self, dt, I_ext_nA=0.0):
        """Advance one timestep (RK4). I_ext_nA in nA, positive=depolarising."""
        dt_ms = dt * 1000.0
        I_ext_density = (I_ext_nA * 1e-6) / self.area_cm2

        state = (self.V, self.m, self.h, self.n, self.m_Ts, self.h_Ts)

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
        self.V, self.m, self.h, self.n, self.m_Ts, self.h_Ts = new_state

        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)
        self.m_Ts = np.clip(self.m_Ts, 0.0, 1.0)
        self.h_Ts = np.clip(self.h_Ts, 0.0, 1.0)

        self.spiked = (self.V_prev < self.spike_threshold
                       and self.V >= self.spike_threshold)

    def apply_noradrenaline(self, excitability_factor):
        """Reduce nRt excitability under NA (decrease T-current gain)."""
        self.na_excitability = excitability_factor
