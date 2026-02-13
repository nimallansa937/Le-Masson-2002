"""
Vectorized population-level thalamic circuit simulator.

All N TC and N nRt neurons updated simultaneously via NumPy array operations.
Synaptic state stored as arrays, not individual objects.
~20-50x faster than the per-object version.

Architecture:
  Retinal_i (independent) --AMPA--> TC_i --AMPA--> nRt_j (divergent)
                                      ^                |
                                      +---GABA_A/B-----+
                                    nRt_j --GABA_A--> nRt_k (intra-reticular)
"""

import sys
import os
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from models.retinal_input import generate_retinal_spikes, spike_times_to_binary
from population.network import build_connectivity, normalize_weights, assign_delays
from population.heterogeneity import (
    sample_tc_parameters, sample_nrt_parameters,
    create_replacement_tc_params, VALIDATED_TC_PARAMS, BASE_NRT_PARAMS,
)
from population.replacement import select_replacement_indices


# -----------------------------------------------------------------------
# Vectorized TC neuron kinetics (all N neurons at once)
# -----------------------------------------------------------------------

class VectorizedTC:
    """Vectorized TC neuron population — all state in NumPy arrays."""

    def __init__(self, params_list):
        """params_list: list of N dicts, one per neuron."""
        N = len(params_list)
        self.N = N

        # Extract per-neuron parameters into arrays
        self.C_m = np.array([p.get('C_m', 1.0) for p in params_list])
        self.g_Na = np.array([p.get('g_Na', 90.0) for p in params_list])
        self.g_K = np.array([p.get('g_K', 10.0) for p in params_list])
        self.g_T = np.array([p.get('g_T', 2.0) for p in params_list])
        self.g_h = np.array([p.get('g_h', 0.05) for p in params_list])
        self.g_L = np.array([p.get('g_L', 0.02) for p in params_list])
        self.g_KL = np.array([p.get('g_KL', 0.03) for p in params_list])
        self.E_Na = np.array([p.get('E_Na', 50.0) for p in params_list])
        self.E_K = np.array([p.get('E_K', -100.0) for p in params_list])
        self.E_Ca = np.array([p.get('E_Ca', 120.0) for p in params_list])
        self.E_h = np.array([p.get('E_h', -40.0) for p in params_list])
        self.E_L = np.array([p.get('E_L', -70.0) for p in params_list])
        self.E_KL = np.array([p.get('E_KL', -100.0) for p in params_list])
        self.area = np.array([p.get('area_cm2', 2.9e-4) for p in params_list])

        temp = np.array([p.get('temperature', 35.0) for p in params_list])
        self.phi_Na = 2.5 ** ((temp - 23.0) / 10.0)
        self.phi_K = 2.5 ** ((temp - 23.0) / 10.0)
        self.phi_T = 3.0 ** ((temp - 24.0) / 10.0)
        self.phi_h = 3.0 ** ((temp - 24.0) / 10.0)

        # State vectors
        V0 = np.array([p.get('V_rest', -64.0) for p in params_list])
        self.V = V0.copy()
        self.V_prev = V0.copy()
        self.m = self._m_inf(V0)
        self.h = self._h_inf(V0)
        self.n = self._n_inf(V0)
        self.m_T = self._mT_inf(V0)
        self.h_T = self._hT_inf(V0)
        self.m_h = self._mh_inf(V0)
        self.spiked = np.zeros(N, dtype=bool)
        self.spike_threshold = -20.0

    # --- Gating functions (vectorized) ---
    def _alpha_m(self, V):
        v = V + 37.0
        safe = np.where(np.abs(v) < 1e-6, 1e-6, v)
        return np.where(np.abs(v) < 1e-6,
                        0.1 * self.phi_Na,
                        0.1 * safe / (1.0 - np.exp(-safe / 10.0)) * self.phi_Na)

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

    def _alpha_n(self, V):
        v = V + 34.0
        safe = np.where(np.abs(v) < 1e-6, 1e-6, v)
        return np.where(np.abs(v) < 1e-6,
                        0.01 * self.phi_K,
                        0.01 * safe / (1.0 - np.exp(-safe / 10.0)) * self.phi_K)

    def _beta_n(self, V):
        return 0.125 * np.exp(-(V + 44.0) / 80.0) * self.phi_K

    def _n_inf(self, V):
        a = self._alpha_n(V)
        return a / (a + self._beta_n(V))

    def _mT_inf(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 59.0) / 6.2))

    def _tau_mT(self, V):
        tau = 0.612 + 1.0 / (np.exp(-(V + 132.0) / 16.7) +
                              np.exp((V + 16.8) / 18.2))
        return tau / self.phi_T

    def _hT_inf(self, V):
        return 1.0 / (1.0 + np.exp((V + 83.0) / 4.0))

    def _tau_hT(self, V):
        tau = np.where(V < -81.0,
                       np.exp((V + 467.0) / 66.6),
                       28.0 + np.exp(-(V + 22.0) / 10.5))
        return tau / self.phi_T

    def _mh_inf(self, V):
        return 1.0 / (1.0 + np.exp((V + 75.0) / 5.5))

    def _tau_mh(self, V):
        tau = 1.0 / (np.exp(-14.59 - 0.086 * V) +
                      np.exp(-1.87 + 0.0701 * V))
        return tau / self.phi_h

    def _derivatives(self, V, m, h, n, m_T, h_T, m_h, I_ext_density):
        """Compute all derivatives for all neurons simultaneously."""
        i_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        i_K = self.g_K * n**4 * (V - self.E_K)
        i_T = self.g_T * m_T**2 * h_T * (V - self.E_Ca)
        i_h = self.g_h * m_h * (V - self.E_h)
        i_L = self.g_L * (V - self.E_L)
        i_KL = self.g_KL * (V - self.E_KL)

        dV = (-i_Na - i_K - i_T - i_h - i_L - i_KL + I_ext_density) / self.C_m

        am = self._alpha_m(V); bm = self._beta_m(V)
        dm = am * (1 - m) - bm * m

        ah = self._alpha_h(V); bh = self._beta_h(V)
        dh = ah * (1 - h) - bh * h

        an = self._alpha_n(V); bn = self._beta_n(V)
        dn = an * (1 - n) - bn * n

        dmT = (self._mT_inf(V) - m_T) / self._tau_mT(V)
        dhT = (self._hT_inf(V) - h_T) / self._tau_hT(V)
        dmh = (self._mh_inf(V) - m_h) / self._tau_mh(V)

        return dV, dm, dh, dn, dmT, dhT, dmh

    def step(self, dt_s, I_ext_nA):
        """RK4 step for all N neurons. I_ext_nA: array of shape (N,)."""
        dt_ms = dt_s * 1000.0
        I_ext_density = (I_ext_nA * 1e-6) / self.area

        state = (self.V, self.m, self.h, self.n, self.m_T, self.h_T, self.m_h)

        k1 = self._derivatives(*state, I_ext_density)
        s2 = tuple(s + 0.5 * dt_ms * k for s, k in zip(state, k1))
        k2 = self._derivatives(*s2, I_ext_density)
        s3 = tuple(s + 0.5 * dt_ms * k for s, k in zip(state, k2))
        k3 = self._derivatives(*s3, I_ext_density)
        s4 = tuple(s + dt_ms * k for s, k in zip(state, k3))
        k4 = self._derivatives(*s4, I_ext_density)

        new = tuple(
            s + (dt_ms / 6.0) * (a + 2*b + 2*c + d)
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)
        )

        self.V_prev = self.V.copy()
        self.V, self.m, self.h, self.n, self.m_T, self.h_T, self.m_h = new

        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)
        self.m_T = np.clip(self.m_T, 0.0, 1.0)
        self.h_T = np.clip(self.h_T, 0.0, 1.0)
        self.m_h = np.clip(self.m_h, 0.0, 1.0)

        self.spiked = (self.V_prev < self.spike_threshold) & \
                      (self.V >= self.spike_threshold)


# -----------------------------------------------------------------------
# Vectorized nRt neuron kinetics
# -----------------------------------------------------------------------

class VectorizedNRt:
    """Vectorized nRt neuron population."""

    def __init__(self, params_list):
        N = len(params_list)
        self.N = N

        self.C_m = np.array([p.get('C_m', 1.0) for p in params_list])
        self.g_Na = np.array([p.get('g_Na', 100.0) for p in params_list])
        self.g_K = np.array([p.get('g_K', 10.0) for p in params_list])
        self.g_Ts = np.array([p.get('g_Ts', 3.0) for p in params_list])
        self.g_L = np.array([p.get('g_L', 0.05) for p in params_list])
        self.E_Na = np.array([p.get('E_Na', 50.0) for p in params_list])
        self.E_K = np.array([p.get('E_K', -100.0) for p in params_list])
        self.E_Ca = np.array([p.get('E_Ca', 120.0) for p in params_list])
        self.E_L = np.array([p.get('E_L', -77.0) for p in params_list])
        self.area = np.array([p.get('area_cm2', 1.43e-4) for p in params_list])

        temp = np.array([p.get('temperature', 35.0) for p in params_list])
        self.phi_Na = 2.5 ** ((temp - 23.0) / 10.0)
        self.phi_K = 2.5 ** ((temp - 23.0) / 10.0)
        self.phi_Ts = 3.0 ** ((temp - 24.0) / 10.0)

        V0 = np.array([p.get('V_rest', -72.0) for p in params_list])
        self.V = V0.copy()
        self.V_prev = V0.copy()
        self.m = self._m_inf(V0)
        self.h = self._h_inf(V0)
        self.n = self._n_inf(V0)
        self.m_Ts = self._mTs_inf(V0)
        self.h_Ts = self._hTs_inf(V0)
        self.spiked = np.zeros(N, dtype=bool)
        self.spike_threshold = -20.0

    def _alpha_m(self, V):
        v = V + 37.0
        safe = np.where(np.abs(v) < 1e-6, 1e-6, v)
        return np.where(np.abs(v) < 1e-6,
                        0.1 * self.phi_Na,
                        0.1 * safe / (1.0 - np.exp(-safe / 10.0)) * self.phi_Na)

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

    def _alpha_n(self, V):
        v = V + 34.0
        safe = np.where(np.abs(v) < 1e-6, 1e-6, v)
        return np.where(np.abs(v) < 1e-6,
                        0.01 * self.phi_K,
                        0.01 * safe / (1.0 - np.exp(-safe / 10.0)) * self.phi_K)

    def _beta_n(self, V):
        return 0.125 * np.exp(-(V + 44.0) / 80.0) * self.phi_K

    def _n_inf(self, V):
        a = self._alpha_n(V)
        return a / (a + self._beta_n(V))

    def _mTs_inf(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 52.0) / 7.4))

    def _tau_mTs(self, V):
        tau = 1.0 + 0.33 / (np.exp((V + 27.0) / 10.0) +
                             np.exp(-(V + 102.0) / 15.0))
        return tau / self.phi_Ts

    def _hTs_inf(self, V):
        return 1.0 / (1.0 + np.exp((V + 80.0) / 5.0))

    def _tau_hTs(self, V):
        tau = 28.3 + 0.33 / (np.exp((V + 48.0) / 4.0) +
                              np.exp(-(V + 407.0) / 50.0))
        return tau / self.phi_Ts

    def _derivatives(self, V, m, h, n, m_Ts, h_Ts, I_ext_density):
        i_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        i_K = self.g_K * n**4 * (V - self.E_K)
        i_Ts = self.g_Ts * m_Ts**2 * h_Ts * (V - self.E_Ca)
        i_L = self.g_L * (V - self.E_L)

        dV = (-i_Na - i_K - i_Ts - i_L + I_ext_density) / self.C_m

        am = self._alpha_m(V); bm = self._beta_m(V)
        dm = am * (1 - m) - bm * m
        ah = self._alpha_h(V); bh = self._beta_h(V)
        dh = ah * (1 - h) - bh * h
        an = self._alpha_n(V); bn = self._beta_n(V)
        dn = an * (1 - n) - bn * n
        dmTs = (self._mTs_inf(V) - m_Ts) / self._tau_mTs(V)
        dhTs = (self._hTs_inf(V) - h_Ts) / self._tau_hTs(V)

        return dV, dm, dh, dn, dmTs, dhTs

    def step(self, dt_s, I_ext_nA):
        dt_ms = dt_s * 1000.0
        I_ext_density = (I_ext_nA * 1e-6) / self.area

        state = (self.V, self.m, self.h, self.n, self.m_Ts, self.h_Ts)

        k1 = self._derivatives(*state, I_ext_density)
        s2 = tuple(s + 0.5 * dt_ms * k for s, k in zip(state, k1))
        k2 = self._derivatives(*s2, I_ext_density)
        s3 = tuple(s + 0.5 * dt_ms * k for s, k in zip(state, k2))
        k3 = self._derivatives(*s3, I_ext_density)
        s4 = tuple(s + dt_ms * k for s, k in zip(state, k3))
        k4 = self._derivatives(*s4, I_ext_density)

        new = tuple(
            s + (dt_ms / 6.0) * (a + 2*b + 2*c + d)
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)
        )

        self.V_prev = self.V.copy()
        self.V, self.m, self.h, self.n, self.m_Ts, self.h_Ts = new

        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)
        self.m_Ts = np.clip(self.m_Ts, 0.0, 1.0)
        self.h_Ts = np.clip(self.h_Ts, 0.0, 1.0)

        self.spiked = (self.V_prev < self.spike_threshold) & \
                      (self.V >= self.spike_threshold)


# -----------------------------------------------------------------------
# Vectorized synapse arrays
# -----------------------------------------------------------------------

class SynapseArray:
    """Vectorized first-order kinetic synapses (AMPA or GABA_A).

    Stores state for M synapses as flat arrays.
    """

    def __init__(self, g_max, E_rev, alpha, beta, T_max=1.0, T_dur_ms=1.0):
        """
        g_max : ndarray (M,) — per-synapse max conductance (nS)
        E_rev : float — reversal potential
        """
        self.g_max = np.asarray(g_max, dtype=np.float64)
        self.M = len(self.g_max)
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.T_max = T_max
        self.T_dur = T_dur_ms

        self.r = np.zeros(self.M)
        self.T = np.zeros(self.M)
        self.T_timer = np.zeros(self.M)

    def activate(self, mask):
        """Activate synapses where mask is True."""
        self.T[mask] = self.T_max
        self.T_timer[mask] = self.T_dur

    def step(self, dt_ms):
        # Decrement T timers
        active = self.T_timer > 0
        self.T_timer[active] -= dt_ms
        expired = active & (self.T_timer <= 0)
        self.T[expired] = 0.0
        self.T_timer[expired] = 0.0

        # Kinetics
        dr = self.alpha * self.T * (1.0 - self.r) - self.beta * self.r
        self.r += dr * dt_ms
        np.clip(self.r, 0.0, 1.0, out=self.r)

    def current_matrix(self, V_post):
        """Compute current for each synapse given postsynaptic V.

        V_post : ndarray matching synapse layout
        Returns: ndarray (M,) of currents in nA
        """
        return -self.g_max * self.r * (V_post - self.E_rev)

    def conductance(self):
        return self.g_max * self.r


class GABABArray:
    """Vectorized GABA_B two-step kinetic synapses."""

    def __init__(self, g_max, E_rev=-110.0, K1=0.09, K2=0.0012,
                 K3=0.18, K4=0.034, n=4, Kd=100.0,
                 T_max=1.0, T_dur_ms=1.0):
        self.g_max = np.asarray(g_max, dtype=np.float64)
        self.M = len(self.g_max)
        self.E_rev = E_rev
        self.K1 = K1; self.K2 = K2; self.K3 = K3; self.K4 = K4
        self.n = n; self.Kd = Kd
        self.T_max = T_max; self.T_dur = T_dur_ms

        self.r = np.zeros(self.M)
        self.s = np.zeros(self.M)
        self.T = np.zeros(self.M)
        self.T_timer = np.zeros(self.M)

    def activate(self, mask):
        self.T[mask] = self.T_max
        self.T_timer[mask] = self.T_dur

    def step(self, dt_ms):
        active = self.T_timer > 0
        self.T_timer[active] -= dt_ms
        expired = active & (self.T_timer <= 0)
        self.T[expired] = 0.0
        self.T_timer[expired] = 0.0

        dr = self.K1 * self.T * (1.0 - self.r) - self.K2 * self.r
        self.r += dr * dt_ms
        np.clip(self.r, 0.0, 1.0, out=self.r)

        ds = self.K3 * self.r - self.K4 * self.s
        self.s += ds * dt_ms
        np.maximum(self.s, 0.0, out=self.s)

    def current_matrix(self, V_post):
        s_n = self.s ** self.n
        g_eff = self.g_max * s_n / (s_n + self.Kd)
        return -g_eff * (V_post - self.E_rev)


# -----------------------------------------------------------------------
# Fast PopulationCircuit
# -----------------------------------------------------------------------

class PopulationCircuit:
    """Vectorized population circuit — 20-50x faster than per-object version."""

    def __init__(self, n_tc=20, n_nrt=20, gaba_gmax_total=36.4,
                 retinal_rate=42.0, gamma_order=1.5,
                 replacement_fraction=0.0, replacement_strategy='random',
                 replacement_seed=42,
                 p_tc_nrt=0.20, p_nrt_tc=0.20, p_nrt_nrt=0.15,
                 gabaa_fraction=0.96, nrt_nrt_gmax=7.0,
                 ampa_ret_tc_gmax=28.0, ampa_tc_nrt_gmax=20.0,
                 dt=0.025e-3, network_seed=42, hetero_seed=42,
                 input_seed=42, homogeneous=False):

        self.n_tc = n_tc
        self.n_nrt = n_nrt
        self.dt = dt
        self.retinal_rate = retinal_rate
        self.gamma_order = gamma_order
        self.gabaa_fraction = gabaa_fraction

        # --- Connectivity ---
        tc_to_nrt, nrt_to_tc, nrt_to_nrt = build_connectivity(
            n_tc, n_nrt, p_tc_nrt, p_nrt_tc, p_nrt_nrt, seed=network_seed)
        self.tc_to_nrt = tc_to_nrt  # (n_nrt, n_tc)
        self.nrt_to_tc = nrt_to_tc  # (n_tc, n_nrt)
        self.nrt_to_nrt = nrt_to_nrt  # (n_nrt, n_nrt)

        # --- Weights ---
        ampa_tc_nrt_w = normalize_weights(tc_to_nrt, ampa_tc_nrt_gmax)
        gaba_nrt_tc_w = normalize_weights(nrt_to_tc, gaba_gmax_total)
        gaba_nrt_nrt_w = normalize_weights(nrt_to_nrt, nrt_nrt_gmax)

        # --- Delays (in timesteps) ---
        dt_ms = dt * 1000.0
        d1 = assign_delays(tc_to_nrt, seed=network_seed)
        d2 = assign_delays(nrt_to_tc, seed=network_seed + 100)
        d3 = assign_delays(nrt_to_nrt, seed=network_seed + 200)
        self.delay_tc_nrt = np.round(d1 / dt_ms).astype(int)
        self.delay_nrt_tc = np.round(d2 / dt_ms).astype(int)
        self.delay_nrt_nrt = np.round(d3 / dt_ms).astype(int)

        self.max_delay = max(
            self.delay_tc_nrt.max() if self.delay_tc_nrt.size else 0,
            self.delay_nrt_tc.max() if self.delay_nrt_tc.size else 0,
            self.delay_nrt_nrt.max() if self.delay_nrt_nrt.size else 0,
            1)

        # --- Replacement ---
        self.replacement_indices = select_replacement_indices(
            n_tc, replacement_fraction,
            strategy=replacement_strategy,
            nrt_to_tc=nrt_to_tc,
            seed=replacement_seed)
        self.is_replacement = np.zeros(n_tc, dtype=bool)
        if len(self.replacement_indices) > 0:
            self.is_replacement[self.replacement_indices] = True

        # --- Neurons ---
        hetero_rng = np.random.default_rng(hetero_seed)
        if homogeneous:
            bio_params = [VALIDATED_TC_PARAMS.copy() for _ in range(n_tc)]
        else:
            bio_params = sample_tc_parameters(n_tc, hetero_rng)
        nrt_params = sample_nrt_parameters(n_nrt, hetero_rng)

        tc_params_final = []
        for i in range(n_tc):
            if self.is_replacement[i]:
                tc_params_final.append(create_replacement_tc_params())
            else:
                tc_params_final.append(bio_params[i])

        self.tc = VectorizedTC(tc_params_final)
        self.nrt = VectorizedNRt(nrt_params)

        # --- Build flat synapse arrays ---
        # Retinal -> TC (one per TC, g_max = ampa_ret_tc_gmax)
        self.ampa_ret = SynapseArray(
            g_max=np.full(n_tc, ampa_ret_tc_gmax),
            E_rev=0.0, alpha=1.1, beta=0.19)

        # TC -> nRt AMPA: flatten the (n_nrt, n_tc) weight matrix
        # Store (post_j, pre_i) pairs for connected synapses
        tn_post, tn_pre = np.where(tc_to_nrt)  # post=nrt, pre=tc
        self.tn_post = tn_post
        self.tn_pre = tn_pre
        self.tn_delays = self.delay_tc_nrt[tn_post, tn_pre]
        tn_g = ampa_tc_nrt_w[tn_post, tn_pre]
        self.ampa_tc_nrt = SynapseArray(
            g_max=tn_g, E_rev=0.0, alpha=1.1, beta=0.19)

        # nRt -> TC GABA_A: flatten
        nt_post, nt_pre = np.where(nrt_to_tc)  # post=tc, pre=nrt
        self.nt_post = nt_post
        self.nt_pre = nt_pre
        self.nt_delays = self.delay_nrt_tc[nt_post, nt_pre]
        nt_g = gaba_nrt_tc_w[nt_post, nt_pre]
        nt_ga = nt_g * gabaa_fraction
        nt_gb = nt_g * (1.0 - gabaa_fraction)
        self.gabaa_nrt_tc = SynapseArray(
            g_max=nt_ga, E_rev=-90.0, alpha=5.0, beta=0.18)
        self.gabab_nrt_tc = GABABArray(g_max=nt_gb)

        # nRt -> nRt GABA_A: flatten
        nn_post, nn_pre = np.where(nrt_to_nrt)
        self.nn_post = nn_post
        self.nn_pre = nn_pre
        self.nn_delays = self.delay_nrt_nrt[nn_post, nn_pre]
        nn_g = gaba_nrt_nrt_w[nn_post, nn_pre]
        self.gabaa_nrt_nrt = SynapseArray(
            g_max=nn_g, E_rev=-90.0, alpha=5.0, beta=0.18)

        self.input_seed = input_seed

    def simulate(self, duration_s, record_dt=None):
        """Run vectorized simulation."""
        dt = self.dt
        n_steps = int(duration_s / dt)
        dt_ms = dt * 1000.0
        n_tc = self.n_tc
        n_nrt = self.n_nrt

        if record_dt is None:
            record_dt = 0.001
        record_every = max(1, int(record_dt / dt))
        n_record = n_steps // record_every + 1

        # Recording
        t_rec = np.zeros(n_record)
        V_tc_rec = np.zeros((n_tc, n_record))
        V_nrt_rec = np.zeros((n_nrt, n_record))
        rec_idx = 0

        # Retinal inputs
        input_rng = np.random.default_rng(self.input_seed)
        ret_spike_times = []
        # Stack all retinal binaries into (n_tc, n_steps) array
        ret_bin = np.zeros((n_tc, n_steps), dtype=bool)
        for i in range(n_tc):
            child_rng = np.random.default_rng(input_rng.integers(0, 2**31))
            from models.retinal_input import generate_retinal_spikes, spike_times_to_binary
            spk = generate_retinal_spikes(
                self.retinal_rate, self.gamma_order, duration_s, rng=child_rng)
            ret_spike_times.append(spk)
            ret_bin[i] = spike_times_to_binary(spk, dt, n_steps)

        # Spike collectors
        tc_spike_times = [[] for _ in range(n_tc)]
        nrt_spike_times = [[] for _ in range(n_nrt)]

        # Spike history ring buffers
        buf_len = self.max_delay + 2
        tc_buf = np.zeros((n_tc, buf_len), dtype=bool)
        nrt_buf = np.zeros((n_nrt, buf_len), dtype=bool)
        buf_ptr = 0  # circular pointer

        for step in range(n_steps):
            t = step * dt

            # --- Record spikes from previous step ---
            tc_spk = self.tc.spiked
            nrt_spk = self.nrt.spiked

            tc_buf[:, buf_ptr] = tc_spk
            nrt_buf[:, buf_ptr] = nrt_spk

            # Collect spike times
            tc_idx = np.where(tc_spk)[0]
            for i in tc_idx:
                tc_spike_times[i].append(t)
            nrt_idx = np.where(nrt_spk)[0]
            for j in nrt_idx:
                nrt_spike_times[j].append(t)

            # --- Activate retinal -> TC ---
            ret_mask = ret_bin[:, step]
            self.ampa_ret.activate(ret_mask)

            # --- TC -> nRt (delayed) ---
            if len(self.tn_pre) > 0:
                delayed_ptr = (buf_ptr - self.tn_delays) % buf_len
                pre_spiked = tc_buf[self.tn_pre, delayed_ptr]
                self.ampa_tc_nrt.activate(pre_spiked)

            # --- nRt -> TC (delayed) ---
            if len(self.nt_pre) > 0:
                delayed_ptr = (buf_ptr - self.nt_delays) % buf_len
                pre_spiked = nrt_buf[self.nt_pre, delayed_ptr]
                self.gabaa_nrt_tc.activate(pre_spiked)
                self.gabab_nrt_tc.activate(pre_spiked)

            # --- nRt -> nRt (delayed) ---
            if len(self.nn_pre) > 0:
                delayed_ptr = (buf_ptr - self.nn_delays) % buf_len
                pre_spiked = nrt_buf[self.nn_pre, delayed_ptr]
                self.gabaa_nrt_nrt.activate(pre_spiked)

            # --- Step all synapses ---
            self.ampa_ret.step(dt_ms)
            self.ampa_tc_nrt.step(dt_ms)
            self.gabaa_nrt_tc.step(dt_ms)
            self.gabab_nrt_tc.step(dt_ms)
            self.gabaa_nrt_nrt.step(dt_ms)

            # --- Compute currents for TC neurons ---
            I_tc = self.ampa_ret.current_matrix(self.tc.V)  # (n_tc,)

            # Add GABA_A and GABA_B from nRt->TC
            if len(self.nt_post) > 0:
                V_post_tc = self.tc.V[self.nt_post]
                I_ga = self.gabaa_nrt_tc.current_matrix(V_post_tc)
                I_gb = self.gabab_nrt_tc.current_matrix(V_post_tc)
                # Accumulate into TC neurons
                np.add.at(I_tc, self.nt_post, I_ga + I_gb)

            # --- Compute currents for nRt neurons ---
            I_nrt = np.zeros(n_nrt)

            # AMPA from TC->nRt
            if len(self.tn_post) > 0:
                V_post_nrt = self.nrt.V[self.tn_post]
                I_ampa_tn = self.ampa_tc_nrt.current_matrix(V_post_nrt)
                np.add.at(I_nrt, self.tn_post, I_ampa_tn)

            # GABA_A from nRt->nRt
            if len(self.nn_post) > 0:
                V_post_nn = self.nrt.V[self.nn_post]
                I_gaba_nn = self.gabaa_nrt_nrt.current_matrix(V_post_nn)
                np.add.at(I_nrt, self.nn_post, I_gaba_nn)

            # --- Step neurons ---
            self.tc.step(dt, I_tc)
            self.nrt.step(dt, I_nrt)

            # --- Record ---
            if step % record_every == 0 and rec_idx < n_record:
                t_rec[rec_idx] = t
                V_tc_rec[:, rec_idx] = self.tc.V
                V_nrt_rec[:, rec_idx] = self.nrt.V
                rec_idx += 1

            # Advance ring buffer
            buf_ptr = (buf_ptr + 1) % buf_len

        t_rec = t_rec[:rec_idx]
        V_tc_rec = V_tc_rec[:, :rec_idx]
        V_nrt_rec = V_nrt_rec[:, :rec_idx]

        return {
            't': t_rec,
            'V_tc': V_tc_rec,
            'V_nrt': V_nrt_rec,
            'tc_spike_times': [np.array(s) for s in tc_spike_times],
            'nrt_spike_times': [np.array(s) for s in nrt_spike_times],
            'retinal_spike_times': ret_spike_times,
            'duration_s': duration_s,
            'n_tc': n_tc,
            'n_nrt': n_nrt,
            'replacement_indices': self.replacement_indices,
            'is_replacement': self.is_replacement,
        }
