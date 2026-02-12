"""
Population-level thalamic circuit simulator.

Wires N TC + N nRt neurons with sparse connectivity, handles
heterogeneous parameters, progressive replacement, and conduction delays.

Uses Option C from guide: reuses existing Rung 1 neuron/synapse classes
directly, parallelizes at the trial level (not within trial).

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

from models.tc_neuron import TCNeuron
from models.nrt_neuron import NRtNeuron
from models.retinal_input import generate_retinal_spikes, spike_times_to_binary
from synapses.ampa import AMPASynapse
from synapses.gabaa import GABAASynapse
from synapses.gabab import GABABSynapse

from population.network import build_connectivity, normalize_weights, assign_delays
from population.heterogeneity import (
    sample_tc_parameters, sample_nrt_parameters,
    create_replacement_tc_params, VALIDATED_TC_PARAMS
)
from population.replacement import select_replacement_indices


class PopulationCircuit:
    """Population-level TC-nRt thalamic circuit.

    Parameters
    ----------
    n_tc : int
        Number of TC neurons.
    n_nrt : int
        Number of nRt neurons.
    gaba_gmax_total : float
        Total GABA conductance per TC neuron (nS).
    retinal_rate : float
        Mean retinal firing rate (Hz).
    gamma_order : float
        ISI gamma shape parameter.
    replacement_fraction : float
        Fraction of TC neurons to replace (0-1).
    replacement_strategy : str
        'random', 'hub_first', 'hub_last', 'spatial_cluster'.
    replacement_seed : int
        Seed for replacement selection.
    p_tc_nrt : float
        TC->nRt connection probability.
    p_nrt_tc : float
        nRt->TC connection probability.
    p_nrt_nrt : float
        nRt->nRt connection probability.
    gabaa_fraction : float
        Fraction of total GABA that is GABA_A.
    nrt_nrt_gmax : float
        Total intra-nRt GABA_A conductance per nRt neuron (nS).
    ampa_ret_tc_gmax : float
        Retinal->TC AMPA conductance (nS).
    ampa_tc_nrt_gmax : float
        Total TC->nRt AMPA conductance per nRt neuron (nS).
    dt : float
        Integration timestep (seconds).
    network_seed : int
        Seed for connectivity generation.
    hetero_seed : int
        Seed for heterogeneity sampling.
    input_seed : int
        Seed for retinal input generation.
    homogeneous : bool
        If True, all bio TC neurons use identical params (Control A).
    """

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
        self.gaba_gmax_total = gaba_gmax_total
        self.gabaa_fraction = gabaa_fraction

        # --- Build connectivity ---
        self.tc_to_nrt, self.nrt_to_tc, self.nrt_to_nrt = build_connectivity(
            n_tc, n_nrt, p_tc_nrt, p_nrt_tc, p_nrt_nrt, seed=network_seed)

        # --- Normalize synaptic weights ---
        # TC->nRt AMPA weights (normalized so each nRt gets ~20 nS total)
        self.ampa_tc_nrt_weights = normalize_weights(
            self.tc_to_nrt, ampa_tc_nrt_gmax)

        # nRt->TC GABA weights (normalized so each TC gets gaba_gmax_total)
        self.gaba_nrt_tc_weights = normalize_weights(
            self.nrt_to_tc, gaba_gmax_total)

        # nRt->nRt GABA_A weights
        self.gaba_nrt_nrt_weights = normalize_weights(
            self.nrt_to_nrt, nrt_nrt_gmax)

        # --- Assign delays ---
        self.delays_tc_nrt = assign_delays(
            self.tc_to_nrt, seed=network_seed)
        self.delays_nrt_tc = assign_delays(
            self.nrt_to_tc, seed=network_seed + 100)
        self.delays_nrt_nrt = assign_delays(
            self.nrt_to_nrt, seed=network_seed + 200)

        # Convert delays to timestep counts
        dt_ms = dt * 1000.0
        self.delay_steps_tc_nrt = np.round(
            self.delays_tc_nrt / dt_ms).astype(int)
        self.delay_steps_nrt_tc = np.round(
            self.delays_nrt_tc / dt_ms).astype(int)
        self.delay_steps_nrt_nrt = np.round(
            self.delays_nrt_nrt / dt_ms).astype(int)

        self.max_delay_steps = max(
            self.delay_steps_tc_nrt.max() if self.delay_steps_tc_nrt.size else 0,
            self.delay_steps_nrt_tc.max() if self.delay_steps_nrt_tc.size else 0,
            self.delay_steps_nrt_nrt.max() if self.delay_steps_nrt_nrt.size else 0,
            1
        )

        # --- Select replacement indices ---
        self.replacement_indices = select_replacement_indices(
            n_tc, replacement_fraction,
            strategy=replacement_strategy,
            nrt_to_tc=self.nrt_to_tc,
            seed=replacement_seed)
        self.is_replacement = np.zeros(n_tc, dtype=bool)
        if len(self.replacement_indices) > 0:
            self.is_replacement[self.replacement_indices] = True

        # --- Create neurons ---
        hetero_rng = np.random.default_rng(hetero_seed)

        if homogeneous:
            bio_tc_params = [VALIDATED_TC_PARAMS.copy() for _ in range(n_tc)]
        else:
            bio_tc_params = sample_tc_parameters(n_tc, hetero_rng)
        nrt_params_list = sample_nrt_parameters(n_nrt, hetero_rng)

        self.tc_neurons = []
        for i in range(n_tc):
            if self.is_replacement[i]:
                p = create_replacement_tc_params()
            else:
                p = bio_tc_params[i]
            self.tc_neurons.append(TCNeuron(params=p))

        self.nrt_neurons = [NRtNeuron(params=p) for p in nrt_params_list]

        # --- Create synapses ---
        # Retinal -> TC: one AMPA per TC (independent)
        self.ampa_ret_tc = [
            AMPASynapse(g_max_nS=ampa_ret_tc_gmax) for _ in range(n_tc)]

        # TC -> nRt: one synapse per connection
        self.ampa_tc_nrt = {}
        for j in range(n_nrt):
            for i in range(n_tc):
                if self.tc_to_nrt[j, i]:
                    g = self.ampa_tc_nrt_weights[j, i]
                    self.ampa_tc_nrt[(i, j)] = AMPASynapse(g_max_nS=g)

        # nRt -> TC: GABA_A + GABA_B per connection
        self.gabaa_nrt_tc = {}
        self.gabab_nrt_tc = {}
        for i in range(n_tc):
            for j in range(n_nrt):
                if self.nrt_to_tc[i, j]:
                    g_total = self.gaba_nrt_tc_weights[i, j]
                    g_a = g_total * gabaa_fraction
                    g_b = g_total * (1.0 - gabaa_fraction)
                    self.gabaa_nrt_tc[(j, i)] = GABAASynapse(g_max_nS=g_a)
                    self.gabab_nrt_tc[(j, i)] = GABABSynapse(g_max_nS=g_b)

        # nRt -> nRt: GABA_A only
        self.gabaa_nrt_nrt = {}
        for j in range(n_nrt):
            for i in range(n_nrt):
                if self.nrt_to_nrt[j, i]:
                    g = self.gaba_nrt_nrt_weights[j, i]
                    self.gabaa_nrt_nrt[(i, j)] = GABAASynapse(
                        g_max_nS=g, E_rev=-90.0)

        # Store seeds for retinal input
        self.input_seed = input_seed

    def simulate(self, duration_s, record_dt=None):
        """Run the population circuit simulation.

        Parameters
        ----------
        duration_s : float
            Simulation duration in seconds.
        record_dt : float, optional
            Recording interval. Default: 0.001s (1 kHz).

        Returns
        -------
        results : dict
            t : time array
            V_tc : (n_tc, n_timepoints) TC voltages
            V_nrt : (n_nrt, n_timepoints) nRt voltages
            tc_spike_times : list of arrays, one per TC neuron
            nrt_spike_times : list of arrays, one per nRt neuron
            retinal_spike_times : list of arrays, one per TC neuron
        """
        dt = self.dt
        n_steps = int(duration_s / dt)
        dt_ms = dt * 1000.0
        n_tc = self.n_tc
        n_nrt = self.n_nrt

        if record_dt is None:
            record_dt = 0.001  # 1 kHz
        record_every = max(1, int(record_dt / dt))
        n_record = n_steps // record_every + 1

        # Recording arrays
        t_rec = np.zeros(n_record)
        V_tc_rec = np.zeros((n_tc, n_record))
        V_nrt_rec = np.zeros((n_nrt, n_record))
        rec_idx = 0

        # Generate independent retinal inputs
        input_rng = np.random.default_rng(self.input_seed)
        retinal_spike_times_list = []
        retinal_binaries = []
        for i in range(n_tc):
            child_rng = np.random.default_rng(
                input_rng.integers(0, 2**31))
            spikes = generate_retinal_spikes(
                self.retinal_rate, self.gamma_order, duration_s, rng=child_rng)
            retinal_spike_times_list.append(spikes)
            retinal_binaries.append(spike_times_to_binary(spikes, dt, n_steps))

        # Spike time collectors
        tc_spike_times = [[] for _ in range(n_tc)]
        nrt_spike_times = [[] for _ in range(n_nrt)]

        # Spike history buffers for delayed transmission
        max_d = self.max_delay_steps + 1
        tc_spike_buffer = np.zeros((n_tc, max_d), dtype=bool)
        nrt_spike_buffer = np.zeros((n_nrt, max_d), dtype=bool)

        # Main integration loop
        for step in range(n_steps):
            t = step * dt

            # Shift spike buffers (circular: newest at index 0)
            tc_spike_buffer = np.roll(tc_spike_buffer, 1, axis=1)
            nrt_spike_buffer = np.roll(nrt_spike_buffer, 1, axis=1)
            tc_spike_buffer[:, 0] = False
            nrt_spike_buffer[:, 0] = False

            # --- Detect spikes and record them ---
            for i in range(n_tc):
                if self.tc_neurons[i].spiked:
                    tc_spike_buffer[i, 0] = True
                    tc_spike_times[i].append(t)

            for j in range(n_nrt):
                if self.nrt_neurons[j].spiked:
                    nrt_spike_buffer[j, 0] = True
                    nrt_spike_times[j].append(t)

            # --- Activate synapses from delayed spikes ---

            # Retinal -> TC (no delay)
            for i in range(n_tc):
                if retinal_binaries[i][step]:
                    self.ampa_ret_tc[i].activate()

            # TC -> nRt (delayed)
            for (tc_i, nrt_j), syn in self.ampa_tc_nrt.items():
                d = self.delay_steps_tc_nrt[nrt_j, tc_i]
                if d < max_d and tc_spike_buffer[tc_i, d]:
                    syn.activate()

            # nRt -> TC (delayed)
            for (nrt_j, tc_i), syn_a in self.gabaa_nrt_tc.items():
                d = self.delay_steps_nrt_tc[tc_i, nrt_j]
                if d < max_d and nrt_spike_buffer[nrt_j, d]:
                    syn_a.activate()
                    self.gabab_nrt_tc[(nrt_j, tc_i)].activate()

            # nRt -> nRt (delayed)
            for (nrt_i, nrt_j), syn in self.gabaa_nrt_nrt.items():
                d = self.delay_steps_nrt_nrt[nrt_j, nrt_i]
                if d < max_d and nrt_spike_buffer[nrt_i, d]:
                    syn.activate()

            # --- Update all synapses ---
            for i in range(n_tc):
                self.ampa_ret_tc[i].step(dt_ms)

            for syn in self.ampa_tc_nrt.values():
                syn.step(dt_ms)

            for syn in self.gabaa_nrt_tc.values():
                syn.step(dt_ms)
            for syn in self.gabab_nrt_tc.values():
                syn.step(dt_ms)

            for syn in self.gabaa_nrt_nrt.values():
                syn.step(dt_ms)

            # --- Compute synaptic currents and update neurons ---

            # TC neurons
            for i in range(n_tc):
                I_total = self.ampa_ret_tc[i].current(self.tc_neurons[i].V)

                # GABA from nRt
                for j in range(n_nrt):
                    if self.nrt_to_tc[i, j]:
                        I_total += self.gabaa_nrt_tc[(j, i)].current(
                            self.tc_neurons[i].V)
                        I_total += self.gabab_nrt_tc[(j, i)].current(
                            self.tc_neurons[i].V)

                self.tc_neurons[i].step(dt, I_ext_nA=I_total)

            # nRt neurons
            for j in range(n_nrt):
                I_total = 0.0

                # AMPA from TC
                for i in range(n_tc):
                    if self.tc_to_nrt[j, i]:
                        I_total += self.ampa_tc_nrt[(i, j)].current(
                            self.nrt_neurons[j].V)

                # GABA from other nRt
                for k in range(n_nrt):
                    if self.nrt_to_nrt[j, k]:
                        I_total += self.gabaa_nrt_nrt[(k, j)].current(
                            self.nrt_neurons[j].V)

                self.nrt_neurons[j].step(dt, I_ext_nA=I_total)

            # --- Record ---
            if step % record_every == 0 and rec_idx < n_record:
                t_rec[rec_idx] = t
                for i in range(n_tc):
                    V_tc_rec[i, rec_idx] = self.tc_neurons[i].V
                for j in range(n_nrt):
                    V_nrt_rec[j, rec_idx] = self.nrt_neurons[j].V
                rec_idx += 1

        # Trim
        t_rec = t_rec[:rec_idx]
        V_tc_rec = V_tc_rec[:, :rec_idx]
        V_nrt_rec = V_nrt_rec[:, :rec_idx]

        # Convert spike lists to arrays
        tc_spike_arrays = [np.array(s) for s in tc_spike_times]
        nrt_spike_arrays = [np.array(s) for s in nrt_spike_times]

        return {
            't': t_rec,
            'V_tc': V_tc_rec,
            'V_nrt': V_nrt_rec,
            'tc_spike_times': tc_spike_arrays,
            'nrt_spike_times': nrt_spike_arrays,
            'retinal_spike_times': retinal_spike_times_list,
            'duration_s': duration_s,
            'n_tc': n_tc,
            'n_nrt': n_nrt,
            'replacement_indices': self.replacement_indices,
            'is_replacement': self.is_replacement,
        }
