"""
Full retinothalamic hybrid circuit assembly.

Architecture:
  Retinal GC --AMPA--> TC --AMPA--> nRt/PGN
                         ^              |
                         +--GABA_A/B----+

Replicates Le Masson et al. 2002 (Nature 417:854-858).
"""

import sys
import os
import numpy as np

# Support both package and standalone imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from models.tc_neuron import TCNeuron
from models.nrt_neuron import NRtNeuron
from models.retinal_input import generate_retinal_spikes, spike_times_to_binary
from synapses.ampa import AMPASynapse
from synapses.gabaa import GABAASynapse
from synapses.gabab import GABABSynapse


class ThalamicCircuit:
    """Complete thalamic relay circuit for Le Masson replication."""

    def __init__(self, gaba_gmax_total=36.4, retinal_rate=42.0,
                 gamma_order=1.5, dt=0.025e-3, temperature=35.0,
                 gabaa_fraction=0.96, tc_params=None, nrt_params=None,
                 ampa_ret_tc_gmax=28.0, ampa_tc_nrt_gmax=20.0,
                 seed=None):
        """
        Parameters
        ----------
        gaba_gmax_total : float
            Total GABA conductance (nS). Split into GABA_A and GABA_B
            by gabaa_fraction.
        retinal_rate : float
            Mean retinal ganglion cell firing rate (Hz).
        gamma_order : float
            ISI distribution shape parameter.
        dt : float
            Integration timestep (seconds).
        temperature : float
            Temperature in Celsius.
        gabaa_fraction : float
            Fraction of total GABA that is GABA_A (default 0.96).
        tc_params : dict, optional
            Override TC neuron parameters.
        nrt_params : dict, optional
            Override nRt neuron parameters.
        ampa_ret_tc_gmax : float
            AMPA Ret->TC maximal conductance (nS).
        ampa_tc_nrt_gmax : float
            AMPA TC->nRt maximal conductance (nS).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.dt = dt
        self.temperature = temperature
        self.retinal_rate = retinal_rate
        self.gamma_order = gamma_order
        self.seed = seed

        # Neuron parameters
        tp = tc_params or {}
        tp.setdefault('temperature', temperature)
        np_ = nrt_params or {}
        np_.setdefault('temperature', temperature)

        # Create neurons
        self.tc = TCNeuron(params=tp)
        self.nrt = NRtNeuron(params=np_)

        # Create synapses
        gabaa_gmax = gaba_gmax_total * gabaa_fraction
        gabab_gmax = gaba_gmax_total * (1.0 - gabaa_fraction)

        self.ampa_ret_tc = AMPASynapse(g_max_nS=ampa_ret_tc_gmax)
        self.ampa_tc_nrt = AMPASynapse(g_max_nS=ampa_tc_nrt_gmax)
        self.gabaa = GABAASynapse(g_max_nS=gabaa_gmax)
        self.gabab = GABABSynapse(g_max_nS=gabab_gmax)

    def simulate(self, duration_s, record_dt=None):
        """Run the full circuit simulation.

        Parameters
        ----------
        duration_s : float
            Simulation duration in seconds.
        record_dt : float, optional
            Recording timestep. If None, records every integration step.
            Use larger values to reduce memory for long simulations.

        Returns
        -------
        results : dict
            t : time array (s)
            V_tc : TC membrane potential (mV)
            V_nrt : nRt membrane potential (mV)
            retinal_spike_times : array of retinal spike times (s)
            tc_spike_times : array of TC spike times (s)
            nrt_spike_times : array of nRt spike times (s)
        """
        dt = self.dt
        n_steps = int(duration_s / dt)
        dt_ms = dt * 1000.0

        # Recording setup
        if record_dt is None:
            record_every = 1
        else:
            record_every = max(1, int(record_dt / dt))

        n_record = n_steps // record_every + 1
        t_rec = np.zeros(n_record)
        V_tc_rec = np.zeros(n_record)
        V_nrt_rec = np.zeros(n_record)
        rec_idx = 0

        # Generate retinal input
        rng = np.random.default_rng(self.seed)
        ret_spike_times = generate_retinal_spikes(
            self.retinal_rate, self.gamma_order, duration_s, rng=rng)
        ret_binary = spike_times_to_binary(ret_spike_times, dt, n_steps)

        # Spike time collectors
        tc_spike_times = []
        nrt_spike_times = []

        # Main integration loop
        for step in range(n_steps):
            t = step * dt

            # --- Check for presynaptic spikes and activate synapses ---
            if ret_binary[step]:
                self.ampa_ret_tc.activate()

            if self.tc.spiked:
                self.ampa_tc_nrt.activate()
                tc_spike_times.append(t)

            if self.nrt.spiked:
                self.gabaa.activate()
                self.gabab.activate()
                nrt_spike_times.append(t)

            # --- Update all synapses ---
            self.ampa_ret_tc.step(dt_ms)
            self.ampa_tc_nrt.step(dt_ms)
            self.gabaa.step(dt_ms)
            self.gabab.step(dt_ms)

            # --- Compute synaptic currents (nA) ---
            I_ampa_tc = self.ampa_ret_tc.current(self.tc.V)
            I_gabaa_tc = self.gabaa.current(self.tc.V)
            I_gabab_tc = self.gabab.current(self.tc.V)
            I_total_tc = I_ampa_tc + I_gabaa_tc + I_gabab_tc

            I_ampa_nrt = self.ampa_tc_nrt.current(self.nrt.V)

            # --- Update neurons ---
            self.tc.step(dt, I_ext_nA=I_total_tc)
            self.nrt.step(dt, I_ext_nA=I_ampa_nrt)

            # --- Record ---
            if step % record_every == 0 and rec_idx < n_record:
                t_rec[rec_idx] = t
                V_tc_rec[rec_idx] = self.tc.V
                V_nrt_rec[rec_idx] = self.nrt.V
                rec_idx += 1

        # Trim recording arrays
        t_rec = t_rec[:rec_idx]
        V_tc_rec = V_tc_rec[:rec_idx]
        V_nrt_rec = V_nrt_rec[:rec_idx]

        return {
            't': t_rec,
            'V_tc': V_tc_rec,
            'V_nrt': V_nrt_rec,
            'retinal_spike_times': ret_spike_times,
            'tc_spike_times': np.array(tc_spike_times),
            'nrt_spike_times': np.array(nrt_spike_times),
            'dt': dt,
            'duration_s': duration_s,
        }
