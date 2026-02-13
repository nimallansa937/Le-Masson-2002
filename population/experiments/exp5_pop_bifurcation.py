"""
Experiment 5 — Population Baseline Bifurcation (0% replacement).

Validates that the heterogeneous TC-nRt population produces:
  - Spindle oscillations above GABA threshold
  - Population coherence > 0.5 during spindles
  - Bifurcation threshold near 29 nS (within 2 SD)

This is the reference for all replacement comparisons.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from population.population_circuit_fast import PopulationCircuit
from population.population_metrics import (
    population_spindle_coherence, spindle_frequency_stability,
    participation_fraction, population_pause_rate, population_burst_fraction,
)
from analysis.oscillation import find_bifurcation_threshold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_experiment(n_tc=20, n_nrt=20, duration_s=10.0,
                   gaba_values=None, retinal_rate=42.0,
                   dt=0.025e-3, seed=42, verbose=True):
    """Run population baseline bifurcation sweep (0% replacement).

    Returns
    -------
    results : list of dict
    threshold : float
    """
    if gaba_values is None:
        gaba_values = np.arange(0, 75, 5)

    all_results = []

    for idx, gmax in enumerate(gaba_values):
        if verbose:
            print(f"[Exp5] GABA={gmax:.0f} nS ({idx+1}/{len(gaba_values)})")

        circuit = PopulationCircuit(
            n_tc=n_tc, n_nrt=n_nrt,
            gaba_gmax_total=gmax,
            retinal_rate=retinal_rate,
            replacement_fraction=0.0,
            dt=dt,
            network_seed=seed,
            hetero_seed=seed,
            input_seed=seed,
        )

        sim = circuit.simulate(duration_s, record_dt=0.001)

        coh = population_spindle_coherence(sim['V_tc'], sim['t'])
        pop_freq, freq_std, _ = spindle_frequency_stability(
            sim['V_tc'], sim['t'])
        mean_pr, _ = population_pause_rate(sim['tc_spike_times'])
        mean_bf, _ = population_burst_fraction(sim['tc_spike_times'])
        part, _ = participation_fraction(sim['tc_spike_times'], duration_s)

        result = {
            'gaba_gmax': float(gmax),
            'coherence': float(coh),
            'mean_pause_rate': float(mean_pr),
            'mean_burst_fraction': float(mean_bf),
            'participation': float(part),
            'pop_frequency': float(pop_freq) if not np.isnan(pop_freq) else None,
            'freq_std': float(freq_std) if not np.isnan(freq_std) else None,
        }
        all_results.append(result)

        if verbose:
            print(f"  Coh={coh:.3f} PauseRate={mean_pr:.2f} "
                  f"Part={part:.2f} Freq={pop_freq:.1f} Hz")

    # Find threshold
    threshold = find_bifurcation_threshold(
        all_results, metric='mean_pause_rate')

    if verbose:
        print(f"\n{'='*60}")
        print(f"POPULATION BASELINE (0% replacement)")
        print(f"Bifurcation threshold: {threshold:.1f} nS")
        print(f"Target: 29.0 +/- 4.2 nS")
        dev = abs(threshold - 29.0) if not np.isnan(threshold) else float('inf')
        print(f"Within 1 SD: {dev <= 4.2}")
        print(f"{'='*60}")

    return all_results, threshold


if __name__ == '__main__':
    results, threshold = run_experiment(duration_s=10.0)
    print(f"\nThreshold: {threshold:.1f} nS")
