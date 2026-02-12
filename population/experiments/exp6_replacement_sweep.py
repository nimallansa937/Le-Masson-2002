"""
Experiment 6 — Progressive Replacement Sweep.

The core Rung 2 experiment: sweep GABA × replacement fraction
to build the 2D bifurcation surface.

Primary deliverable: threshold vs replacement fraction curve.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from population.population_circuit import PopulationCircuit
from population.population_metrics import (
    population_spindle_coherence, population_pause_rate,
    population_burst_fraction, participation_fraction,
    spindle_frequency_stability,
)
from analysis.oscillation import find_bifurcation_threshold


def run_experiment(n_tc=20, n_nrt=20, duration_s=10.0,
                   gaba_values=None, fractions=None,
                   n_seeds=3, retinal_rate=42.0,
                   dt=0.025e-3, base_seed=42, verbose=True):
    """Run progressive replacement sweep.

    Parameters
    ----------
    n_seeds : int
        Number of random replacement seeds per fraction.

    Returns
    -------
    all_results : list of dict
    thresholds : dict (fraction -> threshold_nS)
    """
    if gaba_values is None:
        gaba_values = np.arange(0, 75, 5)
    if fractions is None:
        fractions = [0.0, 0.25, 0.50, 0.75, 1.00]

    seeds = list(range(base_seed, base_seed + n_seeds))
    all_results = []
    total = len(gaba_values) * len(fractions) * len(seeds)
    trial_num = 0

    for frac in fractions:
        for gmax in gaba_values:
            seed_results = []
            for seed in seeds:
                trial_num += 1
                if verbose:
                    print(f"[Exp6] frac={frac:.2f} GABA={gmax:.0f} "
                          f"seed={seed} ({trial_num}/{total})")

                circuit = PopulationCircuit(
                    n_tc=n_tc, n_nrt=n_nrt,
                    gaba_gmax_total=gmax,
                    retinal_rate=retinal_rate,
                    replacement_fraction=frac,
                    replacement_seed=seed,
                    dt=dt,
                    network_seed=base_seed,
                    hetero_seed=base_seed,
                    input_seed=base_seed,
                )

                sim = circuit.simulate(duration_s, record_dt=0.001)

                coh = population_spindle_coherence(sim['V_tc'], sim['t'])
                mean_pr, _ = population_pause_rate(sim['tc_spike_times'])
                mean_bf, _ = population_burst_fraction(sim['tc_spike_times'])
                part, _ = participation_fraction(
                    sim['tc_spike_times'], duration_s)

                seed_results.append({
                    'coherence': coh, 'mean_pause_rate': mean_pr,
                    'mean_burst_fraction': mean_bf, 'participation': part,
                })

            # Average over seeds
            avg = {
                'gaba_gmax': float(gmax),
                'fraction': float(frac),
                'coherence': float(np.mean([r['coherence'] for r in seed_results])),
                'mean_pause_rate': float(np.mean(
                    [r['mean_pause_rate'] for r in seed_results])),
                'mean_burst_fraction': float(np.mean(
                    [r['mean_burst_fraction'] for r in seed_results])),
                'participation': float(np.mean(
                    [r['participation'] for r in seed_results])),
            }
            all_results.append(avg)

    # Find threshold at each fraction
    thresholds = {}
    for frac in fractions:
        frac_data = sorted(
            [r for r in all_results if abs(r['fraction'] - frac) < 0.001],
            key=lambda r: r['gaba_gmax'])
        thresh = find_bifurcation_threshold(
            frac_data, metric='mean_pause_rate')
        thresholds[frac] = round(thresh, 1) if not np.isnan(thresh) else None

    if verbose:
        print(f"\n{'='*60}")
        print("REPLACEMENT SWEEP RESULTS")
        print("=" * 60)
        for frac, thresh in thresholds.items():
            print(f"  Fraction {frac:.2f}: threshold = {thresh} nS")
        print(f"  TARGET: 29.0 +/- 4.2 nS")
        print("=" * 60)

    return all_results, thresholds


if __name__ == '__main__':
    results, thresholds = run_experiment(duration_s=10.0)
