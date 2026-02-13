"""
Experiment 7 — Replacement Strategy Comparison.

Compare replacement strategies at selected GABA values near threshold:
  - Random (primary)
  - Hub-first (most-connected TC neurons replaced first)
  - Hub-last (least-connected first)
  - Spatial cluster (contiguous block)

Tests whether network position matters for replacement.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from population.population_circuit_fast import PopulationCircuit
from population.population_metrics import (
    population_spindle_coherence, population_pause_rate,
    participation_fraction,
)
from analysis.oscillation import find_bifurcation_threshold


STRATEGIES = ['random', 'hub_first', 'hub_last', 'spatial_cluster']


def run_experiment(n_tc=20, n_nrt=20, duration_s=10.0,
                   gaba_values=None, fractions=None,
                   n_seeds=3, retinal_rate=42.0,
                   dt=0.025e-3, base_seed=42, verbose=True):
    """Compare replacement strategies.

    Returns
    -------
    strategy_results : dict
        strategy -> list of {fraction, gaba_gmax, threshold, ...}
    strategy_thresholds : dict
        strategy -> {fraction -> threshold}
    """
    if gaba_values is None:
        gaba_values = np.arange(10, 55, 5)  # Focus near threshold
    if fractions is None:
        fractions = [0.0, 0.25, 0.50, 0.75, 1.00]

    seeds = list(range(base_seed, base_seed + n_seeds))
    strategy_results = {}
    strategy_thresholds = {}

    for strategy in STRATEGIES:
        if verbose:
            print(f"\n--- Strategy: {strategy} ---")

        results = []
        for frac in fractions:
            for gmax in gaba_values:
                seed_prs = []
                seed_cohs = []
                for seed in seeds:
                    circuit = PopulationCircuit(
                        n_tc=n_tc, n_nrt=n_nrt,
                        gaba_gmax_total=gmax,
                        retinal_rate=retinal_rate,
                        replacement_fraction=frac,
                        replacement_strategy=strategy,
                        replacement_seed=seed,
                        dt=dt,
                        network_seed=base_seed,
                        hetero_seed=base_seed,
                        input_seed=base_seed,
                    )
                    sim = circuit.simulate(duration_s, record_dt=0.001)
                    mean_pr, _ = population_pause_rate(sim['tc_spike_times'])
                    coh = population_spindle_coherence(sim['V_tc'], sim['t'])
                    seed_prs.append(mean_pr)
                    seed_cohs.append(coh)

                results.append({
                    'gaba_gmax': float(gmax),
                    'fraction': float(frac),
                    'mean_pause_rate': float(np.mean(seed_prs)),
                    'coherence': float(np.mean(seed_cohs)),
                })

        # Find thresholds
        thresholds = {}
        for frac in fractions:
            frac_data = sorted(
                [r for r in results if abs(r['fraction'] - frac) < 0.001],
                key=lambda r: r['gaba_gmax'])
            if len(frac_data) >= 5:
                thresh = find_bifurcation_threshold(
                    frac_data, metric='mean_pause_rate')
                thresholds[frac] = round(thresh, 1) if not np.isnan(thresh) else None
            else:
                thresholds[frac] = None

        strategy_results[strategy] = results
        strategy_thresholds[strategy] = thresholds

        if verbose:
            for frac, thresh in thresholds.items():
                print(f"  frac={frac:.2f}: threshold = {thresh} nS")

    if verbose:
        print(f"\n{'='*60}")
        print("STRATEGY COMPARISON")
        print("=" * 60)
        print(f"{'Strategy':<18} ", end="")
        for frac in fractions:
            print(f"{'f=' + str(frac):>8}", end="")
        print()
        for strategy in STRATEGIES:
            print(f"{strategy:<18} ", end="")
            for frac in fractions:
                t = strategy_thresholds[strategy].get(frac)
                print(f"{str(t):>8}", end="")
            print()
        print("=" * 60)

    return strategy_results, strategy_thresholds


if __name__ == '__main__':
    results, thresholds = run_experiment(duration_s=10.0, n_seeds=2)
