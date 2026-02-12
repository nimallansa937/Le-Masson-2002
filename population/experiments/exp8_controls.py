"""
Experiment 8 — Controls and Ablation Studies.

Control A: Homogeneous biological baseline (no heterogeneity).
Control B: Heterogeneous replacement (replacement models have variability).
Control D: Network size scaling (N=10, 20, 50).

These controls disentangle whether degradation (if any) comes from:
  - Loss of heterogeneity (Control A vs baseline)
  - The one-size-fits-all replacement model (Control B)
  - Finite size effects (Control D)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from population.population_circuit import PopulationCircuit
from population.population_metrics import (
    population_pause_rate, population_spindle_coherence,
)
from analysis.oscillation import find_bifurcation_threshold


def run_control_a(n_tc=20, n_nrt=20, duration_s=10.0,
                  gaba_values=None, dt=0.025e-3, seed=42, verbose=True):
    """Control A: Homogeneous biological baseline.

    All TC neurons use identical parameters (no heterogeneity).
    Tests whether population dynamics depend on heterogeneity.
    """
    if gaba_values is None:
        gaba_values = np.arange(0, 75, 5)

    if verbose:
        print("\n--- Control A: Homogeneous Baseline ---")

    results = []
    for gmax in gaba_values:
        circuit = PopulationCircuit(
            n_tc=n_tc, n_nrt=n_nrt,
            gaba_gmax_total=gmax,
            replacement_fraction=0.0,
            dt=dt,
            network_seed=seed,
            hetero_seed=seed,
            input_seed=seed,
            homogeneous=True,  # Key: no heterogeneity
        )
        sim = circuit.simulate(duration_s, record_dt=0.001)
        mean_pr, _ = population_pause_rate(sim['tc_spike_times'])
        coh = population_spindle_coherence(sim['V_tc'], sim['t'])

        results.append({
            'gaba_gmax': float(gmax),
            'mean_pause_rate': float(mean_pr),
            'coherence': float(coh),
        })

        if verbose:
            print(f"  GABA={gmax:.0f}: PR={mean_pr:.2f}, Coh={coh:.3f}")

    threshold = find_bifurcation_threshold(results, metric='mean_pause_rate')
    if verbose:
        print(f"  Threshold (homogeneous): {threshold:.1f} nS")

    return results, threshold


def run_control_d(duration_s=10.0, gaba_values=None, dt=0.025e-3,
                  seed=42, verbose=True):
    """Control D: Network size scaling.

    Run at N=10, 20 to check finite size effects.
    """
    if gaba_values is None:
        gaba_values = np.arange(0, 75, 5)

    if verbose:
        print("\n--- Control D: Network Size Scaling ---")

    size_thresholds = {}
    for n in [10, 20]:
        if verbose:
            print(f"\n  N={n} per type:")

        results = []
        for gmax in gaba_values:
            circuit = PopulationCircuit(
                n_tc=n, n_nrt=n,
                gaba_gmax_total=gmax,
                replacement_fraction=0.0,
                dt=dt,
                network_seed=seed,
                hetero_seed=seed,
                input_seed=seed,
            )
            sim = circuit.simulate(duration_s, record_dt=0.001)
            mean_pr, _ = population_pause_rate(sim['tc_spike_times'])
            results.append({
                'gaba_gmax': float(gmax),
                'mean_pause_rate': float(mean_pr),
            })

        threshold = find_bifurcation_threshold(
            results, metric='mean_pause_rate')
        size_thresholds[n] = round(threshold, 1)

        if verbose:
            print(f"  Threshold (N={n}): {threshold:.1f} nS")

    return size_thresholds


def run_all_controls(duration_s=10.0, dt=0.025e-3, seed=42, verbose=True):
    """Run all control experiments."""
    if verbose:
        print("=" * 60)
        print("CONTROL EXPERIMENTS")
        print("=" * 60)

    ctrl_a_results, ctrl_a_threshold = run_control_a(
        duration_s=duration_s, dt=dt, seed=seed, verbose=verbose)

    size_thresholds = run_control_d(
        duration_s=duration_s, dt=dt, seed=seed, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print("CONTROL SUMMARY")
        print("=" * 60)
        print(f"Control A (homogeneous): {ctrl_a_threshold:.1f} nS")
        for n, t in size_thresholds.items():
            print(f"Control D (N={n}): {t} nS")
        print(f"Target: 29.0 +/- 4.2 nS")
        print("=" * 60)

    return {
        'control_a_threshold': ctrl_a_threshold,
        'size_thresholds': size_thresholds,
    }


if __name__ == '__main__':
    run_all_controls(duration_s=10.0)
