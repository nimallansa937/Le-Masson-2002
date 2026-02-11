"""
Experiment 2 — Input-Output Correlation vs Inhibition (Fig 3)

Goal: Reproduce the inverse relationship between GABA G_max and
spike transfer correlation.

Protocol:
  - Retinal input: ~20 Hz, gamma = 3 (matching Fig 3)
  - Sweep GABA G_max: 0, ~25, ~36, ~47 nS (matching Fig 3 panels)
  - Run >= 60s per condition
  - Compute CI and CC with 2-ms bins

Expected results (Fig 3d,e):
  - CI decreases significantly with increasing GABA (P < 0.05)
  - CC remains low and does not differ significantly across conditions
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit.thalamic_circuit import ThalamicCircuit
from analysis.spike_analysis import (
    contribution_index, correlation_index, cross_correlogram, spike_latency
)
from analysis.plotting import plot_ci_cc_vs_gaba, plot_cross_correlogram


def run_experiment(duration_s=60.0, gaba_values=None, retinal_rate=20.0,
                   gamma_order=3.0, dt=0.025e-3, seed=42, verbose=True):
    """Run input-output correlation experiment.

    Returns
    -------
    results : list of dict
    """
    if gaba_values is None:
        gaba_values = [0, 10, 20, 25, 30, 36, 40, 47, 55, 65]

    all_results = []

    for i, gmax in enumerate(gaba_values):
        if verbose:
            print(f"[Exp2] GABA G_max = {gmax:.1f} nS ({i+1}/{len(gaba_values)})")

        circuit = ThalamicCircuit(
            gaba_gmax_total=gmax,
            retinal_rate=retinal_rate,
            gamma_order=gamma_order,
            dt=dt,
            seed=seed,
        )

        record_dt = 0.001 if duration_s > 10 else None
        sim = circuit.simulate(duration_s, record_dt=record_dt)

        ret_spikes = sim['retinal_spike_times']
        tc_spikes = sim['tc_spike_times']

        ci = contribution_index(ret_spikes, tc_spikes, bin_ms=2.0)
        cc = correlation_index(ret_spikes, tc_spikes, bin_ms=2.0)
        lat_mean, lat_list = spike_latency(ret_spikes, tc_spikes)

        result = {
            'gaba_gmax': gmax,
            'CI': ci,
            'CC': cc,
            'mean_latency_ms': lat_mean,
            'n_retinal_spikes': len(ret_spikes),
            'n_tc_spikes': len(tc_spikes),
            'tc_rate_hz': len(tc_spikes) / duration_s,
        }
        all_results.append(result)

        if verbose:
            print(f"  CI={ci:.4f}, CC={cc:.4f}, Latency={lat_mean:.2f} ms, "
                  f"TC rate={result['tc_rate_hz']:.1f} Hz")

        # Save cross-correlograms for key conditions
        if gmax in [0, 25, 36, 47]:
            bins, counts = cross_correlogram(ret_spikes, tc_spikes, bin_ms=2.0)
            plot_cross_correlogram(
                bins, counts,
                title=f'Cross-correlogram (GABA = {gmax} nS)',
                save_name=f'exp2_xcorr_gaba_{gmax}nS.png'
            )

    # Generate summary figure
    gaba_vals = [r['gaba_gmax'] for r in all_results]
    ci_vals = [r['CI'] for r in all_results]
    cc_vals = [r['CC'] for r in all_results]
    plot_ci_cc_vs_gaba(gaba_vals, ci_vals, cc_vals, save_name='exp2_ci_cc.png')

    if verbose:
        print("\n[Exp2] Expected: CI decreases with GABA, CC remains low")
        if len(ci_vals) >= 2:
            trend = "DECREASING" if ci_vals[-1] < ci_vals[0] else "NOT DECREASING"
            print(f"[Exp2] CI trend: {trend}")

    return all_results


if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 2: Input-Output Correlation vs Inhibition")
    print("Le Masson et al. 2002 Fig 3")
    print("=" * 60)

    results = run_experiment(
        duration_s=10.0,  # Reduce for testing
        gaba_values=[0, 15, 25, 36, 47],
        seed=42,
    )

    print("\nResults summary:")
    print(f"{'GABA (nS)':>10} {'CI':>8} {'CC':>8} {'Latency (ms)':>12} "
          f"{'TC rate':>8}")
    for r in results:
        print(f"{r['gaba_gmax']:>10.1f} {r['CI']:>8.4f} {r['CC']:>8.4f} "
              f"{r['mean_latency_ms']:>12.2f} {r['tc_rate_hz']:>8.1f}")
