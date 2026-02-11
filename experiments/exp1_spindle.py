"""
Experiment 1 — Spindle Wave Generation (Fig 1c, Fig 2)

Goal: Reproduce progression from quiescence to spindle oscillations
as GABA G_max increases.

Protocol:
  - Retinal input: 42 Hz, gamma = 1.5 (matching Fig 2)
  - Sweep GABA G_max from 0 to 73 nS
  - For each G_max, run 60 seconds of simulation
  - Measure: oscillation presence, frequency, duration, threshold

Success criterion: Bifurcation threshold within 1 SD of 29 +/- 4.2 nS
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit.thalamic_circuit import ThalamicCircuit
from analysis.oscillation import detect_spindles, spindle_frequency, spindle_duration, is_oscillating, oscillation_power
from analysis.plotting import plot_voltage_traces, ensure_figures_dir


def run_experiment(duration_s=60.0, gaba_values=None, retinal_rate=42.0,
                   gamma_order=1.5, dt=0.025e-3, seed=42, verbose=True):
    """Run spindle wave generation experiment.

    Parameters
    ----------
    duration_s : float
        Duration per trial (seconds).
    gaba_values : array-like, optional
        GABA G_max values to sweep (nS).
    retinal_rate : float
        Retinal firing rate (Hz).
    gamma_order : float
        ISI regularity parameter.

    Returns
    -------
    results : list of dict
        Per-trial results with GABA value and spindle metrics.
    """
    if gaba_values is None:
        # Paper tested 0, 15, 26, 30 nS explicitly in Fig 1c
        gaba_values = [0, 5, 10, 15, 20, 25, 26, 28, 29, 30, 32, 35, 40, 50, 60, 73]

    all_results = []

    for i, gmax in enumerate(gaba_values):
        if verbose:
            print(f"[Exp1] GABA G_max = {gmax:.1f} nS ({i+1}/{len(gaba_values)})")

        circuit = ThalamicCircuit(
            gaba_gmax_total=gmax,
            retinal_rate=retinal_rate,
            gamma_order=gamma_order,
            dt=dt,
            seed=seed,
        )

        # Use larger record_dt for long simulations to save memory
        record_dt = 0.001 if duration_s > 10 else None  # 1 ms recording

        sim = circuit.simulate(duration_s, record_dt=record_dt)

        # Detect spindles
        spindles = detect_spindles(sim['V_tc'], sim['t'])
        freq_mean, freq_std = spindle_frequency(spindles)
        dur_mean, dur_std = spindle_duration(spindles)
        osc = is_oscillating(sim['V_tc'], sim['t'])
        power = oscillation_power(sim['V_tc'], sim['t'])

        result = {
            'gaba_gmax': gmax,
            'n_spindles': len(spindles),
            'oscillating': osc,
            'spindle_freq_mean': freq_mean,
            'spindle_freq_std': freq_std,
            'spindle_dur_mean': dur_mean,
            'spindle_dur_std': dur_std,
            'oscillation_power': power,
            'n_tc_spikes': len(sim['tc_spike_times']),
            'n_nrt_spikes': len(sim['nrt_spike_times']),
        }
        all_results.append(result)

        if verbose:
            print(f"  Oscillating: {osc}, Spindles: {len(spindles)}, "
                  f"Freq: {freq_mean:.2f} Hz, Duration: {dur_mean:.2f} s, "
                  f"Power: {power:.2e}")

        # Save voltage traces for key GABA values
        if gmax in [0, 15, 26, 30, 50]:
            plot_voltage_traces(
                sim, duration_s=5.0,
                title=f'GABA G_max = {gmax} nS',
                save_name=f'exp1_trace_gaba_{gmax}nS.png'
            )

    # Find threshold
    osc_flags = [r['oscillating'] for r in all_results]
    gaba_vals = [r['gaba_gmax'] for r in all_results]
    threshold = find_oscillation_threshold(gaba_vals, osc_flags)

    if verbose:
        print(f"\n[Exp1] Oscillation threshold: {threshold:.1f} nS")
        print(f"[Exp1] Target: 29 +/- 4.2 nS")
        if threshold is not None:
            within_1sd = abs(threshold - 29.0) <= 4.2
            print(f"[Exp1] Within 1 SD: {within_1sd}")

    return all_results, threshold


def find_oscillation_threshold(gaba_values, oscillating_flags):
    """Find GABA G_max value at which oscillations first appear."""
    for gv, osc in zip(gaba_values, oscillating_flags):
        if osc:
            return gv
    return float('nan')


if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 1: Spindle Wave Generation")
    print("Le Masson et al. 2002 Fig 1c, Fig 2")
    print("=" * 60)

    # Quick test with shorter duration
    results, threshold = run_experiment(
        duration_s=10.0,  # Reduce for testing; use 60s for full run
        gaba_values=[0, 10, 20, 25, 28, 29, 30, 35, 40, 50],
        seed=42,
    )

    print("\nResults summary:")
    print(f"{'GABA (nS)':>10} {'Oscillating':>12} {'Spindles':>10} "
          f"{'Freq (Hz)':>10} {'Duration (s)':>12}")
    for r in results:
        print(f"{r['gaba_gmax']:>10.1f} {str(r['oscillating']):>12} "
              f"{r['n_spindles']:>10} {r['spindle_freq_mean']:>10.2f} "
              f"{r['spindle_dur_mean']:>12.2f}")
