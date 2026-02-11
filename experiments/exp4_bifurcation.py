"""
Experiment 4 — Bifurcation Diagram (PRIMARY DELIVERABLE)

The critical validation: does the all-computational circuit's
bifurcation boundary match Le Masson's hybrid result?

X-axis: GABA G_max (nS) from 0 to ~75 nS
Y-axis: Circuit behaviour metric (oscillation power, CI, binary)

Le Masson reports threshold: 29 +/- 4.2 nS (n=9 cells)

IF threshold ~ 29 nS (within ~1 SD):
  -> TC model captures biological TC's dynamical contribution
  -> Substrate independence validated at single-neuron level

IF threshold significantly different:
  -> Identify which TC model property needs adjustment
  -> Iterate on I_T, I_h parameters
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit.thalamic_circuit import ThalamicCircuit
from analysis.oscillation import (
    detect_spindles, spindle_frequency, spindle_duration,
    oscillation_power, is_oscillating
)
from analysis.spike_analysis import contribution_index, correlation_index
from analysis.plotting import plot_bifurcation, plot_voltage_traces, ensure_figures_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_experiment(duration_s=60.0, gaba_values=None, retinal_rate=42.0,
                   gamma_order=1.5, dt=0.025e-3, seed=42, verbose=True):
    """Run full bifurcation diagram experiment.

    Parameters
    ----------
    duration_s : float
        Simulation duration per GABA value (seconds).
    gaba_values : array-like, optional
        GABA G_max values to sweep (nS). Default: 0 to 73 in 2 nS steps.

    Returns
    -------
    results : list of dict
    threshold : float
        Estimated bifurcation threshold (nS).
    """
    if gaba_values is None:
        gaba_values = np.arange(0, 75, 2)

    all_results = []

    for i, gmax in enumerate(gaba_values):
        if verbose:
            print(f"[Bifurcation] GABA G_max = {gmax:.1f} nS "
                  f"({i+1}/{len(gaba_values)})")

        circuit = ThalamicCircuit(
            gaba_gmax_total=gmax,
            retinal_rate=retinal_rate,
            gamma_order=gamma_order,
            dt=dt,
            seed=seed,
        )

        record_dt = 0.001 if duration_s > 10 else None
        sim = circuit.simulate(duration_s, record_dt=record_dt)

        # Oscillation metrics
        spindles = detect_spindles(sim['V_tc'], sim['t'])
        freq_mean, _ = spindle_frequency(spindles)
        dur_mean, _ = spindle_duration(spindles)
        osc = is_oscillating(sim['V_tc'], sim['t'])
        power = oscillation_power(sim['V_tc'], sim['t'])

        # Correlation metrics
        ci = contribution_index(
            sim['retinal_spike_times'], sim['tc_spike_times'])
        cc = correlation_index(
            sim['retinal_spike_times'], sim['tc_spike_times'])

        result = {
            'gaba_gmax': gmax,
            'oscillating': osc,
            'oscillation_power': power,
            'n_spindles': len(spindles),
            'spindle_freq_Hz': freq_mean,
            'spindle_dur_s': dur_mean,
            'CI': ci,
            'CC': cc,
            'n_tc_spikes': len(sim['tc_spike_times']),
            'tc_rate_hz': len(sim['tc_spike_times']) / duration_s,
        }
        all_results.append(result)

        if verbose:
            print(f"  Osc={osc}, Power={power:.2e}, Spindles={len(spindles)}, "
                  f"CI={ci:.4f}, TC rate={result['tc_rate_hz']:.1f} Hz")

    # --- Find bifurcation threshold ---
    gaba_vals = np.array([r['gaba_gmax'] for r in all_results])
    osc_flags = np.array([r['oscillating'] for r in all_results])
    powers = np.array([r['oscillation_power'] for r in all_results])

    threshold = _find_threshold(gaba_vals, osc_flags)

    # --- Generate figures ---
    _generate_figures(all_results, threshold)

    # --- Report ---
    if verbose:
        print("\n" + "=" * 60)
        print("BIFURCATION DIAGRAM RESULTS")
        print("=" * 60)
        print(f"Estimated bifurcation threshold: {threshold:.1f} nS")
        print(f"Le Masson target: 29.0 +/- 4.2 nS (n=9)")
        if not np.isnan(threshold):
            deviation = abs(threshold - 29.0)
            within_1sd = deviation <= 4.2
            within_2sd = deviation <= 8.4
            print(f"Deviation from target: {deviation:.1f} nS")
            print(f"Within 1 SD: {within_1sd}")
            print(f"Within 2 SD: {within_2sd}")

            if within_1sd:
                print("\n*** VALIDATION PASSED ***")
                print("TC model captures biological TC's dynamical contribution.")
                print("Substrate independence validated at single-neuron level.")
            elif within_2sd:
                print("\nPartial validation: within 2 SD.")
                print("Consider tuning I_T and I_h parameters.")
            else:
                print("\nThreshold significantly different from target.")
                print("Sensitivity analysis needed on I_T, I_h, and AMPA parameters.")

    return all_results, threshold


def _find_threshold(gaba_values, oscillating):
    """Find threshold as first GABA value where oscillation occurs."""
    for g, o in zip(gaba_values, oscillating):
        if o:
            return g
    return float('nan')


def _generate_figures(results, threshold):
    """Generate all bifurcation diagram figures."""
    ensure_figures_dir()

    gaba_vals = [r['gaba_gmax'] for r in results]
    powers = [r['oscillation_power'] for r in results]
    ci_vals = [r['CI'] for r in results]
    osc_binary = [1 if r['oscillating'] else 0 for r in results]

    # Fig A: Power bifurcation diagram
    plot_bifurcation(
        gaba_vals, powers,
        metric_name='Oscillation Power (7-14 Hz band)',
        threshold_line=29.0,
        save_name='exp4_bifurcation_power.png'
    )

    # Fig B: Binary oscillation diagram
    plot_bifurcation(
        gaba_vals, osc_binary,
        metric_name='Oscillating (binary)',
        threshold_line=29.0,
        save_name='exp4_bifurcation_binary.png'
    )

    # Fig C: CI bifurcation
    plot_bifurcation(
        gaba_vals, ci_vals,
        metric_name='Contribution Index (CI)',
        threshold_line=29.0,
        save_name='exp4_bifurcation_ci.png'
    )

    # Fig D: Comprehensive multi-panel
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(gaba_vals, powers, 'ko-', markersize=4)
    axes[0].set_ylabel('Oscillation Power')
    axes[0].axvline(29.0, color='red', linestyle='--', label='Le Masson = 29 nS')
    if not np.isnan(threshold):
        axes[0].axvline(threshold, color='blue', linestyle='--',
                        label=f'Our threshold = {threshold} nS')
    axes[0].legend()
    axes[0].set_title('Bifurcation Diagram — Retinothalamic Circuit')

    colors = ['green' if o else 'gray' for o in osc_binary]
    axes[1].bar(gaba_vals, osc_binary, color=colors, width=1.5)
    axes[1].set_ylabel('Oscillating?')
    axes[1].axvline(29.0, color='red', linestyle='--')

    axes[2].plot(gaba_vals, ci_vals, 'bo-', markersize=4)
    axes[2].set_ylabel('Contribution Index')
    axes[2].set_xlabel('GABA G_max (nS)')
    axes[2].axvline(29.0, color='red', linestyle='--')

    plt.tight_layout()
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    plt.savefig(os.path.join(figures_dir, 'exp4_bifurcation_comprehensive.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_sensitivity_analysis(duration_s=30.0, dt=0.025e-3, seed=42, verbose=True):
    """Extended sensitivity analysis across parameter space.

    Sweeps:
      - Retinal rate: 5, 20, 42, 60 Hz
      - Gamma: 1.5, 3.0
      - GABA_A:GABA_B ratio: 90:10, 96:4, 99:1
      - I_T density: 80%, 100%, 120%
      - I_h density: 80%, 100%, 120%
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS")
        print("=" * 60)

    # Retinal rate sweep
    thresholds_rate = {}
    for rate in [5, 20, 42, 60]:
        if verbose:
            print(f"\n--- Retinal rate = {rate} Hz ---")
        gaba_vals = np.arange(10, 60, 5)
        _, thresh = run_experiment(
            duration_s=duration_s, gaba_values=gaba_vals,
            retinal_rate=rate, gamma_order=1.5,
            dt=dt, seed=seed, verbose=verbose
        )
        thresholds_rate[rate] = thresh

    # I_T density sweep
    thresholds_iT = {}
    for factor in [0.8, 1.0, 1.2]:
        if verbose:
            print(f"\n--- I_T density = {factor*100:.0f}% ---")
        gaba_vals = np.arange(10, 60, 5)
        tc_params = {'g_T': 2.2 * factor}
        # Need custom circuit creation here
        results_iT = []
        for gmax in gaba_vals:
            circuit = ThalamicCircuit(
                gaba_gmax_total=gmax, retinal_rate=42.0,
                gamma_order=1.5, dt=dt, seed=seed,
                tc_params=tc_params
            )
            sim = circuit.simulate(duration_s, record_dt=0.001)
            osc = is_oscillating(sim['V_tc'], sim['t'])
            results_iT.append({'gaba_gmax': gmax, 'oscillating': osc})
        osc_flags = [r['oscillating'] for r in results_iT]
        thresh = _find_threshold(gaba_vals, osc_flags)
        thresholds_iT[factor] = thresh
        if verbose:
            print(f"  Threshold: {thresh:.1f} nS")

    if verbose:
        print("\n--- Sensitivity Summary ---")
        print("Retinal rate thresholds:", thresholds_rate)
        print("I_T density thresholds:", thresholds_iT)

    return {'rate': thresholds_rate, 'iT': thresholds_iT}


if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 4: BIFURCATION DIAGRAM (Primary Deliverable)")
    print("Le Masson et al. 2002 — Substrate Independence Test")
    print("=" * 60)

    # Run with reduced duration for testing (use 60s for publication)
    results, threshold = run_experiment(
        duration_s=10.0,
        gaba_values=np.arange(0, 75, 5),  # Coarser for testing
        seed=42,
    )

    print(f"\nFinal threshold: {threshold:.1f} nS")
    print("Figures saved to figures/ directory")
