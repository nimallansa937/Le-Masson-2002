"""
Experiment 3 — Noradrenaline Modulation (Fig 4)

Goal: Show that simulated noradrenaline restores input-output correlation.

Noradrenaline effects modelled:
  1. Depolarisation of TC cell — reduce I_KL (potassium leak)
  2. Decrease nRt gain — reduce nRt excitability / GABA release

Protocol:
  1. Baseline: retinal 20-30 Hz, gamma=3, GABA G_max=36-38 nS (sleep-like)
  2. Apply simulated NA: modify TC leak and/or nRt parameters
  3. Measure CI, CC, spike latency before/after NA

Target results:
  - CC increases significantly under NA (P < 0.05)
  - CI increases in most cells under NA
  - TC spike latency decreases: 5.3 -> 2.8 ms
  - Further improvement when GABA G_max -> 0 under NA
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit.thalamic_circuit import ThalamicCircuit
from analysis.spike_analysis import contribution_index, correlation_index, spike_latency
from analysis.plotting import ensure_figures_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_experiment(duration_s=60.0, gaba_gmax=36.0, retinal_rate=20.0,
                   gamma_order=3.0, dt=0.025e-3, seed=42, verbose=True):
    """Run noradrenaline modulation experiment.

    Tests three conditions:
      1. Control (sleep-like): full GABA, no NA
      2. NA applied: reduced TC KL leak + reduced nRt excitability
      3. NA + no GABA: NA applied and GABA removed

    Returns
    -------
    results : dict with keys 'control', 'na', 'na_no_gaba'
    """
    conditions = {
        'control': {
            'gaba': gaba_gmax,
            'tc_na_factor': 1.0,    # full KL leak
            'nrt_na_factor': 1.0,   # full nRt excitability
        },
        'na': {
            'gaba': gaba_gmax,
            'tc_na_factor': 0.3,    # 70% reduction in KL -> depolarisation
            'nrt_na_factor': 0.5,   # 50% reduction in nRt T-current
        },
        'na_no_gaba': {
            'gaba': 0.0,
            'tc_na_factor': 0.3,
            'nrt_na_factor': 0.5,
        },
    }

    all_results = {}

    for cond_name, params in conditions.items():
        if verbose:
            print(f"[Exp3] Condition: {cond_name}")

        circuit = ThalamicCircuit(
            gaba_gmax_total=params['gaba'],
            retinal_rate=retinal_rate,
            gamma_order=gamma_order,
            dt=dt,
            seed=seed,
        )

        # Apply NA modulation
        circuit.tc.apply_noradrenaline(params['tc_na_factor'])
        circuit.nrt.apply_noradrenaline(params['nrt_na_factor'])

        record_dt = 0.001 if duration_s > 10 else None
        sim = circuit.simulate(duration_s, record_dt=record_dt)

        ret_spikes = sim['retinal_spike_times']
        tc_spikes = sim['tc_spike_times']

        ci = contribution_index(ret_spikes, tc_spikes, bin_ms=2.0)
        cc = correlation_index(ret_spikes, tc_spikes, bin_ms=2.0)
        lat_mean, _ = spike_latency(ret_spikes, tc_spikes)

        result = {
            'CI': ci,
            'CC': cc,
            'mean_latency_ms': lat_mean,
            'n_tc_spikes': len(tc_spikes),
            'tc_rate_hz': len(tc_spikes) / duration_s,
        }
        all_results[cond_name] = result

        if verbose:
            print(f"  CI={ci:.4f}, CC={cc:.4f}, Latency={lat_mean:.2f} ms, "
                  f"TC rate={result['tc_rate_hz']:.1f} Hz")

    # Generate comparison figure
    _plot_na_comparison(all_results)

    if verbose:
        print("\n[Exp3] Summary:")
        print(f"  Control -> NA: CC {all_results['control']['CC']:.4f} -> "
              f"{all_results['na']['CC']:.4f}")
        print(f"  Control -> NA: Latency {all_results['control']['mean_latency_ms']:.2f} -> "
              f"{all_results['na']['mean_latency_ms']:.2f} ms")
        print(f"  Target latency change: 5.3 -> 2.8 ms")

    return all_results


def _plot_na_comparison(results):
    """Bar plot comparing conditions."""
    ensure_figures_dir()
    conditions = ['control', 'na', 'na_no_gaba']
    labels = ['Control\n(Sleep)', 'NA\n(Wake)', 'NA + no GABA\n(Wake, no inh.)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # CI comparison
    ci_vals = [results[c]['CI'] for c in conditions]
    axes[0].bar(labels, ci_vals, color=['steelblue', 'coral', 'gold'])
    axes[0].set_ylabel('Contribution Index (CI)')
    axes[0].set_title('CI')

    # CC comparison
    cc_vals = [results[c]['CC'] for c in conditions]
    axes[1].bar(labels, cc_vals, color=['steelblue', 'coral', 'gold'])
    axes[1].set_ylabel('Correlation Index (CC)')
    axes[1].set_title('CC')

    # Latency comparison
    lat_vals = [results[c]['mean_latency_ms'] for c in conditions]
    axes[2].bar(labels, lat_vals, color=['steelblue', 'coral', 'gold'])
    axes[2].set_ylabel('Mean Latency (ms)')
    axes[2].set_title('Spike Latency')

    plt.suptitle('Noradrenaline Modulation (Fig 4 equivalent)', fontsize=13)
    plt.tight_layout()

    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    plt.savefig(os.path.join(figures_dir, 'exp3_na_modulation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 3: Noradrenaline Modulation")
    print("Le Masson et al. 2002 Fig 4")
    print("=" * 60)

    results = run_experiment(
        duration_s=10.0,  # Reduce for testing
        gaba_gmax=36.0,
        seed=42,
    )
