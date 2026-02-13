"""
Parallel runner for Rung 2 population replacement experiments.

Parallelises the GABA × replacement fraction sweep across CPU cores.
Each (gaba, fraction, seed) combination runs independently.

Usage:
  python run_population.py                        # Quick test (10s, coarse)
  python run_population.py --full                 # Publication (60s, fine)
  python run_population.py --workers 32           # Use 32 cores
  python run_population.py --full --workers 64    # Vast.ai production run
"""

import sys
import os
import argparse
import time
import json
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from population.population_circuit import PopulationCircuit
from population.population_metrics import (
    population_spindle_coherence, spindle_frequency_stability,
    participation_fraction, population_pause_rate, population_burst_fraction,
)
from analysis.oscillation import find_bifurcation_threshold


def run_single_trial(args):
    """Run one (gaba, fraction, seed) trial. For multiprocessing.Pool."""
    (gaba_gmax, fraction, replacement_seed, n_tc, n_nrt,
     duration_s, retinal_rate, gamma_order, dt, network_seed,
     hetero_seed, input_seed, strategy) = args

    t_start = time.time()

    circuit = PopulationCircuit(
        n_tc=n_tc, n_nrt=n_nrt,
        gaba_gmax_total=gaba_gmax,
        retinal_rate=retinal_rate,
        gamma_order=gamma_order,
        replacement_fraction=fraction,
        replacement_strategy=strategy,
        replacement_seed=replacement_seed,
        dt=dt,
        network_seed=network_seed,
        hetero_seed=hetero_seed,
        input_seed=input_seed,
    )

    record_dt = 0.001 if duration_s > 10 else 0.0005
    sim = circuit.simulate(duration_s, record_dt=record_dt)

    # --- Compute population metrics ---
    # Coherence
    coherence = population_spindle_coherence(sim['V_tc'], sim['t'])

    # Frequency stability
    pop_freq, freq_std, _ = spindle_frequency_stability(sim['V_tc'], sim['t'])

    # Pause rate (population mean)
    mean_pr, per_neuron_pr = population_pause_rate(sim['tc_spike_times'])

    # Burst fraction
    mean_bf, _ = population_burst_fraction(sim['tc_spike_times'])

    # Participation
    part_frac, _ = participation_fraction(
        sim['tc_spike_times'], duration_s)

    # TC firing rates
    tc_rates = [len(s) / duration_s for s in sim['tc_spike_times']]
    mean_tc_rate = float(np.mean(tc_rates))

    # nRt firing rates
    nrt_rates = [len(s) / duration_s for s in sim['nrt_spike_times']]
    mean_nrt_rate = float(np.mean(nrt_rates))

    # Progress — simple print, no shared state
    elapsed_trial = time.time() - t_start
    sys.stdout.write(
        f"DONE frac={fraction:.2f} GABA={gaba_gmax:.0f} seed={replacement_seed} "
        f"PR={mean_pr:.2f} Coh={coherence:.3f} ({elapsed_trial:.0f}s)\n")
    sys.stdout.flush()

    return {
        'gaba_gmax': float(gaba_gmax),
        'fraction': float(fraction),
        'replacement_seed': int(replacement_seed),
        'strategy': strategy,
        'coherence': float(coherence),
        'pop_frequency': float(pop_freq) if not np.isnan(pop_freq) else None,
        'freq_std': float(freq_std) if not np.isnan(freq_std) else None,
        'mean_pause_rate': float(mean_pr),
        'mean_burst_fraction': float(mean_bf),
        'participation': float(part_frac),
        'mean_tc_rate': mean_tc_rate,
        'mean_nrt_rate': mean_nrt_rate,
        'n_tc': n_tc,
        'n_nrt': n_nrt,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Rung 2: Population replacement bifurcation sweep')
    parser.add_argument('--full', action='store_true',
                        help='60s simulations, fine grid (publication quality)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--n-tc', type=int, default=20)
    parser.add_argument('--n-nrt', type=int, default=20)
    parser.add_argument('--gaba-min', type=float, default=0)
    parser.add_argument('--gaba-max', type=float, default=74)
    parser.add_argument('--gaba-step', type=float, default=5)
    parser.add_argument('--fractions', type=str, default='coarse',
                        help='coarse (5) or fine (11) replacement fractions')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Random seeds per fraction')
    parser.add_argument('--rate', type=float, default=42.0)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--strategy', type=str, default='random',
                        choices=['random', 'hub_first', 'hub_last',
                                 'spatial_cluster'])
    parser.add_argument('--network-seed', type=int, default=42)
    parser.add_argument('--hetero-seed', type=int, default=42)
    parser.add_argument('--input-seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='population_results.json')
    args = parser.parse_args()

    n_workers = args.workers or cpu_count()
    dt = 0.025e-3

    if args.full:
        duration = 60.0
        gaba_step = 2 if args.gaba_step == 5 else args.gaba_step
        n_seeds = 5 if args.n_seeds == 3 else args.n_seeds
    else:
        duration = 10.0
        gaba_step = args.gaba_step
        n_seeds = args.n_seeds

    gaba_values = np.arange(args.gaba_min, args.gaba_max + 0.1, gaba_step)

    if args.fractions == 'fine':
        fractions = np.arange(0.0, 1.05, 0.10)
    else:
        fractions = np.array([0.0, 0.25, 0.50, 0.75, 1.00])

    seeds = list(range(42, 42 + n_seeds))

    total_trials = len(gaba_values) * len(fractions) * len(seeds)

    print("=" * 70)
    print("Rung 2: Population Replacement — Bifurcation Surface")
    print("=" * 70)
    print(f"Network:       {args.n_tc} TC + {args.n_nrt} nRt neurons")
    print(f"Workers:       {n_workers} cores")
    print(f"GABA range:    {args.gaba_min} to {args.gaba_max} nS "
          f"(step {gaba_step}, {len(gaba_values)} values)")
    print(f"Fractions:     {[f'{f:.2f}' for f in fractions]}")
    print(f"Seeds/frac:    {n_seeds}")
    print(f"Strategy:      {args.strategy}")
    print(f"Duration:      {duration}s per trial")
    print(f"Total trials:  {total_trials}")
    print(f"Output:        {args.output}")
    print("=" * 70)

    # Build task list
    task_args = []
    for gaba in gaba_values:
        for frac in fractions:
            for seed in seeds:
                task_args.append((
                    float(gaba), float(frac), seed,
                    args.n_tc, args.n_nrt,
                    duration, args.rate, args.gamma, dt,
                    args.network_seed, args.hetero_seed, args.input_seed,
                    args.strategy
                ))

    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single_trial, task_args)

    elapsed = time.time() - t0

    print(f"\nCompleted {len(results)} trials in {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")

    # --- Aggregate results: average over seeds ---
    aggregated = {}
    for r in results:
        key = (r['fraction'], r['gaba_gmax'])
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(r)

    summary = []
    for (frac, gaba), trials in sorted(aggregated.items()):
        avg = {
            'gaba_gmax': gaba,
            'fraction': frac,
            'n_seeds': len(trials),
            'coherence': float(np.mean([t['coherence'] for t in trials])),
            'mean_pause_rate': float(np.mean(
                [t['mean_pause_rate'] for t in trials])),
            'mean_burst_fraction': float(np.mean(
                [t['mean_burst_fraction'] for t in trials])),
            'participation': float(np.mean(
                [t['participation'] for t in trials])),
            'mean_tc_rate': float(np.mean(
                [t['mean_tc_rate'] for t in trials])),
        }
        freqs = [t['pop_frequency'] for t in trials if t['pop_frequency'] is not None]
        avg['pop_frequency'] = float(np.mean(freqs)) if freqs else None
        summary.append(avg)

    # --- Find threshold at each fraction ---
    thresholds = {}
    for frac in fractions:
        frac_data = sorted(
            [s for s in summary if abs(s['fraction'] - frac) < 0.001],
            key=lambda s: s['gaba_gmax'])
        if len(frac_data) >= 5:
            thresh = find_bifurcation_threshold(
                frac_data, metric='mean_pause_rate')
            thresholds[f"{frac:.2f}"] = round(thresh, 1)
        else:
            thresholds[f"{frac:.2f}"] = None

    # --- Print summary ---
    print(f"\n{'Frac':>6} {'GABA':>6} {'Coh':>6} {'PauseR':>8} "
          f"{'BurstF':>7} {'Part':>6} {'TCHz':>6}")
    print("-" * 55)
    for s in summary:
        print(f"{s['fraction']:>6.2f} {s['gaba_gmax']:>6.1f} "
              f"{s['coherence']:>6.3f} {s['mean_pause_rate']:>8.2f} "
              f"{s['mean_burst_fraction']:>7.2f} {s['participation']:>6.2f} "
              f"{s['mean_tc_rate']:>6.1f}")

    print(f"\n{'=' * 70}")
    print("BIFURCATION THRESHOLDS BY REPLACEMENT FRACTION")
    print("=" * 70)
    for frac_str, thresh in thresholds.items():
        marker = ""
        if thresh is not None:
            dev = abs(thresh - 29.0)
            if dev <= 4.2:
                marker = " (within 1 SD)"
            elif dev <= 8.4:
                marker = " (within 2 SD)"
        print(f"  Fraction {frac_str}: {thresh} nS{marker}")
    print(f"  TARGET: 29.0 +/- 4.2 nS")
    print("=" * 70)

    # --- Save ---
    output = {
        'metadata': {
            'n_tc': args.n_tc,
            'n_nrt': args.n_nrt,
            'duration_s': duration,
            'retinal_rate_hz': args.rate,
            'gamma_order': args.gamma,
            'strategy': args.strategy,
            'dt': dt,
            'n_workers': n_workers,
            'elapsed_s': elapsed,
            'network_seed': args.network_seed,
            'hetero_seed': args.hetero_seed,
            'input_seed': args.input_seed,
        },
        'gaba_values': gaba_values.tolist(),
        'fractions': fractions.tolist(),
        'thresholds': thresholds,
        'target_threshold_nS': 29.0,
        'target_threshold_sd': 4.2,
        'summary': summary,
        'raw_results': results,
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # --- Generate figures ---
    try:
        _generate_figures(summary, fractions, gaba_values, thresholds)
        print("Figures saved to figures/")
    except Exception as e:
        print(f"Figure generation failed (non-critical): {e}")


def _generate_figures(summary, fractions, gaba_values, thresholds):
    """Generate population bifurcation figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # --- Figure 1: Bifurcation Surface (heatmap) ---
    n_frac = len(fractions)
    n_gaba = len(gaba_values)
    surface = np.full((n_frac, n_gaba), np.nan)

    for s in summary:
        fi = np.argmin(np.abs(fractions - s['fraction']))
        gi = np.argmin(np.abs(gaba_values - s['gaba_gmax']))
        surface[fi, gi] = s['mean_pause_rate']

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(surface, aspect='auto', origin='lower',
                   extent=[gaba_values[0], gaba_values[-1],
                           fractions[0] * 100, fractions[-1] * 100],
                   cmap='viridis')
    ax.set_xlabel('GABA G_max (nS)')
    ax.set_ylabel('Replacement Fraction (%)')
    ax.set_title('Bifurcation Surface — Mean Pause Rate (Hz)')
    plt.colorbar(im, label='Pause Rate (Hz)')

    # Overlay threshold contour
    for frac_str, thresh in thresholds.items():
        if thresh is not None:
            frac_val = float(frac_str) * 100
            ax.plot(thresh, frac_val, 'r*', markersize=15)

    ax.axvline(29.0, color='red', linestyle='--', linewidth=1,
               label='Le Masson = 29 nS')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pop_bifurcation_surface.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: Threshold vs Replacement Fraction (money plot) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    frac_vals = []
    thresh_vals = []
    for frac_str, thresh in thresholds.items():
        if thresh is not None:
            frac_vals.append(float(frac_str) * 100)
            thresh_vals.append(thresh)

    if frac_vals:
        ax.plot(frac_vals, thresh_vals, 'ko-', markersize=8, linewidth=2)
        ax.axhline(29.0, color='red', linestyle='--', linewidth=1.5,
                   label='Le Masson target = 29.0 nS')
        ax.axhspan(29.0 - 4.2, 29.0 + 4.2, alpha=0.15, color='red',
                   label='± 1 SD (4.2 nS)')
        ax.set_xlabel('Replacement Fraction (%)')
        ax.set_ylabel('Bifurcation Threshold (nS)')
        ax.set_title('Substrate Independence Composition Test')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pop_threshold_vs_fraction.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 3: Coherence vs Replacement Fraction ---
    fig, ax = plt.subplots(figsize=(8, 6))
    # Group by fraction, pick high-GABA values (oscillating regime)
    high_gaba = gaba_values[gaba_values >= 30]
    if len(high_gaba) > 0:
        for frac in fractions:
            coh_vals = [s['coherence'] for s in summary
                        if abs(s['fraction'] - frac) < 0.001
                        and s['gaba_gmax'] >= 30]
            if coh_vals:
                ax.bar(frac * 100, np.mean(coh_vals), width=8,
                       alpha=0.7, color='steelblue',
                       edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Replacement Fraction (%)')
    ax.set_ylabel('Mean Coherence (κ)')
    ax.set_title('Population Coherence in Oscillating Regime (GABA ≥ 30 nS)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pop_coherence_vs_fraction.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 4: Multi-panel pause rate curves ---
    fig, axes = plt.subplots(1, len(fractions), figsize=(4 * len(fractions), 5),
                             sharey=True)
    if len(fractions) == 1:
        axes = [axes]

    for idx, frac in enumerate(fractions):
        frac_data = sorted(
            [s for s in summary if abs(s['fraction'] - frac) < 0.001],
            key=lambda s: s['gaba_gmax'])
        gabas = [s['gaba_gmax'] for s in frac_data]
        prs = [s['mean_pause_rate'] for s in frac_data]
        axes[idx].plot(gabas, prs, 'ko-', markersize=3)
        axes[idx].axvline(29.0, color='red', linestyle='--', linewidth=1)
        thresh = thresholds.get(f"{frac:.2f}")
        if thresh is not None:
            axes[idx].axvline(thresh, color='blue', linestyle='--', linewidth=1)
        axes[idx].set_title(f'{frac * 100:.0f}% replaced')
        axes[idx].set_xlabel('GABA (nS)')
        if idx == 0:
            axes[idx].set_ylabel('Mean Pause Rate (Hz)')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Pause Rate Bifurcation Curves by Replacement Fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pop_pause_rate_panels.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
