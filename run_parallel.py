"""
Parallel runner for Vast.ai / multi-core machines.

Parallelises the GABA sweep across CPU cores using multiprocessing.
Each GABA value runs independently — perfect for cloud deployment.

Usage:
  python run_parallel.py                    # Auto-detect cores, 10s sims
  python run_parallel.py --full             # 60s sims (publication quality)
  python run_parallel.py --workers 32       # Use 32 cores
  python run_parallel.py --full --workers 64
"""

import sys
import os
import argparse
import time
import json
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit.thalamic_circuit import ThalamicCircuit
from analysis.oscillation import (
    detect_spindles, spindle_frequency, spindle_duration,
    oscillation_power, is_oscillating
)
from analysis.spike_analysis import contribution_index, correlation_index, spike_latency


def run_single_gaba(args):
    """Run one GABA value. Designed for multiprocessing.Pool.map()."""
    gmax, duration_s, retinal_rate, gamma_order, dt, seed = args

    circuit = ThalamicCircuit(
        gaba_gmax_total=gmax,
        retinal_rate=retinal_rate,
        gamma_order=gamma_order,
        dt=dt,
        seed=seed,
    )

    record_dt = 0.001 if duration_s > 10 else 0.0005
    sim = circuit.simulate(duration_s, record_dt=record_dt)

    # Oscillation metrics
    spindles = detect_spindles(sim['V_tc'], sim['t'])
    freq_mean, freq_std = spindle_frequency(spindles)
    dur_mean, dur_std = spindle_duration(spindles)
    osc = is_oscillating(sim['V_tc'], sim['t'])
    power = oscillation_power(sim['V_tc'], sim['t'])

    # Correlation metrics
    ci = contribution_index(sim['retinal_spike_times'], sim['tc_spike_times'])
    cc = correlation_index(sim['retinal_spike_times'], sim['tc_spike_times'])
    lat_mean, _ = spike_latency(sim['retinal_spike_times'], sim['tc_spike_times'])

    return {
        'gaba_gmax': float(gmax),
        'oscillating': bool(osc),
        'oscillation_power': float(power),
        'n_spindles': len(spindles),
        'spindle_freq_Hz': float(freq_mean) if not np.isnan(freq_mean) else None,
        'spindle_freq_std': float(freq_std) if not np.isnan(freq_std) else None,
        'spindle_dur_s': float(dur_mean) if not np.isnan(dur_mean) else None,
        'spindle_dur_std': float(dur_std) if not np.isnan(dur_std) else None,
        'CI': float(ci),
        'CC': float(cc),
        'mean_latency_ms': float(lat_mean) if not np.isnan(lat_mean) else None,
        'n_tc_spikes': len(sim['tc_spike_times']),
        'n_nrt_spikes': len(sim['nrt_spike_times']),
        'tc_rate_hz': float(len(sim['tc_spike_times']) / duration_s),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Le Masson 2002 bifurcation sweep')
    parser.add_argument('--full', action='store_true',
                        help='60s simulations (publication quality)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: all cores)')
    parser.add_argument('--gaba-min', type=float, default=0,
                        help='Min GABA G_max (nS)')
    parser.add_argument('--gaba-max', type=float, default=74,
                        help='Max GABA G_max (nS)')
    parser.add_argument('--gaba-step', type=float, default=2,
                        help='GABA step size (nS)')
    parser.add_argument('--rate', type=float, default=42.0,
                        help='Retinal firing rate (Hz)')
    parser.add_argument('--gamma', type=float, default=1.5,
                        help='ISI gamma parameter')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    n_workers = args.workers or cpu_count()
    duration = 60.0 if args.full else 10.0
    dt = 0.025e-3

    gaba_values = np.arange(args.gaba_min, args.gaba_max + 0.1, args.gaba_step)

    print("=" * 60)
    print("Le Masson 2002 — Parallel Bifurcation Sweep")
    print("=" * 60)
    print(f"Workers:       {n_workers} cores")
    print(f"GABA range:    {args.gaba_min} to {args.gaba_max} nS "
          f"(step {args.gaba_step}, {len(gaba_values)} values)")
    print(f"Duration:      {duration}s per trial")
    print(f"Retinal rate:  {args.rate} Hz, gamma={args.gamma}")
    print(f"Output:        {args.output}")
    print("=" * 60)

    # Build argument list for pool.map
    task_args = [
        (gmax, duration, args.rate, args.gamma, dt, args.seed)
        for gmax in gaba_values
    ]

    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single_gaba, task_args)

    elapsed = time.time() - t0

    # Sort by GABA value
    results.sort(key=lambda r: r['gaba_gmax'])

    # Find threshold
    threshold = None
    for r in results:
        if r['oscillating']:
            threshold = r['gaba_gmax']
            break

    # Print summary
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n{'GABA (nS)':>10} {'Osc':>5} {'Power':>12} {'Spindles':>9} "
          f"{'Freq (Hz)':>10} {'CI':>8} {'TC rate':>8}")
    print("-" * 75)
    for r in results:
        freq = f"{r['spindle_freq_Hz']:.2f}" if r['spindle_freq_Hz'] else "N/A"
        print(f"{r['gaba_gmax']:>10.1f} {'Y' if r['oscillating'] else 'N':>5} "
              f"{r['oscillation_power']:>12.2e} {r['n_spindles']:>9} "
              f"{freq:>10} {r['CI']:>8.4f} {r['tc_rate_hz']:>8.1f}")

    print(f"\n{'=' * 60}")
    print(f"BIFURCATION THRESHOLD: {threshold} nS")
    print(f"TARGET (Le Masson):    29.0 +/- 4.2 nS")
    if threshold is not None:
        dev = abs(threshold - 29.0)
        print(f"DEVIATION:             {dev:.1f} nS")
        print(f"WITHIN 1 SD:           {dev <= 4.2}")
    print(f"{'=' * 60}")

    # Save results
    output = {
        'metadata': {
            'duration_s': duration,
            'retinal_rate_hz': args.rate,
            'gamma_order': args.gamma,
            'dt': dt,
            'seed': args.seed,
            'n_workers': n_workers,
            'elapsed_s': elapsed,
        },
        'threshold_nS': threshold,
        'target_threshold_nS': 29.0,
        'target_threshold_sd': 4.2,
        'results': results,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate figures
    try:
        _generate_figures(results, threshold)
        print("Figures saved to figures/")
    except Exception as e:
        print(f"Figure generation failed (non-critical): {e}")


def _generate_figures(results, threshold):
    """Generate bifurcation plots from results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    gaba = [r['gaba_gmax'] for r in results]
    power = [r['oscillation_power'] for r in results]
    ci = [r['CI'] for r in results]
    osc = [1 if r['oscillating'] else 0 for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(gaba, power, 'ko-', markersize=4)
    axes[0].set_ylabel('Oscillation Power (7-14 Hz)')
    axes[0].axvline(29.0, color='red', linestyle='--', linewidth=1.5,
                    label='Le Masson = 29 nS')
    if threshold is not None:
        axes[0].axvline(threshold, color='blue', linestyle='--', linewidth=1.5,
                        label=f'Model = {threshold} nS')
    axes[0].legend()
    axes[0].set_title('Bifurcation Diagram — Le Masson 2002 Replication')
    axes[0].grid(True, alpha=0.3)

    colors = ['green' if o else 'gray' for o in osc]
    axes[1].bar(gaba, osc, color=colors, width=1.5)
    axes[1].set_ylabel('Oscillating?')
    axes[1].axvline(29.0, color='red', linestyle='--')
    if threshold:
        axes[1].axvline(threshold, color='blue', linestyle='--')

    axes[2].plot(gaba, ci, 'bo-', markersize=4)
    axes[2].set_ylabel('Contribution Index (CI)')
    axes[2].set_xlabel('GABA G_max (nS)')
    axes[2].axvline(29.0, color='red', linestyle='--')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'bifurcation_parallel.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
