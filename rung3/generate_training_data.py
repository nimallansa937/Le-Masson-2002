"""
Phase 0: Generate training data — simulate 0%-replacement trials and save HDF5.

Usage:
    python -m rung3.generate_training_data --workers 8
    python -m rung3.generate_training_data --quick  # 1 trial for testing

Generates ~190 trials (38 GABA values × 5 seeds) with intermediate recordings.
Each trial saves V_tc, V_nrt, spike times, and biological intermediates
(gating variables, summed synaptic conductances) for Rung 3 model comparison.
"""

import sys
import os
import time
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population.population_circuit_fast import PopulationCircuit
from rung3.config import (
    N_TC, N_NRT, RETINAL_RATE, GAMMA_ORDER, DT, RECORD_DT,
    DURATION_S, GABA_VALUES, ALL_SEEDS, DATA_DIR,
)
from rung3.phase0_recording import save_trial_hdf5, trial_filename


def run_single_trial(args):
    """Run one trial and save to HDF5. Worker function for Pool.map()."""
    gaba_gmax, seed, duration_s, data_dir, trial_idx, total = args

    filepath = trial_filename(gaba_gmax, seed, data_dir)

    # Skip if already generated
    if os.path.exists(filepath):
        sz = os.path.getsize(filepath) / 1e6
        sys.stdout.write(
            f"[{trial_idx+1}/{total}] SKIP GABA={gaba_gmax:.0f} seed={seed} "
            f"({sz:.1f} MB exists)\n")
        sys.stdout.flush()
        return filepath

    t0 = time.time()
    sys.stdout.write(
        f"[{trial_idx+1}/{total}] START GABA={gaba_gmax:.0f} seed={seed}\n")
    sys.stdout.flush()

    circuit = PopulationCircuit(
        n_tc=N_TC, n_nrt=N_NRT,
        gaba_gmax_total=gaba_gmax,
        retinal_rate=RETINAL_RATE,
        gamma_order=GAMMA_ORDER,
        replacement_fraction=0.0,
        dt=DT,
        network_seed=seed,
        hetero_seed=seed,
        input_seed=seed,
    )

    sim = circuit.simulate(
        duration_s,
        record_dt=RECORD_DT,
        record_intermediates=True,
    )

    save_trial_hdf5(filepath, sim, gaba_gmax, seed)

    elapsed = time.time() - t0
    sz = os.path.getsize(filepath) / 1e6
    sys.stdout.write(
        f"[{trial_idx+1}/{total}] DONE  GABA={gaba_gmax:.0f} seed={seed} "
        f"({elapsed:.0f}s, {sz:.1f} MB)\n")
    sys.stdout.flush()

    return filepath


def main():
    parser = argparse.ArgumentParser(description='Generate Rung 3 training data')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--duration', type=float, default=DURATION_S,
                        help='Simulation duration per trial (seconds)')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Output directory for HDF5 files')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1 trial (GABA=30, seed=42)')
    parser.add_argument('--gaba-min', type=float, default=None)
    parser.add_argument('--gaba-max', type=float, default=None)
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (e.g., "42,43")')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.quick:
        gaba_vals = [30.0]
        seeds = [42]
    else:
        gaba_vals = GABA_VALUES
        if args.gaba_min is not None:
            gaba_vals = gaba_vals[gaba_vals >= args.gaba_min]
        if args.gaba_max is not None:
            gaba_vals = gaba_vals[gaba_vals <= args.gaba_max]
        seeds = ALL_SEEDS
        if args.seeds:
            seeds = [int(s) for s in args.seeds.split(',')]

    # Build task list
    tasks = []
    idx = 0
    total = len(gaba_vals) * len(seeds)
    for gaba in gaba_vals:
        for seed in seeds:
            tasks.append((float(gaba), seed, args.duration, args.data_dir,
                          idx, total))
            idx += 1

    print(f"Generating {total} trials ({len(gaba_vals)} GABA × {len(seeds)} seeds)")
    print(f"Duration: {args.duration}s per trial | Workers: {args.workers}")
    print(f"Output: {args.data_dir}/")
    print(f"{'='*60}")

    t_start = time.time()

    if args.workers <= 1:
        results = [run_single_trial(t) for t in tasks]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(run_single_trial, tasks)

    elapsed = time.time() - t_start
    n_done = sum(1 for r in results if r is not None)
    print(f"\n{'='*60}")
    print(f"COMPLETE: {n_done}/{total} trials in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min)")
    total_size = sum(os.path.getsize(r) for r in results if r and os.path.exists(r))
    print(f"Total disk: {total_size/1e9:.2f} GB")


if __name__ == '__main__':
    main()
