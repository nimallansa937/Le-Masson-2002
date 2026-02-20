"""
DESCARTES-NeuralODE: Run the gap-guided architecture search.

Usage:
    # Full run on real data (CPU, 2h per model)
    python run_descartes_neural_ode.py --data_dir path/to/rung3_data --device cpu

    # Quick test (subsample 50 train windows, 0.05h per model)
    python run_descartes_neural_ode.py --data_dir path/to/rung3_data --device cpu --subsample 50 --max_hours_per_model 0.05

    # Synthetic data test (no HDF5 needed)
    python run_descartes_neural_ode.py --data_dir nonexistent --device cpu --max_iterations 3

Requires:
    - A-R2 data with ground truth intermediates (Phase 0 of transformation guide)
    - PyTorch, torchdiffeq, scikit-learn, scipy, numpy, h5py
"""
import argparse
import json
import numpy as np
from pathlib import Path
from orchestrator import DescartesNeuralODEOrchestrator


def main():
    parser = argparse.ArgumentParser(description="DESCARTES-NeuralODE Architecture Search")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing trial_gaba*.h5 files')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda (default: cpu)')
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='Maximum DESCARTES iterations (default: 20)')
    parser.add_argument('--max_hours_per_model', type=float, default=2.0,
                        help='Max training hours per architecture (default: 2.0)')
    parser.add_argument('--target_recovery', type=int, default=120,
                        help='Target bio vars to recover out of 160 (default: 120)')
    parser.add_argument('--subsample', type=int, default=0,
                        help='Subsample N training windows for quick tests (0=use all)')
    parser.add_argument('--output', type=str, default='descartes_results.json')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    # Load data
    from data.ar3_data_loader import load_ar2_data
    train_data, val_data, bio_gt = load_ar2_data(args.data_dir)

    # Subsample for quick testing
    if args.subsample > 0:
        n_train = train_data['X_train'].shape[0]
        n_val = val_data['X_val'].shape[0]
        n_sub_train = min(args.subsample, n_train)
        n_sub_val = min(max(args.subsample // 3, 10), n_val)

        np.random.seed(42)
        train_idx = np.random.choice(n_train, n_sub_train, replace=False)
        val_idx = np.random.choice(n_val, n_sub_val, replace=False)

        train_data['X_train'] = train_data['X_train'][train_idx]
        train_data['Y_train'] = train_data['Y_train'][train_idx]
        train_data['X_short'] = train_data['X_train'][:, :200, :]
        train_data['Y_short'] = train_data['Y_train'][:, :200, :]
        val_data['X_val'] = val_data['X_val'][val_idx]
        val_data['Y_val'] = val_data['Y_val'][val_idx]

        print(f"\nSubsampled: {n_sub_train} train, {n_sub_val} val windows")

    print()
    print("+" + "=" * 67 + "+")
    print("|  DESCARTES-NeuralODE: Gap-Guided Architecture Search            |")
    print("+" + "=" * 67 + "+")
    print("|  Foundation: Balloon Principle (DESCARTES-COLLATZ v2)            |")
    print("|  Enhancement 1: BioVar Pattern Extraction (DreamCoder analog)   |")
    print("|  Enhancement 2: 160-dim Biological Variable Recovery Space      |")
    print("|  Enhancement 3: Short-Segment Verification (Z3-C1 analog)       |")
    print("|  Enhancement 4: Gap-Directed Architecture Generation            |")
    print("+" + "=" * 67 + "+")
    print(f"|  Data: {train_data['X_train'].shape[0]} train, "
          f"{val_data['X_val'].shape[0]} val windows"
          f"{' (subsampled)' if args.subsample > 0 else '':>20} |")
    print(f"|  Device: {args.device:<10}  Budget: {args.max_hours_per_model:.1f}h/model × "
          f"{args.max_iterations} iterations     |")
    print("+" + "=" * 67 + "+")

    # Run
    orchestrator = DescartesNeuralODEOrchestrator(
        train_data=train_data,
        val_data=val_data,
        bio_ground_truth=bio_gt,
        device=args.device,
        max_training_hours=args.max_hours_per_model,
        verbose=args.verbose
    )

    result = orchestrator.run(
        max_iterations=args.max_iterations,
        target_recovery=args.target_recovery
    )

    # Report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best architecture: {result.best_architecture}")
    print(f"Best spike correlation: {result.best_spike_correlation:.3f}")
    print(f"Best bio var recovery: {result.best_biovar_recovery}/160")
    print(f"Total iterations: {result.total_iterations}")
    print(f"Balloon expansions: {result.balloon_expansions}")
    print(f"Remaining gap: {result.gap_remaining:.1%}")
    print(f"Patterns discovered: {result.patterns_discovered}")

    # Compare to A-R3 baselines
    print("\n" + "-" * 70)
    print("COMPARISON TO A-R3 BASELINES")
    print("-" * 70)
    print(f"{'Architecture':<30} {'Spike Corr':>12} {'Bio Vars':>10}")
    print(f"{'Volterra (A-R3)':<30} {'0.549':>12} {'89/160':>10}")
    print(f"{'LSTM (A-R3)':<30} {'0.451':>12} {'~20/160':>10}")
    print(f"{'Neural ODE (A-R3)':<30} {'0.012':>12} {'0/160':>10}")
    print(f"{'DESCARTES Best':<30} {result.best_spike_correlation:>12.3f} "
          f"{result.best_biovar_recovery:>7}/160")

    # Save
    output = {
        'best_architecture': result.best_architecture,
        'best_spike_correlation': result.best_spike_correlation,
        'best_biovar_recovery': result.best_biovar_recovery,
        'iterations': result.total_iterations,
        'balloon_expansions': result.balloon_expansions,
        'gap_remaining': result.gap_remaining,
        'patterns': result.patterns_discovered,
        'log': result.iteration_log
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
