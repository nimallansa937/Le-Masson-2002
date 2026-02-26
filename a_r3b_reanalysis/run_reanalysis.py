"""
A-R3b Zombie Probe Re-Analysis — CLI Entry Point.

Usage:
    python -m a_r3b_reanalysis.run_reanalysis --phase targets
    python -m a_r3b_reanalysis.run_reanalysis --phase extract --device cuda
    python -m a_r3b_reanalysis.run_reanalysis --phase probe --n-perms 200
    python -m a_r3b_reanalysis.run_reanalysis --phase report
    python -m a_r3b_reanalysis.run_reanalysis --phase all
    python -m a_r3b_reanalysis.run_reanalysis --phase quick  # validation run

Phases:
    targets  — Phase 0: Compute identifiable combination targets from HDF5
    extract  — Phase 1: Extract hidden states + untrained baselines
    probe    — Phase 4: Run probes with selectivity (includes Phases 2-3)
    report   — Phases 5-6: Diagnostics, classification, and plots
    all      — Full pipeline (0 → 1 → 4 → 5 → 6)
    quick    — Quick validation (1 arch, 1 target, 10 perms)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description='A-R3b Zombie Probe Re-Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--phase', required=True,
        choices=['targets', 'extract', 'probe', 'report', 'all', 'quick'],
        help='Which phase to run',
    )
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--arch', nargs='*', default=None,
                        help='Architectures to process')
    parser.add_argument('--n-perms', type=int, default=None,
                        help='Number of permutations for selectivity')
    parser.add_argument('--n-neurons', type=int, default=None,
                        help='Number of neurons to probe (default: all 20)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from incremental checkpoint (probe phase)')
    args = parser.parse_args()

    phase = args.phase

    if phase in ('targets', 'all'):
        print("\n" + "=" * 60)
        print("PHASE 0: Computing identifiable targets")
        print("=" * 60)
        from a_r3b_reanalysis.phase0_identifiable_targets import compute_all_targets
        compute_all_targets(verbose=True)

    if phase in ('extract', 'all'):
        print("\n" + "=" * 60)
        print("PHASE 1: Extracting hidden states + untrained baselines")
        print("=" * 60)
        from a_r3b_reanalysis.phase1_extract_hidden import extract_all_hidden_states
        extract_all_hidden_states(
            architectures=args.arch,
            device=args.device,
            verbose=True,
        )

    if phase in ('probe', 'all'):
        print("\n" + "=" * 60)
        print("PHASE 4: Running probes with selectivity")
        print("=" * 60)
        from a_r3b_reanalysis.phase4_pipeline import run_pipeline
        results_df, _ = run_pipeline(
            architectures=args.arch,
            n_neurons=args.n_neurons,
            n_permutations=args.n_perms,
            device=args.device,
            verbose=True,
            resume=args.resume,
        )

    if phase in ('report', 'all'):
        print("\n" + "=" * 60)
        print("PHASES 5-6: Diagnostics + Plots")
        print("=" * 60)

        # Load results if not already in memory
        if phase == 'report':
            import pandas as pd
            from a_r3b_reanalysis.config import OUTPUT_DIR
            results_path = OUTPUT_DIR / 'a_r3b_full_results.csv'
            incremental_path = OUTPUT_DIR / 'a_r3b_full_results_incremental.csv'
            if results_path.exists():
                results_df = pd.read_csv(str(results_path))
            elif incremental_path.exists():
                print(f"  (using incremental checkpoint — run may be incomplete)")
                results_df = pd.read_csv(str(incremental_path))
            else:
                print(f"ERROR: Results not found at {results_path}")
                print("Run --phase probe first.")
                sys.exit(1)

        from a_r3b_reanalysis.phase5_diagnostics import save_diagnostics
        from a_r3b_reanalysis.phase6_plots import generate_all_plots

        save_diagnostics(results_df, verbose=True)
        generate_all_plots(results_df, verbose=True)

    if phase == 'quick':
        # Quick mode runs ALL prerequisite phases on a minimal subset
        print("\n" + "=" * 60)
        print("QUICK VALIDATION: Phase 0 — Targets")
        print("=" * 60)
        from a_r3b_reanalysis.config import TARGETS_DIR, HIDDEN_STATES_DIR
        if not list(TARGETS_DIR.glob('targets_*.npz')):
            from a_r3b_reanalysis.phase0_identifiable_targets import compute_all_targets
            compute_all_targets(verbose=True)
        else:
            print("  (targets already cached, skipping)")

        print("\n" + "=" * 60)
        print("QUICK VALIDATION: Phase 1 — Hidden states (volterra only)")
        print("=" * 60)
        volterra_path = HIDDEN_STATES_DIR / 'volterra_hidden_states.npz'
        if not volterra_path.exists():
            from a_r3b_reanalysis.phase1_extract_hidden import extract_all_hidden_states
            extract_all_hidden_states(
                architectures=['volterra'],
                device=args.device,
                verbose=True,
            )
        else:
            print("  (volterra hidden states already cached, skipping)")

        print("\n" + "=" * 60)
        print("QUICK VALIDATION: Phase 4 — Probing")
        print("=" * 60)
        from a_r3b_reanalysis.phase4_pipeline import run_quick_validation
        results_df, _ = run_quick_validation(device=args.device, verbose=True)

        from a_r3b_reanalysis.phase5_diagnostics import save_diagnostics
        save_diagnostics(results_df, verbose=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
