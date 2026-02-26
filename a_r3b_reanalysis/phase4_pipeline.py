"""
Phase 4: Full Reanalysis Orchestrator.

For each (architecture x target x probe_type), compute:
  1. R²_trained via temporal_block_cv (Phase 2)
  2. R²_untrained via temporal_block_cv on untrained baseline (Phase 2)
  3. Selectivity via block permutation (Phase 3)
  4. delta_R² = R²_trained - R²_untrained

Iterates over all 20 TC neurons for each target variable.
Results saved as a comprehensive CSV for diagnostic analysis.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a_r3b_reanalysis.config import (
    TARGET_REGISTRY, ARCHITECTURE_CHECKPOINTS, PROBE_TYPES,
    HIDDEN_STATES_DIR, TARGETS_DIR, OUTPUT_DIR, VAL_SEEDS, N_TC,
)
from a_r3b_reanalysis.phase2_probes import temporal_block_cv
from a_r3b_reanalysis.phase3_selectivity import compute_selectivity


def load_hidden_states(arch_name):
    """Load pre-extracted hidden states from Phase 1.

    Returns
    -------
    data : dict with keys:
        'trained_hidden': ndarray (n_windows, window_bins, hidden_dim)
        'untrained_hidden': ndarray (n_windows, window_bins, hidden_dim)
        'trained_preds': ndarray (n_windows, window_bins, output_dim)
        'trial_ids': ndarray (n_windows,)
    """
    path = HIDDEN_STATES_DIR / f'{arch_name}_hidden_states.npz'
    if not path.exists():
        raise FileNotFoundError(
            f"Hidden states not found: {path}\n"
            f"Run Phase 1 first: python -m a_r3b_reanalysis.phase1_extract_hidden"
        )
    data = np.load(str(path), allow_pickle=True)
    return {
        'trained_hidden': data['trained_hidden'],
        'untrained_hidden': data['untrained_hidden'],
        'trained_preds': data['trained_preds'],
        'trial_ids': data['trial_ids'],
    }


def load_targets():
    """Load pre-computed identifiable targets from Phase 0.

    Concatenates all trial target files in the same order as
    the hidden states were extracted.

    Returns
    -------
    all_targets : dict
        target_name -> ndarray (n_windows_total, window_bins, n_neurons)
    """
    target_files = sorted(TARGETS_DIR.glob('targets_*.npz'))
    if not target_files:
        raise FileNotFoundError(
            f"No target files in {TARGETS_DIR}\n"
            f"Run Phase 0 first: python -m a_r3b_reanalysis.phase0_identifiable_targets"
        )

    # Collect targets per variable across trials
    all_targets = {}
    first = True

    for tf in target_files:
        data = np.load(str(tf), allow_pickle=True)
        for key in TARGET_REGISTRY:
            if key in data:
                arr = data[key]  # (n_windows, window_bins, n_neurons)
                if key not in all_targets:
                    all_targets[key] = [arr]
                else:
                    all_targets[key].append(arr)

        if first:
            first = False

    # Concatenate across trials
    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key], axis=0)

    return all_targets


def run_pipeline(architectures=None, target_names=None, probe_types=None,
                 n_neurons=None, n_permutations=None,
                 device='cpu', verbose=True, resume=False):
    """Run the full A-R3b reanalysis pipeline.

    Parameters
    ----------
    architectures : list of str, optional
        Which architectures to analyze. Defaults to all.
    target_names : list of str, optional
        Which targets to probe. Defaults to all in TARGET_REGISTRY.
    probe_types : list of str, optional
        Which probe types. Defaults to PROBE_TYPES.
    n_neurons : int, optional
        How many neurons to probe (0 to n_neurons-1). Defaults to N_TC (20).
    n_permutations : int, optional
        For selectivity. Defaults to config.
    device : str
    verbose : bool
    resume : bool
        If True, load incremental checkpoint and skip already-completed combos.

    Returns
    -------
    results_df : pd.DataFrame
        Full results table.
    output_path : Path
        Where CSV was saved.
    """
    if architectures is None:
        architectures = list(ARCHITECTURE_CHECKPOINTS.keys())
    if target_names is None:
        target_names = list(TARGET_REGISTRY.keys())
    if probe_types is None:
        probe_types = PROBE_TYPES
    if n_neurons is None:
        n_neurons = N_TC

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load targets (shared across all architectures)
    if verbose:
        print("Phase 4: Loading targets...")
    all_targets = load_targets()
    if verbose:
        print(f"  Loaded {len(all_targets)} target variables")
        for k, v in list(all_targets.items())[:3]:
            print(f"    {k}: {v.shape}")

    rows = []
    total_combos = (len(architectures) * len(target_names)
                    * len(probe_types) * n_neurons)
    combo_i = 0

    # Incremental save path — checkpoint every N combos so progress
    # survives crashes. Final CSV overwrites this at the end.
    incremental_path = OUTPUT_DIR / 'a_r3b_full_results_incremental.csv'
    SAVE_EVERY = 30  # save checkpoint every 30 combos (~every 10 minutes)

    # Resume from checkpoint if requested
    completed_keys = set()
    if resume and incremental_path.exists():
        prev_df = pd.read_csv(str(incremental_path))
        rows = prev_df.to_dict('records')
        for r in rows:
            completed_keys.add(
                (r['architecture'], r['target'], r['neuron_idx'], r['probe_type']))
        if verbose:
            print(f"  Resuming: loaded {len(rows)} rows from checkpoint, "
                  f"skipping {len(completed_keys)} completed combos")

    for arch_name in architectures:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Architecture: {arch_name}")
            print(f"{'='*70}")

        # Load hidden states for this architecture
        try:
            hs_data = load_hidden_states(arch_name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        trained_hidden = hs_data['trained_hidden']
        untrained_hidden = hs_data['untrained_hidden']
        trial_ids = hs_data['trial_ids']

        if verbose:
            print(f"  Trained hidden: {trained_hidden.shape}")
            print(f"  Untrained hidden: {untrained_hidden.shape}")

        for target_name in target_names:
            if target_name not in all_targets:
                if verbose:
                    print(f"  SKIP target {target_name}: not in loaded targets")
                continue

            targets = all_targets[target_name]
            target_info = TARGET_REGISTRY[target_name]

            # Verify window count alignment
            if targets.shape[0] != trained_hidden.shape[0]:
                if verbose:
                    print(f"  SKIP {target_name}: window count mismatch "
                          f"(targets={targets.shape[0]}, "
                          f"hidden={trained_hidden.shape[0]})")
                continue

            for neuron_idx in range(min(n_neurons, targets.shape[2])):
                # Skip if already completed (resume mode)
                ridge_key = (arch_name, target_name, neuron_idx, 'ridge')
                if ridge_key in completed_keys:
                    combo_i += 3  # ridge + mlp_1 + mlp_2
                    continue

                # --- Ridge: full selectivity (permutation null) ---
                combo_i += 1
                if verbose and combo_i % 20 == 0:
                    print(f"  Progress: {combo_i}/{total_combos} "
                          f"({target_name}, neuron={neuron_idx})")

                sel_result = compute_selectivity(
                    trained_hidden, targets, trial_ids,
                    probe_type='ridge', neuron_idx=neuron_idx,
                    n_permutations=n_permutations, device=device,
                    verbose=False,
                )

                untrained_ridge = temporal_block_cv(
                    untrained_hidden, targets, trial_ids,
                    probe_type='ridge', neuron_idx=neuron_idx,
                    device=device, verbose=False,
                )

                r2_trained = sel_result['r2_real']
                r2_untrained = untrained_ridge['r2_mean']

                rows.append({
                    'architecture': arch_name,
                    'target': target_name,
                    'tier': target_info['tier'],
                    'group': target_info['group'],
                    'timescale_ms': target_info['timescale_ms'],
                    'neuron_idx': neuron_idx,
                    'probe_type': 'ridge',
                    'r2_trained': r2_trained,
                    'r2_untrained': r2_untrained,
                    'delta_r2': r2_trained - r2_untrained,
                    'selectivity': sel_result['selectivity'],
                    'p_value': sel_result['p_value'],
                    'r2_null_mean': sel_result['r2_null_mean'],
                    'r2_null_std': sel_result['r2_null_std'],
                    'block_size': sel_result['block_size'],
                    'tau_e': sel_result['tau_e'],
                    'pearson_trained': sel_result['real_cv_result']['pearson_mean'],
                    'pearson_untrained': untrained_ridge['pearson_mean'],
                })

                # --- MLP probes: R² only (no permutations) ---
                # Selectivity already established by Ridge. MLPs test
                # for nonlinear encoding (MLP R² > Ridge R²).
                for mlp_type in ('mlp_1', 'mlp_2'):
                    combo_i += 1

                    trained_result = temporal_block_cv(
                        trained_hidden, targets, trial_ids,
                        probe_type=mlp_type, neuron_idx=neuron_idx,
                        device=device, verbose=False,
                    )

                    untrained_result = temporal_block_cv(
                        untrained_hidden, targets, trial_ids,
                        probe_type=mlp_type, neuron_idx=neuron_idx,
                        device=device, verbose=False,
                    )

                    r2_t = trained_result['r2_mean']
                    r2_u = untrained_result['r2_mean']

                    rows.append({
                        'architecture': arch_name,
                        'target': target_name,
                        'tier': target_info['tier'],
                        'group': target_info['group'],
                        'timescale_ms': target_info['timescale_ms'],
                        'neuron_idx': neuron_idx,
                        'probe_type': mlp_type,
                        'r2_trained': r2_t,
                        'r2_untrained': r2_u,
                        'delta_r2': r2_t - r2_u,
                        # Reuse Ridge selectivity for MLP rows
                        'selectivity': sel_result['selectivity'],
                        'p_value': sel_result['p_value'],
                        'r2_null_mean': sel_result['r2_null_mean'],
                        'r2_null_std': sel_result['r2_null_std'],
                        'block_size': sel_result['block_size'],
                        'tau_e': sel_result['tau_e'],
                        'pearson_trained': trained_result['pearson_mean'],
                        'pearson_untrained': untrained_result['pearson_mean'],
                    })

                # Incremental checkpoint save
                if combo_i % SAVE_EVERY == 0 and len(rows) > 0:
                    pd.DataFrame(rows).to_csv(
                        str(incremental_path), index=False)
                    if verbose:
                        print(f"  [checkpoint] Saved {len(rows)} rows "
                              f"to {incremental_path.name}")

    results_df = pd.DataFrame(rows)

    # Save
    output_path = OUTPUT_DIR / 'a_r3b_full_results.csv'
    results_df.to_csv(str(output_path), index=False)

    if verbose:
        print(f"\nPhase 4 complete: {len(rows)} results")
        print(f"Saved to: {output_path}")
        _print_summary(results_df)

    return results_df, output_path


def _print_summary(df):
    """Print a quick overview of the results."""
    if df.empty:
        print("  (no results)")
        return

    print(f"\n--- Quick Summary ---")
    for arch in df['architecture'].unique():
        adf = df[df['architecture'] == arch]
        print(f"\n  {arch}:")
        for probe in adf['probe_type'].unique():
            pdf = adf[adf['probe_type'] == probe]
            print(f"    {probe}:")
            print(f"      R²_trained: {pdf['r2_trained'].mean():.4f} "
                  f"(+/- {pdf['r2_trained'].std():.4f})")
            print(f"      delta_R²:   {pdf['delta_r2'].mean():.4f}")
            print(f"      selectivity: {pdf['selectivity'].mean():.4f}")
            sig = (pdf['p_value'] < 0.05).sum()
            print(f"      significant (p<0.05): {sig}/{len(pdf)}")


def run_quick_validation(arch_name='volterra', target_name='gabab_per_tc',
                         n_neurons=1, n_permutations=10,
                         device='cpu', verbose=True):
    """Quick validation run on a single arch/target for sanity checking.

    Uses Volterra + GABA_B (known recoverable from A-R3) as default
    to verify the pipeline produces reasonable R² values.
    """
    if verbose:
        print("=" * 60)
        print("QUICK VALIDATION RUN")
        print(f"  arch={arch_name}, target={target_name}, "
              f"n_neurons={n_neurons}, n_perms={n_permutations}")
        print("=" * 60)

    return run_pipeline(
        architectures=[arch_name],
        target_names=[target_name],
        n_neurons=n_neurons,
        n_permutations=n_permutations,
        device=device,
        verbose=verbose,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', nargs='*', default=None)
    parser.add_argument('--target', nargs='*', default=None)
    parser.add_argument('--probe', nargs='*', default=None)
    parser.add_argument('--n-neurons', type=int, default=None)
    parser.add_argument('--n-perms', type=int, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation run (1 arch, 1 target, 10 perms)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from incremental checkpoint')
    args = parser.parse_args()

    if args.quick:
        run_quick_validation(device=args.device)
    else:
        run_pipeline(
            architectures=args.arch,
            target_names=args.target,
            probe_types=args.probe,
            n_neurons=args.n_neurons,
            n_permutations=args.n_perms,
            device=args.device,
            resume=args.resume,
        )
