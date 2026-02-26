"""
Phase 0: Compute identifiable combination targets from existing ground truth.

From the A-R2 biological ground truth stored in HDF5 trial files, compute
the physically identifiable combinations that any system has informational
basis to represent. These serve as calibration targets for probing.

Tier 1 — Identifiable combinations:
    G_T(t) = g_T_bar * m_T(t)^2 * h_T(t)     # effective T-conductance
    G_h(t) = g_h_bar * m_H(t)                  # effective H-conductance
    I_T(t) = G_T(t) * (V(t) - E_Ca)           # T-current
    I_h(t) = G_h(t) * (V(t) - E_h)            # H-current
    I_GABA_A(t) = gaba_a(t) * (V(t) - E_GABA_A)
    I_GABA_B(t) = gaba_b(t) * (V(t) - E_GABA_B)

Tier 3 — Shape-normalized individual gates:
    z-scored and rank-transformed per-neuron versions of m_T, h_T, m_H
"""

import sys
import os
import numpy as np
from pathlib import Path
from scipy.stats import rankdata

# Ensure parent package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rung3.phase0_recording import load_trial_hdf5, list_trials
from a_r3b_reanalysis.config import (
    DATA_DIR, TARGETS_DIR, VAL_SEEDS,
    g_T_bar, g_h_bar, E_Ca, E_h, E_GABA_A, E_GABA_B,
    N_TC, WINDOW_SIZE_MS, WINDOW_STRIDE_MS, BIN_DT_MS,
)


def zscore_per_neuron(arr):
    """Z-score each neuron's trace independently.

    Parameters
    ----------
    arr : ndarray (n_neurons, T)

    Returns
    -------
    z : ndarray (n_neurons, T)
    """
    mu = arr.mean(axis=1, keepdims=True)
    sigma = arr.std(axis=1, keepdims=True)
    sigma[sigma < 1e-10] = 1.0
    return (arr - mu) / sigma


def rank_transform_per_neuron(arr):
    """Convert each neuron's trace to rank-order (0-1 scale).

    Parameters
    ----------
    arr : ndarray (n_neurons, T)

    Returns
    -------
    ranked : ndarray (n_neurons, T)
    """
    ranked = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        ranked[i] = rankdata(arr[i]) / len(arr[i])
    return ranked


def compute_trial_targets(trial_data):
    """Compute all identifiable targets for a single trial.

    Parameters
    ----------
    trial_data : dict
        Output from load_trial_hdf5(). Must contain 'intermediates' and 'V_tc'.

    Returns
    -------
    targets : dict
        Keys are target names from TARGET_REGISTRY, values are (n_neurons, T).
    """
    inter = trial_data['intermediates']
    V_tc = trial_data['V_tc']  # (n_tc, T)

    # Raw gating variables from HDF5
    tc_m_T = inter['tc_m_T']      # (20, T)
    tc_h_T = inter['tc_h_T']      # (20, T)
    tc_m_h = inter['tc_m_h']      # (20, T)
    gaba_a = inter['gabaa_per_tc']  # (20, T)
    gaba_b = inter['gabab_per_tc']  # (20, T)

    targets = {}

    # --- Tier 1: Identifiable combinations ---
    G_T = g_T_bar * (tc_m_T ** 2) * tc_h_T
    G_h = g_h_bar * tc_m_h
    targets['G_T'] = G_T
    targets['G_h'] = G_h

    # Ionic currents (require V_tc)
    targets['I_T'] = G_T * (V_tc - E_Ca)
    targets['I_h'] = G_h * (V_tc - E_h)
    targets['I_GABA_A'] = gaba_a * (V_tc - E_GABA_A)
    targets['I_GABA_B'] = gaba_b * (V_tc - E_GABA_B)

    # --- Tier 2: Raw individual gates (direct from HDF5) ---
    targets['tc_m_T'] = tc_m_T
    targets['tc_h_T'] = tc_h_T
    targets['tc_m_h'] = tc_m_h

    # --- Tier 3: Shape-normalized gates ---
    targets['tc_m_T_zscore'] = zscore_per_neuron(tc_m_T)
    targets['tc_h_T_zscore'] = zscore_per_neuron(tc_h_T)
    targets['tc_m_h_zscore'] = zscore_per_neuron(tc_m_h)
    targets['tc_m_T_rank'] = rank_transform_per_neuron(tc_m_T)
    targets['tc_h_T_rank'] = rank_transform_per_neuron(tc_h_T)
    targets['tc_m_h_rank'] = rank_transform_per_neuron(tc_m_h)

    # --- Existing synaptic conductances ---
    targets['gabaa_per_tc'] = gaba_a
    targets['gabab_per_tc'] = gaba_b

    return targets


def window_targets(targets, n_timepoints):
    """Window target arrays using the same scheme as rung3 preprocessing.

    Parameters
    ----------
    targets : dict
        Keys are target names, values are (n_neurons, T_full) at 1ms resolution.
    n_timepoints : int
        Full recording length in bins.

    Returns
    -------
    windowed : dict
        Keys are target names, values are (n_windows, window_bins, n_neurons).
    trial_window_count : int
        Number of windows extracted from this trial.
    """
    window_bins = int(WINDOW_SIZE_MS / BIN_DT_MS)
    stride_bins = int(WINDOW_STRIDE_MS / BIN_DT_MS)

    starts = list(range(0, n_timepoints - window_bins + 1, stride_bins))
    n_windows = len(starts)

    windowed = {}
    for key, arr in targets.items():
        # arr: (n_neurons, T_full) — trim to n_timepoints
        n_bins = min(arr.shape[1], n_timepoints)
        arr_t = arr[:, :n_bins].T  # (n_bins, n_neurons)

        w = np.zeros((n_windows, window_bins, arr.shape[0]), dtype=np.float32)
        for i, s in enumerate(starts):
            w[i] = arr_t[s:s + window_bins]
        windowed[key] = w

    return windowed, n_windows


def compute_all_targets(seeds=None, verbose=True):
    """Compute identifiable targets for all validation trials and save to disk.

    Parameters
    ----------
    seeds : list of int, optional
        Which seeds to process. Defaults to VAL_SEEDS.
    verbose : bool

    Returns
    -------
    output_dir : Path
    """
    if seeds is None:
        seeds = VAL_SEEDS

    TARGETS_DIR.mkdir(parents=True, exist_ok=True)

    all_trials = list_trials(str(DATA_DIR))
    selected = [t for t in all_trials if t['seed'] in seeds]

    if verbose:
        print(f"Phase 0: Computing identifiable targets for {len(selected)} trials")
        print(f"  Params: g_T={g_T_bar}, g_h={g_h_bar}, "
              f"E_Ca={E_Ca}, E_h={E_h}, E_GABA_A={E_GABA_A}, E_GABA_B={E_GABA_B}")

    trial_metadata = []

    for i, trial_info in enumerate(selected):
        if verbose:
            print(f"  [{i+1}/{len(selected)}] "
                  f"gaba={trial_info['gaba_gmax']}, seed={trial_info['seed']}")

        data = load_trial_hdf5(trial_info['filepath'])

        if not data.get('intermediates'):
            if verbose:
                print(f"    SKIP: no intermediates in {trial_info['filepath']}")
            continue

        # Compute all target arrays
        targets = compute_trial_targets(data)

        # Determine n_timepoints from the recording
        n_timepoints = data['V_tc'].shape[1]

        # Window targets (same scheme as rung3)
        windowed, n_windows = window_targets(targets, n_timepoints)

        # Save
        fname = (f"targets_gaba{trial_info['gaba_gmax']:05.1f}"
                 f"_seed{trial_info['seed']}.npz")
        outpath = TARGETS_DIR / fname

        save_dict = {}
        for key, arr in windowed.items():
            save_dict[key] = arr
        save_dict['gaba_gmax'] = trial_info['gaba_gmax']
        save_dict['seed'] = trial_info['seed']
        save_dict['n_windows'] = n_windows

        np.savez_compressed(str(outpath), **save_dict)

        trial_metadata.append({
            'filepath': str(outpath),
            'gaba_gmax': trial_info['gaba_gmax'],
            'seed': trial_info['seed'],
            'n_windows': n_windows,
        })

        if verbose and i == 0:
            _sanity_check(targets, windowed)

    if verbose:
        total_windows = sum(m['n_windows'] for m in trial_metadata)
        print(f"\nPhase 0 complete: {len(trial_metadata)} trials, "
              f"{total_windows} total windows")
        print(f"Saved to: {TARGETS_DIR}")

    return TARGETS_DIR


def _sanity_check(targets, windowed):
    """Print sanity checks for the first trial."""
    G_T = targets['G_T']
    G_h = targets['G_h']
    I_T = targets['I_T']

    print(f"    Sanity checks:")
    print(f"      G_T range: [{G_T.min():.4f}, {G_T.max():.4f}] "
          f"(expected [0, {g_T_bar}])")
    print(f"      G_h range: [{G_h.min():.6f}, {G_h.max():.6f}] "
          f"(expected [0, {g_h_bar}])")
    print(f"      I_T range: [{I_T.min():.2f}, {I_T.max():.2f}] "
          f"(expected mostly negative — inward Ca current)")

    for key, arr in windowed.items():
        print(f"      {key}: windowed shape = {arr.shape}")
        break  # Just show one


if __name__ == '__main__':
    compute_all_targets(verbose=True)
