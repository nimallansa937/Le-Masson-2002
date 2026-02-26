"""
Phase 3: Block Permutation Selectivity Controls.

For every R² value, compute a matched null distribution via block-permuted
targets. Block permutation preserves autocorrelation structure while destroying
the temporal alignment between hidden states and targets.

Selectivity = R²_real - mean(R²_control)
p_value = (sum(R²_control >= R²_real) + 1) / (n_perms + 1)

ADAPTIVE BLOCK SIZE: Compute empirical autocorrelation of each target trace
and set block_size = max(50, 3 * tau_e) where tau_e is the lag at which the
ACF drops below 1/e. This ensures blocks are long enough to preserve the
temporal structure that could produce spurious correlations.
"""

import numpy as np
from a_r3b_reanalysis.phase2_probes import temporal_block_cv


# ============================================================
# Autocorrelation-based adaptive block size
# ============================================================

def estimate_autocorrelation_decay(trace, max_lag=500):
    """Estimate the autocorrelation decay timescale of a signal.

    Parameters
    ----------
    trace : ndarray (T,)
        1D signal.
    max_lag : int
        Maximum lag to compute ACF.

    Returns
    -------
    tau_e : int
        Lag at which ACF drops below 1/e (~0.368).
        If ACF never drops below 1/e within max_lag, returns max_lag.
    """
    trace = trace - np.mean(trace)
    var = np.var(trace)
    if var < 1e-10:
        return 1  # Constant signal

    T = len(trace)
    max_lag = min(max_lag, T // 2)
    threshold = 1.0 / np.e  # ~0.368

    for lag in range(1, max_lag):
        acf = np.mean(trace[:T - lag] * trace[lag:]) / var
        if acf < threshold:
            return lag

    return max_lag


def compute_adaptive_block_size(targets, neuron_idx=0, min_block=50):
    """Determine block size from target autocorrelation.

    The block size must be long enough to preserve the temporal structure
    that could produce spurious correlation. Using 3 * tau_e ensures
    blocks span ~3 autocorrelation times — shuffling these blocks
    genuinely disrupts the signal-hidden alignment.

    Parameters
    ----------
    targets : ndarray (n_windows, window_bins, n_neurons)
    neuron_idx : int
    min_block : int

    Returns
    -------
    block_size : int
    tau_e : int
    """
    # Concatenate all windows for this neuron
    trace = targets[:, :, neuron_idx].ravel()

    tau_e = estimate_autocorrelation_decay(trace)
    block_size = max(min_block, 3 * tau_e)

    return int(block_size), int(tau_e)


# ============================================================
# Block Permutation
# ============================================================

def block_permute_targets(targets, block_size, rng):
    """Block-permute target arrays along the time axis within each window.

    Preserves autocorrelation within blocks but destroys the temporal
    alignment between hidden states and the biological target.

    Parameters
    ----------
    targets : ndarray (n_windows, window_bins, n_neurons)
    block_size : int
    rng : np.random.Generator

    Returns
    -------
    permuted : ndarray (n_windows, window_bins, n_neurons)
    """
    n_windows, window_bins, n_neurons = targets.shape
    permuted = np.empty_like(targets)

    for w in range(n_windows):
        # Split this window's target into blocks
        n_blocks = max(1, window_bins // block_size)
        # Create block indices
        block_starts = list(range(0, window_bins, block_size))

        # Extract blocks
        blocks = []
        for bs in block_starts:
            be = min(bs + block_size, window_bins)
            blocks.append(targets[w, bs:be, :].copy())

        # Shuffle block order
        perm_order = rng.permutation(len(blocks))
        shuffled_blocks = [blocks[i] for i in perm_order]

        # Reassemble
        idx = 0
        for blk in shuffled_blocks:
            blk_len = blk.shape[0]
            permuted[w, idx:idx + blk_len, :] = blk
            idx += blk_len

    return permuted


# ============================================================
# Selectivity Computation
# ============================================================

def compute_selectivity(hidden_states, targets, trial_ids,
                        probe_type='ridge', neuron_idx=0,
                        n_permutations=None, device='cpu',
                        verbose=False):
    """Compute selectivity via block-permutation null distribution.

    For each permutation:
      1. Block-permute the target traces (preserving autocorrelation)
      2. Re-run temporal_block_cv with the permuted targets
      3. Collect null R² values

    Then: selectivity = R²_real - mean(R²_null)
          p_value = (sum(R²_null >= R²_real) + 1) / (n_perms + 1)

    Parameters
    ----------
    hidden_states : ndarray (n_windows, window_bins, hidden_dim)
    targets : ndarray (n_windows, window_bins, n_neurons)
    trial_ids : ndarray (n_windows,)
    probe_type : str
    neuron_idx : int
    n_permutations : int, optional
    device : str
    verbose : bool

    Returns
    -------
    result : dict
        'r2_real': float — actual R² from temporal_block_cv
        'r2_null_mean': float — mean null R²
        'r2_null_std': float — std of null R²
        'r2_null_values': list — all null R² values
        'selectivity': float — r2_real - r2_null_mean
        'p_value': float — permutation p-value
        'block_size': int — adaptive block size used
        'tau_e': int — autocorrelation decay lag
    """
    from a_r3b_reanalysis.config import SELECTIVITY_N_PERMS_QUICK

    if n_permutations is None:
        n_permutations = SELECTIVITY_N_PERMS_QUICK

    # Step 1: Compute actual R²
    real_result = temporal_block_cv(
        hidden_states, targets, trial_ids,
        probe_type=probe_type, neuron_idx=neuron_idx,
        device=device, verbose=False,
    )
    r2_real = real_result['r2_mean']

    if verbose:
        print(f"    Real R² = {r2_real:.4f}")

    # Step 2: Adaptive block size
    block_size, tau_e = compute_adaptive_block_size(targets, neuron_idx)

    if verbose:
        print(f"    tau_e = {tau_e}, block_size = {block_size}")

    # Step 3: Null distribution via block permutation
    null_r2_values = []
    rng = np.random.default_rng(seed=42)

    for perm_i in range(n_permutations):
        # Block-permute targets
        targets_perm = block_permute_targets(targets, block_size, rng)

        # Rerun CV with permuted targets
        perm_result = temporal_block_cv(
            hidden_states, targets_perm, trial_ids,
            probe_type=probe_type, neuron_idx=neuron_idx,
            device=device, verbose=False,
        )
        null_r2_values.append(perm_result['r2_mean'])

        if verbose and (perm_i + 1) % 10 == 0:
            print(f"    Permutation {perm_i + 1}/{n_permutations}: "
                  f"null R² = {perm_result['r2_mean']:.4f}")

    null_r2 = np.array(null_r2_values)

    # Step 4: Compute selectivity and p-value
    selectivity = r2_real - float(np.mean(null_r2))
    p_value = (np.sum(null_r2 >= r2_real) + 1) / (n_permutations + 1)

    return {
        'r2_real': r2_real,
        'r2_null_mean': float(np.mean(null_r2)),
        'r2_null_std': float(np.std(null_r2)),
        'r2_null_values': null_r2_values,
        'selectivity': selectivity,
        'p_value': float(p_value),
        'block_size': block_size,
        'tau_e': tau_e,
        'real_cv_result': real_result,
    }
