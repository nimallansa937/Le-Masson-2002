"""
Progressive replacement selection logic.

Selects which TC neurons to replace with the validated computational model.
Supports multiple strategies: random, hub-first, hub-last, spatial cluster.
"""

import numpy as np


def select_replacement_indices(n_tc, fraction, strategy='random',
                               nrt_to_tc=None, seed=42):
    """Select which TC neurons to replace.

    Parameters
    ----------
    n_tc : int
        Total number of TC neurons.
    fraction : float
        Fraction to replace (0.0 to 1.0).
    strategy : str
        'random': random selection (primary).
        'hub_first': most-connected TC neurons first.
        'hub_last': least-connected TC neurons first.
        'spatial_cluster': contiguous block starting at index 0.
    nrt_to_tc : ndarray (n_tc, n_nrt), optional
        Connectivity matrix. Required for hub strategies.
    seed : int
        Random seed (for random strategy).

    Returns
    -------
    indices : ndarray of int
        Indices of TC neurons to replace.
    """
    n_replace = int(round(n_tc * fraction))
    n_replace = min(n_replace, n_tc)

    if n_replace == 0:
        return np.array([], dtype=int)
    if n_replace == n_tc:
        return np.arange(n_tc)

    if strategy == 'random':
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_tc, n_replace, replace=False))

    elif strategy == 'hub_first':
        if nrt_to_tc is None:
            raise ValueError("nrt_to_tc required for hub_first strategy")
        degree = nrt_to_tc.sum(axis=1)  # total GABA inputs per TC
        return np.argsort(degree)[-n_replace:]  # highest degree first

    elif strategy == 'hub_last':
        if nrt_to_tc is None:
            raise ValueError("nrt_to_tc required for hub_last strategy")
        degree = nrt_to_tc.sum(axis=1)
        return np.argsort(degree)[:n_replace]  # lowest degree first

    elif strategy == 'spatial_cluster':
        return np.arange(n_replace)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


REPLACEMENT_FRACTIONS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
                          0.60, 0.70, 0.80, 0.90, 1.00]

REPLACEMENT_FRACTIONS_COARSE = [0.0, 0.25, 0.50, 0.75, 1.00]
