"""
Network topology builder for TC-nRt population circuit.

Builds sparse random connectivity matrices, normalizes synaptic weights
by convergence, and assigns conduction delays.

Connectivity rules (from population_replacement_guide.md):
  TC -> nRt: AMPA, p=0.20, total G_AMPA = 20 nS per nRt
  nRt -> TC: GABA_A + GABA_B, p=0.20, total G_GABA = gaba_gmax per TC
  nRt -> nRt: GABA_A only, p=0.15, G = 5-10 nS per nRt (no autapses)
  Retinal -> TC: independent per TC, G_AMPA = 28 nS (not in connectivity matrix)
  No TC -> TC connections.
"""

import numpy as np


def build_connectivity(n_tc, n_nrt, p_tc_nrt=0.20, p_nrt_tc=0.20,
                       p_nrt_nrt=0.15, seed=42):
    """Build sparse random connectivity matrices.

    Parameters
    ----------
    n_tc : int
        Number of TC neurons.
    n_nrt : int
        Number of nRt neurons.
    p_tc_nrt : float
        TC -> nRt connection probability.
    p_nrt_tc : float
        nRt -> TC connection probability.
    p_nrt_nrt : float
        nRt -> nRt connection probability (no autapses).
    seed : int
        Random seed.

    Returns
    -------
    tc_to_nrt : ndarray (n_nrt, n_tc)
        Binary matrix. tc_to_nrt[j, i] = 1 means TC_i -> nRt_j.
    nrt_to_tc : ndarray (n_tc, n_nrt)
        Binary matrix. nrt_to_tc[i, j] = 1 means nRt_j -> TC_i.
    nrt_to_nrt : ndarray (n_nrt, n_nrt)
        Binary matrix. nrt_to_nrt[j, i] = 1 means nRt_i -> nRt_j. No autapses.
    """
    rng = np.random.default_rng(seed)

    tc_to_nrt = (rng.random((n_nrt, n_tc)) < p_tc_nrt).astype(np.int8)
    nrt_to_tc = (rng.random((n_tc, n_nrt)) < p_nrt_tc).astype(np.int8)
    nrt_to_nrt = (rng.random((n_nrt, n_nrt)) < p_nrt_nrt).astype(np.int8)
    np.fill_diagonal(nrt_to_nrt, 0)  # no autapses

    return tc_to_nrt, nrt_to_tc, nrt_to_nrt


def normalize_weights(connectivity_matrix, total_conductance):
    """Compute per-synapse weights normalized by convergence.

    Each postsynaptic neuron receives total_conductance nS total,
    divided equally among its presynaptic inputs.

    Parameters
    ----------
    connectivity_matrix : ndarray (n_post, n_pre)
        Binary connectivity.
    total_conductance : float
        Target total conductance per postsynaptic neuron (nS).

    Returns
    -------
    weights : ndarray (n_post, n_pre)
        Per-synapse conductance in nS. weights[j, i] is the conductance
        from pre_i to post_j.
    """
    n_inputs = connectivity_matrix.sum(axis=1)  # convergence per post neuron
    n_inputs_safe = np.maximum(n_inputs, 1)
    per_synapse = total_conductance / n_inputs_safe
    weights = connectivity_matrix.astype(np.float64) * per_synapse[:, np.newaxis]
    return weights


def assign_delays(connectivity_matrix, mean_ms=1.0, std_ms=0.2,
                  min_ms=0.5, seed=42):
    """Assign conduction delays to existing connections.

    Parameters
    ----------
    connectivity_matrix : ndarray
        Binary connectivity matrix.
    mean_ms : float
        Mean conduction delay (ms).
    std_ms : float
        Std of conduction delay (ms).
    min_ms : float
        Minimum delay (ms), clips below this.
    seed : int
        Random seed.

    Returns
    -------
    delays_ms : ndarray
        Delay matrix (ms). Zero where no connection exists.
    """
    rng = np.random.default_rng(seed + 1000)
    delays = rng.normal(mean_ms, std_ms, size=connectivity_matrix.shape)
    delays = np.clip(delays, min_ms, None)
    delays *= connectivity_matrix  # zero where no connection
    return delays


def connectivity_stats(tc_to_nrt, nrt_to_tc, nrt_to_nrt):
    """Print connectivity statistics for debugging."""
    n_nrt, n_tc = tc_to_nrt.shape

    print(f"Network: {n_tc} TC, {n_nrt} nRt")
    print(f"TC->nRt: {tc_to_nrt.sum()} connections, "
          f"p_eff={tc_to_nrt.sum() / (n_tc * n_nrt):.3f}, "
          f"mean convergence={tc_to_nrt.sum(axis=1).mean():.1f}")
    print(f"nRt->TC: {nrt_to_tc.sum()} connections, "
          f"p_eff={nrt_to_tc.sum() / (n_tc * n_nrt):.3f}, "
          f"mean convergence={nrt_to_tc.sum(axis=1).mean():.1f}")
    print(f"nRt->nRt: {nrt_to_nrt.sum()} connections, "
          f"p_eff={nrt_to_nrt.sum() / (n_nrt * (n_nrt - 1)):.3f}, "
          f"mean convergence={nrt_to_nrt.sum(axis=1).mean():.1f}")
