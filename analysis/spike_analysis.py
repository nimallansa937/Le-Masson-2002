"""
Spike detection and cross-correlation analysis tools.

Implements the contribution index (CI) and correlation index (CC)
from Le Masson et al. 2002 Fig 3.
"""

import numpy as np


def detect_spikes(V, t, threshold=-20.0):
    """Detect spikes from membrane potential trace.

    Parameters
    ----------
    V : np.ndarray
        Membrane potential trace (mV).
    t : np.ndarray
        Time array (s).
    threshold : float
        Spike detection threshold (mV).

    Returns
    -------
    spike_times : np.ndarray
        Times of positive-going threshold crossings (s).
    """
    crossings = np.where((V[:-1] < threshold) & (V[1:] >= threshold))[0]
    return t[crossings]


def cross_correlogram(spike_times_a, spike_times_b, bin_ms=2.0, window_ms=150.0):
    """Compute cross-correlogram between two spike trains.

    Counts how many spikes in train B occur at delay tau after
    each spike in train A.

    Parameters
    ----------
    spike_times_a : np.ndarray
        Reference spike train times (s).
    spike_times_b : np.ndarray
        Target spike train times (s).
    bin_ms : float
        Bin width in ms.
    window_ms : float
        Half-window in ms (correlogram spans -window to +window).

    Returns
    -------
    bins_ms : np.ndarray
        Bin centres in ms.
    counts : np.ndarray
        Spike counts per bin.
    """
    bin_s = bin_ms / 1000.0
    window_s = window_ms / 1000.0
    n_bins = int(2 * window_ms / bin_ms) + 1
    edges = np.linspace(-window_s, window_s, n_bins + 1)
    counts = np.zeros(n_bins)

    for t_a in spike_times_a:
        # Find spikes in B within window of this spike in A
        diffs = spike_times_b - t_a
        mask = (diffs >= -window_s) & (diffs <= window_s)
        relevant = diffs[mask]
        if len(relevant) > 0:
            indices = np.digitize(relevant, edges) - 1
            indices = indices[(indices >= 0) & (indices < n_bins)]
            for idx in indices:
                counts[idx] += 1

    bins_ms_centres = (edges[:-1] + edges[1:]) / 2 * 1000.0
    return bins_ms_centres, counts


def contribution_index(retinal_spikes, tc_spikes, bin_ms=2.0, window_ms=150.0):
    """Contribution Index (CI).

    CI = peak of cross-correlation / N_tc_spikes
    Measures what fraction of TC spikes were driven by retinal input.
    High CI = reliable transfer.

    From Le Masson 2002 Fig 3d.
    """
    if len(tc_spikes) == 0:
        return 0.0
    _, counts = cross_correlogram(retinal_spikes, tc_spikes, bin_ms, window_ms)
    # Look at positive lags only (TC spikes after retinal spikes)
    n_bins = len(counts)
    mid = n_bins // 2
    positive_lag_counts = counts[mid:]
    peak = np.max(positive_lag_counts) if len(positive_lag_counts) > 0 else 0.0
    return peak / len(tc_spikes)


def correlation_index(retinal_spikes, tc_spikes, bin_ms=2.0, window_ms=150.0):
    """Correlation Index (CC).

    CC = peak of cross-correlation / N_retinal_spikes
    Measures what fraction of retinal spikes produced a TC spike.
    High CC = efficient transfer.

    From Le Masson 2002 Fig 3e.
    """
    if len(retinal_spikes) == 0:
        return 0.0
    _, counts = cross_correlogram(retinal_spikes, tc_spikes, bin_ms, window_ms)
    n_bins = len(counts)
    mid = n_bins // 2
    positive_lag_counts = counts[mid:]
    peak = np.max(positive_lag_counts) if len(positive_lag_counts) > 0 else 0.0
    return peak / len(retinal_spikes)


def spike_latency(retinal_spikes, tc_spikes, max_lag_ms=20.0):
    """Compute mean spike latency from retinal to TC spikes.

    For each retinal spike, finds the first TC spike within max_lag_ms.

    Returns
    -------
    mean_latency_ms : float
    latencies : list of float
        Individual latencies in ms.
    """
    max_lag_s = max_lag_ms / 1000.0
    latencies = []

    tc_idx = 0
    for t_ret in retinal_spikes:
        # Advance tc_idx to first spike >= t_ret
        while tc_idx < len(tc_spikes) and tc_spikes[tc_idx] < t_ret:
            tc_idx += 1
        if tc_idx < len(tc_spikes):
            lag = tc_spikes[tc_idx] - t_ret
            if lag <= max_lag_s:
                latencies.append(lag * 1000.0)  # convert to ms

    if len(latencies) == 0:
        return float('nan'), []
    return np.mean(latencies), latencies
