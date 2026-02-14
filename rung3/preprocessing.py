"""
Preprocessing pipeline for Rung 3.

Transforms raw simulation outputs (spike times, voltages) into windowed
input/output pairs for model training.

Input features:  20 retinal smoothed rates + 1 GABA scalar = 21D per timestep
Output targets:  20 TC smoothed rates
Windows:         2s (2000 bins at 1ms) with 500ms stride
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from rung3.config import (
    BIN_DT_MS, SMOOTH_SIGMA_MS, WINDOW_SIZE_MS, WINDOW_STRIDE_MS,
    INPUT_DIM, OUTPUT_DIM,
)


def bin_spike_trains(spike_times_list, duration_s, bin_dt_ms=BIN_DT_MS):
    """Convert spike times to binary matrix.

    Parameters
    ----------
    spike_times_list : list of ndarray
        Spike times in seconds per channel/neuron.
    duration_s : float
        Total duration.
    bin_dt_ms : float
        Bin width in ms.

    Returns
    -------
    binary : ndarray (n_channels, n_bins)
        Binary spike matrix (1 = spike in bin, 0 = no spike).
    """
    n_channels = len(spike_times_list)
    n_bins = int(duration_s * 1000.0 / bin_dt_ms)
    binary = np.zeros((n_channels, n_bins), dtype=np.float32)

    for i, spk in enumerate(spike_times_list):
        if len(spk) == 0:
            continue
        # Convert spike times (s) to bin indices
        bin_idx = (spk * 1000.0 / bin_dt_ms).astype(int)
        bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]
        binary[i, bin_idx] = 1.0

    return binary


def smooth_rates(binary, sigma_ms=SMOOTH_SIGMA_MS, bin_dt_ms=BIN_DT_MS):
    """Gaussian-smooth binary spike trains into firing rates.

    Parameters
    ----------
    binary : ndarray (n_channels, n_bins)
    sigma_ms : float
        Gaussian kernel σ in ms.
    bin_dt_ms : float

    Returns
    -------
    rates : ndarray (n_channels, n_bins)
        Smoothed firing rates (spikes/bin → continuous).
    """
    sigma_bins = sigma_ms / bin_dt_ms
    rates = np.zeros_like(binary, dtype=np.float32)
    for i in range(binary.shape[0]):
        rates[i] = gaussian_filter1d(binary[i].astype(np.float64),
                                      sigma=sigma_bins).astype(np.float32)
    return rates


def build_input_features(retinal_spike_times, gaba_gmax, duration_s,
                          bin_dt_ms=BIN_DT_MS, sigma_ms=SMOOTH_SIGMA_MS):
    """Build input feature matrix: 20 retinal rates + 1 GABA scalar.

    Parameters
    ----------
    retinal_spike_times : list of ndarray
        Per-channel retinal spike times (s).
    gaba_gmax : float
        GABA conductance for this trial (nS).
    duration_s : float
    bin_dt_ms : float
    sigma_ms : float

    Returns
    -------
    features : ndarray (n_bins, 21)
        Input features per timestep: [retinal_0, ..., retinal_19, gaba_scalar].
    """
    ret_binary = bin_spike_trains(retinal_spike_times, duration_s, bin_dt_ms)
    ret_rates = smooth_rates(ret_binary, sigma_ms, bin_dt_ms)

    n_bins = ret_rates.shape[1]

    # Normalize GABA to [0, 1] range (max ~74 nS)
    gaba_norm = np.float32(gaba_gmax / 74.0)
    gaba_channel = np.full((1, n_bins), gaba_norm, dtype=np.float32)

    # Stack: (21, n_bins) then transpose to (n_bins, 21)
    features = np.vstack([ret_rates, gaba_channel]).T

    return features


def build_output_targets(tc_spike_times, duration_s,
                          bin_dt_ms=BIN_DT_MS, sigma_ms=SMOOTH_SIGMA_MS):
    """Build output target matrix: 20 TC smoothed rates.

    Parameters
    ----------
    tc_spike_times : list of ndarray
        Per-neuron TC spike times (s).
    duration_s : float

    Returns
    -------
    targets : ndarray (n_bins, 20)
        Target firing rates per timestep.
    """
    tc_binary = bin_spike_trains(tc_spike_times, duration_s, bin_dt_ms)
    tc_rates = smooth_rates(tc_binary, sigma_ms, bin_dt_ms)

    return tc_rates.T  # (n_bins, 20)


def build_output_binary(tc_spike_times, duration_s, bin_dt_ms=BIN_DT_MS):
    """Build binary spike target matrix (for BCE loss).

    Returns
    -------
    binary : ndarray (n_bins, 20)
    """
    tc_binary = bin_spike_trains(tc_spike_times, duration_s, bin_dt_ms)
    return tc_binary.T  # (n_bins, 20)


def create_windows(features, targets, binary_targets=None,
                    window_ms=WINDOW_SIZE_MS, stride_ms=WINDOW_STRIDE_MS,
                    bin_dt_ms=BIN_DT_MS):
    """Slice time series into overlapping windows.

    Parameters
    ----------
    features : ndarray (n_bins, input_dim)
    targets : ndarray (n_bins, output_dim)
    binary_targets : ndarray (n_bins, output_dim), optional
    window_ms : int
    stride_ms : int

    Returns
    -------
    X_windows : ndarray (n_windows, window_bins, input_dim)
    Y_rate_windows : ndarray (n_windows, window_bins, output_dim)
    Y_binary_windows : ndarray or None (n_windows, window_bins, output_dim)
    """
    window_bins = int(window_ms / bin_dt_ms)
    stride_bins = int(stride_ms / bin_dt_ms)
    n_bins = features.shape[0]

    starts = list(range(0, n_bins - window_bins + 1, stride_bins))
    n_windows = len(starts)

    X_windows = np.zeros((n_windows, window_bins, features.shape[1]),
                          dtype=np.float32)
    Y_rate_windows = np.zeros((n_windows, window_bins, targets.shape[1]),
                               dtype=np.float32)

    for i, s in enumerate(starts):
        X_windows[i] = features[s:s + window_bins]
        Y_rate_windows[i] = targets[s:s + window_bins]

    Y_binary_windows = None
    if binary_targets is not None:
        Y_binary_windows = np.zeros(
            (n_windows, window_bins, binary_targets.shape[1]),
            dtype=np.float32)
        for i, s in enumerate(starts):
            Y_binary_windows[i] = binary_targets[s:s + window_bins]

    return X_windows, Y_rate_windows, Y_binary_windows


def preprocess_trial(trial_data):
    """Full preprocessing pipeline for one trial.

    Parameters
    ----------
    trial_data : dict
        From load_trial_hdf5().

    Returns
    -------
    X : ndarray (n_windows, window_bins, 21)
    Y_rate : ndarray (n_windows, window_bins, 20)
    Y_binary : ndarray (n_windows, window_bins, 20)
    intermediates_windows : dict or None
        Windowed intermediate variables (if present in trial_data).
    """
    features = build_input_features(
        trial_data['retinal_spike_times'],
        trial_data['gaba_gmax'],
        trial_data['duration_s'])

    targets = build_output_targets(
        trial_data['tc_spike_times'],
        trial_data['duration_s'])

    binary = build_output_binary(
        trial_data['tc_spike_times'],
        trial_data['duration_s'])

    X, Y_rate, Y_binary = create_windows(features, targets, binary)

    # Window intermediates too if present
    intermediates_windows = None
    if trial_data.get('intermediates'):
        intermediates_windows = {}
        inter = trial_data['intermediates']
        n_timepoints = inter['tc_m_T'].shape[1]
        window_bins = int(WINDOW_SIZE_MS / BIN_DT_MS)
        stride_bins = int(WINDOW_STRIDE_MS / BIN_DT_MS)

        # Intermediates are at record_dt (1ms) — same as our bin_dt_ms
        # Shape: (n_neurons, n_timepoints) → window as (n_windows, window_bins, n_neurons)
        for key, arr in inter.items():
            # arr: (n_neurons, n_timepoints), need min(n_timepoints, n_bins)
            n_bins = min(arr.shape[1], features.shape[0])
            arr_t = arr[:, :n_bins].T  # (n_bins, n_neurons)
            starts = list(range(0, n_bins - window_bins + 1, stride_bins))
            windowed = np.zeros((len(starts), window_bins, arr.shape[0]),
                                dtype=np.float32)
            for i, s in enumerate(starts):
                windowed[i] = arr_t[s:s + window_bins]
            intermediates_windows[key] = windowed

    return X, Y_rate, Y_binary, intermediates_windows
