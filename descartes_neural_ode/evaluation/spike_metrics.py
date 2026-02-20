"""
Spike Train Evaluation Metrics for DESCARTES-NeuralODE.

Measures output-level fidelity between model predictions and
biological ground truth TC neuron responses:
  1. Spike train correlation (Pearson on smoothed rates)
  2. Victor-Purpura distance (temporal precision)
  3. Per-neuron and population-level summaries

These measure FUNCTIONAL equivalence — does the model produce
the same input→output mapping as the biological circuit?

Adapted from rung3/evaluation/output_metrics.py but simplified
for the DESCARTES search loop (no bifurcation or coherence tests —
those are in bifurcation_test.py).
"""
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional


def spike_train_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    smooth_bins: int = 5,
) -> Tuple[float, List[float]]:
    """Pearson correlation between smoothed firing rates.

    Parameters
    ----------
    y_true : ndarray (seq_len, n_neurons) or (n_neurons, seq_len)
        Ground truth smoothed rates.
    y_pred : ndarray same shape
        Model predictions.
    smooth_bins : int
        Moving average kernel width (in timestep bins).

    Returns
    -------
    mean_corr : float
        Mean correlation across neurons.
    per_neuron : list of float
    """
    # Ensure (seq_len, n_neurons) orientation
    if y_true.ndim == 2 and y_true.shape[0] < y_true.shape[1]:
        y_true = y_true.T
        y_pred = y_pred.T

    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]

    n_neurons = y_true.shape[1]
    per_neuron = []

    for i in range(n_neurons):
        t = y_true[:, i]
        p = y_pred[:, i]

        # Smoothing
        if smooth_bins > 1:
            kernel = np.ones(smooth_bins) / smooth_bins
            t = np.convolve(t, kernel, mode='same')
            p = np.convolve(p, kernel, mode='same')

        if np.std(t) < 1e-10 or np.std(p) < 1e-10:
            per_neuron.append(0.0)
        else:
            c = np.corrcoef(t, p)[0, 1]
            per_neuron.append(float(c) if not np.isnan(c) else 0.0)

    return float(np.mean(per_neuron)), per_neuron


def victor_purpura_distance(
    spike_times_a: np.ndarray,
    spike_times_b: np.ndarray,
    cost: float = 2.0,
) -> float:
    """Victor-Purpura distance between two spike trains.

    The VP distance is the minimum cost of transforming one spike train
    into another via insertions (cost 1), deletions (cost 1), and
    shifts (cost = q * |dt| where dt is the time shift).

    Parameters
    ----------
    spike_times_a : ndarray
        Spike times in seconds.
    spike_times_b : ndarray
        Spike times in seconds.
    cost : float
        Cost parameter q (1/s). Higher = more sensitive to timing.

    Returns
    -------
    distance : float
    """
    n_a = len(spike_times_a)
    n_b = len(spike_times_b)

    if n_a == 0:
        return float(n_b)
    if n_b == 0:
        return float(n_a)

    D = np.zeros((n_a + 1, n_b + 1))
    D[:, 0] = np.arange(n_a + 1)
    D[0, :] = np.arange(n_b + 1)

    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            shift_cost = cost * abs(spike_times_a[i - 1] - spike_times_b[j - 1])
            D[i, j] = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + shift_cost,
            )

    return float(D[n_a, n_b])


def mean_vp_distance(
    spikes_true: List[np.ndarray],
    spikes_pred: List[np.ndarray],
    cost: float = 2.0,
) -> Tuple[float, List[float]]:
    """Mean VP distance across all neurons.

    Parameters
    ----------
    spikes_true : list of ndarray
        Per-neuron spike times (biological).
    spikes_pred : list of ndarray
        Per-neuron spike times (model).
    cost : float

    Returns
    -------
    mean_dist : float
    per_neuron : list of float
    """
    per_neuron = []
    for s_true, s_pred in zip(spikes_true, spikes_pred):
        d = victor_purpura_distance(np.asarray(s_true), np.asarray(s_pred), cost)
        per_neuron.append(d)
    return float(np.mean(per_neuron)) if per_neuron else 0.0, per_neuron


def rate_to_spike_times(
    rates: np.ndarray,
    threshold: Optional[float] = None,
    dt_s: float = 0.001,
) -> List[np.ndarray]:
    """Convert rate predictions to approximate spike times.

    Parameters
    ----------
    rates : ndarray (seq_len, n_neurons)
        Predicted firing rates (0-1).
    threshold : float or None
        If None, uses mean + 1*std per neuron.
    dt_s : float
        Time bin width in seconds.

    Returns
    -------
    spike_times : list of ndarray
        Per-neuron spike times in seconds.
    """
    if rates.ndim == 1:
        rates = rates[:, np.newaxis]

    n_neurons = rates.shape[1]
    spike_times = []

    for i in range(n_neurons):
        r = rates[:, i]
        if threshold is None:
            thr = np.mean(r) + np.std(r)
        else:
            thr = threshold

        # Find threshold crossings (rising edge)
        above = r > thr
        crossings = np.diff(above.astype(int))
        spike_indices = np.where(crossings == 1)[0] + 1
        spike_times.append(spike_indices.astype(float) * dt_s)

    return spike_times


def evaluate_spike_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    smooth_bins: int = 5,
    vp_cost: float = 2.0,
    dt_s: float = 0.001,
) -> Dict:
    """Complete spike metrics evaluation.

    Parameters
    ----------
    y_true : ndarray (seq_len, n_neurons)
    y_pred : ndarray (seq_len, n_neurons)
    smooth_bins : int
    vp_cost : float
    dt_s : float

    Returns
    -------
    metrics : dict
    """
    mean_corr, per_neuron_corr = spike_train_correlation(
        y_true, y_pred, smooth_bins
    )

    # VP distance on converted spike trains
    spikes_true = rate_to_spike_times(y_true, dt_s=dt_s)
    spikes_pred = rate_to_spike_times(y_pred, dt_s=dt_s)
    mean_vp, per_neuron_vp = mean_vp_distance(spikes_true, spikes_pred, vp_cost)

    return {
        'spike_correlation_mean': mean_corr,
        'spike_correlation_per_neuron': per_neuron_corr,
        'vp_distance_mean': mean_vp,
        'vp_distance_per_neuron': per_neuron_vp,
        'n_neurons': y_true.shape[1] if y_true.ndim == 2 else 1,
    }
