"""
Bifurcation Threshold Test for DESCARTES-NeuralODE.

Tests whether a model preserves the GABA bifurcation threshold
that defines the transition from tonic to oscillatory firing.

Biological target: 29 +/- 4 nS (from La Masson et al. 2002).

Method:
  1. Run model at multiple GABA conductance levels
  2. Detect transition from tonic to oscillatory regime
  3. Estimate EC50 (half-maximal effect concentration)
  4. Compare to biological target

This is a FUNCTIONAL test — it doesn't care about internal
representations, only whether the model produces the correct
bifurcation behavior.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit


# Biological target from La Masson et al. 2002
BIF_TARGET_NS = 29.0
BIF_TARGET_SD = 4.0


def sigmoid(x, L, x0, k, b):
    """Sigmoid function for EC50 fitting."""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def detect_oscillation_rate(
    predictions: np.ndarray,
    dt_ms: float = 1.0,
    freq_band: Tuple[float, float] = (7.0, 14.0),
) -> float:
    """Detect oscillation (spindle) rate from model predictions.

    Uses spectral analysis to detect power in the spindle frequency band
    (7-14 Hz). Higher power = more oscillatory = past bifurcation point.

    Parameters
    ----------
    predictions : ndarray (seq_len, n_neurons)
        Model output for one GABA level.
    dt_ms : float
        Timestep in milliseconds.
    freq_band : tuple
        Frequency band of interest (Hz).

    Returns
    -------
    oscillation_power : float
        Normalized power in the spindle band.
    """
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]

    fs = 1000.0 / dt_ms  # Sampling frequency in Hz
    n_neurons = predictions.shape[1]
    powers = []

    for i in range(n_neurons):
        signal = predictions[:, i]
        signal = signal - np.mean(signal)

        if np.std(signal) < 1e-10:
            powers.append(0.0)
            continue

        # FFT
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
        fft_vals = np.abs(np.fft.rfft(signal)) ** 2

        # Band power
        band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        total_power = np.sum(fft_vals[1:])  # Exclude DC
        band_power = np.sum(fft_vals[band_mask])

        if total_power > 0:
            powers.append(band_power / total_power)
        else:
            powers.append(0.0)

    return float(np.mean(powers))


def detect_pause_rate(
    predictions: np.ndarray,
    dt_ms: float = 1.0,
) -> float:
    """Detect pause rate from predictions (alternative to spectral method).

    Pauses are periods where activity drops below a threshold.
    In oscillatory regime, pauses occur regularly (spindle bursts).

    Parameters
    ----------
    predictions : ndarray (seq_len, n_neurons)
    dt_ms : float

    Returns
    -------
    pause_rate : float
        Pauses per second across neurons.
    """
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]

    duration_s = predictions.shape[0] * dt_ms / 1000.0
    n_neurons = predictions.shape[1]
    total_pauses = 0

    for i in range(n_neurons):
        r = predictions[:, i]
        threshold = np.mean(r) - 0.5 * np.std(r)
        below = r < threshold
        transitions = np.diff(below.astype(int))
        n_pauses = np.sum(transitions == 1)
        total_pauses += n_pauses

    return total_pauses / (n_neurons * duration_s) if duration_s > 0 else 0.0


def estimate_bifurcation_threshold(
    gaba_values: np.ndarray,
    oscillation_metric: np.ndarray,
) -> Tuple[float, bool]:
    """Estimate bifurcation threshold via sigmoid EC50 fit.

    Parameters
    ----------
    gaba_values : ndarray
        GABA conductance levels tested.
    oscillation_metric : ndarray
        Oscillation metric (power or pause rate) at each GABA level.

    Returns
    -------
    threshold : float
        Estimated bifurcation threshold in nS.
    fit_success : bool
    """
    if len(gaba_values) < 4:
        return float('nan'), False

    # Sort by GABA value
    sort_idx = np.argsort(gaba_values)
    x = np.array(gaba_values)[sort_idx]
    y = np.array(oscillation_metric)[sort_idx]

    # Normalize y to [0, 1] range
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return float('nan'), False
    y_norm = (y - y_min) / (y_max - y_min)

    try:
        popt, _ = curve_fit(
            sigmoid, x, y_norm,
            p0=[1.0, np.median(x), 0.1, 0.0],
            maxfev=5000,
        )
        threshold = popt[1]  # x0 = EC50
        return float(threshold), True
    except (RuntimeError, ValueError):
        # Fallback: find GABA level where metric crosses 50%
        midpoint = 0.5 * (y_norm.max() + y_norm.min())
        crossings = np.where(np.diff(np.sign(y_norm - midpoint)))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            # Linear interpolation
            if idx + 1 < len(x):
                frac = (midpoint - y_norm[idx]) / (y_norm[idx + 1] - y_norm[idx] + 1e-10)
                threshold = x[idx] + frac * (x[idx + 1] - x[idx])
                return float(threshold), True
        return float('nan'), False


def run_bifurcation_test(
    model,
    X_by_gaba: Dict[float, np.ndarray],
    dt_ms: float = 1.0,
    method: str = 'spectral',
) -> Dict:
    """Run bifurcation threshold test on a model.

    Parameters
    ----------
    model : nn.Module
        Trained model with forward() method.
    X_by_gaba : dict
        Maps GABA conductance (nS) -> input array (batch, seq_len, input_dim).
    dt_ms : float
        Timestep in milliseconds.
    method : str
        'spectral' or 'pause'.

    Returns
    -------
    result : dict
    """
    import torch

    gaba_values = sorted(X_by_gaba.keys())
    metrics = []

    model.eval()
    for gaba in gaba_values:
        X = X_by_gaba[gaba]
        if isinstance(X, np.ndarray):
            X_t = torch.tensor(X, dtype=torch.float32)
        else:
            X_t = X

        device = next(model.parameters()).device
        X_t = X_t.to(device)

        with torch.no_grad():
            output = model(X_t)
            y_pred = output[0] if isinstance(output, tuple) else output
            y_np = y_pred.cpu().numpy()

        # Average across batch
        y_mean = np.mean(y_np, axis=0)  # (seq_len, n_neurons)

        if method == 'spectral':
            m = detect_oscillation_rate(y_mean, dt_ms)
        else:
            m = detect_pause_rate(y_mean, dt_ms)

        metrics.append(m)

    gaba_arr = np.array(gaba_values)
    metric_arr = np.array(metrics)

    threshold, fit_ok = estimate_bifurcation_threshold(gaba_arr, metric_arr)

    within_1sd = abs(threshold - BIF_TARGET_NS) <= BIF_TARGET_SD if not np.isnan(threshold) else False
    error_ns = abs(threshold - BIF_TARGET_NS) if not np.isnan(threshold) else float('inf')

    return {
        'bifurcation_threshold': float(threshold) if not np.isnan(threshold) else None,
        'target_ns': BIF_TARGET_NS,
        'target_sd': BIF_TARGET_SD,
        'error_ns': error_ns,
        'within_1sd': within_1sd,
        'fit_success': fit_ok,
        'gaba_values': gaba_values,
        'oscillation_metrics': metrics,
        'method': method,
    }
