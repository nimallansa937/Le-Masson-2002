"""
Population Coherence Test for DESCARTES-NeuralODE.

During the oscillatory regime (high GABA conductance), the biological
TC-nRt feedback loop synchronizes TC neuron output into population-level
oscillatory coherence. This module tests whether a learned model preserves
that synchronization.

Key question: Does the model produce individually correct but collectively
incoherent output (i.e., it learned per-neuron statistics but missed the
coupling that synchronizes them)?

Method: Pairwise Phase Consistency (PPC)
  1. Bandpass filter predicted spike trains in the oscillation band (7-14 Hz)
  2. Extract instantaneous phase via Hilbert transform
  3. Compute mean pairwise phase consistency (circular variance)
  4. Compare to biological baseline

Per Phase 6.3 of the transformation_replacement_guide:
  - 'preserved' if model coherence >= 0.7 * bio coherence
"""
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from typing import Dict, List, Optional, Tuple


# Defaults
OSCILLATION_BAND_HZ = (7.0, 14.0)  # TC-nRt oscillation frequency band
SAMPLING_RATE_HZ = 1000.0           # 1 ms bins → 1000 Hz
COHERENCE_PRESERVATION_THRESHOLD = 0.7  # Model must reach 70% of bio


def _bandpass_filter(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : ndarray (T,)
    low_hz, high_hz : float
    fs_hz : float
    order : int

    Returns
    -------
    filtered : ndarray (T,)
    """
    nyq = fs_hz / 2.0
    low = low_hz / nyq
    high = high_hz / nyq

    # Clamp to valid range
    low = max(low, 0.001)
    high = min(high, 0.999)

    if low >= high:
        return signal

    b, a = butter(order, [low, high], btype='band')
    try:
        return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))
    except ValueError:
        return signal


def _instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """Extract instantaneous phase via Hilbert transform.

    Parameters
    ----------
    signal : ndarray (T,)

    Returns
    -------
    phase : ndarray (T,) in radians [-π, π]
    """
    analytic = hilbert(signal)
    return np.angle(analytic)


def pairwise_phase_consistency(
    phases: np.ndarray,
) -> float:
    """Compute Pairwise Phase Consistency (PPC).

    PPC is a bias-free measure of phase synchronization across neurons.
    It ranges from 0 (no synchronization) to 1 (perfect synchronization).

    For each timepoint, compute the mean cosine of pairwise phase differences.
    Then average across time.

    Parameters
    ----------
    phases : ndarray (n_neurons, T)
        Instantaneous phase for each neuron.

    Returns
    -------
    ppc : float
    """
    n_neurons, T = phases.shape
    if n_neurons < 2:
        return 0.0

    # For each timepoint, compute mean cos(phase_i - phase_j) for all pairs
    ppc_per_t = np.zeros(T)
    n_pairs = n_neurons * (n_neurons - 1) / 2

    for t in range(T):
        total = 0.0
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                total += np.cos(phases[i, t] - phases[j, t])
        ppc_per_t[t] = total / n_pairs

    return float(np.mean(ppc_per_t))


def pairwise_phase_consistency_fast(
    phases: np.ndarray,
) -> float:
    """Vectorized PPC computation (faster for large n_neurons).

    Uses the identity: PPC = (|sum(e^{i*phi})|^2 - N) / (N*(N-1))

    Parameters
    ----------
    phases : ndarray (n_neurons, T)

    Returns
    -------
    ppc : float
    """
    n_neurons, T = phases.shape
    if n_neurons < 2:
        return 0.0

    # Complex unit vectors on the circle
    z = np.exp(1j * phases)  # (n_neurons, T)

    # Resultant vector magnitude squared
    R_squared = np.abs(np.sum(z, axis=0)) ** 2  # (T,)

    # PPC = (R^2 - N) / (N*(N-1))
    ppc_per_t = (R_squared - n_neurons) / (n_neurons * (n_neurons - 1))

    return float(np.mean(ppc_per_t))


def compute_population_coherence(
    spike_rates: np.ndarray,
    band_hz: Tuple[float, float] = OSCILLATION_BAND_HZ,
    fs_hz: float = SAMPLING_RATE_HZ,
) -> Dict:
    """Compute population coherence for a set of neuron outputs.

    Parameters
    ----------
    spike_rates : ndarray (T, n_neurons) or (n_neurons, T)
        Smoothed firing rates or spike trains.
    band_hz : tuple of float
        Oscillation frequency band for bandpass filtering.
    fs_hz : float
        Sampling rate in Hz.

    Returns
    -------
    result : dict
        'ppc': float — pairwise phase consistency
        'mean_power_in_band': float — average oscillatory power
        'per_neuron_power': list of float
    """
    # Ensure (n_neurons, T) orientation
    if spike_rates.ndim == 2 and spike_rates.shape[0] > spike_rates.shape[1]:
        spike_rates = spike_rates.T

    n_neurons, T = spike_rates.shape

    if T < 100:
        return {
            'ppc': 0.0,
            'mean_power_in_band': 0.0,
            'per_neuron_power': [0.0] * n_neurons,
        }

    # Bandpass filter each neuron
    filtered = np.zeros_like(spike_rates)
    for i in range(n_neurons):
        filtered[i] = _bandpass_filter(spike_rates[i], band_hz[0], band_hz[1], fs_hz)

    # Extract instantaneous phase
    phases = np.zeros_like(filtered)
    for i in range(n_neurons):
        phases[i] = _instantaneous_phase(filtered[i])

    # Compute PPC
    ppc = pairwise_phase_consistency_fast(phases)

    # Power in oscillation band (RMS of bandpass-filtered signal)
    per_neuron_power = []
    for i in range(n_neurons):
        power = float(np.sqrt(np.mean(filtered[i] ** 2)))
        per_neuron_power.append(power)
    mean_power = float(np.mean(per_neuron_power))

    return {
        'ppc': ppc,
        'mean_power_in_band': mean_power,
        'per_neuron_power': per_neuron_power,
    }


def test_coherence_preservation(
    model_rates: np.ndarray,
    bio_rates: np.ndarray,
    band_hz: Tuple[float, float] = OSCILLATION_BAND_HZ,
    fs_hz: float = SAMPLING_RATE_HZ,
    preservation_threshold: float = COHERENCE_PRESERVATION_THRESHOLD,
) -> Dict:
    """Test whether model preserves population-level oscillatory coherence.

    Per Phase 6.3 of the guide: during oscillatory regime (high GABA),
    the model's output should show population-level coherence at >= 70%
    of the biological baseline.

    Parameters
    ----------
    model_rates : ndarray (T, n_neurons) or (n_neurons, T)
        Model predicted firing rates.
    bio_rates : ndarray same shape
        Biological ground truth firing rates.
    band_hz : tuple of float
    fs_hz : float
    preservation_threshold : float
        Fraction of bio coherence that model must reach (default 0.7).

    Returns
    -------
    result : dict
    """
    bio_result = compute_population_coherence(bio_rates, band_hz, fs_hz)
    model_result = compute_population_coherence(model_rates, band_hz, fs_hz)

    bio_coherence = bio_result['ppc']
    model_coherence = model_result['ppc']

    if bio_coherence > 1e-6:
        ratio = model_coherence / bio_coherence
    else:
        # No coherence in biology — can't compute ratio
        ratio = 1.0 if model_coherence < 1e-6 else 0.0

    return {
        'model_coherence': model_coherence,
        'bio_coherence': bio_coherence,
        'ratio': ratio,
        'preserved': ratio >= preservation_threshold,
        'model_power': model_result['mean_power_in_band'],
        'bio_power': bio_result['mean_power_in_band'],
    }


def test_coherence_across_gaba(
    model,
    X_data: np.ndarray,
    Y_data: np.ndarray,
    gaba_levels: np.ndarray,
    band_hz: Tuple[float, float] = OSCILLATION_BAND_HZ,
    fs_hz: float = SAMPLING_RATE_HZ,
    high_gaba_threshold: float = 25.0,
) -> Dict:
    """Test coherence preservation across GABA levels.

    Particularly important at high GABA (oscillatory regime) where
    population synchronization is most prominent.

    Parameters
    ----------
    model : nn.Module
        Trained model with .predict() or forward method.
    X_data : ndarray (n_windows, T, input_dim)
        Input data (retinal + GABA).
    Y_data : ndarray (n_windows, T, output_dim)
        Biological target outputs.
    gaba_levels : ndarray (n_windows,)
        GABA level for each window.
    band_hz : tuple of float
    fs_hz : float
    high_gaba_threshold : float
        GABA level above which oscillations are expected.

    Returns
    -------
    result : dict
    """
    import torch

    unique_gaba = np.unique(gaba_levels)
    per_gaba_results = {}

    for gaba in unique_gaba:
        mask = gaba_levels == gaba
        X_gaba = X_data[mask]
        Y_gaba = Y_data[mask]

        # Get model predictions
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_gaba, dtype=torch.float32)
            output = model(x_tensor)
            y_pred = output[0] if isinstance(output, tuple) else output
            y_pred_np = y_pred.cpu().numpy()

        # Average coherence across windows at this GABA level
        model_ppcs = []
        bio_ppcs = []
        for w in range(Y_gaba.shape[0]):
            bio_coh = compute_population_coherence(Y_gaba[w], band_hz, fs_hz)
            model_coh = compute_population_coherence(y_pred_np[w], band_hz, fs_hz)
            bio_ppcs.append(bio_coh['ppc'])
            model_ppcs.append(model_coh['ppc'])

        per_gaba_results[float(gaba)] = {
            'mean_bio_ppc': float(np.mean(bio_ppcs)),
            'mean_model_ppc': float(np.mean(model_ppcs)),
            'ratio': float(np.mean(model_ppcs)) / max(float(np.mean(bio_ppcs)), 1e-6),
            'n_windows': int(mask.sum()),
        }

    # Aggregate for high-GABA (oscillatory) regime
    high_gaba_mask = gaba_levels >= high_gaba_threshold
    if high_gaba_mask.any():
        high_gaba_bio = []
        high_gaba_model = []
        for gaba in unique_gaba:
            if gaba >= high_gaba_threshold:
                high_gaba_bio.append(per_gaba_results[float(gaba)]['mean_bio_ppc'])
                high_gaba_model.append(per_gaba_results[float(gaba)]['mean_model_ppc'])
        oscillatory_ratio = float(np.mean(high_gaba_model)) / max(float(np.mean(high_gaba_bio)), 1e-6)
    else:
        oscillatory_ratio = 0.0

    return {
        'per_gaba': per_gaba_results,
        'oscillatory_regime_ratio': oscillatory_ratio,
        'oscillatory_preserved': oscillatory_ratio >= COHERENCE_PRESERVATION_THRESHOLD,
    }
