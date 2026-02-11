"""
Spindle oscillation detection and characterisation.

Targets from Le Masson 2002:
  Spindle frequency: 9.26 +/- 0.87 Hz (n=27)
  Spindle duration:  1.74 +/- 0.36 s  (n=27)
"""

import numpy as np
from scipy import signal


def bandpass_filter(x, fs, low=5.0, high=15.0, order=4):
    """Bandpass filter signal in spindle frequency range."""
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, x)


def detect_spindles(V_tc, t, fs=None, threshold_sd=1.5, min_duration_s=0.3,
                    freq_low=5.0, freq_high=15.0):
    """Detect spindle oscillation epochs in TC voltage trace.

    Parameters
    ----------
    V_tc : np.ndarray
        TC membrane potential (mV).
    t : np.ndarray
        Time array (s).
    fs : float, optional
        Sampling frequency. Computed from t if not given.
    threshold_sd : float
        Number of SDs above mean of bandpass envelope for detection.
    min_duration_s : float
        Minimum spindle duration to count (s).
    freq_low, freq_high : float
        Bandpass filter bounds (Hz).

    Returns
    -------
    spindles : list of dict
        Each dict has keys: start_s, end_s, duration_s, peak_freq_Hz
    """
    if fs is None:
        fs = 1.0 / (t[1] - t[0])

    # Bandpass filter in spindle range
    filtered = bandpass_filter(V_tc, fs, freq_low, freq_high)

    # Analytic signal envelope
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    # Smooth envelope
    smooth_samples = int(0.1 * fs)  # 100 ms smoothing
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        envelope = np.convolve(envelope, kernel, mode='same')

    # Threshold
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    thresh = mean_env + threshold_sd * std_env

    above = envelope > thresh

    # Find contiguous regions above threshold
    spindles = []
    in_spindle = False
    start_idx = 0

    for i in range(len(above)):
        if above[i] and not in_spindle:
            start_idx = i
            in_spindle = True
        elif not above[i] and in_spindle:
            in_spindle = False
            duration = t[i] - t[start_idx]
            if duration >= min_duration_s:
                # Estimate peak frequency via FFT of this segment
                seg = V_tc[start_idx:i]
                if len(seg) > 10:
                    freqs = np.fft.rfftfreq(len(seg), d=1.0/fs)
                    power = np.abs(np.fft.rfft(seg - np.mean(seg)))**2
                    # Restrict to spindle band
                    mask = (freqs >= freq_low) & (freqs <= freq_high)
                    if np.any(mask):
                        peak_freq = freqs[mask][np.argmax(power[mask])]
                    else:
                        peak_freq = float('nan')
                else:
                    peak_freq = float('nan')

                spindles.append({
                    'start_s': t[start_idx],
                    'end_s': t[i],
                    'duration_s': duration,
                    'peak_freq_Hz': peak_freq,
                })

    # Handle case where spindle extends to end
    if in_spindle:
        duration = t[-1] - t[start_idx]
        if duration >= min_duration_s:
            seg = V_tc[start_idx:]
            if len(seg) > 10:
                freqs = np.fft.rfftfreq(len(seg), d=1.0/fs)
                power = np.abs(np.fft.rfft(seg - np.mean(seg)))**2
                mask = (freqs >= freq_low) & (freqs <= freq_high)
                if np.any(mask):
                    peak_freq = freqs[mask][np.argmax(power[mask])]
                else:
                    peak_freq = float('nan')
            else:
                peak_freq = float('nan')
            spindles.append({
                'start_s': t[start_idx],
                'end_s': t[-1],
                'duration_s': duration,
                'peak_freq_Hz': peak_freq,
            })

    return spindles


def spindle_frequency(spindles):
    """Mean spindle frequency across detected epochs."""
    freqs = [s['peak_freq_Hz'] for s in spindles if not np.isnan(s['peak_freq_Hz'])]
    if len(freqs) == 0:
        return float('nan'), float('nan')
    return np.mean(freqs), np.std(freqs)


def spindle_duration(spindles):
    """Mean spindle duration across detected epochs."""
    durs = [s['duration_s'] for s in spindles]
    if len(durs) == 0:
        return float('nan'), float('nan')
    return np.mean(durs), np.std(durs)


def oscillation_power(V_tc, t, freq_low=7.0, freq_high=14.0, fs=None):
    """Compute power in spindle frequency band (7-14 Hz).

    Useful as a continuous metric for bifurcation diagrams.
    """
    if fs is None:
        fs = 1.0 / (t[1] - t[0])

    # Remove mean
    V_centered = V_tc - np.mean(V_tc)

    # FFT
    freqs = np.fft.rfftfreq(len(V_centered), d=1.0/fs)
    power = np.abs(np.fft.rfft(V_centered))**2 / len(V_centered)

    # Band power
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0
    return np.sum(power[mask])


def is_oscillating(V_tc, t, threshold_power=None, **kwargs):
    """Binary classification: is the circuit oscillating?

    If threshold_power is None, uses a heuristic based on
    the ratio of spindle-band power to total power.
    """
    fs = 1.0 / (t[1] - t[0])
    V_centered = V_tc - np.mean(V_tc)
    freqs = np.fft.rfftfreq(len(V_centered), d=1.0/fs)
    power = np.abs(np.fft.rfft(V_centered))**2 / len(V_centered)

    total_power = np.sum(power[1:])  # exclude DC
    band_mask = (freqs >= 7.0) & (freqs <= 14.0)
    band_power = np.sum(power[band_mask])

    if total_power == 0:
        return False

    ratio = band_power / total_power

    if threshold_power is not None:
        return band_power > threshold_power
    else:
        return ratio > 0.15  # >15% of power in spindle band
