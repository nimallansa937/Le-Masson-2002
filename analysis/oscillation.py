"""
Spindle oscillation detection and characterisation.

Targets from Le Masson 2002:
  Spindle frequency: 9.26 +/- 0.87 Hz (n=27)
  Spindle duration:  1.74 +/- 0.36 s  (n=27)

Detection strategy:
  The retinothalamic circuit under continuous retinal drive operates in a
  mixed mode where tonic firing and oscillatory bursts coexist. Unlike
  classic thalamic slices, spindle epochs are not clean silence→burst
  alternations but rather periods where GABA feedback produces rhythmic
  burst-pause sequences superimposed on the retinal drive.

  We detect oscillatory state by:
  1. Burst-pause analysis: count inhibition-mediated pauses (ISI > threshold)
     in TC spike train. More pauses = more oscillatory.
  2. Pause rate: pauses per second. The transition from relay to oscillation
     is marked by a sharp increase in pause rate.
  3. Spectral power: 7-14 Hz band power as continuous metric.
"""

import numpy as np
from scipy import signal


def bandpass_filter(x, fs, low=5.0, high=15.0, order=4):
    """Bandpass filter signal in spindle frequency range."""
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, x)


# ---- Burst-pause analysis (primary oscillation detector) ----

def analyse_burst_pauses(tc_spike_times, pause_threshold_ms=50.0):
    """Analyse TC spike train for burst-pause structure.

    In the oscillating circuit, GABA feedback creates periodic pauses
    in TC firing. A 'pause' is an ISI longer than pause_threshold_ms,
    indicating the TC cell was silenced by inhibition.

    Parameters
    ----------
    tc_spike_times : np.ndarray
        TC spike times in seconds.
    pause_threshold_ms : float
        ISI threshold for a 'pause' (ms). Default 50ms corresponds to
        GABA_A-mediated inhibition silencing TC for at least one
        oscillation half-cycle.

    Returns
    -------
    stats : dict
        n_pauses: number of inhibition-mediated pauses
        pause_rate_hz: pauses per second
        n_bursts: number of burst ISIs (<8ms)
        burst_fraction: fraction of ISIs that are burst-like
        mean_pause_ms: mean pause duration
        pause_intervals_s: intervals between successive pauses (for freq)
        estimated_freq_hz: estimated oscillation frequency from pause timing
    """
    if len(tc_spike_times) < 3:
        return {
            'n_pauses': 0, 'pause_rate_hz': 0.0, 'n_bursts': 0,
            'burst_fraction': 0.0, 'mean_pause_ms': 0.0,
            'pause_intervals_s': np.array([]),
            'estimated_freq_hz': float('nan'),
        }

    isis_ms = np.diff(tc_spike_times) * 1000.0
    duration_s = tc_spike_times[-1] - tc_spike_times[0]

    pause_mask = isis_ms > pause_threshold_ms
    n_pauses = int(np.sum(pause_mask))
    pause_rate = n_pauses / duration_s if duration_s > 0 else 0.0

    burst_mask = isis_ms < 8.0
    n_bursts = int(np.sum(burst_mask))
    burst_fraction = n_bursts / len(isis_ms) if len(isis_ms) > 0 else 0.0

    pause_durations = isis_ms[pause_mask]
    mean_pause = float(np.mean(pause_durations)) if n_pauses > 0 else 0.0

    # Find times of pauses and compute inter-pause intervals
    pause_indices = np.where(pause_mask)[0]
    pause_times = tc_spike_times[pause_indices]  # start of each pause
    if len(pause_times) > 1:
        pause_intervals = np.diff(pause_times)
        estimated_freq = 1.0 / np.mean(pause_intervals)
    else:
        pause_intervals = np.array([])
        estimated_freq = float('nan')

    return {
        'n_pauses': n_pauses,
        'pause_rate_hz': pause_rate,
        'n_bursts': n_bursts,
        'burst_fraction': burst_fraction,
        'mean_pause_ms': mean_pause,
        'pause_intervals_s': pause_intervals,
        'estimated_freq_hz': estimated_freq,
    }


def detect_spindles(V_tc, t, fs=None, threshold_sd=1.5, min_duration_s=0.3,
                    freq_low=5.0, freq_high=15.0):
    """Detect spindle oscillation epochs in TC voltage trace.

    Uses bandpass envelope method. Works best for classic spindle
    morphology (clear burst-silence alternation). For the mixed-mode
    retinothalamic circuit, use analyse_burst_pauses() instead.
    """
    if fs is None:
        fs = 1.0 / (t[1] - t[0])

    filtered = bandpass_filter(V_tc, fs, freq_low, freq_high)
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    smooth_samples = int(0.1 * fs)
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        envelope = np.convolve(envelope, kernel, mode='same')

    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    thresh = mean_env + threshold_sd * std_env
    above = envelope > thresh

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
                seg = V_tc[start_idx:i]
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
                    'start_s': t[start_idx], 'end_s': t[i],
                    'duration_s': duration, 'peak_freq_Hz': peak_freq,
                })

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
                'start_s': t[start_idx], 'end_s': t[-1],
                'duration_s': duration, 'peak_freq_Hz': peak_freq,
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
    """Compute power in spindle frequency band (7-14 Hz)."""
    if fs is None:
        fs = 1.0 / (t[1] - t[0])
    V_centered = V_tc - np.mean(V_tc)
    freqs = np.fft.rfftfreq(len(V_centered), d=1.0/fs)
    power = np.abs(np.fft.rfft(V_centered))**2 / len(V_centered)
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0
    return np.sum(power[mask])


def is_oscillating(tc_spike_times, duration_s, pause_threshold_ms=50.0,
                   min_pause_rate_hz=1.0):
    """Binary classification: is the circuit in oscillatory mode?

    Uses burst-pause analysis of TC spike train rather than voltage
    trace morphology. The circuit is oscillating when GABA feedback
    produces a sustained rate of inhibition-mediated pauses.

    Parameters
    ----------
    tc_spike_times : np.ndarray
        TC spike times in seconds.
    duration_s : float
        Simulation duration.
    pause_threshold_ms : float
        ISI threshold defining an inhibitory pause (ms).
    min_pause_rate_hz : float
        Minimum pause rate (pauses/second) to classify as oscillating.
        Default 1.0 means at least 1 GABA-mediated pause per second.

    Returns
    -------
    bool
    """
    stats = analyse_burst_pauses(tc_spike_times, pause_threshold_ms)
    return stats['pause_rate_hz'] >= min_pause_rate_hz
