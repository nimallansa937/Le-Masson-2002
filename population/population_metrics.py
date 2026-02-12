"""
Population-level metrics for Rung 2 analysis.

Measures emergent properties that single-neuron experiments cannot capture:
  - Population spindle coherence (PPC via Hilbert transform)
  - Spindle frequency stability (cross-neuron consistency)
  - Recruitment cascade integrity (latency spread)
  - Participation fraction (what fraction of TC neurons oscillate)
"""

import numpy as np
from scipy import signal


def population_spindle_coherence(V_tc, t, spindle_band=(7, 14), fs=None):
    """Compute population coherence during oscillatory epochs.

    Uses Pairwise Phase Consistency (PPC):
    1. Bandpass each TC trace in spindle band
    2. Hilbert transform -> instantaneous phase
    3. Compute mean resultant length (MRL) of pairwise phase differences

    Parameters
    ----------
    V_tc : ndarray (n_tc, n_timepoints)
        TC voltage traces.
    t : ndarray
        Time array.
    spindle_band : tuple
        (low, high) frequency band in Hz.
    fs : float, optional
        Sampling frequency. Inferred from t if not provided.

    Returns
    -------
    kappa : float
        Mean population coherence (0 = desynchronized, 1 = perfectly locked).
    """
    if fs is None:
        fs = 1.0 / (t[1] - t[0])

    n_tc = V_tc.shape[0]
    if n_tc < 2:
        return 0.0

    nyq = fs / 2.0
    low = spindle_band[0] / nyq
    high = spindle_band[1] / nyq

    if high >= 1.0:
        high = 0.99
    if low <= 0:
        low = 0.01

    b, a = signal.butter(4, [low, high], btype='band')

    phases = []
    for i in range(n_tc):
        filtered = signal.filtfilt(b, a, V_tc[i])
        analytic = signal.hilbert(filtered)
        phases.append(np.angle(analytic))
    phases = np.array(phases)  # (n_tc, n_timepoints)

    # Compute pairwise MRL across all TC pairs
    mrl_values = []
    for i in range(n_tc):
        for j in range(i + 1, n_tc):
            phase_diff = phases[i] - phases[j]
            mrl = np.abs(np.mean(np.exp(1j * phase_diff)))
            mrl_values.append(mrl)

    return float(np.mean(mrl_values)) if mrl_values else 0.0


def spindle_frequency_stability(V_tc, t, fmin=5.0, fmax=20.0, fs=None):
    """Compute dominant oscillation frequency and cross-neuron variance.

    Parameters
    ----------
    V_tc : ndarray (n_tc, n_timepoints)
    t : ndarray
    fmin, fmax : float
        Frequency band to search for peak.

    Returns
    -------
    pop_peak_freq : float
        Peak frequency of population-averaged signal.
    per_neuron_std : float
        Standard deviation of per-neuron peak frequencies.
    per_neuron_freqs : list of float
        Peak frequency per TC neuron.
    """
    if fs is None:
        fs = 1.0 / (t[1] - t[0])

    n_tc = V_tc.shape[0]
    n_pts = V_tc.shape[1]

    # Population-averaged signal
    pop_avg = np.mean(V_tc, axis=0)
    pop_avg -= np.mean(pop_avg)
    freqs = np.fft.rfftfreq(n_pts, 1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(mask):
        return float('nan'), float('nan'), []

    power_pop = np.abs(np.fft.rfft(pop_avg)) ** 2
    pop_peak_freq = float(freqs[mask][np.argmax(power_pop[mask])])

    per_neuron_freqs = []
    for i in range(n_tc):
        trace = V_tc[i] - np.mean(V_tc[i])
        power_i = np.abs(np.fft.rfft(trace)) ** 2
        peak_i = float(freqs[mask][np.argmax(power_i[mask])])
        per_neuron_freqs.append(peak_i)

    per_neuron_std = float(np.std(per_neuron_freqs))
    return pop_peak_freq, per_neuron_std, per_neuron_freqs


def recruitment_latency(tc_spike_times_list, spindle_onset_times,
                        window_s=0.2):
    """Measure recruitment cascade integrity.

    For each spindle onset, find when each TC neuron first spikes.
    Compute spread of recruitment latencies.

    Parameters
    ----------
    tc_spike_times_list : list of ndarray
        Spike times per TC neuron.
    spindle_onset_times : ndarray
        Onset times of detected spindle epochs.
    window_s : float
        Window after onset to look for first spike.

    Returns
    -------
    mean_spread_ms : float
        Mean spread (std) of recruitment latencies across spindles.
    mean_latency_ms : float
        Mean first-spike latency across all TC neurons and spindles.
    participation : float
        Mean fraction of TC neurons that spike within window.
    """
    if len(spindle_onset_times) == 0:
        return float('nan'), float('nan'), 0.0

    n_tc = len(tc_spike_times_list)
    spreads = []
    latencies_all = []
    participations = []

    for onset in spindle_onset_times:
        first_spikes = []
        for spikes in tc_spike_times_list:
            post = spikes[(spikes > onset) & (spikes < onset + window_s)]
            if len(post) > 0:
                first_spikes.append((post[0] - onset) * 1000.0)  # ms

        if len(first_spikes) > 1:
            spreads.append(np.std(first_spikes))
        latencies_all.extend(first_spikes)
        participations.append(len(first_spikes) / n_tc)

    mean_spread = float(np.mean(spreads)) if spreads else float('nan')
    mean_lat = float(np.mean(latencies_all)) if latencies_all else float('nan')
    mean_part = float(np.mean(participations)) if participations else 0.0

    return mean_spread, mean_lat, mean_part


def participation_fraction(tc_spike_times_list, duration_s,
                           pause_threshold_ms=50.0, min_pause_rate_hz=1.0):
    """Compute what fraction of TC neurons are in oscillatory mode.

    Uses per-neuron burst-pause analysis.

    Parameters
    ----------
    tc_spike_times_list : list of ndarray
    duration_s : float
    pause_threshold_ms : float
    min_pause_rate_hz : float

    Returns
    -------
    fraction : float
        Fraction of TC neurons classified as oscillating.
    per_neuron_osc : list of bool
    """
    n_tc = len(tc_spike_times_list)
    per_neuron_osc = []

    for spikes in tc_spike_times_list:
        if len(spikes) < 3:
            per_neuron_osc.append(False)
            continue
        isis_ms = np.diff(spikes) * 1000.0
        dur = spikes[-1] - spikes[0]
        if dur <= 0:
            per_neuron_osc.append(False)
            continue
        n_pauses = np.sum(isis_ms > pause_threshold_ms)
        pause_rate = n_pauses / dur
        per_neuron_osc.append(pause_rate >= min_pause_rate_hz)

    fraction = sum(per_neuron_osc) / n_tc if n_tc > 0 else 0.0
    return fraction, per_neuron_osc


def population_pause_rate(tc_spike_times_list, pause_threshold_ms=50.0):
    """Compute mean pause rate across TC population.

    Parameters
    ----------
    tc_spike_times_list : list of ndarray
    pause_threshold_ms : float

    Returns
    -------
    mean_pause_rate : float
        Mean pause rate across TC neurons (Hz).
    per_neuron_rates : list of float
    """
    rates = []
    for spikes in tc_spike_times_list:
        if len(spikes) < 3:
            rates.append(0.0)
            continue
        isis_ms = np.diff(spikes) * 1000.0
        dur = spikes[-1] - spikes[0]
        if dur <= 0:
            rates.append(0.0)
            continue
        n_pauses = int(np.sum(isis_ms > pause_threshold_ms))
        rates.append(n_pauses / dur)

    return float(np.mean(rates)), rates


def population_burst_fraction(tc_spike_times_list):
    """Compute mean burst fraction across TC population.

    Returns
    -------
    mean_bf : float
    per_neuron_bf : list of float
    """
    bfs = []
    for spikes in tc_spike_times_list:
        if len(spikes) < 3:
            bfs.append(0.0)
            continue
        isis_ms = np.diff(spikes) * 1000.0
        n_burst = int(np.sum(isis_ms < 8.0))
        bfs.append(n_burst / len(isis_ms))

    return float(np.mean(bfs)), bfs
