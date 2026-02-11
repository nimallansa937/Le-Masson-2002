"""
Retinal ganglion cell spike train generator.

ISI distribution: Gamma (Erlang) renewal process.
  gamma_order controls regularity: low = irregular, high = regular.
  Physiological range: gamma = 0.7 to 12.
  Paper uses gamma = 1.5 (Fig 2) and gamma = 3 (Figs 3, 4).
  Mean firing rates: 5-60 Hz (in vivo range).
"""

import numpy as np


def generate_retinal_spikes(rate_hz, gamma_order, duration_s, rng=None):
    """Generate spike train with gamma-distributed ISIs.

    Parameters
    ----------
    rate_hz : float
        Mean firing rate in Hz.
    gamma_order : float
        Shape parameter (gamma). Higher = more regular.
    duration_s : float
        Simulation duration in seconds.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    spike_times : np.ndarray
        Array of spike times in seconds.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rate_hz <= 0:
        return np.array([])

    mean_isi = 1.0 / rate_hz
    scale = mean_isi / gamma_order

    # Pre-allocate generously
    expected_n = int(rate_hz * duration_s * 1.5) + 100
    isis = rng.gamma(gamma_order, scale, size=expected_n)

    spike_times = np.cumsum(isis)
    spike_times = spike_times[spike_times < duration_s]

    return spike_times


def retinal_spike_at(t, spike_times, dt):
    """Check if there is a retinal spike in the time bin [t, t+dt)."""
    if len(spike_times) == 0:
        return False
    idx = np.searchsorted(spike_times, t)
    if idx < len(spike_times) and spike_times[idx] < t + dt:
        return True
    # Also check if a spike falls at exactly t
    if idx > 0 and spike_times[idx - 1] >= t and spike_times[idx - 1] < t + dt:
        return True
    return False


def spike_times_to_binary(spike_times, dt, n_steps):
    """Convert spike times to binary array for fast lookup.

    Returns an array of length n_steps where entry i is True
    if there is a spike in the interval [i*dt, (i+1)*dt).
    """
    binary = np.zeros(n_steps, dtype=bool)
    if len(spike_times) == 0:
        return binary
    indices = (spike_times / dt).astype(int)
    indices = indices[(indices >= 0) & (indices < n_steps)]
    binary[indices] = True
    return binary
