"""
Load A-R2 data with ground truth biological variables.

Reads per-trial HDF5 files (trial_gaba{X}_seed{N}.h5) from the rung3_data
directory. Each file contains spike times, voltages, and intermediate
biological variables for one simulation trial at a specific GABA conductance.

Data pipeline:
  1. Load spike times from HDF5
  2. Convert to 1ms-binned binary spike trains
  3. Smooth with Gaussian kernel (sigma=5ms) to get firing rates
  4. Construct input: 20 retinal rates + 1 GABA level = 21 channels
  5. Construct output: 20 TC firing rates = 20 channels
  6. Window into 2000-step (2s) overlapping windows (stride=500ms)
  7. Extract intermediates as ground truth biological variables
"""
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
import re


# ── Constants matching rung3 preprocessing ──────────────────────────
BIN_DT_MS = 1.0            # 1 ms bins
SMOOTH_SIGMA_MS = 5.0      # 5 ms Gaussian smoothing
WINDOW_SIZE_MS = 2000       # 2 second windows → 2000 bins
WINDOW_STRIDE_MS = 500      # 500 ms stride → 500 bins
GABA_MAX_NS = 74.0          # Max GABA for normalization
N_TC = 20
N_NRT = 20
N_RETINAL = 20

TRAIN_SEEDS = [42, 43, 44]
VAL_SEEDS = [45, 46]


def load_ar2_data(data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load A-R2 data for DESCARTES architecture search.

    Args:
        data_dir: Path to directory containing trial_gaba*.h5 files

    Returns:
        train_data: dict with X_train, Y_train, X_short, Y_short
        val_data: dict with X_val, Y_val, T
        bio_ground_truth: dict of variable_name -> (n_neurons, T) arrays
    """
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob('trial_gaba*.h5'))

    if not h5_files:
        print(f"WARNING: No trial_gaba*.h5 files found in {data_dir}")
        print("Generating synthetic data for testing...")
        return _generate_synthetic_data()

    print(f"Found {len(h5_files)} trial files in {data_dir}")

    # Parse filenames to get gaba value and seed
    trials = []
    for f in h5_files:
        match = re.match(r'trial_gaba([\d.]+)_seed(\d+)\.h5', f.name)
        if match:
            trials.append({
                'filepath': f,
                'gaba': float(match.group(1)),
                'seed': int(match.group(2)),
            })

    if not trials:
        print("WARNING: Could not parse trial filenames. Falling back to synthetic.")
        return _generate_synthetic_data()

    # Split by seed
    train_trials = [t for t in trials if t['seed'] in TRAIN_SEEDS]
    val_trials = [t for t in trials if t['seed'] in VAL_SEEDS]

    print(f"  Train trials: {len(train_trials)} (seeds {TRAIN_SEEDS})")
    print(f"  Val trials:   {len(val_trials)} (seeds {VAL_SEEDS})")

    # Process train trials
    print("Processing training trials...")
    X_train_list, Y_train_list, Yb_train_list, inter_train = _process_trials(train_trials)

    # Process val trials
    print("Processing validation trials...")
    X_val_list, Y_val_list, Yb_val_list, inter_val = _process_trials(val_trials)

    # Concatenate
    X_train = np.concatenate(X_train_list, axis=0)   # (N_win, 2000, 21)
    Y_train = np.concatenate(Y_train_list, axis=0)    # (N_win, 2000, 20) rates
    Yb_train = np.concatenate(Yb_train_list, axis=0)  # (N_win, 2000, 20) binary
    X_val = np.concatenate(X_val_list, axis=0)
    Y_val = np.concatenate(Y_val_list, axis=0)
    Yb_val = np.concatenate(Yb_val_list, axis=0)

    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_val:   {X_val.shape}, Y_val:   {Y_val.shape}")

    # Short segments for verifier (first 200 steps = 200ms of each window)
    # 200ms is enough temporal context for meaningful thalamic dynamics
    # while keeping verification fast (~30-120s per architecture)
    X_short = X_train[:, :200, :]
    Y_short = Y_train[:, :200, :]

    train_data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'Y_binary_train': Yb_train,
        'X_short': X_short,
        'Y_short': Y_short,
    }
    val_data = {
        'X_val': X_val,
        'Y_val': Y_val,
        'Y_binary_val': Yb_val,
        'T': WINDOW_SIZE_MS,
    }

    # Build ground truth biological variables from intermediates
    # Use ALL trials (all seeds) for the most complete GT
    bio_gt = _build_bio_ground_truth(trials)
    print(f"  Bio ground truth: {len(bio_gt)} variable groups")
    for k, v in bio_gt.items():
        print(f"    {k}: {v.shape}")

    return train_data, val_data, bio_gt


def _process_trials(trials: List[Dict]) -> Tuple[List, List, List, List]:
    """Process a list of trials into windowed X, Y, Y_binary arrays.

    Returns
    -------
    X_list : list of (n_win, window, 21) arrays — input features
    Y_list : list of (n_win, window, 20) arrays — smoothed rate targets
    Yb_list : list of (n_win, window, 20) arrays — binary spike targets
    inter_list : list of dicts — intermediate variables
    """
    X_list = []
    Y_list = []
    Yb_list = []
    inter_list = []

    for i, trial_info in enumerate(trials):
        filepath = trial_info['filepath']
        gaba = trial_info['gaba']

        with h5py.File(filepath, 'r') as f:
            duration_s = float(f['meta'].attrs['duration_s'])

            # Load spike times
            retinal_spikes = _load_spike_times(f, 'retinal_spike_times', 'channel', N_RETINAL)
            tc_spikes = _load_spike_times(f, 'tc_spike_times', 'neuron', N_TC)

            # Convert to rates (smoothed) AND binary
            ret_rates = _spikes_to_rates(retinal_spikes, duration_s)  # (20, T)
            tc_rates = _spikes_to_rates(tc_spikes, duration_s)        # (20, T)
            tc_binary = _spikes_to_binary(tc_spikes, duration_s)      # (20, T)

            # Build input features: 20 retinal + 1 GABA = 21
            n_bins = ret_rates.shape[1]
            gaba_norm = gaba / GABA_MAX_NS
            gaba_channel = np.full((1, n_bins), gaba_norm, dtype=np.float32)
            features = np.vstack([ret_rates, gaba_channel]).T    # (T, 21)
            targets_rate = tc_rates.T                             # (T, 20)
            targets_binary = tc_binary.T                          # (T, 20)

            # Window
            X_win, Yr_win, Yb_win = _create_windows(features, targets_rate, targets_binary)
            X_list.append(X_win)
            Y_list.append(Yr_win)
            Yb_list.append(Yb_win)

            # Load intermediates for this trial
            intermediates = {}
            if 'intermediates' in f:
                for key in f['intermediates']:
                    intermediates[key] = f[f'intermediates/{key}'][:].astype(np.float32)
            inter_list.append(intermediates)

        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(trials)} trials")

    return X_list, Y_list, Yb_list, inter_list


def _load_spike_times(f: h5py.File, group: str, prefix: str, n_channels: int) -> List[np.ndarray]:
    """Load spike times from HDF5 group."""
    spikes = []
    for i in range(n_channels):
        key = f'{group}/{prefix}_{i}'
        if key in f:
            spikes.append(f[key][:])
        else:
            spikes.append(np.array([]))
    return spikes


def _spikes_to_rates(spike_times_list: List[np.ndarray],
                     duration_s: float) -> np.ndarray:
    """
    Convert spike times (seconds) → binary (1ms bins) → smoothed rates.

    Returns:
        rates: (n_channels, n_bins) float32 smoothed firing rates
    """
    n_bins = int(duration_s * 1000.0 / BIN_DT_MS)
    n_channels = len(spike_times_list)
    binary = np.zeros((n_channels, n_bins), dtype=np.float32)

    for i, spk in enumerate(spike_times_list):
        if len(spk) > 0:
            bin_idx = (spk * 1000.0 / BIN_DT_MS).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            binary[i, bin_idx] = 1.0

    # Gaussian smoothing (sigma in bins = sigma_ms / bin_dt_ms)
    sigma_bins = SMOOTH_SIGMA_MS / BIN_DT_MS
    rates = gaussian_filter1d(binary, sigma=sigma_bins, axis=1).astype(np.float32)

    return rates


def _spikes_to_binary(spike_times_list: List[np.ndarray],
                      duration_s: float) -> np.ndarray:
    """
    Convert spike times (seconds) → binary matrix (1ms bins, no smoothing).

    Returns:
        binary: (n_channels, n_bins) float32 binary spike trains
    """
    n_bins = int(duration_s * 1000.0 / BIN_DT_MS)
    n_channels = len(spike_times_list)
    binary = np.zeros((n_channels, n_bins), dtype=np.float32)

    for i, spk in enumerate(spike_times_list):
        if len(spk) > 0:
            bin_idx = (spk * 1000.0 / BIN_DT_MS).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            binary[i, bin_idx] = 1.0

    return binary


def _create_windows(features: np.ndarray, targets_rate: np.ndarray,
                    targets_binary: Optional[np.ndarray] = None,
                    ) -> Tuple[np.ndarray, ...]:
    """
    Create overlapping windows from full trial.

    Args:
        features: (T, 21) input features
        targets_rate: (T, 20) smoothed rate targets
        targets_binary: (T, 20) binary spike targets (optional)

    Returns:
        X_windows: (n_windows, window_size, 21)
        Yr_windows: (n_windows, window_size, 20) — smoothed rates
        Yb_windows: (n_windows, window_size, 20) — binary spikes (only if provided)
    """
    T = features.shape[0]
    window_bins = int(WINDOW_SIZE_MS / BIN_DT_MS)
    stride_bins = int(WINDOW_STRIDE_MS / BIN_DT_MS)

    starts = list(range(0, T - window_bins + 1, stride_bins))
    n_windows = len(starts)

    X_windows = np.zeros((n_windows, window_bins, features.shape[1]), dtype=np.float32)
    Yr_windows = np.zeros((n_windows, window_bins, targets_rate.shape[1]), dtype=np.float32)

    for i, start in enumerate(starts):
        X_windows[i] = features[start:start + window_bins]
        Yr_windows[i] = targets_rate[start:start + window_bins]

    if targets_binary is not None:
        Yb_windows = np.zeros((n_windows, window_bins, targets_binary.shape[1]), dtype=np.float32)
        for i, start in enumerate(starts):
            Yb_windows[i] = targets_binary[start:start + window_bins]
        return X_windows, Yr_windows, Yb_windows

    return X_windows, Yr_windows, np.zeros_like(Yr_windows)


def _build_bio_ground_truth(trials: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Build ground truth biological variable arrays from intermediates.

    Uses the first available trial for each variable (they represent
    the same biological system, just different random seeds).
    Returns a representative sample aggregated across GABA levels.

    Maps to the 160-dimensional recovery space:
      - tc_m_T (20 neurons)   → tc_gating
      - tc_h_T (20 neurons)   → tc_gating
      - tc_m_h (20 neurons)   → tc_gating
      - nrt_m_Ts (20 neurons) → nrt_state
      - nrt_h_Ts (20 neurons) → nrt_state
      - V_nrt (20 neurons)    → nrt_state
      - gabaa_per_tc (20 neurons) → synaptic
      - gabab_per_tc (20 neurons) → synaptic
    """
    bio_gt = {}

    # Collect from multiple GABA levels for diversity
    # Use seed 42 across different GABA levels
    seed42_trials = [t for t in trials if t['seed'] == 42]
    seed42_trials.sort(key=lambda t: t['gaba'])

    # Sample a few representative GABA levels
    if len(seed42_trials) >= 5:
        # Pick 5 evenly spaced
        indices = np.linspace(0, len(seed42_trials) - 1, 5, dtype=int)
        sample_trials = [seed42_trials[i] for i in indices]
    else:
        sample_trials = seed42_trials[:5] if seed42_trials else trials[:5]

    for trial_info in sample_trials:
        with h5py.File(trial_info['filepath'], 'r') as f:
            gaba = trial_info['gaba']
            suffix = f"_gaba{gaba:.0f}"

            # TC gating variables (3 × 20 = 60)
            if 'intermediates' in f:
                for var_name in ['tc_m_T', 'tc_h_T', 'tc_m_h',
                                 'nrt_m_Ts', 'nrt_h_Ts',
                                 'gabaa_per_tc', 'gabab_per_tc',
                                 'ampa_per_nrt']:
                    key = f'intermediates/{var_name}'
                    if key in f:
                        bio_gt[f'{var_name}{suffix}'] = f[key][:].astype(np.float32)

            # Voltages
            if 'V_nrt' in f:
                bio_gt[f'V_nrt{suffix}'] = f['V_nrt'][:].astype(np.float32)
            if 'V_tc' in f:
                bio_gt[f'V_tc{suffix}'] = f['V_tc'][:].astype(np.float32)

    # Also provide a single canonical set (gaba=30 if available, or middle trial)
    canonical = None
    for t in seed42_trials:
        if abs(t['gaba'] - 30.0) < 1.0:
            canonical = t
            break
    if canonical is None and seed42_trials:
        canonical = seed42_trials[len(seed42_trials) // 2]

    if canonical:
        with h5py.File(canonical['filepath'], 'r') as f:
            if 'intermediates' in f:
                for var_name in ['tc_m_T', 'tc_h_T', 'tc_m_h',
                                 'nrt_m_Ts', 'nrt_h_Ts',
                                 'gabaa_per_tc', 'gabab_per_tc',
                                 'ampa_per_nrt']:
                    key = f'intermediates/{var_name}'
                    if key in f:
                        bio_gt[var_name] = f[key][:].astype(np.float32)

            if 'V_nrt' in f:
                bio_gt['V_nrt'] = f['V_nrt'][:].astype(np.float32)

    return bio_gt


def _generate_synthetic_data():
    """Generate synthetic data for testing the framework."""
    n_windows = 100
    T_full = 2000
    T_short = 50
    n_in = 21
    n_out = 20
    n_neurons = 20

    train_data = {
        'X_train': np.random.randn(n_windows, T_full, n_in).astype(np.float32),
        'Y_train': np.random.rand(n_windows, T_full, n_out).astype(np.float32),
        'X_short': np.random.randn(n_windows, T_short, n_in).astype(np.float32),
        'Y_short': np.random.rand(n_windows, T_short, n_out).astype(np.float32),
    }
    val_data = {
        'X_val': np.random.randn(20, T_full, n_in).astype(np.float32),
        'Y_val': np.random.rand(20, T_full, n_out).astype(np.float32),
        'T': T_full,
    }

    # Synthetic ground truth (160 variables)
    T_sub = T_full * 10
    bio_gt = {
        'tc_m_T': np.random.rand(n_neurons, T_sub).astype(np.float32),
        'tc_h_T': np.random.rand(n_neurons, T_sub).astype(np.float32),
        'tc_m_h': np.random.rand(n_neurons, T_sub).astype(np.float32),
        'V_nrt': np.random.randn(n_neurons, T_sub).astype(np.float32) * 20 - 60,
        'nrt_m_Ts': np.random.rand(n_neurons, T_sub).astype(np.float32),
        'nrt_h_Ts': np.random.rand(n_neurons, T_sub).astype(np.float32),
        'gabaa_per_tc': np.random.rand(n_neurons, T_sub).astype(np.float32) * 10,
        'gabab_per_tc': np.random.rand(n_neurons, T_sub).astype(np.float32) * 5,
    }

    return train_data, val_data, bio_gt
