"""
Bio-aligned data loader for A-R3b: GRU-ODE with Auxiliary Biological Loss.

CRITICAL DESIGN: Each training window gets bio targets from its OWN
source trial at its OWN temporal offset. This ensures perfect alignment
between spike data and biological variable targets.

Previous version (BUGGY): Used canonical trial's bio variables for ALL
windows, regardless of which GABA level or seed produced the spikes.
This meant windows from trial_gaba74_seed43 got bio targets from
trial_gaba30_seed42 — completely wrong dynamics.

Architecture:
  1. Reuse load_ar2_data() for spike data (unchanged, well-tested)
  2. Re-scan the same trial HDF5 files to load intermediates per-trial
  3. Build a window-to-trial mapping (trial_idx, temporal_offset)
  4. Normalize bio variables globally across all training trials
  5. In __getitem__, slice the correct trial at the correct offset

Memory efficiency: stores per-trial category arrays (~730 MB for 114
training trials) rather than pre-windowed bio data (~2.5 GB), and
slices on-the-fly in __getitem__.
"""
import numpy as np
import h5py
import re
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

# Import constants from existing data loader (must match exactly for
# window alignment — same bin size, window size, stride, seed splits)
from .ar3_data_loader import (
    load_ar2_data,
    BIN_DT_MS, WINDOW_SIZE_MS, WINDOW_STRIDE_MS,
    TRAIN_SEEDS, VAL_SEEDS,
)


# ── Variable organization ────────────────────────────────────────────

# Variable-to-category mapping (DESCARTES 160-dim recovery space)
VAR_TO_CATEGORY = {
    'tc_m_T': 'tc_gating',
    'tc_h_T': 'tc_gating',
    'tc_m_h': 'tc_gating',
    'nrt_m_Ts': 'nrt_state',
    'nrt_h_Ts': 'nrt_state',
    'V_nrt': 'nrt_state',
    'gabaa_per_tc': 'synaptic',
    'gabab_per_tc': 'synaptic',
}

# Ordered variable names per category (order matters for consistent
# concatenation across trials — must be the same everywhere)
CATEGORY_VARS = {
    'tc_gating': ['tc_m_T', 'tc_h_T', 'tc_m_h'],        # 3 × 20 = 60 vars
    'nrt_state': ['nrt_m_Ts', 'nrt_h_Ts', 'V_nrt'],      # 3 × 20 = 60 vars
    'synaptic': ['gabaa_per_tc', 'gabab_per_tc'],          # 2 × 20 = 40 vars
}

EXPECTED_DIMS = {
    'tc_gating': 60,
    'nrt_state': 60,
    'synaptic': 40,
}


# ── Trial parsing ────────────────────────────────────────────────────

def _parse_trials(data_dir):
    """Parse trial HDF5 filenames into structured list.

    Must produce the EXACT same ordering as ar3_data_loader.load_ar2_data()
    to ensure window indices match between spike and bio data. Both use
    sorted(glob('trial_gaba*.h5')) and filter by seed.
    """
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob('trial_gaba*.h5'))

    trials = []
    for f in h5_files:
        match = re.match(r'trial_gaba([\d.]+)_seed(\d+)\.h5', f.name)
        if match:
            trials.append({
                'filepath': f,
                'gaba': float(match.group(1)),
                'seed': int(match.group(2)),
            })
    return trials


# ── Per-trial intermediate loading ───────────────────────────────────

def _load_trial_intermediates(filepath):
    """Load the 8 biological intermediate variables from one HDF5 trial.

    Returns dict of var_name -> (n_neurons, T) numpy array.
    Handles V_nrt being at top level (not in intermediates/ group).
    """
    intermediates = {}
    with h5py.File(filepath, 'r') as f:
        for var_name in VAR_TO_CATEGORY.keys():
            if var_name == 'V_nrt':
                # V_nrt is stored at top level, not in intermediates/
                if 'V_nrt' in f:
                    intermediates['V_nrt'] = f['V_nrt'][:].astype(np.float32)
            else:
                key = f'intermediates/{var_name}'
                if key in f:
                    intermediates[var_name] = f[key][:].astype(np.float32)
    return intermediates


def _organize_into_categories(intermediates):
    """Organize per-trial intermediates into category arrays.

    Args:
        intermediates: dict of var_name -> (n_neurons, T) arrays

    Returns:
        categories: dict of cat_name -> (T, n_vars) arrays
            e.g., 'tc_gating' -> (10000, 60) for 3 vars × 20 neurons
    """
    categories = {}

    for cat_name, var_names in CATEGORY_VARS.items():
        arrays = []
        for var_name in var_names:
            if var_name in intermediates:
                # (n_neurons, T) → (T, n_neurons) for temporal-first layout
                arrays.append(intermediates[var_name].T)

        if arrays:
            # Concatenate along var axis: (T, 20) + (T, 20) + ... = (T, n_vars)
            categories[cat_name] = np.concatenate(arrays, axis=1)

    return categories


def _load_all_trial_categories(trials):
    """Load intermediates for all trials and organize into categories.

    Returns list of dicts, one per trial, each dict: cat_name -> (T, n_vars).
    """
    all_cats = []
    for i, trial_info in enumerate(trials):
        intermediates = _load_trial_intermediates(trial_info['filepath'])
        categories = _organize_into_categories(intermediates)
        all_cats.append(categories)

        if (i + 1) % 20 == 0:
            print(f"    Bio loaded {i + 1}/{len(trials)} trials")

    return all_cats


# ── Normalization ────────────────────────────────────────────────────

def _compute_global_normalization(trial_cats):
    """Compute z-score parameters per variable across ALL training trials.

    Global normalization ensures consistent z-scores across trials,
    regardless of GABA level. This is critical because:
      - Voltages range ~50mV, gating vars ~0-1, conductances ~0-10 nS
      - MSE loss needs all variables on the same scale
      - Per-trial normalization would break cross-trial comparisons

    Uses Welford-style accumulation (sum + sum_of_squares) for numerical
    stability with large datasets.
    """
    cat_sums = {}
    cat_sq_sums = {}
    cat_counts = {}

    for trial_cat in trial_cats:
        for cat_name, data in trial_cat.items():
            n_vars = data.shape[1]
            if cat_name not in cat_sums:
                cat_sums[cat_name] = np.zeros(n_vars, dtype=np.float64)
                cat_sq_sums[cat_name] = np.zeros(n_vars, dtype=np.float64)
                cat_counts[cat_name] = 0

            cat_sums[cat_name] += data.sum(axis=0).astype(np.float64)
            cat_sq_sums[cat_name] += (data.astype(np.float64) ** 2).sum(axis=0)
            cat_counts[cat_name] += data.shape[0]

    norm_params = {}
    for cat_name in cat_sums:
        n = cat_counts[cat_name]
        mean = (cat_sums[cat_name] / n).astype(np.float32)
        variance = (cat_sq_sums[cat_name] / n
                     - mean.astype(np.float64) ** 2).astype(np.float32)
        std = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
        std[std < 1e-8] = 1.0  # Prevent division by zero

        norm_params[cat_name] = {'mean': mean, 'std': std}
        print(f"  [bio_loader] {cat_name}: {len(mean)} vars, "
              f"mean=[{mean.min():.3f}, {mean.max():.3f}], "
              f"std=[{std.min():.4f}, {std.max():.4f}]")

    return norm_params


def _normalize_trial_categories(trial_cats, norm_params):
    """Apply z-score normalization to all trial categories in-place.

    After this, each variable has approximately zero mean and unit
    variance across the training set. Val data uses training statistics
    (no data leakage).
    """
    for trial_cat in trial_cats:
        for cat_name in list(trial_cat.keys()):
            if cat_name in norm_params:
                mean = norm_params[cat_name]['mean']   # (n_vars,)
                std = norm_params[cat_name]['std']
                trial_cat[cat_name] = ((trial_cat[cat_name] - mean) / std
                                       ).astype(np.float32)


# ── Window-to-trial mapping ─────────────────────────────────────────

def _build_window_trial_map(trials):
    """Build mapping from window index to (trial_idx, temporal_offset).

    Uses the EXACT same windowing logic as ar3_data_loader._create_windows()
    to ensure window indices align with the spike data (X_train/X_val).

    Each 10s trial (10000 bins at 1ms) with window=2000, stride=500:
      starts = [0, 500, 1000, ..., 8000] → 17 windows per trial.

    The mapping is ordered: all windows from trial 0, then trial 1, etc.
    This matches np.concatenate(X_list) in _process_trials().
    """
    window_bins = int(WINDOW_SIZE_MS / BIN_DT_MS)     # 2000
    stride_bins = int(WINDOW_STRIDE_MS / BIN_DT_MS)   # 500

    window_map = []

    for trial_idx, trial_info in enumerate(trials):
        # Read trial duration to compute correct window count
        with h5py.File(trial_info['filepath'], 'r') as f:
            duration_s = float(f['meta'].attrs['duration_s'])

        T = int(duration_s * 1000.0 / BIN_DT_MS)  # Total bins
        starts = list(range(0, T - window_bins + 1, stride_bins))

        for start in starts:
            window_map.append((trial_idx, start))

    return window_map


# ── Dataset ──────────────────────────────────────────────────────────

class BioAlignedDataset(Dataset):
    """Dataset with per-window bio targets properly aligned per-trial.

    Each window's bio targets come from its ACTUAL source trial at its
    ACTUAL temporal offset — not from a shared canonical trial.

    Memory-efficient: stores per-trial category arrays (not per-window)
    and slices on-the-fly in __getitem__.
    """

    def __init__(self, X, Y, Y_binary, trial_categories, window_trial_map,
                 seq_len=2000):
        """
        Args:
            X: (n_windows, seq_len, input_dim) — retinal spike inputs
            Y: (n_windows, seq_len, output_dim) — TC rate targets
            Y_binary: (n_windows, seq_len, output_dim) — TC binary spike targets
            trial_categories: list of dicts per trial, each dict:
                cat_name -> (T, n_vars) NORMALIZED numpy array
            window_trial_map: list of (trial_idx, offset) tuples, one per window
            seq_len: window length in bins (2000)
        """
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.Y = torch.FloatTensor(Y) if not isinstance(Y, torch.Tensor) else Y
        self.Y_binary = (torch.FloatTensor(Y_binary) if not isinstance(Y_binary, torch.Tensor)
                         else Y_binary)

        self.trial_categories = trial_categories
        self.window_trial_map = window_trial_map
        self.seq_len = seq_len
        self.n_windows = len(X)

        # Verify alignment
        assert len(window_trial_map) == self.n_windows, (
            f"Window count mismatch: spike data has {self.n_windows} windows "
            f"but bio mapping has {len(window_trial_map)}. Trial ordering diverged!"
        )

    def get_bio_dims(self):
        """Return dict of category -> n_variables for model construction."""
        dims = {}
        if self.trial_categories:
            for cat_name, data in self.trial_categories[0].items():
                dims[cat_name] = data.shape[1]
        return dims

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        """
        Returns:
            x: (seq_len, input_dim) — input spikes
            y: (seq_len, output_dim) — smoothed rate targets
            y_binary: (seq_len, output_dim) — binary spike targets
            bio: dict of cat_name -> (seq_len, n_vars) tensors
        """
        trial_idx, offset = self.window_trial_map[idx]

        bio = {}
        for cat_name, cat_data in self.trial_categories[trial_idx].items():
            # Slice the exact temporal window from this trial's bio data
            bio_slice = cat_data[offset:offset + self.seq_len]
            bio[cat_name] = torch.FloatTensor(bio_slice)

        return self.X[idx], self.Y[idx], self.Y_binary[idx], bio


# ── Collate function ─────────────────────────────────────────────────

def bio_collate_fn(batch):
    """Custom collate for BioAlignedDataset batches.

    Handles the dict-of-tensors bio targets by stacking each
    category separately into (batch, seq_len, n_vars) tensors.
    """
    xs, ys, y_bins, bios = zip(*batch)

    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    y_bin = torch.stack(y_bins, dim=0)

    bio_collated = {}
    if bios[0]:
        for cat_name in bios[0].keys():
            cat_tensors = [b[cat_name] for b in bios if cat_name in b]
            if cat_tensors:
                bio_collated[cat_name] = torch.stack(cat_tensors, dim=0)

    return x, y, y_bin, bio_collated


# ── Main entry point ─────────────────────────────────────────────────

def load_bio_aligned_data(data_dir, batch_size=32, num_workers=0):
    """
    Load A-R2 data with per-window bio targets for A-R3b training.

    This is the correct entry point for the A-R3b experiment. It:
      1. Loads spike data via load_ar2_data() (unchanged, well-tested)
      2. Re-scans trial HDF5 files to load biological intermediates
      3. Builds per-window bio targets aligned to each window's source trial
      4. Normalizes bio variables globally across training trials
      5. Creates DataLoaders with (x, y, y_binary, bio_targets) batches

    Args:
        data_dir: Path to directory containing trial_gaba*.h5 files
        batch_size: batch size for DataLoader
        num_workers: number of data loading workers

    Returns:
        train_loader: DataLoader yielding (x, y, y_binary, bio_targets)
        val_loader: DataLoader for validation
        bio_dims: dict of category -> n_vars (for model construction)
        bio_gt: dict of var_name -> (n_neurons, T) (for legacy evaluation)
        data_info: dict with input_dim, output_dim, n_train, n_val
    """
    # ── Step 1: Load spike data (reuses existing well-tested code) ──
    print("\n  Loading spike data via load_ar2_data()...")
    train_data, val_data, bio_gt = load_ar2_data(data_dir)

    print(f"\n  Spike data loaded:")
    print(f"    Train: X={train_data['X_train'].shape}, "
          f"Y={train_data['Y_train'].shape}")
    print(f"    Val:   X={val_data['X_val'].shape}, "
          f"Y={val_data['Y_val'].shape}")

    # ── Step 2: Re-scan trials for intermediates ──────────────────
    trials = _parse_trials(data_dir)
    if not trials:
        raise ValueError(f"No trial_gaba*.h5 files found in {data_dir}")

    train_trials = [t for t in trials if t['seed'] in TRAIN_SEEDS]
    val_trials = [t for t in trials if t['seed'] in VAL_SEEDS]

    print(f"\n  Loading biological intermediates...")
    print(f"    Train: {len(train_trials)} trials")
    train_trial_cats = _load_all_trial_categories(train_trials)

    print(f"    Val: {len(val_trials)} trials")
    val_trial_cats = _load_all_trial_categories(val_trials)

    # Verify consistent dimensions across trials
    _verify_category_dims(train_trial_cats, "train")
    _verify_category_dims(val_trial_cats, "val")

    # ── Step 3: Compute normalization from training data only ─────
    print(f"\n  Computing global normalization (training trials)...")
    norm_params = _compute_global_normalization(train_trial_cats)

    # Apply to both train and val (val uses training statistics — no leakage)
    _normalize_trial_categories(train_trial_cats, norm_params)
    _normalize_trial_categories(val_trial_cats, norm_params)

    # ── Step 4: Build window-to-trial mappings ────────────────────
    print(f"\n  Building window-trial alignment...")
    train_map = _build_window_trial_map(train_trials)
    val_map = _build_window_trial_map(val_trials)

    print(f"    Train: {len(train_map)} windows from "
          f"{len(train_trials)} trials")
    print(f"    Val:   {len(val_map)} windows from "
          f"{len(val_trials)} trials")

    # Verify window counts match spike data
    n_spike_train = train_data['X_train'].shape[0]
    n_spike_val = val_data['X_val'].shape[0]

    if len(train_map) != n_spike_train:
        raise ValueError(
            f"ALIGNMENT ERROR: Spike data has {n_spike_train} train windows "
            f"but bio mapping has {len(train_map)}. "
            f"Trial parsing order diverged between ar3_data_loader and "
            f"bio_data_loader!"
        )
    if len(val_map) != n_spike_val:
        raise ValueError(
            f"ALIGNMENT ERROR: Spike data has {n_spike_val} val windows "
            f"but bio mapping has {len(val_map)}. "
            f"Trial parsing order diverged!"
        )

    print(f"    Window counts verified: "
          f"train={n_spike_train}, val={n_spike_val}")

    # ── Step 5: Create datasets and dataloaders ───────────────────
    seq_len = train_data['X_train'].shape[1]

    Y_binary_train = train_data.get(
        'Y_binary_train',
        (train_data['Y_train'] > 0.5).astype(np.float32)
    )
    Y_binary_val = val_data.get(
        'Y_binary_val',
        (val_data['Y_val'] > 0.5).astype(np.float32)
    )

    train_dataset = BioAlignedDataset(
        X=train_data['X_train'],
        Y=train_data['Y_train'],
        Y_binary=Y_binary_train,
        trial_categories=train_trial_cats,
        window_trial_map=train_map,
        seq_len=seq_len,
    )

    val_dataset = BioAlignedDataset(
        X=val_data['X_val'],
        Y=val_data['Y_val'],
        Y_binary=Y_binary_val,
        trial_categories=val_trial_cats,
        window_trial_map=val_map,
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=bio_collate_fn, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=bio_collate_fn, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    bio_dims = train_dataset.get_bio_dims()

    print(f"\n  Bio-aligned data ready:")
    print(f"    Categories: {bio_dims}")
    print(f"    Total bio vars: {sum(bio_dims.values())}")
    print(f"    Train: {len(train_dataset)} windows, "
          f"Val: {len(val_dataset)} windows")

    data_info = {
        'input_dim': int(train_data['X_train'].shape[-1]),
        'output_dim': int(train_data['Y_train'].shape[-1]),
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'norm_params': {k: {kk: vv.tolist() for kk, vv in v.items()}
                        for k, v in norm_params.items()},
    }

    return train_loader, val_loader, bio_dims, bio_gt, data_info


def _verify_category_dims(trial_cats, split_name):
    """Verify all trials have consistent category dimensions."""
    if not trial_cats:
        return

    ref_dims = {cat: data.shape[1]
                for cat, data in trial_cats[0].items()}

    for i, trial_cat in enumerate(trial_cats[1:], 1):
        for cat_name, data in trial_cat.items():
            if data.shape[1] != ref_dims.get(cat_name, data.shape[1]):
                raise ValueError(
                    f"Dimension mismatch in {split_name} trial {i}: "
                    f"{cat_name} has {data.shape[1]} vars but trial 0 "
                    f"has {ref_dims[cat_name]}"
                )

    print(f"    {split_name}: all {len(trial_cats)} trials have "
          f"consistent dims: {ref_dims}")
