"""
Extended data loader that provides biological variable targets
aligned to training windows for the A-R3b experiment.

In A-R3, bio variables were only used post-hoc for evaluation.
In A-R3b, they are provided AS TRAINING TARGETS alongside spike data.

Design choice: bio targets are drawn from the canonical trial's
biological variables and shared across all windows. This is a valid
simplification because:
  1. We're testing whether the model CAN encode biology, not whether
     it can track per-trial variability
  2. The canonical bio represents the same biological system
  3. Each window covers 2s of the same 10s simulation
  4. Different windows see different 2s segments (aligned by offset)

Normalization: each variable is z-scored (zero mean, unit variance)
so that MSE loss weights all variables equally regardless of their
natural scale (voltages ~50mV, gating ~0.5, conductances ~10nS).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional


# Variable-to-category mapping (matches DESCARTES 160-dim recovery space)
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

# Expected dimensions per category
EXPECTED_DIMS = {
    'tc_gating': 60,   # 3 vars x 20 neurons
    'nrt_state': 60,   # 3 vars x 20 neurons
    'synaptic': 40,    # 2 vars x 20 neurons
}


class BioAlignedDataset(Dataset):
    """
    Dataset that provides (input_spikes, target_spikes, bio_targets) tuples.

    bio_targets is a dict with keys 'tc_gating', 'nrt_state', 'synaptic',
    each containing a tensor of shape (seq_len, n_vars_in_category).

    The bio targets are drawn from the canonical trial and aligned to the
    training window's temporal position. All windows in a given epoch see
    the SAME bio dynamics but at DIFFERENT time offsets (because they are
    different 2s windows from a 10s simulation).
    """

    def __init__(self, X, Y, Y_binary, bio_ground_truth, seq_len=2000,
                 window_stride=500):
        """
        Args:
            X: (n_windows, seq_len, input_dim) — retinal spike inputs
            Y: (n_windows, seq_len, output_dim) — TC rate targets
            Y_binary: (n_windows, seq_len, output_dim) — TC binary spike targets
            bio_ground_truth: dict from load_ar2_data(), keys like
                'tc_m_T' -> (20, T_full) for canonical, or
                'tc_m_T_gaba0' -> (20, T_full) for GABA-specific
            seq_len: training window length (should match X.shape[1])
            window_stride: stride used during windowing (for offset computation)
        """
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.Y = torch.FloatTensor(Y) if not isinstance(Y, torch.Tensor) else Y
        self.Y_binary = torch.FloatTensor(Y_binary) if not isinstance(Y_binary, torch.Tensor) else Y_binary
        self.seq_len = seq_len
        self.window_stride = window_stride
        self.n_windows = len(X)

        # Build category tensors from bio ground truth
        self.bio_categories, self.norm_params = self._organize_bio_vars(
            bio_ground_truth
        )

        # Compute window offsets for bio variable alignment
        # Each window starts at window_idx * window_stride in the simulation
        # Bio variables are at 1ms resolution (same as training data bins)
        self.window_offsets = [i * window_stride for i in range(self.n_windows)]

    def _organize_bio_vars(self, bio_gt):
        """
        Organize raw bio ground truth into category tensors with normalization.

        Uses bare-name variables (canonical trial, no GABA suffix) when
        available, falling back to the first available GABA-specific version.

        Returns:
            categories: dict of cat_name -> (T_full, n_vars) numpy array
            norm_params: dict of cat_name -> (means, stds) for denormalization
        """
        # Collect arrays per category
        cat_arrays = {
            'tc_gating': [],
            'nrt_state': [],
            'synaptic': [],
        }

        for var_name, category in VAR_TO_CATEGORY.items():
            # Prefer bare name (canonical), fall back to first GABA-specific
            if var_name in bio_gt:
                data = bio_gt[var_name]  # (20, T_full)
            else:
                # Try to find a GABA-specific version
                found = False
                for key in sorted(bio_gt.keys()):
                    if key.startswith(var_name + '_gaba'):
                        data = bio_gt[key]
                        found = True
                        break
                if not found:
                    print(f"  [bio_loader] WARNING: {var_name} not found in bio_gt, skipping")
                    continue

            # data shape: (20_neurons, T_full_timesteps)
            # Transpose to (T_full, 20) for temporal-first layout
            cat_arrays[category].append(data.T)

        # Concatenate within categories and normalize
        categories = {}
        norm_params = {}

        for cat_name, arrays in cat_arrays.items():
            if not arrays:
                print(f"  [bio_loader] WARNING: no variables found for {cat_name}")
                continue

            # Concatenate along variable axis: (T_full, n_vars)
            cat_data = np.concatenate(arrays, axis=1).astype(np.float32)

            # Z-score normalization per variable (critical for balanced MSE)
            means = cat_data.mean(axis=0, keepdims=True)
            stds = cat_data.std(axis=0, keepdims=True)
            stds[stds < 1e-8] = 1.0  # Prevent division by zero

            cat_normalized = (cat_data - means) / stds
            categories[cat_name] = cat_normalized
            norm_params[cat_name] = {'mean': means, 'std': stds}

            print(f"  [bio_loader] {cat_name}: {cat_data.shape[1]} vars, "
                  f"T={cat_data.shape[0]}, range=[{cat_data.min():.3f}, {cat_data.max():.3f}]")

        return categories, norm_params

    def get_bio_dims(self):
        """Return dict of category -> n_variables for model construction."""
        return {k: v.shape[1] for k, v in self.bio_categories.items()}

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        """
        Returns:
            x: (seq_len, input_dim) — input spikes
            y: (seq_len, output_dim) — smoothed rate targets
            y_binary: (seq_len, output_dim) — binary spike targets
            bio: dict of category_name -> (seq_len, n_vars) tensors
        """
        x = self.X[idx]
        y = self.Y[idx]
        y_bin = self.Y_binary[idx]

        # Get bio targets aligned to this window's temporal position
        offset = self.window_offsets[min(idx, len(self.window_offsets) - 1)]
        bio = {}

        for cat_name, cat_data in self.bio_categories.items():
            T_full = cat_data.shape[0]
            start = offset
            end = start + self.seq_len

            if end <= T_full:
                # Normal case: window fits within simulation
                bio_slice = cat_data[start:end]
            elif start < T_full:
                # Window extends beyond simulation — pad with last value
                available = cat_data[start:T_full]
                pad_len = self.seq_len - available.shape[0]
                pad = np.tile(available[-1:], (pad_len, 1))
                bio_slice = np.vstack([available, pad])
            else:
                # Window is entirely beyond simulation — use last seq_len
                bio_slice = cat_data[-self.seq_len:]

            bio[cat_name] = torch.FloatTensor(bio_slice)

        return x, y, y_bin, bio


def bio_collate_fn(batch):
    """
    Custom collate function for BioAlignedDataset.

    Handles the dict-of-tensors in the bio targets by stacking
    each category separately.

    Returns:
        x: (batch, seq_len, input_dim)
        y: (batch, seq_len, output_dim)
        y_binary: (batch, seq_len, output_dim)
        bio: dict of category -> (batch, seq_len, n_vars)
    """
    xs, ys, y_bins, bios = zip(*batch)

    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    y_bin = torch.stack(y_bins, dim=0)

    # Collate bio dicts
    bio_collated = {}
    if bios[0]:  # At least one category exists
        for cat_name in bios[0].keys():
            cat_tensors = [b[cat_name] for b in bios if cat_name in b]
            if cat_tensors:
                bio_collated[cat_name] = torch.stack(cat_tensors, dim=0)

    return x, y, y_bin, bio_collated


def create_bio_dataloaders(train_data, val_data, bio_gt,
                           batch_size=32, num_workers=0):
    """
    Create train and val DataLoaders with bio targets.

    Args:
        train_data: dict from load_ar2_data() with X_train, Y_train, Y_binary_train
        val_data: dict from load_ar2_data() with X_val, Y_val, Y_binary_val
        bio_gt: dict from load_ar2_data() bio ground truth
        batch_size: batch size for DataLoader
        num_workers: number of data loading workers

    Returns:
        train_loader: DataLoader yielding (x, y, y_binary, bio_targets)
        val_loader: DataLoader yielding (x, y, y_binary, bio_targets)
        bio_dims: dict of category -> n_vars (for model construction)
    """
    train_dataset = BioAlignedDataset(
        X=train_data['X_train'],
        Y=train_data['Y_train'],
        Y_binary=train_data.get('Y_binary_train',
                                (train_data['Y_train'] > 0.5).astype(np.float32)),
        bio_ground_truth=bio_gt,
        seq_len=train_data['X_train'].shape[1],
    )

    val_dataset = BioAlignedDataset(
        X=val_data['X_val'],
        Y=val_data['Y_val'],
        Y_binary=val_data.get('Y_binary_val',
                              (val_data['Y_val'] > 0.5).astype(np.float32)),
        bio_ground_truth=bio_gt,
        seq_len=val_data['X_val'].shape[1],
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
    print(f"  [bio_loader] Bio dims: {bio_dims}")
    print(f"  [bio_loader] Train: {len(train_dataset)} windows, "
          f"Val: {len(val_dataset)} windows")

    return train_loader, val_loader, bio_dims
