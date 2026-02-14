"""
PyTorch Dataset and DataModule for Rung 3 training.

Loads preprocessed windows from HDF5 trials, splits by seed
(train seeds: 42-44, val seeds: 45-46).
"""

import os
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from rung3.config import (
    TRAIN_SEEDS, VAL_SEEDS, BATCH_SIZE, NUM_WORKERS, DATA_DIR,
)
from rung3.phase0_recording import load_trial_hdf5, list_trials
from rung3.preprocessing import preprocess_trial


class ThalamicDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed trial windows.

    Each item: (input_seq, rate_target, binary_target)
      input_seq:     (window_bins, 21)
      rate_target:   (window_bins, 20)
      binary_target: (window_bins, 20)
    """

    def __init__(self, X_all, Y_rate_all, Y_binary_all,
                 intermediates_all=None):
        """
        Parameters
        ----------
        X_all : ndarray (total_windows, window_bins, 21)
        Y_rate_all : ndarray (total_windows, window_bins, 20)
        Y_binary_all : ndarray (total_windows, window_bins, 20)
        intermediates_all : dict of ndarray, optional
            Each value: (total_windows, window_bins, n_dims)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required: pip install torch")

        self.X = torch.from_numpy(X_all)
        self.Y_rate = torch.from_numpy(Y_rate_all)
        self.Y_binary = torch.from_numpy(Y_binary_all)
        self.intermediates = None
        if intermediates_all:
            self.intermediates = {
                k: torch.from_numpy(v) for k, v in intermediates_all.items()
            }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            'input': self.X[idx],
            'rate_target': self.Y_rate[idx],
            'binary_target': self.Y_binary[idx],
        }
        if self.intermediates:
            item['intermediates'] = {
                k: v[idx] for k, v in self.intermediates.items()
            }
        return item


def load_and_preprocess_trials(seeds, data_dir=DATA_DIR,
                                include_intermediates=False, verbose=True):
    """Load all HDF5 trials for given seeds, preprocess, and concatenate.

    Parameters
    ----------
    seeds : list of int
    data_dir : str
    include_intermediates : bool
    verbose : bool

    Returns
    -------
    X : ndarray (total_windows, window_bins, 21)
    Y_rate : ndarray (total_windows, window_bins, 20)
    Y_binary : ndarray (total_windows, window_bins, 20)
    intermediates : dict of ndarray or None
    """
    all_trials = list_trials(data_dir)
    selected = [t for t in all_trials if t['seed'] in seeds]

    if verbose:
        print(f"Loading {len(selected)} trials for seeds {seeds}")

    X_list, Yr_list, Yb_list = [], [], []
    inter_lists = {} if include_intermediates else None

    for i, trial_info in enumerate(selected):
        if verbose:
            print(f"  [{i+1}/{len(selected)}] {os.path.basename(trial_info['filepath'])}")

        data = load_trial_hdf5(trial_info['filepath'])
        X, Y_rate, Y_binary, inter_w = preprocess_trial(data)

        X_list.append(X)
        Yr_list.append(Y_rate)
        Yb_list.append(Y_binary)

        if include_intermediates and inter_w:
            for key, val in inter_w.items():
                if key not in inter_lists:
                    inter_lists[key] = []
                inter_lists[key].append(val)

    X_all = np.concatenate(X_list, axis=0)
    Yr_all = np.concatenate(Yr_list, axis=0)
    Yb_all = np.concatenate(Yb_list, axis=0)

    inter_all = None
    if include_intermediates and inter_lists:
        inter_all = {k: np.concatenate(v, axis=0)
                     for k, v in inter_lists.items()}

    if verbose:
        print(f"Total windows: {X_all.shape[0]}")
        print(f"Input shape:  {X_all.shape}")
        print(f"Target shape: {Yr_all.shape}")

    return X_all, Yr_all, Yb_all, inter_all


def create_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        include_intermediates=False, verbose=True):
    """Create train and validation DataLoaders.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")

    # Load and preprocess
    X_train, Yr_train, Yb_train, inter_train = load_and_preprocess_trials(
        TRAIN_SEEDS, data_dir, include_intermediates, verbose)
    X_val, Yr_val, Yb_val, inter_val = load_and_preprocess_trials(
        VAL_SEEDS, data_dir, include_intermediates, verbose)

    # Create datasets
    train_ds = ThalamicDataset(X_train, Yr_train, Yb_train, inter_train)
    val_ds = ThalamicDataset(X_val, Yr_val, Yb_val, inter_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
