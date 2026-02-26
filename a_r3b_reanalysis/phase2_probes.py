"""
Phase 2: Probe Complexity Ladder — Ridge + MLP probes with temporal block CV.

The probe ladder separates four failure modes:
  1. Low Ridge, Low MLP → variable genuinely not encoded (GENUINE_ZOMBIE)
  2. Low Ridge, High MLP → nonlinear encoding invisible to linear probes
  3. High Ridge, High MLP → linear encoding (strongest evidence)
  4. High trained, High untrained → spurious (structured input artifact)

Cross-validation MUST split across TRIALS, not timepoints:
  - Window overlap: window_size=2000ms, stride=500ms → 75% overlap
  - Adjacent windows share most data → random CV leaks train→test
  - Fix: all windows from the same HDF5 trial stay in the same fold
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


# ============================================================
# RidgeProbe
# ============================================================

class RidgeProbe:
    """Linear probe (Ridge regression with CV-selected alpha).

    Wraps sklearn RidgeCV with input standardization.
    The simplest possible readout — if this fails, the encoding
    is either nonlinear or absent.
    """

    def __init__(self, alphas=None):
        from a_r3b_reanalysis.config import RIDGE_ALPHAS
        self.alphas = alphas if alphas is not None else RIDGE_ALPHAS
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=self.alphas, store_cv_results=True)

    def fit(self, X, y):
        """Fit Ridge probe on training data.

        Parameters
        ----------
        X : ndarray (n_samples, hidden_dim)
        y : ndarray (n_samples,) or (n_samples, n_targets)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score(self, X, y):
        """R² on test data."""
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)

    @property
    def best_alpha(self):
        return self.model.alpha_


# ============================================================
# MLPProbe (PyTorch)
# ============================================================

class MLPProbeNet(nn.Module):
    """MLP probe network with 1 or 2 hidden layers.

    Architecture follows the guide: hidden_dim=64, dropout=0.3, ReLU.
    """

    def __init__(self, input_dim, output_dim=1, hidden_dim=64,
                 n_layers=1, dropout=0.3):
        super().__init__()
        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Optional second hidden layer
        if n_layers >= 2:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPProbe:
    """MLP nonlinear probe with training loop and early stopping.

    Detects nonlinear encoding that Ridge misses — the critical
    addition from A-R3b that the original analysis lacked.
    """

    def __init__(self, input_dim, output_dim=1, n_layers=1,
                 hidden_dim=None, dropout=None, lr=None,
                 weight_decay=None, epochs=None, patience=None,
                 device='cpu'):
        from a_r3b_reanalysis.config import (
            MLP_HIDDEN_DIM, MLP_DROPOUT, MLP_LR,
            MLP_WEIGHT_DECAY, MLP_EPOCHS, MLP_PATIENCE,
        )
        self.hidden_dim = hidden_dim or MLP_HIDDEN_DIM
        self.dropout = dropout if dropout is not None else MLP_DROPOUT
        self.lr = lr or MLP_LR
        self.weight_decay = weight_decay or MLP_WEIGHT_DECAY
        self.epochs = epochs or MLP_EPOCHS
        self.patience = patience or MLP_PATIENCE
        self.device = device
        self.output_dim = output_dim

        self.model = MLPProbeNet(
            input_dim, output_dim, self.hidden_dim,
            n_layers, self.dropout
        ).to(device)

        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train MLP probe with early stopping.

        Parameters
        ----------
        X_train : ndarray (n_train, hidden_dim)
        y_train : ndarray (n_train,) or (n_train, n_targets)
        X_val : ndarray, optional — for early stopping
        y_val : ndarray, optional
        """
        # Standardize input
        X_train_s = self.scaler.fit_transform(X_train)
        X_t = torch.from_numpy(X_train_s).float().to(self.device)
        y_t = torch.from_numpy(y_train).float().to(self.device)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(1)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_s = self.scaler.transform(X_val)
            X_v = torch.from_numpy(X_val_s).float().to(self.device)
            y_v = torch.from_numpy(y_val).float().to(self.device)
            if y_v.dim() == 1:
                y_v = y_v.unsqueeze(1)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_v)
                    val_loss = criterion(val_pred, y_v).item()
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.model.eval()
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        X_t = torch.from_numpy(X_s).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy()
        return pred.squeeze()

    def score(self, X, y):
        """R² on test data."""
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        return 1.0 - ss_res / ss_tot


# ============================================================
# Temporal Block Cross-Validation
# ============================================================

def temporal_block_cv(hidden_states, targets, trial_ids, probe_type='ridge',
                      n_splits=None, neuron_idx=0, device='cpu',
                      verbose=False):
    """Cross-validate a probe with trial-level grouping.

    This is the critical fix to the original A-R3 analysis: split across
    TRIALS so that overlapping windows never leak across train/test.

    Parameters
    ----------
    hidden_states : ndarray (n_windows, window_bins, hidden_dim)
        Hidden state trajectories from a model.
    targets : ndarray (n_windows, window_bins, n_neurons)
        Biological target traces (from phase0).
    trial_ids : ndarray (n_windows,)
        Integer trial ID for each window — all windows from the same trial
        get the same ID so they stay together in CV splits.
    probe_type : str
        'ridge', 'mlp_1' (1-layer MLP), or 'mlp_2' (2-layer MLP)
    n_splits : int
        Number of CV folds. Defaults to config CV_N_SPLITS.
    neuron_idx : int
        Which neuron's trace to probe (0-19).
    device : str
        Device for MLP probes.
    verbose : bool

    Returns
    -------
    result : dict
        'r2_mean': mean R² across folds
        'r2_std': std of R² across folds
        'r2_folds': list of per-fold R²
        'pearson_mean': mean Pearson r across folds
        'pearson_folds': list of per-fold Pearson r
        'probe_type': str
    """
    from a_r3b_reanalysis.config import CV_N_SPLITS, TEMPORAL_SUBSAMPLE

    if n_splits is None:
        n_splits = CV_N_SPLITS

    # Temporal downsampling: take every Nth bin to shrink probe matrix.
    # 1ms bins × TEMPORAL_SUBSAMPLE(20) = 20ms resolution.
    # Reduces 2.58M rows → 129K rows while preserving encoding signal.
    step = TEMPORAL_SUBSAMPLE
    n_windows, window_bins, hidden_dim = hidden_states.shape
    H_sub = hidden_states[:, ::step, :]           # (n_windows, bins_sub, hidden_dim)
    T_sub = targets[:, ::step, neuron_idx]        # (n_windows, bins_sub)
    bins_sub = H_sub.shape[1]

    X_all = H_sub.reshape(-1, hidden_dim)         # (n_windows * bins_sub, hidden_dim)
    y_all = T_sub.reshape(-1)                     # (n_windows * bins_sub,)

    # Expand trial_ids to per-timepoint: each window's trial_id repeated bins_sub times
    trial_ids_expanded = np.repeat(trial_ids, bins_sub)

    # Skip if target is constant
    if np.std(y_all) < 1e-10:
        return _empty_result(probe_type, n_splits)

    # Get unique trial IDs for splitting
    unique_trials = np.unique(trial_ids)
    n_actual_splits = min(n_splits, len(unique_trials))

    if n_actual_splits < 2:
        return _empty_result(probe_type, n_splits)

    kf = KFold(n_splits=n_actual_splits, shuffle=True, random_state=42)

    r2_folds = []
    pearson_folds = []

    for fold_i, (train_trial_idx, test_trial_idx) in enumerate(kf.split(unique_trials)):
        train_trials = set(unique_trials[train_trial_idx])
        test_trials = set(unique_trials[test_trial_idx])

        # Build masks: all timepoints from train/test trials
        train_mask = np.isin(trial_ids_expanded, list(train_trials))
        test_mask = np.isin(trial_ids_expanded, list(test_trials))

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        if len(X_test) == 0 or np.std(y_test) < 1e-10:
            continue

        # Fit probe
        if probe_type == 'ridge':
            probe = RidgeProbe()
            probe.fit(X_train, y_train)
            r2 = probe.score(X_test, y_test)
            pred = probe.predict(X_test)

        elif probe_type in ('mlp_1', 'mlp_2'):
            n_layers = 1 if probe_type == 'mlp_1' else 2
            probe = MLPProbe(
                input_dim=hidden_dim,
                output_dim=1,
                n_layers=n_layers,
                device=device,
            )
            # Use a fraction of training data for early stopping
            n_train = len(X_train)
            n_val_split = max(1, int(0.15 * n_train))
            perm = np.random.RandomState(42 + fold_i).permutation(n_train)
            val_idx = perm[:n_val_split]
            train_idx = perm[n_val_split:]

            probe.fit(
                X_train[train_idx], y_train[train_idx],
                X_val=X_train[val_idx], y_val=y_train[val_idx],
            )
            r2 = probe.score(X_test, y_test)
            pred = probe.predict(X_test)

        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        # Pearson r
        if np.std(pred) > 1e-10 and np.std(y_test) > 1e-10:
            r, _ = pearsonr(pred, y_test)
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0

        r2_folds.append(r2)
        pearson_folds.append(abs(r))

        if verbose:
            print(f"    Fold {fold_i}: R²={r2:.4f}, |r|={abs(r):.4f} "
                  f"(train={len(X_train)}, test={len(X_test)})")

    if len(r2_folds) == 0:
        return _empty_result(probe_type, n_splits)

    return {
        'r2_mean': float(np.mean(r2_folds)),
        'r2_std': float(np.std(r2_folds)),
        'r2_folds': r2_folds,
        'pearson_mean': float(np.mean(pearson_folds)),
        'pearson_folds': pearson_folds,
        'probe_type': probe_type,
        'n_folds_completed': len(r2_folds),
    }


def _empty_result(probe_type, n_splits):
    """Return a zeroed result for constant targets or insufficient data."""
    return {
        'r2_mean': 0.0,
        'r2_std': 0.0,
        'r2_folds': [0.0] * n_splits,
        'pearson_mean': 0.0,
        'pearson_folds': [0.0] * n_splits,
        'probe_type': probe_type,
        'n_folds_completed': 0,
    }


def probe_single_variable(hidden_states, targets, trial_ids,
                           target_name, neuron_idx=0,
                           probe_types=None, device='cpu',
                           verbose=False):
    """Run all probe types on one target variable.

    Parameters
    ----------
    hidden_states : ndarray (n_windows, window_bins, hidden_dim)
    targets : ndarray (n_windows, window_bins, n_neurons)
    trial_ids : ndarray (n_windows,)
    target_name : str
    neuron_idx : int
    probe_types : list of str
    device : str
    verbose : bool

    Returns
    -------
    results : dict
        probe_type -> temporal_block_cv result dict
    """
    from a_r3b_reanalysis.config import PROBE_TYPES

    if probe_types is None:
        probe_types = PROBE_TYPES

    results = {}
    for pt in probe_types:
        if verbose:
            print(f"  [{target_name}] neuron={neuron_idx}, probe={pt}")

        result = temporal_block_cv(
            hidden_states, targets, trial_ids,
            probe_type=pt, neuron_idx=neuron_idx,
            device=device, verbose=verbose,
        )
        results[pt] = result

    return results
