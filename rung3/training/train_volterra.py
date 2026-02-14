"""
Volterra-Laguerre training script.

Fits the GLVM via ridge regression (CPU only, no iterative training).
"""

import os
import time
import json
import numpy as np

from rung3.config import (
    DATA_DIR, CHECKPOINT_DIR, TRAIN_SEEDS, VAL_SEEDS,
    VOLTERRA_N_BASES, VOLTERRA_ALPHA, VOLTERRA_MEMORY_MS,
    VOLTERRA_ORDER, VOLTERRA_RIDGE_ALPHA,
)
from rung3.phase0_recording import load_trial_hdf5, list_trials
from rung3.preprocessing import preprocess_trial
from rung3.models.volterra_laguerre import VolterraLaguerre


def train_volterra(data_dir=DATA_DIR, checkpoint_dir=CHECKPOINT_DIR,
                    verbose=True):
    """Fit Volterra-Laguerre model on training data.

    Returns
    -------
    model : VolterraLaguerre
    results : dict
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = VolterraLaguerre(
        n_bases=VOLTERRA_N_BASES,
        alpha=VOLTERRA_ALPHA,
        memory_ms=VOLTERRA_MEMORY_MS,
        order=VOLTERRA_ORDER,
        ridge_alpha=VOLTERRA_RIDGE_ALPHA,
    )

    if verbose:
        print(f"Volterra-Laguerre Model")
        print(f"  Bases: {VOLTERRA_N_BASES}, α={VOLTERRA_ALPHA}")
        print(f"  Memory: {VOLTERRA_MEMORY_MS}ms, Order: {VOLTERRA_ORDER}")
        print(f"  Features: {model.n_features}")
        print(f"  Ridge α: {VOLTERRA_RIDGE_ALPHA}")

    # Load training data
    all_trials = list_trials(data_dir)
    train_trials = [t for t in all_trials if t['seed'] in TRAIN_SEEDS]
    val_trials = [t for t in all_trials if t['seed'] in VAL_SEEDS]

    if verbose:
        print(f"\nTraining trials: {len(train_trials)}")
        print(f"Validation trials: {len(val_trials)}")

    # Preprocess training data
    X_train_list, Y_train_list = [], []
    for i, trial_info in enumerate(train_trials):
        if verbose:
            print(f"  Loading train [{i+1}/{len(train_trials)}] "
                  f"{os.path.basename(trial_info['filepath'])}")
        data = load_trial_hdf5(trial_info['filepath'])
        X, Y_rate, _, _ = preprocess_trial(data)
        # Flatten windows for Volterra (it can handle full sequences)
        for w in range(X.shape[0]):
            X_train_list.append(X[w])
            Y_train_list.append(Y_rate[w])

    if verbose:
        print(f"  Total training windows: {len(X_train_list)}")

    # Fit
    t0 = time.time()
    if verbose:
        print("\nFitting ridge regression...")

    train_metrics = model.fit(X_train_list, Y_train_list)
    fit_time = time.time() - t0

    if verbose:
        print(f"  Train MSE: {train_metrics['mse']:.6f}")
        print(f"  Train Correlation: {train_metrics['correlation']:.4f}")
        print(f"  Fit time: {fit_time:.1f}s")

    # Evaluate on validation set
    if verbose:
        print("\nEvaluating on validation set...")

    val_mses, val_corrs = [], []
    for i, trial_info in enumerate(val_trials):
        data = load_trial_hdf5(trial_info['filepath'])
        X, Y_rate, _, _ = preprocess_trial(data)
        for w in range(X.shape[0]):
            y_pred = model.forward(X[w])
            mse = float(np.mean((Y_rate[w] - y_pred) ** 2))
            # Per-channel correlation
            corrs = []
            for ch in range(Y_rate.shape[-1]):
                if np.std(Y_rate[w][:, ch]) > 1e-8:
                    c = np.corrcoef(Y_rate[w][:, ch], y_pred[:, ch])[0, 1]
                    corrs.append(c)
            val_mses.append(mse)
            if corrs:
                val_corrs.append(np.mean(corrs))

    val_mse = float(np.mean(val_mses))
    val_corr = float(np.mean(val_corrs)) if val_corrs else 0.0

    if verbose:
        print(f"  Val MSE: {val_mse:.6f}")
        print(f"  Val Correlation: {val_corr:.4f}")

    # Save model
    model_path = os.path.join(checkpoint_dir, 'volterra_model.npz')
    model.save(model_path)
    if verbose:
        print(f"\nModel saved: {model_path}")

    results = {
        'model_type': 'volterra_laguerre',
        'n_bases': VOLTERRA_N_BASES,
        'alpha': VOLTERRA_ALPHA,
        'n_features': model.n_features,
        'train_mse': train_metrics['mse'],
        'train_corr': train_metrics['correlation'],
        'val_mse': val_mse,
        'val_corr': val_corr,
        'fit_time_s': fit_time,
        'n_train_windows': len(X_train_list),
    }

    results_path = os.path.join(checkpoint_dir, 'volterra_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return model, results


if __name__ == '__main__':
    model, results = train_volterra()
    print(f"\nDone. Val correlation: {results['val_corr']:.4f}")
