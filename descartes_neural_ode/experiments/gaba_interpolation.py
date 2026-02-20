"""
GABA Interpolation Hold-Out Test (from Guide Phase 5.1)

Train on GABA = [0, 5, 10, 15, 25, 30, 35], hold out GABA = 20.
Tests whether the model can interpolate to unseen GABA conductance levels.

This mirrors how a real prosthesis would need to adapt to changing
neuromodulatory state. If the model can only reproduce output at training
GABA levels but fails at interpolated levels, it has memorized specific
operating points rather than learning the underlying transformation.

The test is especially important because GABA=20 nS is near the
bifurcation threshold (~29 nS) — the transition region between tonic
and oscillatory regimes where the circuit's behavior changes most
dramatically.
"""
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ar3_data_loader import (
    _process_trials, _generate_synthetic_data,
    TRAIN_SEEDS, VAL_SEEDS, GABA_MAX_NS
)
from core.full_training_pipeline import FullTrainingPipeline
from evaluation.spike_metrics import spike_train_correlation, evaluate_spike_metrics
from evaluation.coherence_test import test_coherence_preservation

import re
import h5py


# Default held-out GABA level
HOLDOUT_GABA = 20.0
# GABA levels to train on (guide Phase 5.1)
TRAIN_GABA_LEVELS = [0, 5, 10, 15, 25, 30, 35]


def load_data_with_gaba_holdout(
    data_dir: str,
    holdout_gaba: float = HOLDOUT_GABA,
    train_gaba: Optional[List[float]] = None,
    tolerance: float = 1.0,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load data with GABA interpolation hold-out.

    Parameters
    ----------
    data_dir : str
        Path to directory with trial HDF5 files.
    holdout_gaba : float
        GABA level to hold out for interpolation test.
    train_gaba : list of float, optional
        GABA levels for training. If None, uses all except holdout.
    tolerance : float
        Tolerance for GABA level matching.

    Returns
    -------
    train_data : dict
        Training data (all seeds, excluding holdout GABA).
    val_data : dict
        Validation data (held-out seeds, excluding holdout GABA).
    holdout_data : dict
        Held-out GABA level data (all seeds at holdout GABA).
    bio_gt : dict
        Bio ground truth variables.
    """
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob('trial_gaba*.h5'))

    if not h5_files:
        print(f"WARNING: No trial files found in {data_dir}")
        # Return synthetic
        td, vd, bg = _generate_synthetic_data()
        return td, vd, vd, bg

    # Parse filenames
    trials = []
    for f in h5_files:
        match = re.match(r'trial_gaba([\d.]+)_seed(\d+)\.h5', f.name)
        if match:
            trials.append({
                'filepath': f,
                'gaba': float(match.group(1)),
                'seed': int(match.group(2)),
            })

    # Split into training GABA levels and holdout
    if train_gaba is None:
        train_gaba = TRAIN_GABA_LEVELS

    train_trials = []
    val_trials = []
    holdout_trials = []

    for t in trials:
        is_holdout = abs(t['gaba'] - holdout_gaba) < tolerance

        if is_holdout:
            holdout_trials.append(t)
        elif t['seed'] in TRAIN_SEEDS:
            if any(abs(t['gaba'] - g) < tolerance for g in train_gaba):
                train_trials.append(t)
        elif t['seed'] in VAL_SEEDS:
            if any(abs(t['gaba'] - g) < tolerance for g in train_gaba):
                val_trials.append(t)

    print(f"GABA Interpolation Split:")
    print(f"  Train GABA levels: {train_gaba}")
    print(f"  Holdout GABA: {holdout_gaba}")
    print(f"  Train trials: {len(train_trials)}")
    print(f"  Val trials:   {len(val_trials)}")
    print(f"  Holdout trials: {len(holdout_trials)}")

    # Process train/val/holdout
    if train_trials:
        X_train_list, Y_train_list, Yb_train_list, _ = _process_trials(train_trials)
        X_train = np.concatenate(X_train_list, axis=0)
        Y_train = np.concatenate(Y_train_list, axis=0)
        Yb_train = np.concatenate(Yb_train_list, axis=0)
    else:
        print("WARNING: No training trials found. Using synthetic data.")
        td, vd, bg = _generate_synthetic_data()
        return td, vd, vd, bg

    if val_trials:
        X_val_list, Y_val_list, Yb_val_list, _ = _process_trials(val_trials)
        X_val = np.concatenate(X_val_list, axis=0)
        Y_val = np.concatenate(Y_val_list, axis=0)
        Yb_val = np.concatenate(Yb_val_list, axis=0)
    else:
        # Use a subset of training data for validation
        n = max(1, X_train.shape[0] // 5)
        X_val, Y_val, Yb_val = X_train[:n], Y_train[:n], Yb_train[:n]
        X_train, Y_train, Yb_train = X_train[n:], Y_train[n:], Yb_train[n:]

    train_data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'Y_binary_train': Yb_train,
        'X_short': X_train[:, :200, :],
        'Y_short': Y_train[:, :200, :],
    }
    val_data = {
        'X_val': X_val,
        'Y_val': Y_val,
        'Y_binary_val': Yb_val,
        'T': 2000,
    }

    # Process holdout data
    if holdout_trials:
        X_ho_list, Y_ho_list, Yb_ho_list, _ = _process_trials(holdout_trials)
        X_holdout = np.concatenate(X_ho_list, axis=0)
        Y_holdout = np.concatenate(Y_ho_list, axis=0)
        Yb_holdout = np.concatenate(Yb_ho_list, axis=0)
    else:
        print("WARNING: No holdout trials found. Using val data for holdout test.")
        X_holdout, Y_holdout, Yb_holdout = X_val, Y_val, Yb_val

    holdout_data = {
        'X_holdout': X_holdout,
        'Y_holdout': Y_holdout,
        'Y_binary_holdout': Yb_holdout,
        'gaba_level': holdout_gaba,
    }

    # Bio ground truth (placeholder — use full dataset)
    from data.ar3_data_loader import _build_bio_ground_truth
    bio_gt = _build_bio_ground_truth(trials)

    return train_data, val_data, holdout_data, bio_gt


def evaluate_interpolation(
    model: torch.nn.Module,
    holdout_data: Dict,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """Evaluate model on held-out GABA level.

    Parameters
    ----------
    model : nn.Module
    holdout_data : dict
    device : str
    verbose : bool

    Returns
    -------
    result : dict
    """
    X = holdout_data['X_holdout']
    Y = holdout_data['Y_holdout']
    gaba = holdout_data.get('gaba_level', 'unknown')

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, device=device, dtype=torch.float32)
        output = model(x_tensor)
        y_pred = output[0] if isinstance(output, tuple) else output
        y_pred_np = y_pred.cpu().numpy()

    # Spike correlation
    per_window_corrs = []
    for w in range(Y.shape[0]):
        corr, _ = spike_train_correlation(Y[w], y_pred_np[w])
        per_window_corrs.append(corr)

    mean_corr = float(np.mean(per_window_corrs))
    std_corr = float(np.std(per_window_corrs))

    # MSE on held-out data
    mse = float(np.mean((y_pred_np - Y) ** 2))

    # Coherence test on holdout (averaged across windows)
    coherence_ratios = []
    for w in range(min(Y.shape[0], 10)):  # Sample up to 10 windows
        coh = test_coherence_preservation(y_pred_np[w], Y[w])
        coherence_ratios.append(coh['ratio'])
    mean_coherence_ratio = float(np.mean(coherence_ratios)) if coherence_ratios else 0.0

    result = {
        'holdout_gaba': gaba,
        'n_windows': int(X.shape[0]),
        'spike_correlation_mean': mean_corr,
        'spike_correlation_std': std_corr,
        'mse': mse,
        'coherence_ratio': mean_coherence_ratio,
        'per_window_correlations': per_window_corrs,
        'interpolation_passed': mean_corr > 0.3,  # Minimum threshold for interpolation
    }

    if verbose:
        print(f"\n  GABA Interpolation Test (GABA={gaba} nS):")
        print(f"    Spike correlation: {mean_corr:.4f} ± {std_corr:.4f}")
        print(f"    MSE: {mse:.6f}")
        print(f"    Coherence ratio: {mean_coherence_ratio:.3f}")
        print(f"    Passed: {result['interpolation_passed']}")

    return result


def run_interpolation_experiment(
    arch_id: str,
    data_dir: str,
    holdout_gaba: float = HOLDOUT_GABA,
    device: str = 'cpu',
    max_hours: float = 1.0,
    max_epochs: int = 200,
    patience: int = 25,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run full GABA interpolation experiment for one architecture.

    Parameters
    ----------
    arch_id : str
    data_dir : str
    holdout_gaba : float
    device : str
    max_hours : float
    max_epochs : int
    patience : int
    output_path : str, optional
    verbose : bool

    Returns
    -------
    result : dict
    """
    from experiments.hidden_sweep import build_model_with_dim

    # Load data with holdout
    train_data, val_data, holdout_data, bio_gt = load_data_with_gaba_holdout(
        data_dir, holdout_gaba=holdout_gaba
    )

    if verbose:
        print(f"\nTraining {arch_id} without GABA={holdout_gaba} nS...")

    # Build model (default latent dim)
    model = build_model_with_dim(arch_id, latent_dim=64)
    if model is None:
        print(f"  ERROR: Could not build {arch_id}")
        return {'error': f'Could not build {arch_id}'}

    # Train
    pipeline = FullTrainingPipeline(
        train_data, val_data, device=device, max_hours=max_hours
    )
    model, train_result = pipeline.train(
        model, template=None, lr=5e-4, batch_size=32,
        max_epochs=max_epochs, patience=patience, verbose=verbose
    )

    # Evaluate on validation set (seen GABA levels)
    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(val_data['X_val'], device=device, dtype=torch.float32)
        output = model(x_val)
        y_pred = output[0] if isinstance(output, tuple) else output
        y_pred_np = y_pred.cpu().numpy()

    val_corrs = []
    for w in range(val_data['Y_val'].shape[0]):
        c, _ = spike_train_correlation(val_data['Y_val'][w], y_pred_np[w])
        val_corrs.append(c)
    val_corr_mean = float(np.mean(val_corrs))

    # Evaluate on holdout GABA level
    holdout_result = evaluate_interpolation(model, holdout_data, device, verbose)

    result = {
        'architecture': arch_id,
        'holdout_gaba': holdout_gaba,
        'training': {
            'spike_correlation': train_result.spike_correlation,
            'best_val_loss': train_result.best_val_loss,
            'training_hours': train_result.training_hours,
            'total_epochs': train_result.total_epochs,
            'converged': train_result.converged,
        },
        'val_seen_gaba': {
            'spike_correlation': val_corr_mean,
        },
        'holdout_interpolation': holdout_result,
        'interpolation_gap': val_corr_mean - holdout_result['spike_correlation_mean'],
    }

    if output_path:
        with open(output_path, 'w') as f:
            # Convert numpy to python types for JSON serialization
            json.dump(_to_json_serializable(result), f, indent=2)
        if verbose:
            print(f"\nResults saved to {output_path}")

    return result


def _to_json_serializable(obj):
    """Recursively convert numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GABA Interpolation Hold-Out Test')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--holdout-gaba', type=float, default=HOLDOUT_GABA)
    parser.add_argument('--max-hours', type=float, default=1.0)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--output', type=str, default='gaba_interpolation_results.json')
    parser.add_argument('--architectures', nargs='+',
                       default=['ltc_network', 'gru_ode', 'hybrid_lstm_ode'])
    args = parser.parse_args()

    all_results = {}
    for arch_id in args.architectures:
        result = run_interpolation_experiment(
            arch_id=arch_id,
            data_dir=args.data_dir,
            holdout_gaba=args.holdout_gaba,
            device=args.device,
            max_hours=args.max_hours,
            max_epochs=args.max_epochs,
            patience=args.patience,
            verbose=True,
        )
        all_results[arch_id] = result

    # Save combined results
    with open(args.output, 'w') as f:
        json.dump(_to_json_serializable(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"GABA Interpolation Summary (holdout GABA={args.holdout_gaba} nS)")
    print(f"{'='*60}")
    for arch_id, res in all_results.items():
        if 'error' in res:
            print(f"  {arch_id}: ERROR - {res['error']}")
            continue
        seen = res.get('val_seen_gaba', {}).get('spike_correlation', 0)
        holdout = res.get('holdout_interpolation', {}).get('spike_correlation_mean', 0)
        gap = res.get('interpolation_gap', 0)
        print(f"  {arch_id}: seen_corr={seen:.4f} holdout_corr={holdout:.4f} gap={gap:.4f}")
