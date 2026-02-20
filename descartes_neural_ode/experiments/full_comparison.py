"""
Full Architecture Comparison (Guide Task 7: exp13_full_comparison.py)

Load best models from each architecture, run all metrics head-to-head,
generate comparison table for Paper 1 results section.

Metrics compared:
  Output quality:
    - Spike correlation (Pearson on smoothed rates)
    - Victor-Purpura distance
    - Bifurcation threshold error
    - Coherence preservation ratio
  Latent alignment (NOVEL):
    - CCA mean canonical correlation
    - RSA correlation
    - Per-variable recovery (n/160 at |r| > 0.5)
  Training efficiency:
    - Training time
    - Convergence epoch
"""
import sys
import os
import json
import numpy as np
import torch
import time
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.spike_metrics import evaluate_spike_metrics, spike_train_correlation
from evaluation.bifurcation_test import run_bifurcation_test
from evaluation.coherence_test import test_coherence_preservation, compute_population_coherence
from evaluation.latent_comparison import full_latent_comparison, per_variable_correlation
from evaluation.biovar_recovery_scorer import compute_detailed_recovery, compare_architectures_recovery


def load_model_checkpoint(checkpoint_path: str, arch_id: str, device: str = 'cpu'):
    """Load a trained model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint file.
    arch_id : str
        Architecture ID for model reconstruction.
    device : str

    Returns
    -------
    model : nn.Module or None
    metadata : dict
    """
    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None, {}

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine model class and build
    from experiments.hidden_sweep import build_model_with_dim
    latent_dim = checkpoint.get('latent_dim', 64)
    model = build_model_with_dim(arch_id, latent_dim)

    if model is None:
        return None, {}

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Try loading directly
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"  Could not load state dict for {arch_id}: {e}")
            return None, {}

    model = model.to(device)
    model.eval()

    metadata = {
        'arch_id': arch_id,
        'latent_dim': latent_dim,
        'best_val_loss': checkpoint.get('best_val_loss', None),
        'best_epoch': checkpoint.get('best_epoch', None),
        'spike_correlation': checkpoint.get('spike_correlation', None),
    }

    return model, metadata


def evaluate_architecture(
    model: torch.nn.Module,
    arch_id: str,
    val_data: Dict,
    bio_ground_truth: Optional[Dict] = None,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """Run full evaluation suite on one architecture.

    Parameters
    ----------
    model : nn.Module
    arch_id : str
    val_data : dict
    bio_ground_truth : dict, optional
    device : str
    verbose : bool

    Returns
    -------
    result : dict
    """
    result = {'architecture': arch_id}

    X_val = val_data['X_val']
    Y_val = val_data['Y_val']

    # === Output Quality ===
    if verbose:
        print(f"\n  [{arch_id}] Output quality evaluation...")

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_val, device=device, dtype=torch.float32)
        output = model(x_tensor)
        y_pred = output[0] if isinstance(output, tuple) else output
        y_pred_np = y_pred.cpu().numpy()

    # Spike correlation
    per_window_corrs = []
    for w in range(Y_val.shape[0]):
        c, _ = spike_train_correlation(Y_val[w], y_pred_np[w])
        per_window_corrs.append(c)
    result['spike_correlation_mean'] = float(np.mean(per_window_corrs))
    result['spike_correlation_std'] = float(np.std(per_window_corrs))

    # VP distance (on a sample — VP is O(n^2) per neuron pair)
    n_sample = min(5, Y_val.shape[0])
    vp_distances = []
    for w in range(n_sample):
        metrics = evaluate_spike_metrics(Y_val[w], y_pred_np[w])
        vp_distances.append(metrics['vp_distance_mean'])
    result['vp_distance_mean'] = float(np.mean(vp_distances))

    # Coherence test (high-GABA windows)
    coherence_ratios = []
    for w in range(min(10, Y_val.shape[0])):
        coh = test_coherence_preservation(y_pred_np[w], Y_val[w])
        coherence_ratios.append(coh['ratio'])
    result['coherence_ratio'] = float(np.mean(coherence_ratios))
    result['coherence_preserved'] = result['coherence_ratio'] >= 0.7

    if verbose:
        print(f"    Spike corr: {result['spike_correlation_mean']:.4f}")
        print(f"    VP distance: {result['vp_distance_mean']:.4f}")
        print(f"    Coherence ratio: {result['coherence_ratio']:.3f}")

    # === Latent Alignment (NOVEL) ===
    if bio_ground_truth:
        if verbose:
            print(f"  [{arch_id}] Latent alignment evaluation...")

        # Extract latents
        from experiments.hidden_sweep import extract_latents
        latents = extract_latents(model, X_val, device)

        if latents is not None:
            # Build bio matrix from ground truth
            bio_arrays = []
            bio_names = []
            for key, arr in bio_ground_truth.items():
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    # (n_neurons, T) -> columns
                    if arr.shape[0] < arr.shape[1]:
                        for n in range(arr.shape[0]):
                            bio_arrays.append(arr[n, :])
                            bio_names.append(f"{key}_n{n}")
                    else:
                        for n in range(arr.shape[1]):
                            bio_arrays.append(arr[:, n])
                            bio_names.append(f"{key}_d{n}")

            if bio_arrays:
                T_min = min(latents.shape[0], min(a.shape[0] for a in bio_arrays))
                bio_matrix = np.column_stack([a[:T_min] for a in bio_arrays])
                lat_aligned = latents[:T_min]

                try:
                    comparison = full_latent_comparison(
                        lat_aligned, bio_matrix, bio_names, verbose=verbose
                    )
                    result['latent_comparison'] = comparison
                except Exception as e:
                    result['latent_comparison'] = {'error': str(e)}

                # Per-variable correlation
                try:
                    pvc = per_variable_correlation(lat_aligned, bio_matrix, bio_names)
                    result['n_recovered_05'] = pvc['n_above_05']
                    result['n_recovered_03'] = pvc['n_above_03']
                    result['mean_abs_correlation'] = pvc['mean_abs_r']
                except Exception as e:
                    result['n_recovered_05'] = 0
                    result['n_recovered_03'] = 0
                    result['mean_abs_correlation'] = 0.0
        else:
            result['latent_comparison'] = {'error': 'Could not extract latents'}
            result['n_recovered_05'] = 0
            result['n_recovered_03'] = 0
            result['mean_abs_correlation'] = 0.0

    return result


def run_full_comparison(
    checkpoint_dir: str,
    val_data: Dict,
    bio_ground_truth: Optional[Dict] = None,
    architectures: Optional[List[str]] = None,
    device: str = 'cpu',
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run complete architecture comparison.

    Parameters
    ----------
    checkpoint_dir : str
    val_data : dict
    bio_ground_truth : dict, optional
    architectures : list of str, optional
    device : str
    output_path : str, optional
    verbose : bool

    Returns
    -------
    comparison : dict
    """
    if architectures is None:
        architectures = [
            'standard_ode_baseline', 'segmented_ode', 'ltc_network',
            'neural_cde', 'coupled_oscillatory', 'gru_ode',
            'hybrid_lstm_ode', 'volterra_distilled_ode', 's4_mamba'
        ]

    results = {}
    ckpt_path = Path(checkpoint_dir)

    for arch_id in architectures:
        # Look for checkpoint
        possible_names = [
            f'{arch_id}_best.pt',
            f'{arch_id}_checkpoint.pt',
            f'{arch_id}.pt',
        ]

        model = None
        metadata = {}
        for name in possible_names:
            fpath = ckpt_path / name
            if fpath.exists():
                model, metadata = load_model_checkpoint(str(fpath), arch_id, device)
                break

        if model is None:
            if verbose:
                print(f"\n  [{arch_id}] No checkpoint found, skipping.")
            results[arch_id] = {'architecture': arch_id, 'error': 'No checkpoint found'}
            continue

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Evaluating: {arch_id}")
            print(f"{'='*60}")

        try:
            result = evaluate_architecture(
                model, arch_id, val_data, bio_ground_truth, device, verbose
            )
            result['metadata'] = metadata
            results[arch_id] = result
        except Exception as e:
            print(f"  ERROR evaluating {arch_id}: {e}")
            results[arch_id] = {'architecture': arch_id, 'error': str(e)}

    # Build comparison table
    comparison = _build_comparison_table(results)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(_to_json_safe(comparison), f, indent=2)
        if verbose:
            print(f"\nComparison saved to {output_path}")

    if verbose:
        _print_comparison_table(comparison)

    return comparison


def _build_comparison_table(results: Dict) -> Dict:
    """Build structured comparison from per-architecture results."""
    table = {
        'architectures': {},
        'best_output': None,
        'best_latent': None,
    }

    best_corr = -1
    best_recovered = -1

    for arch_id, res in results.items():
        if 'error' in res:
            table['architectures'][arch_id] = {'error': res['error']}
            continue

        entry = {
            'spike_correlation': res.get('spike_correlation_mean', 0),
            'spike_correlation_std': res.get('spike_correlation_std', 0),
            'vp_distance': res.get('vp_distance_mean', float('inf')),
            'coherence_ratio': res.get('coherence_ratio', 0),
            'coherence_preserved': res.get('coherence_preserved', False),
            'n_recovered_05': res.get('n_recovered_05', 0),
            'n_recovered_03': res.get('n_recovered_03', 0),
            'mean_abs_correlation': res.get('mean_abs_correlation', 0),
        }

        # Add latent comparison if available
        lc = res.get('latent_comparison', {})
        if 'cca' in lc:
            entry['cca_correlation'] = lc['cca'].get('mean_correlation', 0)
            entry['cca_significant'] = lc['cca'].get('significant', False)
        if 'rsa' in lc:
            entry['rsa_correlation'] = lc['rsa'].get('correlation', 0)

        table['architectures'][arch_id] = entry

        # Track best
        if entry['spike_correlation'] > best_corr:
            best_corr = entry['spike_correlation']
            table['best_output'] = arch_id
        if entry['n_recovered_05'] > best_recovered:
            best_recovered = entry['n_recovered_05']
            table['best_latent'] = arch_id

    return table


def _print_comparison_table(comparison: Dict):
    """Print formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"{'Architecture':<25} {'SpikeCorr':>9} {'VP Dist':>8} {'Coh%':>5} "
          f"{'Recovered':>9} {'CCA':>6} {'RSA':>6}")
    print(f"{'='*90}")

    for arch_id, entry in comparison['architectures'].items():
        if 'error' in entry:
            print(f"{arch_id:<25} ERROR: {entry['error']}")
            continue

        corr = entry.get('spike_correlation', 0)
        vp = entry.get('vp_distance', float('inf'))
        coh = entry.get('coherence_ratio', 0)
        rec = entry.get('n_recovered_05', 0)
        cca = entry.get('cca_correlation', 0)
        rsa = entry.get('rsa_correlation', 0)

        print(f"{arch_id:<25} {corr:9.4f} {vp:8.2f} {coh:5.2f} "
              f"{rec:>5}/160 {cca:6.3f} {rsa:6.3f}")

    print(f"{'='*90}")
    print(f"Best output: {comparison.get('best_output', 'N/A')}")
    print(f"Best latent: {comparison.get('best_latent', 'N/A')}")


def _to_json_safe(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
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

    parser = argparse.ArgumentParser(description='Full Architecture Comparison')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='architecture_comparison.json')
    parser.add_argument('--architectures', nargs='+', default=None)
    args = parser.parse_args()

    from data.ar3_data_loader import load_ar2_data

    print("Loading data...")
    train_data, val_data, bio_gt = load_ar2_data(args.data_dir)

    comparison = run_full_comparison(
        checkpoint_dir=args.checkpoint_dir,
        val_data=val_data,
        bio_ground_truth=bio_gt,
        architectures=args.architectures,
        device=args.device,
        output_path=args.output,
    )
