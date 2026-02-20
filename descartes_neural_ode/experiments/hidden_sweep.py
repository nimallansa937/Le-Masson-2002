"""
Hidden Dimension Sweep (Guide Task 6: exp12_hidden_sweep.py)

Sweep latent/hidden dimensions for each architecture to test whether
biological variable recovery correlates with model capacity.

Key hypothesis from the guide:
  If latent-biological correlation peaks near biological dimensionality (~240),
  this suggests the model naturally discovers the biological representation
  when given matching capacity. If it's equally high at 512, the mapping is
  robust. If low everywhere — zombie.

LSTM sweep: hidden_size = [32, 64, 128, 256, 512]
Neural ODE sweep: latent_dim = [16, 32, 64, 128, 256]

For each size: train, evaluate output accuracy, evaluate latent alignment.
"""
import sys
import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.full_training_pipeline import FullTrainingPipeline, TrainingResult
from evaluation.spike_metrics import spike_train_correlation
from evaluation.latent_comparison import per_variable_correlation


# ============================================================
# Sweep Configuration
# ============================================================

# Hidden dimensions to sweep per architecture family
LSTM_HIDDEN_SIZES = [32, 64, 128, 256, 512]
NODE_LATENT_DIMS = [16, 32, 64, 128, 256]

# All architectures and their sweep configs
SWEEP_CONFIGS = {
    # LSTM-like architectures
    'hybrid_lstm_ode': {
        'dims': LSTM_HIDDEN_SIZES,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
    'gru_ode': {
        'dims': LSTM_HIDDEN_SIZES,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
    # ODE-based architectures
    'standard_ode_baseline': {
        'dims': NODE_LATENT_DIMS,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
    'segmented_ode': {
        'dims': NODE_LATENT_DIMS,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
    'ltc_network': {
        'dims': NODE_LATENT_DIMS,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
    'neural_cde': {
        'dims': NODE_LATENT_DIMS,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
    'coupled_oscillatory': {
        'dims': NODE_LATENT_DIMS,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
    's4_mamba': {
        'dims': NODE_LATENT_DIMS,
        'param_name': 'latent_dim',
        'bio_dims': 240,
    },
}


@dataclass
class SweepPoint:
    """Result for one (architecture, hidden_dim) combination."""
    architecture: str
    hidden_dim: int
    spike_correlation: float
    best_val_loss: float
    n_recovered_05: int       # biovars recovered at |r| > 0.5
    n_recovered_03: int       # biovars recovered at |r| > 0.3
    mean_abs_correlation: float
    training_hours: float
    converged: bool
    total_epochs: int


def build_model_with_dim(arch_id: str, latent_dim: int, n_input: int = 21, n_output: int = 20):
    """Build a model with a specific latent dimension.

    Parameters
    ----------
    arch_id : str
        Architecture template ID.
    latent_dim : int
        Latent/hidden dimension to use.
    n_input, n_output : int

    Returns
    -------
    model : nn.Module or None
    """
    try:
        if arch_id in ('standard_ode_baseline', 'segmented_ode'):
            from architectures.base_ode import TCReplacementNeuralODE
            return TCReplacementNeuralODE(
                n_input=n_input, n_output=n_output, latent_dim=latent_dim)

        elif arch_id == 'ltc_network':
            from architectures.ltc_network import LTCModel
            return LTCModel(n_input=n_input, n_output=n_output, latent_dim=latent_dim)

        elif arch_id == 'neural_cde':
            from architectures.neural_cde import NeuralCDEModel
            return NeuralCDEModel(n_input=n_input, n_output=n_output, latent_dim=latent_dim)

        elif arch_id == 'coupled_oscillatory':
            from architectures.coRNN import CoRNNModel
            return CoRNNModel(n_input=n_input, n_output=n_output, latent_dim=latent_dim)

        elif arch_id == 'gru_ode':
            from architectures.gru_ode import GRUODEModel
            return GRUODEModel(n_input=n_input, n_output=n_output, latent_dim=latent_dim)

        elif arch_id == 'hybrid_lstm_ode':
            from architectures.hybrid_lstm_ode import HybridLSTMODEModel
            return HybridLSTMODEModel(n_input=n_input, n_output=n_output, latent_dim=latent_dim)

        elif arch_id == 's4_mamba':
            from architectures.s4_mamba import S4MambaModel
            return S4MambaModel(n_input=n_input, n_output=n_output, latent_dim=latent_dim)

    except (ImportError, Exception) as e:
        print(f"  Could not build {arch_id} with dim={latent_dim}: {e}")
        return None

    return None


def extract_latents(model, X_val, device='cpu'):
    """Extract latent trajectories from model.

    Parameters
    ----------
    model : nn.Module
    X_val : ndarray (n_windows, T, input_dim)
    device : str

    Returns
    -------
    latents : ndarray (T, latent_dim) or None
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_val[:1], device=device, dtype=torch.float32)
        output = model(x)

        if isinstance(output, tuple) and len(output) >= 2:
            lat = output[1]
            if isinstance(lat, torch.Tensor):
                lat_np = lat.cpu().numpy()
                if lat_np.ndim == 3:
                    return lat_np[0]  # (T, latent_dim)
                return lat_np
            elif isinstance(lat, dict) and 'hidden' in lat:
                h = lat['hidden']
                if isinstance(h, torch.Tensor):
                    h = h.cpu().numpy()
                if h.ndim == 3:
                    return h[0]
                return h

    return None


def run_sweep_point(
    arch_id: str,
    hidden_dim: int,
    train_data: Dict,
    val_data: Dict,
    bio_ground_truth: Optional[np.ndarray] = None,
    device: str = 'cpu',
    max_hours: float = 0.5,
    max_epochs: int = 100,
    patience: int = 20,
    verbose: bool = True,
) -> SweepPoint:
    """Train and evaluate one (architecture, hidden_dim) combination.

    Parameters
    ----------
    arch_id : str
    hidden_dim : int
    train_data, val_data : dict
    bio_ground_truth : ndarray (T, n_bio_dims), optional
    device : str
    max_hours : float
    max_epochs : int
    patience : int
    verbose : bool

    Returns
    -------
    point : SweepPoint
    """
    if verbose:
        print(f"\n  {arch_id} dim={hidden_dim}...")

    model = build_model_with_dim(arch_id, hidden_dim)
    if model is None:
        return SweepPoint(
            architecture=arch_id, hidden_dim=hidden_dim,
            spike_correlation=0.0, best_val_loss=float('inf'),
            n_recovered_05=0, n_recovered_03=0, mean_abs_correlation=0.0,
            training_hours=0.0, converged=False, total_epochs=0,
        )

    # Train
    pipeline = FullTrainingPipeline(
        train_data, val_data, device=device, max_hours=max_hours
    )
    model, result = pipeline.train(
        model, template=None, lr=5e-4, batch_size=32,
        max_epochs=max_epochs, patience=patience, verbose=verbose
    )

    # Evaluate latent alignment (if bio ground truth available)
    n_recovered_05 = 0
    n_recovered_03 = 0
    mean_abs_r = 0.0

    if bio_ground_truth is not None:
        latents = extract_latents(model, val_data['X_val'], device)
        if latents is not None:
            # Align lengths
            T_min = min(latents.shape[0], bio_ground_truth.shape[0])
            lat_aligned = latents[:T_min]
            bio_aligned = bio_ground_truth[:T_min]

            pvc = per_variable_correlation(lat_aligned, bio_aligned)
            n_recovered_05 = pvc['n_above_05']
            n_recovered_03 = pvc['n_above_03']
            mean_abs_r = pvc['mean_abs_r']

    return SweepPoint(
        architecture=arch_id,
        hidden_dim=hidden_dim,
        spike_correlation=result.spike_correlation,
        best_val_loss=result.best_val_loss,
        n_recovered_05=n_recovered_05,
        n_recovered_03=n_recovered_03,
        mean_abs_correlation=mean_abs_r,
        training_hours=result.training_hours,
        converged=result.converged,
        total_epochs=result.total_epochs,
    )


def run_full_sweep(
    architectures: List[str],
    train_data: Dict,
    val_data: Dict,
    bio_ground_truth: Optional[np.ndarray] = None,
    device: str = 'cpu',
    max_hours_per_point: float = 0.5,
    max_epochs: int = 100,
    patience: int = 20,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run complete hidden dimension sweep across architectures.

    Parameters
    ----------
    architectures : list of str
        Architecture IDs to sweep.
    train_data, val_data : dict
    bio_ground_truth : ndarray, optional
    device : str
    max_hours_per_point : float
    max_epochs : int
    patience : int
    output_path : str, optional
    verbose : bool

    Returns
    -------
    results : dict
    """
    all_points = []
    start_time = time.time()

    for arch_id in architectures:
        if arch_id not in SWEEP_CONFIGS:
            print(f"  Warning: no sweep config for {arch_id}, skipping")
            continue

        config = SWEEP_CONFIGS[arch_id]
        dims = config['dims']

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Sweeping {arch_id}: dims={dims}")
            print(f"{'='*60}")

        for dim in dims:
            try:
                point = run_sweep_point(
                    arch_id, dim, train_data, val_data,
                    bio_ground_truth=bio_ground_truth,
                    device=device, max_hours=max_hours_per_point,
                    max_epochs=max_epochs, patience=patience,
                    verbose=verbose,
                )
                all_points.append(point)

                if verbose:
                    print(f"    dim={dim}: spike_corr={point.spike_correlation:.4f} "
                          f"val_loss={point.best_val_loss:.5f} "
                          f"biovars(|r|>0.5)={point.n_recovered_05}")
            except Exception as e:
                print(f"    Error at {arch_id} dim={dim}: {e}")
                all_points.append(SweepPoint(
                    architecture=arch_id, hidden_dim=dim,
                    spike_correlation=0.0, best_val_loss=float('inf'),
                    n_recovered_05=0, n_recovered_03=0, mean_abs_correlation=0.0,
                    training_hours=0.0, converged=False, total_epochs=0,
                ))

    total_hours = (time.time() - start_time) / 3600

    # Organize results
    results = {
        'sweep_points': [asdict(p) for p in all_points],
        'total_hours': total_hours,
        'summary': _summarize_sweep(all_points),
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nSweep results saved to {output_path}")

    return results


def _summarize_sweep(points: List[SweepPoint]) -> Dict:
    """Summarize sweep results per architecture."""
    summary = {}
    for p in points:
        if p.architecture not in summary:
            summary[p.architecture] = {'dims': [], 'spike_corr': [],
                                        'n_recovered': [], 'mean_abs_r': []}
        summary[p.architecture]['dims'].append(p.hidden_dim)
        summary[p.architecture]['spike_corr'].append(p.spike_correlation)
        summary[p.architecture]['n_recovered'].append(p.n_recovered_05)
        summary[p.architecture]['mean_abs_r'].append(p.mean_abs_correlation)

    for arch_id in summary:
        s = summary[arch_id]
        # Find optimal dim
        best_idx = int(np.argmax(s['spike_corr']))
        s['best_dim_for_output'] = s['dims'][best_idx]
        s['best_spike_corr'] = s['spike_corr'][best_idx]

        if any(r > 0 for r in s['n_recovered']):
            best_bio_idx = int(np.argmax(s['n_recovered']))
            s['best_dim_for_biovars'] = s['dims'][best_bio_idx]
            s['best_n_recovered'] = s['n_recovered'][best_bio_idx]
        else:
            s['best_dim_for_biovars'] = None
            s['best_n_recovered'] = 0

    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hidden Dimension Sweep')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing HDF5 trial files')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max-hours', type=float, default=0.5,
                       help='Max training hours per (arch, dim) point')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--output', type=str, default='hidden_sweep_results.json')
    parser.add_argument('--architectures', nargs='+', default=None,
                       help='Architectures to sweep (default: all)')
    args = parser.parse_args()

    from data.ar3_data_loader import load_ar2_data

    print("Loading data...")
    train_data, val_data, intermediates = load_ar2_data(args.data_dir)

    # Build bio ground truth matrix from intermediates if available
    bio_gt = None
    if intermediates:
        # Flatten all intermediate variables into (T, n_bio) matrix
        bio_arrays = []
        for key, arr in intermediates.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                bio_arrays.append(arr.T if arr.shape[0] < arr.shape[1] else arr)
        if bio_arrays:
            T_min = min(a.shape[0] for a in bio_arrays)
            bio_gt = np.hstack([a[:T_min] for a in bio_arrays])

    architectures = args.architectures or list(SWEEP_CONFIGS.keys())

    results = run_full_sweep(
        architectures=architectures,
        train_data=train_data,
        val_data=val_data,
        bio_ground_truth=bio_gt,
        device=args.device,
        max_hours_per_point=args.max_hours,
        max_epochs=args.max_epochs,
        patience=args.patience,
        output_path=args.output,
    )

    print(f"\nSweep complete. Total time: {results['total_hours']:.2f} hours")
    for arch_id, s in results['summary'].items():
        print(f"  {arch_id}: best_output_dim={s['best_dim_for_output']} "
              f"(corr={s['best_spike_corr']:.4f}), "
              f"best_bio_dim={s.get('best_dim_for_biovars', 'N/A')}")
