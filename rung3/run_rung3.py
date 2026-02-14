#!/usr/bin/env python
"""
Master runner for Rung 3: Thalamic Transformation Replacement.

Usage:
    # Full pipeline
    python -m rung3.run_rung3 --phase all --workers 8

    # Individual phases
    python -m rung3.run_rung3 --phase data --workers 12     # CPU: generate training data
    python -m rung3.run_rung3 --phase volterra              # CPU: fit Volterra model
    python -m rung3.run_rung3 --phase lstm --device cuda     # GPU: train LSTM
    python -m rung3.run_rung3 --phase node --device cuda     # GPU: train Neural ODE
    python -m rung3.run_rung3 --phase evaluate               # Evaluate all models
    python -m rung3.run_rung3 --phase compare                # Architecture comparison

Vast.ai workflow:
    1. CPU instance: --phase data --workers 12
    2. CPU instance: --phase volterra
    3. GPU instance: --phase lstm --device cuda
    4. GPU instance: --phase node --device cuda
    5. Any instance:  --phase evaluate
    6. Any instance:  --phase compare
"""

import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rung3.config import (
    DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR,
    GABA_VALUES, VAL_SEEDS, DURATION_S,
)


def phase_data(args):
    """Phase 0: Generate training data with intermediate recordings."""
    print("=" * 60)
    print("PHASE 0: Generate Training Data")
    print("=" * 60)

    from rung3.generate_training_data import main as gen_main
    # Patch sys.argv for the sub-main
    sys.argv = ['generate_training_data',
                '--workers', str(args.workers),
                '--duration', str(args.duration),
                '--data-dir', args.data_dir]
    if args.quick:
        sys.argv.append('--quick')
    gen_main()


def phase_volterra(args):
    """Phase 1: Fit Volterra-Laguerre model."""
    print("=" * 60)
    print("PHASE 1: Volterra-Laguerre Model")
    print("=" * 60)

    from rung3.training.train_volterra import train_volterra
    model, results = train_volterra(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir)

    # Run evaluation
    _evaluate_model_full(model, 'volterra', args)

    return model, results


def phase_lstm(args):
    """Phase 2: Train LSTM model."""
    print("=" * 60)
    print("PHASE 2: LSTM Model")
    print("=" * 60)

    from rung3.training.train_pytorch import train_lstm
    model, history = train_lstm(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device)

    # Run evaluation
    _evaluate_model_full(model, 'lstm', args)

    return model, history


def phase_node(args):
    """Phase 3: Train Neural ODE model."""
    print("=" * 60)
    print("PHASE 3: Neural ODE Model")
    print("=" * 60)

    from rung3.training.train_pytorch import train_neural_ode
    model, history = train_neural_ode(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device)

    # Run evaluation
    _evaluate_model_full(model, 'neural_ode', args)

    return model, history


def _evaluate_model_full(model, model_name, args):
    """Run full evaluation (output + latent) for a trained model."""
    from rung3.evaluation.output_metrics import evaluate_output_quality
    from rung3.evaluation.latent_comparison import (
        full_latent_comparison, extract_model_latent, prepare_bio_intermediates
    )
    from rung3.phase0_recording import load_trial_hdf5, list_trials
    from rung3.preprocessing import preprocess_trial

    print(f"\n--- Evaluating {model_name} ---")

    # Output metrics
    output_metrics = evaluate_output_quality(
        model, model_name, args.data_dir, VAL_SEEDS,
        gaba_values=GABA_VALUES)

    # Latent comparison on a sample trial
    all_trials = list_trials(args.data_dir)
    # Pick a trial with GABA near threshold (30 nS, seed 45)
    sample_trial = None
    for t in all_trials:
        if t['seed'] == 45 and t['gaba_gmax'] is not None:
            if abs(t['gaba_gmax'] - 30.0) < 3.0:
                sample_trial = t
                break

    latent_metrics = {}
    if sample_trial:
        data = load_trial_hdf5(sample_trial['filepath'])
        X, Y_rate, _, inter_w = preprocess_trial(data)

        if inter_w and X.shape[0] > 0:
            # Use first window
            latent_metrics = full_latent_comparison(
                model, model_name, X[0], inter_w_to_single(inter_w, 0))

    # Combine and save
    combined = {**output_metrics, **latent_metrics}
    save_path = os.path.join(args.checkpoint_dir,
                              f'{model_name}_evaluation.json')
    with open(save_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Saved: {save_path}")


def inter_w_to_single(inter_w, window_idx):
    """Extract a single window from windowed intermediates dict."""
    return {k: v[window_idx] for k, v in inter_w.items()}


def phase_evaluate(args):
    """Evaluate all trained models."""
    print("=" * 60)
    print("EVALUATION: All Models")
    print("=" * 60)

    model_configs = [
        ('volterra', _load_volterra),
        ('lstm', _load_lstm),
        ('neural_ode', _load_node),
    ]

    for model_name, load_fn in model_configs:
        try:
            model = load_fn(args)
            if model is not None:
                _evaluate_model_full(model, model_name, args)
        except Exception as e:
            print(f"  {model_name}: FAILED — {e}")


def _load_volterra(args):
    """Load trained Volterra model."""
    from rung3.models.volterra_laguerre import VolterraLaguerre
    model = VolterraLaguerre()
    path = os.path.join(args.checkpoint_dir, 'volterra_model.npz')
    if os.path.exists(path):
        model.load(path)
        print(f"  Loaded Volterra from {path}")
        return model
    print(f"  Volterra not found at {path}")
    return None


def _load_lstm(args):
    """Load trained LSTM model."""
    import torch
    from rung3.models.lstm_model import ThalamicLSTM
    model = ThalamicLSTM()
    path = os.path.join(args.checkpoint_dir, 'lstm_best.pt')
    if os.path.exists(path):
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(path, map_location=device,
                                          weights_only=True))
        model = model.to(device)
        model.eval()
        print(f"  Loaded LSTM from {path}")
        return model
    print(f"  LSTM not found at {path}")
    return None


def _load_node(args):
    """Load trained Neural ODE model."""
    import torch
    from rung3.models.neural_ode_model import ThalamicNeuralODE
    model = ThalamicNeuralODE()
    path = os.path.join(args.checkpoint_dir, 'neural_ode_best.pt')
    if os.path.exists(path):
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(path, map_location=device,
                                          weights_only=True))
        model = model.to(device)
        model.eval()
        print(f"  Loaded Neural ODE from {path}")
        return model
    print(f"  Neural ODE not found at {path}")
    return None


def phase_compare(args):
    """Architecture comparison with figures."""
    print("=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)

    from rung3.evaluation.architecture_comparison import run_comparison
    run_comparison(args.checkpoint_dir, args.figure_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Rung 3: Thalamic Transformation Replacement')
    parser.add_argument('--phase', required=True,
                        choices=['data', 'volterra', 'lstm', 'node',
                                 'evaluate', 'compare', 'all'],
                        help='Which phase to run')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--duration', type=float, default=DURATION_S)
    parser.add_argument('--device', type=str, default=None,
                        help='cpu or cuda')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--figure-dir', type=str, default=FIGURE_DIR)
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    t_start = time.time()

    if args.phase == 'data':
        phase_data(args)
    elif args.phase == 'volterra':
        phase_volterra(args)
    elif args.phase == 'lstm':
        phase_lstm(args)
    elif args.phase == 'node':
        phase_node(args)
    elif args.phase == 'evaluate':
        phase_evaluate(args)
    elif args.phase == 'compare':
        phase_compare(args)
    elif args.phase == 'all':
        phase_data(args)
        phase_volterra(args)
        phase_lstm(args)
        phase_node(args)
        phase_evaluate(args)
        phase_compare(args)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Phase '{args.phase}' completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
