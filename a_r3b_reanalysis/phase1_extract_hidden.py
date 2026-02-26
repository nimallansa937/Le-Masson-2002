"""
Phase 1: Hidden state extraction + untrained baselines.

For each trained architecture, run forward pass on validation data and save
hidden state trajectories to disk. Also extract from untrained (randomly
initialized) copies — the mandatory baseline that the original A-R3 was missing.

Any R² from a TRAINED model must exceed R² from its UNTRAINED counterpart
to be meaningful: random networks produce structured hidden states from
structured inputs (Rahaman et al. 2019).
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rung3.models.lstm_model import ThalamicLSTM
from rung3.models.neural_ode_model import ThalamicNeuralODE
from rung3.models.volterra_laguerre import VolterraLaguerre
from rung3.dataset import load_and_preprocess_trials
from rung3.config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    NODE_LATENT_DIM, NODE_HIDDEN_DIM, NODE_N_HIDDEN, NODE_SOLVER,
    VOLTERRA_N_BASES, VOLTERRA_ALPHA, VOLTERRA_MEMORY_MS,
    VOLTERRA_ORDER, VOLTERRA_RIDGE_ALPHA,
    VOLTERRA_OUTPUT_FEEDBACK, VOLTERRA_FB_N_BASES,
    INPUT_DIM, OUTPUT_DIM,
)
from a_r3b_reanalysis.config import (
    DATA_DIR, CHECKPOINT_DIR, HIDDEN_STATES_DIR,
    VAL_SEEDS, ARCHITECTURE_CHECKPOINTS,
)


def _load_trained_model(arch_name, device='cpu'):
    """Load a trained model from its checkpoint."""
    ckpt_name = ARCHITECTURE_CHECKPOINTS[arch_name]
    ckpt_path = CHECKPOINT_DIR / ckpt_name

    if arch_name == 'lstm':
        model = ThalamicLSTM()
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        # Handle different checkpoint formats
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.to(device).eval()
        return model

    elif arch_name == 'neural_ode':
        model = ThalamicNeuralODE()
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.to(device).eval()
        return model

    elif arch_name == 'volterra':
        # .load() is an instance method — create with matching config, then load W
        model = VolterraLaguerre(
            n_bases=VOLTERRA_N_BASES,
            alpha=VOLTERRA_ALPHA,
            memory_ms=VOLTERRA_MEMORY_MS,
            order=VOLTERRA_ORDER,
            ridge_alpha=VOLTERRA_RIDGE_ALPHA,
            output_feedback=VOLTERRA_OUTPUT_FEEDBACK,
            fb_n_bases=VOLTERRA_FB_N_BASES,
        )
        model.load(str(ckpt_path))
        return model

    raise ValueError(f"Unknown architecture: {arch_name}")


def _create_untrained_model(arch_name, device='cpu'):
    """Create a fresh model with identical architecture but random weights."""
    if arch_name == 'lstm':
        model = ThalamicLSTM()
        model.to(device).eval()
        return model

    elif arch_name == 'neural_ode':
        model = ThalamicNeuralODE()
        model.to(device).eval()
        return model

    elif arch_name == 'volterra':
        model = VolterraLaguerre(
            n_bases=VOLTERRA_N_BASES,
            alpha=VOLTERRA_ALPHA,
            memory_ms=VOLTERRA_MEMORY_MS,
            order=VOLTERRA_ORDER,
            ridge_alpha=VOLTERRA_RIDGE_ALPHA,
            output_feedback=VOLTERRA_OUTPUT_FEEDBACK,
            fb_n_bases=VOLTERRA_FB_N_BASES,
        )
        # Randomize weights (not fitted, so W is None — set to random)
        # Forward checks self.W is None, so setting W enables forward pass
        model.W = np.random.randn(model.n_features, OUTPUT_DIM).astype(np.float32) * 0.01
        return model

    raise ValueError(f"Unknown architecture: {arch_name}")


def _extract_hidden(model, X_input, arch_name, device='cpu'):
    """Extract hidden state trajectory from a model.

    Parameters
    ----------
    model : trained or untrained model
    X_input : ndarray (n_windows, window_bins, input_dim)
    arch_name : str
    device : str

    Returns
    -------
    all_hidden : list of ndarray (window_bins, hidden_dim)
        One per window.
    all_preds : list of ndarray (window_bins, output_dim)
        Model predictions per window.
    """
    all_hidden = []
    all_preds = []

    for w_idx in range(X_input.shape[0]):
        x_window = X_input[w_idx:w_idx+1]  # (1, window_bins, input_dim)

        if arch_name in ('lstm', 'neural_ode'):
            x_t = torch.from_numpy(x_window).float().to(device)
            model.eval()
            with torch.no_grad():
                out, latent_dict = model(x_t, return_latent=True)
                h = latent_dict['hidden'][0].cpu().numpy()  # (window_bins, hidden_dim)
                pred = out[0].cpu().numpy()  # (window_bins, output_dim)
        elif arch_name == 'volterra':
            out, latent_dict = model.forward(x_window, return_latent=True)
            h = latent_dict['hidden'][0]  # (window_bins, latent_dim)
            pred = out[0]  # (window_bins, output_dim)
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")

        all_hidden.append(h)
        all_preds.append(pred)

    return all_hidden, all_preds


def extract_all_hidden_states(architectures=None, device='cpu', verbose=True):
    """Extract hidden states for all architectures and save to disk.

    Parameters
    ----------
    architectures : list of str, optional
        Which architectures to process. Defaults to all in ARCHITECTURE_CHECKPOINTS.
    device : str
    verbose : bool

    Returns
    -------
    output_dir : Path
    """
    if architectures is None:
        architectures = list(ARCHITECTURE_CHECKPOINTS.keys())

    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation data
    if verbose:
        print(f"Phase 1: Loading validation data (seeds {VAL_SEEDS})...")

    X_val, Y_rate_val, Y_binary_val, _ = load_and_preprocess_trials(
        VAL_SEEDS, str(DATA_DIR), include_intermediates=False, verbose=verbose)

    if verbose:
        print(f"  Validation windows: {X_val.shape}")

    # Track which windows came from which trial for CV grouping
    # Recompute per-trial window counts
    from rung3.phase0_recording import list_trials as _list_trials
    all_trials = _list_trials(str(DATA_DIR))
    val_trials = [t for t in all_trials if t['seed'] in VAL_SEEDS]
    from a_r3b_reanalysis.config import WINDOW_SIZE_MS, WINDOW_STRIDE_MS, BIN_DT_MS
    trial_window_counts = []
    window_bins = int(WINDOW_SIZE_MS / BIN_DT_MS)
    stride_bins = int(WINDOW_STRIDE_MS / BIN_DT_MS)

    for trial_info in val_trials:
        from rung3.phase0_recording import load_trial_hdf5
        data = load_trial_hdf5(trial_info['filepath'])
        n_timepoints = data['V_tc'].shape[1]
        n_bins = min(n_timepoints, int(data['duration_s'] * 1000 / BIN_DT_MS))
        n_windows = len(range(0, n_bins - window_bins + 1, stride_bins))
        trial_window_counts.append({
            'gaba_gmax': trial_info['gaba_gmax'],
            'seed': trial_info['seed'],
            'n_windows': n_windows,
        })

    # Build trial_ids array: which trial each window belongs to
    trial_ids = []
    for t_idx, tc in enumerate(trial_window_counts):
        trial_ids.extend([t_idx] * tc['n_windows'])
    trial_ids = np.array(trial_ids)

    for arch_name in architectures:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Extracting: {arch_name}")
            print(f"{'='*60}")

        # --- TRAINED model ---
        try:
            trained_model = _load_trained_model(arch_name, device)
        except Exception as e:
            print(f"  ERROR loading trained {arch_name}: {e}")
            continue

        if verbose:
            print(f"  Trained model loaded from {ARCHITECTURE_CHECKPOINTS[arch_name]}")

        trained_hidden, trained_preds = _extract_hidden(
            trained_model, X_val, arch_name, device)

        if verbose:
            print(f"  Trained hidden: {len(trained_hidden)} windows, "
                  f"dim={trained_hidden[0].shape[1]}")

        # --- UNTRAINED baseline ---
        untrained_model = _create_untrained_model(arch_name, device)

        if verbose:
            print(f"  Extracting untrained baseline...")

        untrained_hidden, _ = _extract_hidden(
            untrained_model, X_val, arch_name, device)

        if verbose:
            print(f"  Untrained hidden: {len(untrained_hidden)} windows, "
                  f"dim={untrained_hidden[0].shape[1]}")

        # Save
        outpath = HIDDEN_STATES_DIR / f'{arch_name}_hidden_states.npz'
        np.savez_compressed(
            str(outpath),
            trained_hidden=np.array(trained_hidden),
            trained_preds=np.array(trained_preds),
            untrained_hidden=np.array(untrained_hidden),
            trial_ids=trial_ids,
        )

        if verbose:
            print(f"  Saved to {outpath}")
            # Sanity check: compare trained vs untrained magnitudes
            t_mag = np.mean([np.std(h) for h in trained_hidden])
            u_mag = np.mean([np.std(h) for h in untrained_hidden])
            print(f"  Hidden state std — trained: {t_mag:.4f}, untrained: {u_mag:.4f}")

    return HIDDEN_STATES_DIR


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--arch', nargs='*', default=None)
    args = parser.parse_args()
    extract_all_hidden_states(architectures=args.arch, device=args.device)
