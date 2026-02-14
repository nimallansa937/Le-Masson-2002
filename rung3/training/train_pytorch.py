"""
Unified PyTorch trainer for LSTM and Neural ODE models.

Loss: 0.7×BCE(spike predictions) + 0.3×MSE(smoothed rates)
Optimizer: AdamW with gradient clipping
Scheduler: ReduceLROnPlateau
Early stopping: patience=15
"""

import sys
import os
import time
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rung3.config import (
    DATA_DIR, CHECKPOINT_DIR,
    LOSS_BCE_WEIGHT, LOSS_MSE_WEIGHT,
    LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP_NORM,
    SCHEDULER_PATIENCE, SCHEDULER_FACTOR, MIN_LR,
    EARLY_STOP_PATIENCE, MIN_DELTA,
    BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS,
)
from rung3.dataset import create_dataloaders


class CombinedLoss(nn.Module):
    """Combined BCE + MSE loss for spike prediction."""

    def __init__(self, bce_weight=LOSS_BCE_WEIGHT, mse_weight=LOSS_MSE_WEIGHT):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, rate_target, binary_target):
        """
        pred: (batch, seq_len, 20) — model output (sigmoid-activated)
        rate_target: (batch, seq_len, 20) — smoothed firing rates
        binary_target: (batch, seq_len, 20) — binary spike trains
        """
        # Clamp predictions for numerical stability
        pred_clamped = torch.clamp(pred, 1e-7, 1 - 1e-7)
        loss_bce = self.bce(pred_clamped, binary_target)
        loss_mse = self.mse(pred, rate_target)
        return self.bce_weight * loss_bce + self.mse_weight * loss_mse


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(model, model_name, data_dir=DATA_DIR,
                checkpoint_dir=CHECKPOINT_DIR, num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                device=None, verbose=True):
    """Train a PyTorch model (LSTM or Neural ODE).

    Parameters
    ----------
    model : nn.Module
        Must implement forward(x) → (batch, seq_len, 20)
    model_name : str
        'lstm' or 'neural_ode'
    device : str or None
        'cuda', 'cpu', or None for auto-detect.

    Returns
    -------
    model : nn.Module (trained)
    history : dict
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if verbose:
        print(f"Training {model_name} on {device}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

    model = model.to(device)

    # Data
    train_loader, val_loader = create_dataloaders(
        data_dir, batch_size, NUM_WORKERS, verbose=verbose)

    # Loss, optimizer, scheduler
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR, min_lr=MIN_LR)
    early_stopping = EarlyStopping()

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_corr': [], 'val_corr': [],
        'lr': [], 'best_epoch': 0, 'best_val_loss': float('inf'),
    }

    best_model_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')

    for epoch in range(num_epochs):
        t0 = time.time()

        # --- Training ---
        model.train()
        train_losses = []
        train_corrs = []

        for batch in train_loader:
            x = batch['input'].to(device)
            y_rate = batch['rate_target'].to(device)
            y_binary = batch['binary_target'].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y_rate, y_binary)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            train_losses.append(loss.item())

            # Correlation (on CPU, detached)
            with torch.no_grad():
                p = pred.cpu().numpy()
                y = y_rate.cpu().numpy()
                # Mean correlation across channels and batch
                corrs = []
                for b in range(p.shape[0]):
                    for ch in range(p.shape[2]):
                        if np.std(y[b, :, ch]) > 1e-8:
                            c = np.corrcoef(y[b, :, ch], p[b, :, ch])[0, 1]
                            if not np.isnan(c):
                                corrs.append(c)
                if corrs:
                    train_corrs.append(np.mean(corrs))

        # --- Validation ---
        model.eval()
        val_losses = []
        val_corrs = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                y_rate = batch['rate_target'].to(device)
                y_binary = batch['binary_target'].to(device)

                pred = model(x)
                loss = criterion(pred, y_rate, y_binary)
                val_losses.append(loss.item())

                p = pred.cpu().numpy()
                y = y_rate.cpu().numpy()
                corrs = []
                for b in range(p.shape[0]):
                    for ch in range(p.shape[2]):
                        if np.std(y[b, :, ch]) > 1e-8:
                            c = np.corrcoef(y[b, :, ch], p[b, :, ch])[0, 1]
                            if not np.isnan(c):
                                corrs.append(c)
                if corrs:
                    val_corrs.append(np.mean(corrs))

        # Epoch stats
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_corr = np.mean(train_corrs) if train_corrs else 0.0
        epoch_val_corr = np.mean(val_corrs) if val_corrs else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))
        history['train_corr'].append(float(epoch_train_corr))
        history['val_corr'].append(float(epoch_val_corr))
        history['lr'].append(float(current_lr))

        # Save best model
        if epoch_val_loss < history['best_val_loss']:
            history['best_val_loss'] = float(epoch_val_loss)
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), best_model_path)

        # Scheduler & early stopping
        scheduler.step(epoch_val_loss)
        early_stopping.step(epoch_val_loss)

        if verbose:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {epoch_train_loss:.5f} (r={epoch_train_corr:.3f}) | "
                  f"Val: {epoch_val_loss:.5f} (r={epoch_val_corr:.3f}) | "
                  f"LR: {current_lr:.1e} | {elapsed:.1f}s")

        if early_stopping.should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Save history
    history_path = os.path.join(checkpoint_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"\nBest model at epoch {history['best_epoch']+1}")
        print(f"  Val loss: {history['best_val_loss']:.5f}")
        print(f"  Val corr: {history['val_corr'][history['best_epoch']]:.4f}")
        print(f"  Saved: {best_model_path}")

    return model, history


def train_lstm(data_dir=DATA_DIR, checkpoint_dir=CHECKPOINT_DIR,
               device=None, verbose=True):
    """Train LSTM model."""
    from rung3.models.lstm_model import ThalamicLSTM
    model = ThalamicLSTM()
    return train_model(model, 'lstm', data_dir, checkpoint_dir,
                       device=device, verbose=verbose)


def train_neural_ode(data_dir=DATA_DIR, checkpoint_dir=CHECKPOINT_DIR,
                      device=None, verbose=True):
    """Train Neural ODE model."""
    from rung3.models.neural_ode_model import ThalamicNeuralODE
    model = ThalamicNeuralODE()
    return train_model(model, 'neural_ode', data_dir, checkpoint_dir,
                       device=device, verbose=verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['lstm', 'neural_ode'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    if args.model == 'lstm':
        model, history = train_lstm(device=args.device)
    else:
        model, history = train_neural_ode(device=args.device)
