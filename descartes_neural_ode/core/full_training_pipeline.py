"""
Layer 3: Full Training Pipeline (Z3-C2 Analog)

After short-segment verification passes, this trains the architecture
to convergence on the full 2000-step sequences and evaluates recovery
of biological variables.

Training protocol:
  - Loss: 0.7*BCE(binary spike targets) + 0.3*MSE(smoothed rate targets)
    BCE is applied to BINARY spike trains (0/1), NOT to smoothed rates.
    MSE is applied to smoothed firing rates.
    This matches the rung3 CombinedLoss from train_pytorch.py.
  - Optimizer: AdamW with gradient clipping
  - Scheduler: ReduceLROnPlateau
  - Early stopping: patience configurable per architecture

NOTE: If binary targets are not provided (Y_binary_train missing from data),
falls back to MSE-only loss on smoothed rates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


BCE_WEIGHT = 0.7
MSE_WEIGHT = 0.3


@dataclass
class TrainingResult:
    """Full training result with diagnostics."""
    spike_correlation: float
    bifurcation_error: float
    training_hours: float
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    failure_regimes: list      # Time windows with highest loss
    gaba_level_losses: dict    # Per-GABA-level loss breakdown
    converged: bool
    model_state_dict: Optional[dict] = None


class FullTrainingPipeline:
    """
    Z3-C2 analog: Full training and evaluation.

    Trains architecture to convergence, records:
    - Which GABA levels the model fails on (counterexamples)
    - Which time windows have highest loss (failure regimes)
    - Training convergence curve for DreamCoder analysis
    """

    def __init__(self, train_data: Dict, val_data: Dict,
                 device: str = 'cuda', max_hours: float = 2.0):
        self.device = device
        self.max_hours = max_hours

        # Full training data — smoothed rate targets
        self.X_train = torch.tensor(train_data['X_train'], device=device)
        self.Y_train = torch.tensor(train_data['Y_train'], device=device)

        # Binary spike targets (for BCE component of combined loss)
        self.has_binary = 'Y_binary_train' in train_data
        if self.has_binary:
            self.Yb_train = torch.tensor(train_data['Y_binary_train'], device=device)
        else:
            self.Yb_train = None

        # Validation data
        self.X_val = torch.tensor(val_data['X_val'], device=device)
        self.Y_val = torch.tensor(val_data['Y_val'], device=device)

        if self.has_binary and 'Y_binary_val' in val_data:
            self.Yb_val = torch.tensor(val_data['Y_binary_val'], device=device)
        else:
            self.Yb_val = None

    def _compute_loss(self, y_pred, y_rate, y_binary, bce_fn, mse_fn):
        """Combined loss: 0.7*BCE(binary) + 0.3*MSE(rates).

        If no binary targets, falls back to MSE-only.
        """
        if y_binary is not None:
            y_pred_clamped = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
            loss = (BCE_WEIGHT * bce_fn(y_pred_clamped, y_binary) +
                    MSE_WEIGHT * mse_fn(y_pred, y_rate))
        else:
            loss = mse_fn(y_pred, y_rate)
        return loss

    def train(self, model: nn.Module, template,
              lr: float = 5e-4, batch_size: int = 32,
              max_epochs: int = 300, patience: int = 30,
              verbose: bool = True) -> Tuple[nn.Module, TrainingResult]:
        """
        Full training pipeline.

        Returns trained model and TrainingResult with diagnostics.
        """
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
        )
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        if self.has_binary and verbose:
            print("  Using combined loss: 0.7*BCE(binary) + 0.3*MSE(rates)")
        elif verbose:
            print("  Using MSE-only loss (no binary targets)")

        best_val_loss = float('inf')
        best_state = None
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()

        n_train = self.X_train.shape[0]

        for epoch in range(max_epochs):
            # Check time budget
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > self.max_hours:
                if verbose:
                    print(f"  Time budget exhausted ({elapsed_hours:.1f}h)")
                break

            # --- Training ---
            model.train()
            epoch_losses = []
            indices = torch.randperm(n_train)

            for i in range(0, n_train, batch_size):
                batch_idx = indices[i:i + batch_size]
                x = self.X_train[batch_idx]
                y_rate = self.Y_train[batch_idx]
                y_binary = self.Yb_train[batch_idx] if self.Yb_train is not None else None

                optimizer.zero_grad()
                output = model(x)
                y_pred = output[0] if isinstance(output, tuple) else output

                loss = self._compute_loss(y_pred, y_rate, y_binary, bce_loss, mse_loss)

                if torch.isnan(loss):
                    if verbose:
                        print(f"  NaN loss at epoch {epoch + 1}")
                    return model, TrainingResult(
                        spike_correlation=0.0, bifurcation_error=float('inf'),
                        training_hours=elapsed_hours, best_val_loss=float('inf'),
                        best_epoch=0, total_epochs=epoch,
                        failure_regimes=[], gaba_level_losses={},
                        converged=False
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_output = model(self.X_val)
                val_pred = val_output[0] if isinstance(val_output, tuple) else val_output
                val_loss = self._compute_loss(
                    val_pred, self.Y_val, self.Yb_val, bce_loss, mse_loss
                ).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                train_loss = np.mean(epoch_losses)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch + 1}: train={train_loss:.5f} val={val_loss:.5f} lr={current_lr:.1e}")

        elapsed_hours = (time.time() - start_time) / 3600

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Compute final metrics on rate targets
        model.eval()
        with torch.no_grad():
            val_output = model(self.X_val)
            val_pred = val_output[0] if isinstance(val_output, tuple) else val_output
            val_np = val_pred.cpu().numpy()
            val_true = self.Y_val.cpu().numpy()

        # Spike correlation (on smoothed rates)
        corrs = []
        for b in range(val_np.shape[0]):
            for ch in range(val_np.shape[2]):
                if np.std(val_true[b, :, ch]) > 1e-8:
                    c = np.corrcoef(val_true[b, :, ch], val_np[b, :, ch])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
        spike_corr = float(np.mean(corrs)) if corrs else 0.0

        # Failure regimes: find time windows with highest per-step loss
        per_step_loss = np.mean((val_np - val_true) ** 2, axis=(0, 2))
        top_failure_windows = np.argsort(per_step_loss)[-10:].tolist()

        return model, TrainingResult(
            spike_correlation=spike_corr,
            bifurcation_error=float('inf'),  # Computed separately
            training_hours=elapsed_hours,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            total_epochs=epoch + 1,
            failure_regimes=top_failure_windows,
            gaba_level_losses={},  # TODO: per-GABA analysis
            converged=patience_counter < patience,
            model_state_dict=best_state
        )
