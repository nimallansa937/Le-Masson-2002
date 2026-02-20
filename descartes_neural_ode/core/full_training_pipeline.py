"""
Layer 3: Full Training Pipeline (Z3-C2 Analog)

After short-segment verification passes, this trains the architecture
to convergence on the full 2000-step sequences and evaluates recovery
of biological variables.

Training protocol:
  - Progressive training: start with truncated sequences (200 steps),
    extend to 500, then 1000, then full 2000 as the model improves.
    This gives ODE architectures time to learn short-range dynamics first.
  - Loss: 0.7*BCE(binary spike targets) + 0.3*MSE(smoothed rate targets)
    BCE is applied to BINARY spike trains (0/1), NOT to smoothed rates.
    MSE is applied to smoothed firing rates.
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

# Progressive training schedule: (seq_len, fraction_of_budget)
# Spend 20% of time on 200-step, 20% on 500-step, 20% on 1000-step, 40% on full
PROGRESSIVE_SCHEDULE = [
    (200,  0.15),
    (500,  0.20),
    (1000, 0.25),
    (2000, 0.40),
]


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
        Full training pipeline with progressive sequence length.

        Starts with short sequences (200 steps) and progressively extends
        to full 2000 steps. This lets models learn short-range dynamics
        first, critical for ODE architectures that are slow on long sequences.

        Returns trained model and TrainingResult with diagnostics.
        """
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
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
        total_epochs = 0
        start_time = time.time()

        full_seq_len = self.X_train.shape[1]  # Usually 2000
        n_train = self.X_train.shape[0]

        # Build progressive schedule based on actual sequence length
        schedule = []
        for seq_len, frac in PROGRESSIVE_SCHEDULE:
            if seq_len <= full_seq_len:
                schedule.append((min(seq_len, full_seq_len), frac))
        # Ensure last stage uses full sequence
        if schedule and schedule[-1][0] < full_seq_len:
            schedule.append((full_seq_len, 0.20))
        # Normalize fractions
        total_frac = sum(f for _, f in schedule)
        schedule = [(s, f / total_frac) for s, f in schedule]

        for stage_idx, (seq_len, time_frac) in enumerate(schedule):
            stage_budget_hours = self.max_hours * time_frac
            stage_start = time.time()

            if verbose:
                print(f"  Stage {stage_idx + 1}/{len(schedule)}: "
                      f"seq_len={seq_len}, budget={stage_budget_hours * 60:.0f}min")

            # Truncate data to current sequence length
            X_tr = self.X_train[:, :seq_len, :]
            Y_tr = self.Y_train[:, :seq_len, :]
            Yb_tr = self.Yb_train[:, :seq_len, :] if self.Yb_train is not None else None
            X_vl = self.X_val[:, :seq_len, :]
            Y_vl = self.Y_val[:, :seq_len, :]
            Yb_vl = self.Yb_val[:, :seq_len, :] if self.Yb_val is not None else None

            stage_epochs = 0

            for epoch in range(max_epochs):
                # Check total time budget
                elapsed_total = (time.time() - start_time) / 3600
                if elapsed_total > self.max_hours:
                    if verbose:
                        print(f"  Total time budget exhausted ({elapsed_total:.2f}h)")
                    break

                # Check stage time budget
                elapsed_stage = (time.time() - stage_start) / 3600
                if elapsed_stage > stage_budget_hours:
                    if verbose:
                        print(f"  Stage budget exhausted after {stage_epochs} epochs")
                    break

                # --- Training ---
                model.train()
                epoch_losses = []
                indices = torch.randperm(n_train)
                epoch_start = time.time()

                for i in range(0, n_train, batch_size):
                    batch_idx = indices[i:i + batch_size]
                    x = X_tr[batch_idx]
                    y_rate = Y_tr[batch_idx]
                    y_binary = Yb_tr[batch_idx] if Yb_tr is not None else None

                    optimizer.zero_grad()
                    output = model(x)
                    y_pred = output[0] if isinstance(output, tuple) else output

                    # Handle seq_len mismatch if model outputs different length
                    if y_pred.shape[1] != y_rate.shape[1]:
                        min_len = min(y_pred.shape[1], y_rate.shape[1])
                        y_pred = y_pred[:, :min_len, :]
                        y_rate = y_rate[:, :min_len, :]
                        if y_binary is not None:
                            y_binary = y_binary[:, :min_len, :]

                    loss = self._compute_loss(y_pred, y_rate, y_binary, bce_loss, mse_loss)

                    if torch.isnan(loss):
                        if verbose:
                            print(f"  NaN loss at epoch {total_epochs + 1}")
                        elapsed_hours = (time.time() - start_time) / 3600
                        return model, TrainingResult(
                            spike_correlation=0.0, bifurcation_error=float('inf'),
                            training_hours=elapsed_hours, best_val_loss=float('inf'),
                            best_epoch=0, total_epochs=total_epochs,
                            failure_regimes=[], gaba_level_losses={},
                            converged=False
                        )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())

                # --- Validation (batched to avoid OOM) ---
                model.eval()
                with torch.no_grad():
                    val_preds = []
                    n_val = X_vl.shape[0]
                    for vi in range(0, n_val, batch_size):
                        vx = X_vl[vi:vi + batch_size]
                        vo = model(vx)
                        vp = vo[0] if isinstance(vo, tuple) else vo
                        if vp.shape[1] != seq_len:
                            vp = vp[:, :min(vp.shape[1], seq_len), :]
                        val_preds.append(vp)
                    val_pred = torch.cat(val_preds, dim=0)
                    # Ensure shapes match
                    min_len = min(val_pred.shape[1], Y_vl.shape[1])
                    val_loss = self._compute_loss(
                        val_pred[:, :min_len, :],
                        Y_vl[:, :min_len, :],
                        Yb_vl[:, :min_len, :] if Yb_vl is not None else None,
                        bce_loss, mse_loss
                    ).item()

                scheduler.step(val_loss)
                epoch_time = time.time() - epoch_start

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = total_epochs
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {total_epochs + 1}")
                        break

                total_epochs += 1
                stage_epochs += 1

                # Print every 5 epochs or every epoch for first 5
                if verbose and (total_epochs <= 5 or total_epochs % 5 == 0):
                    train_loss = np.mean(epoch_losses)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"    Ep {total_epochs} (len={seq_len}): "
                          f"train={train_loss:.5f} val={val_loss:.5f} "
                          f"lr={current_lr:.1e} [{epoch_time:.1f}s/ep]")

            # Reset patience between stages (model needs to re-adapt)
            patience_counter = max(0, patience_counter - 10)

        elapsed_hours = (time.time() - start_time) / 3600

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Compute final metrics on FULL-LENGTH rate targets (batched)
        model.eval()
        with torch.no_grad():
            val_preds_final = []
            n_val = self.X_val.shape[0]
            for vi in range(0, n_val, batch_size):
                vx = self.X_val[vi:vi + batch_size]
                vo = model(vx)
                vp = vo[0] if isinstance(vo, tuple) else vo
                val_preds_final.append(vp.cpu())
            val_pred = torch.cat(val_preds_final, dim=0)
            val_np = val_pred.numpy()
            val_true = self.Y_val.cpu().numpy()

        # Align lengths for evaluation
        min_len = min(val_np.shape[1], val_true.shape[1])
        val_np = val_np[:, :min_len, :]
        val_true = val_true[:, :min_len, :]

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

        if verbose:
            print(f"  Training complete: {total_epochs} epochs in {elapsed_hours:.2f}h")
            print(f"  Final spike_corr={spike_corr:.4f}, best_val_loss={best_val_loss:.5f}")

        return model, TrainingResult(
            spike_correlation=spike_corr,
            bifurcation_error=float('inf'),  # Computed separately
            training_hours=elapsed_hours,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            total_epochs=total_epochs,
            failure_regimes=top_failure_windows,
            gaba_level_losses={},  # TODO: per-GABA analysis
            converged=patience_counter < patience,
            model_state_dict=best_state
        )
