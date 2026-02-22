"""
Training loop for GRU-ODE with auxiliary biological loss (A-R3b).

Uses the same 4-stage progressive curriculum as DESCARTES A-R3:
  Stage 1: seq_len=200,  budget=15% (18min of 2h)
  Stage 2: seq_len=500,  budget=20% (24min)
  Stage 3: seq_len=1000, budget=25% (30min)
  Stage 4: seq_len=2000, budget=40% (48min)

Key difference from A-R3: the loss function includes biological
variable reconstruction as a secondary objective, controlled by
the alpha mixing parameter.

Hyperparameters follow user-specified values from previous experiments:
  lr_patience=20, min_lr=1e-5, early_stop_patience=40
These were chosen to prevent premature learning rate collapse that
plagued earlier LTC sweep and GRU-ODE analysis runs.
"""
import torch
import time
import json
import numpy as np
from pathlib import Path


# Progressive curriculum: (seq_len, fraction_of_total_budget)
PROGRESSIVE_SCHEDULE = [
    (200,  0.15),
    (500,  0.20),
    (1000, 0.25),
    (2000, 0.40),
]


def train_gru_ode_bio(
    model,
    train_loader,
    val_loader,
    loss_fn,
    alpha,
    device='cuda',
    total_budget_hours=2.0,
    output_dir='./results',
    lr=5e-4,
    lr_patience=20,
    min_lr=1e-5,
    early_stop_patience=40,
    grad_clip=1.0,
    seed=42,
):
    """
    Train GRU-ODE-Bio with combined spike + bio loss.

    Uses progressive curriculum (short -> long sequences) for stable
    convergence, exactly matching the DESCARTES A-R3 training protocol.

    Args:
        model: GRUODEBio instance
        train_loader: DataLoader yielding (x, y, y_binary, bio_targets) batches
        val_loader: DataLoader for validation
        loss_fn: CombinedBioLoss instance with alpha set
        alpha: loss mixing parameter (for logging, loss_fn has it internally)
        device: 'cuda' or 'cpu'
        total_budget_hours: maximum training time
        output_dir: where to save checkpoints and logs
        lr: initial learning rate
        lr_patience: epochs before LR reduction
        min_lr: minimum learning rate
        early_stop_patience: epochs before early stopping
        grad_clip: gradient norm clipping value
        seed: random seed for reproducibility

    Returns:
        results: dict with final training metrics
    """
    # Set seeds for reproducibility across alpha conditions
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=lr_patience, factor=0.5, min_lr=min_lr
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    total_epochs = 0
    start_time = time.time()
    training_log = []

    total_budget_s = total_budget_hours * 3600

    for stage_idx, (seq_len, budget_frac) in enumerate(PROGRESSIVE_SCHEDULE):
        stage_budget_s = total_budget_s * budget_frac
        stage_start = time.time()
        stage_epochs = 0

        print(f"\n  Stage {stage_idx+1}/4: seq_len={seq_len}, "
              f"budget={budget_frac*100:.0f}% ({stage_budget_s/60:.0f}min)")

        # Reset early stopping counter at each stage transition
        # (longer sequences change the loss landscape)
        epochs_no_improve = 0

        while time.time() - stage_start < stage_budget_s:
            # Check total time budget
            elapsed_total = time.time() - start_time
            if elapsed_total > total_budget_s:
                print(f"    Total time budget exhausted ({elapsed_total/3600:.2f}h)")
                break

            total_epochs += 1
            stage_epochs += 1
            ep_start = time.time()

            # === Training epoch ===
            model.train()
            train_losses = []
            train_components = {}

            for batch in train_loader:
                x, y, y_binary, bio_targets = batch

                # Truncate to current stage's seq_len
                x = x[:, :seq_len].to(device)
                y = y[:, :seq_len].to(device)
                y_bin = y_binary[:, :seq_len].to(device)

                # Move bio targets to device and truncate
                bio_t = {}
                for cat_name, cat_tensor in bio_targets.items():
                    bio_t[cat_name] = cat_tensor[:, :seq_len].to(device)

                # Forward pass
                spike_pred, bio_preds = model(x)

                # Truncate predictions to match truncated inputs
                spike_pred = spike_pred[:, :seq_len]
                bio_preds_trunc = {}
                for k, v in bio_preds.items():
                    bio_preds_trunc[k] = v[:, :seq_len]

                # Combined loss
                loss, components = loss_fn(
                    spike_pred, y_bin, y, bio_preds_trunc, bio_t
                )

                # Backward + optimize
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                train_losses.append(loss.item())
                for k, v in components.items():
                    if k not in train_components:
                        train_components[k] = []
                    train_components[k].append(v)

            # === Validation epoch ===
            model.eval()
            val_losses = []
            val_components = {}

            with torch.no_grad():
                for batch in val_loader:
                    x, y, y_binary, bio_targets = batch
                    x = x[:, :seq_len].to(device)
                    y = y[:, :seq_len].to(device)
                    y_bin = y_binary[:, :seq_len].to(device)

                    bio_t = {}
                    for cat_name, cat_tensor in bio_targets.items():
                        bio_t[cat_name] = cat_tensor[:, :seq_len].to(device)

                    spike_pred, bio_preds = model(x)
                    spike_pred = spike_pred[:, :seq_len]
                    bio_preds_trunc = {}
                    for k, v in bio_preds.items():
                        bio_preds_trunc[k] = v[:, :seq_len]

                    loss, comp = loss_fn(
                        spike_pred, y_bin, y, bio_preds_trunc, bio_t
                    )
                    val_losses.append(loss.item())
                    for k, v in comp.items():
                        if k not in val_components:
                            val_components[k] = []
                        val_components[k].append(v)

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            ep_time = time.time() - ep_start
            current_lr = optimizer.param_groups[0]['lr']

            # Build log entry
            log_entry = {
                'epoch': total_epochs,
                'stage': stage_idx + 1,
                'seq_len': seq_len,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr,
                'time_s': ep_time,
                'alpha': alpha,
                'train_components': {k: float(np.mean(v))
                                     for k, v in train_components.items()},
                'val_components': {k: float(np.mean(v))
                                   for k, v in val_components.items()},
            }
            training_log.append(log_entry)

            # Print progress
            if total_epochs % 5 == 0 or total_epochs <= 3 or stage_epochs <= 2:
                comp_str = ""
                tc = log_entry['train_components']
                if 'spike' in tc:
                    comp_str += f" spike={tc['spike']:.5f}"
                if 'bio_total' in tc:
                    comp_str += f" bio={tc['bio_total']:.5f}"
                print(f"    Ep {total_epochs} (S{stage_idx+1} len={seq_len}): "
                      f"train={train_loss:.5f} val={val_loss:.5f} "
                      f"lr={current_lr:.1e} [{ep_time:.1f}s]{comp_str}")

            # LR scheduling
            scheduler.step(val_loss)

            # Early stopping / best model tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                # Save best checkpoint
                torch.save(best_model_state,
                           output_dir / f'best_model_alpha{alpha:.2f}.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"    Early stopping at epoch {total_epochs} "
                          f"(no improvement for {early_stop_patience} epochs)")
                    break

        # Check total budget after stage
        elapsed_total = time.time() - start_time
        if elapsed_total > total_budget_s:
            print(f"  Total budget exhausted after stage {stage_idx+1}")
            break
        print(f"  Stage {stage_idx+1} done: {stage_epochs} epochs in "
              f"{(time.time() - stage_start)/60:.1f}min")

    # Load best model for downstream evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    elapsed_h = (time.time() - start_time) / 3600
    print(f"\n  Training complete: {total_epochs} epochs in {elapsed_h:.2f}h")
    print(f"  Best val loss: {best_val_loss:.6f}")

    # Save training log
    log_path = output_dir / f'training_log_alpha{alpha:.2f}.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"  Log saved to {log_path}")

    return {
        'total_epochs': total_epochs,
        'best_val_loss': best_val_loss,
        'training_hours': elapsed_h,
        'alpha': alpha,
        'final_lr': optimizer.param_groups[0]['lr'],
    }
