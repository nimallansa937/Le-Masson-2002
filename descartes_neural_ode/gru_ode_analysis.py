"""
GRU-ODE Biological Alignment Analysis
=======================================
Critical comparison experiment: Does GRU-ODE genuinely encode biological
variables, or does its high Pearson r=0.98 on GABA_B reflect shared trends?

Trains GRU-ODE (latent_dim=32), then runs the full alignment suite:
  1. Pearson |r| (1-to-1 alignment)
  2. CKA (Centered Kernel Alignment)
  3. Mutual Information
  4. Ridge Regression R² (the definitive test)

If Ridge R² is high → confirmed genuine biological encoding
If Ridge R² is low but Pearson is high → shared trends, not real encoding

Usage:
  python gru_ode_analysis.py --data_dir /root/rung3_data --device cuda
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ar3_data_loader import load_ar2_data
from architectures.gru_ode import GRUODEModel
from core.biovar_recovery_space import build_biovar_registry, score_biovar_recovery

# ─────────────────────────────────────────────────────────────────────
# Constants (identical to main DESCARTES pipeline)
# ─────────────────────────────────────────────────────────────────────
BCE_WEIGHT = 0.7
MSE_WEIGHT = 0.3
PROGRESSIVE_SCHEDULE = [
    (200,  0.15),
    (500,  0.20),
    (1000, 0.25),
    (2000, 0.40),
]


# ─────────────────────────────────────────────────────────────────────
# Alignment metrics (same as ltc_dimension_sweep.py)
# ─────────────────────────────────────────────────────────────────────

def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X
    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_mutual_info(latents: np.ndarray, bio_matrix: np.ndarray,
                        top_k: int = 20) -> Dict:
    """MI between latent dims and top bio variables."""
    from sklearn.feature_selection import mutual_info_regression
    bio_var = np.var(bio_matrix, axis=1)
    valid_mask = bio_var > 1e-10
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return {'mean_mi': 0.0, 'max_mi': 0.0, 'per_var': {}}
    sorted_idx = valid_indices[np.argsort(bio_var[valid_indices])[::-1]]
    top_indices = sorted_idx[:top_k]
    X = latents.T
    mi_scores = {}
    for bio_idx in top_indices:
        y = bio_matrix[bio_idx]
        if np.std(y) < 1e-10:
            continue
        mi = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
        mi_scores[int(bio_idx)] = float(np.max(mi))
    mean_mi = float(np.mean(list(mi_scores.values()))) if mi_scores else 0.0
    max_mi = float(np.max(list(mi_scores.values()))) if mi_scores else 0.0
    return {'mean_mi': mean_mi, 'max_mi': max_mi, 'per_var': mi_scores}


def compute_ridge_r2(latents: np.ndarray, bio_matrix: np.ndarray,
                     alpha: float = 1.0) -> Dict:
    """Ridge regression R² — can we linearly decode each bio var from latents?"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    X = latents.T
    n_bio = bio_matrix.shape[0]
    T = X.shape[0]
    if T > 5000:
        subsample_idx = np.random.RandomState(42).choice(T, 5000, replace=False)
        X = X[subsample_idx]
        bio_sub = bio_matrix[:, subsample_idx]
    else:
        bio_sub = bio_matrix
    r2_scores = {}
    ridge = Ridge(alpha=alpha)
    for i in range(n_bio):
        y = bio_sub[i]
        if np.std(y) < 1e-10:
            r2_scores[i] = 0.0
            continue
        try:
            scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
            r2_scores[i] = max(0.0, float(np.mean(scores)))
        except Exception:
            r2_scores[i] = 0.0
    r2_values = list(r2_scores.values())
    mean_r2 = float(np.mean(r2_values))
    return {
        'mean_r2': mean_r2,
        'n_decodable_r2_gt_025': sum(1 for v in r2_values if v > 0.25),
        'n_decodable_r2_gt_050': sum(1 for v in r2_values if v > 0.50),
        'per_var': {int(k): round(v, 4) for k, v in r2_scores.items()},
    }


# ─────────────────────────────────────────────────────────────────────
# Latent extraction
# ─────────────────────────────────────────────────────────────────────

def extract_latents(model: nn.Module, X_val: torch.Tensor,
                    device: str = 'cuda', batch_size: int = 16) -> np.ndarray:
    """Extract latent trajectories. Returns (latent_dim, T)."""
    model.eval()
    all_latents = []
    with torch.no_grad():
        for i in range(0, X_val.shape[0], batch_size):
            batch = X_val[i:i + batch_size].to(device)
            output = model(batch, return_latent=True)
            if isinstance(output, tuple) and len(output) >= 2:
                latent = output[1]
            else:
                print("  WARNING: Model did not return latent trajectory!")
                return None
            all_latents.append(latent.cpu().numpy())
    latents = np.concatenate(all_latents, axis=0)
    mean_latent = np.mean(latents, axis=0)
    result = mean_latent.T
    print(f"  [latent] Shape: {result.shape}")
    print(f"  [latent] Stats: mean={result.mean():.4f}, std={result.std():.4f}, "
          f"min={result.min():.4f}, max={result.max():.4f}")
    per_dim_std = np.std(result, axis=1)
    n_constant = np.sum(per_dim_std < 1e-6)
    if n_constant > 0:
        print(f"  [latent] WARNING: {n_constant}/{result.shape[0]} dims near-constant")
    print(f"  [latent] Active dims: {result.shape[0] - n_constant}/{result.shape[0]}")
    return result


# ─────────────────────────────────────────────────────────────────────
# Bio ground truth alignment
# ─────────────────────────────────────────────────────────────────────

def align_bio_matrix(bio_ground_truth: Dict, registry, target_T: int) -> np.ndarray:
    """Build (n_bio, T) matrix from ground truth, aligned to target_T."""
    n_bio = len(registry)
    bio_matrix = np.zeros((n_bio, target_T), dtype=np.float32)
    matched = 0
    for i, var in enumerate(registry):
        if ':' in var.name:
            gt_key, neuron_str = var.name.rsplit(':', 1)
            try:
                neuron_idx = int(neuron_str)
            except ValueError:
                continue
        else:
            parts = var.name.rsplit('_', 1)
            if len(parts) != 2:
                continue
            gt_key, neuron_str = parts
            try:
                neuron_idx = int(neuron_str)
            except ValueError:
                continue
        if gt_key not in bio_ground_truth:
            continue
        gt_array = bio_ground_truth[gt_key]
        if neuron_idx >= gt_array.shape[0]:
            continue
        signal = gt_array[neuron_idx]
        if len(signal) != target_T:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, target_T)
            f = interp1d(x_old, signal, kind='linear', fill_value='extrapolate')
            signal = f(x_new)
        bio_matrix[i] = signal
        matched += 1
    print(f"  [bio] Matched {matched}/{n_bio} variables to ground truth")
    return bio_matrix


# ─────────────────────────────────────────────────────────────────────
# Training (identical protocol to DESCARTES pipeline)
# ─────────────────────────────────────────────────────────────────────

def train_gru_ode(model: nn.Module, train_data: Dict, val_data: Dict,
                  device: str = 'cuda', max_hours: float = 2.0,
                  lr: float = 5e-4, batch_size: int = 32,
                  max_epochs: int = 300, patience: int = 40):
    """Train GRU-ODE with progressive sequence length schedule."""
    model = model.to(device)

    X_train = torch.tensor(train_data['X_train'], device=device)
    Y_train = torch.tensor(train_data['Y_train'], device=device)
    Yb_train = torch.tensor(train_data['Y_binary_train'], device=device) \
        if 'Y_binary_train' in train_data else None
    X_val = torch.tensor(val_data['X_val'], device=device)
    Y_val = torch.tensor(val_data['Y_val'], device=device)
    Yb_val = torch.tensor(val_data['Y_binary_val'], device=device) \
        if 'Y_binary_val' in val_data else None

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=20, factor=0.5, min_lr=1e-5
    )
    bce_fn = nn.BCELoss()
    mse_fn = nn.MSELoss()
    has_binary = Yb_train is not None
    print(f"  Loss: {'0.7*BCE + 0.3*MSE' if has_binary else 'MSE-only'}")

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    total_epochs = 0
    start_time = time.time()
    n_train = X_train.shape[0]
    full_seq_len = X_train.shape[1]

    schedule = []
    for seq_len, frac in PROGRESSIVE_SCHEDULE:
        if seq_len <= full_seq_len:
            schedule.append((min(seq_len, full_seq_len), frac))
    if schedule and schedule[-1][0] < full_seq_len:
        schedule.append((full_seq_len, 0.20))
    total_frac = sum(f for _, f in schedule)
    schedule = [(s, f / total_frac) for s, f in schedule]

    for stage_idx, (seq_len, time_frac) in enumerate(schedule):
        stage_budget = max_hours * time_frac
        stage_start = time.time()
        print(f"  Stage {stage_idx + 1}/{len(schedule)}: "
              f"seq_len={seq_len}, budget={stage_budget * 60:.0f}min")

        X_tr = X_train[:, :seq_len, :]
        Y_tr = Y_train[:, :seq_len, :]
        Yb_tr = Yb_train[:, :seq_len, :] if Yb_train is not None else None
        X_vl = X_val[:, :seq_len, :]
        Y_vl = Y_val[:, :seq_len, :]
        Yb_vl = Yb_val[:, :seq_len, :] if Yb_val is not None else None

        stage_epochs = 0
        for epoch in range(max_epochs):
            elapsed_total = (time.time() - start_time) / 3600
            if elapsed_total > max_hours:
                print(f"  Total budget exhausted ({elapsed_total:.2f}h)")
                break
            elapsed_stage = (time.time() - stage_start) / 3600
            if elapsed_stage > stage_budget:
                print(f"  Stage budget exhausted after {stage_epochs} epochs")
                break

            model.train()
            epoch_losses = []
            indices = torch.randperm(n_train, device=device)
            epoch_start = time.time()

            for i in range(0, n_train, batch_size):
                batch_idx = indices[i:i + batch_size]
                x = X_tr[batch_idx]
                y_rate = Y_tr[batch_idx]
                y_bin = Yb_tr[batch_idx] if Yb_tr is not None else None

                optimizer.zero_grad()
                output = model(x)
                y_pred = output[0] if isinstance(output, tuple) else output

                if y_pred.shape[1] != y_rate.shape[1]:
                    ml = min(y_pred.shape[1], y_rate.shape[1])
                    y_pred = y_pred[:, :ml, :]
                    y_rate = y_rate[:, :ml, :]
                    if y_bin is not None:
                        y_bin = y_bin[:, :ml, :]

                y_pred_c = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
                if has_binary and y_bin is not None:
                    loss = BCE_WEIGHT * bce_fn(y_pred_c, y_bin) + \
                           MSE_WEIGHT * mse_fn(y_pred, y_rate)
                else:
                    loss = mse_fn(y_pred, y_rate)

                if torch.isnan(loss):
                    print(f"  NaN loss at epoch {total_epochs + 1}")
                    return model, {'spike_corr': 0.0, 'total_epochs': total_epochs,
                                   'hours': (time.time() - start_time) / 3600,
                                   'converged': False}

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            # Validate
            model.eval()
            with torch.no_grad():
                val_preds = []
                for vi in range(0, X_vl.shape[0], batch_size):
                    vx = X_vl[vi:vi + batch_size]
                    vo = model(vx)
                    vp = vo[0] if isinstance(vo, tuple) else vo
                    if vp.shape[1] != seq_len:
                        vp = vp[:, :min(vp.shape[1], seq_len), :]
                    val_preds.append(vp)
                val_pred = torch.cat(val_preds, dim=0)
                ml = min(val_pred.shape[1], Y_vl.shape[1])
                vp_c = torch.clamp(val_pred[:, :ml, :], 1e-7, 1 - 1e-7)
                if has_binary and Yb_vl is not None:
                    val_loss = (
                        BCE_WEIGHT * bce_fn(vp_c, Yb_vl[:, :ml, :]) +
                        MSE_WEIGHT * mse_fn(val_pred[:, :ml, :], Y_vl[:, :ml, :])
                    ).item()
                else:
                    val_loss = mse_fn(val_pred[:, :ml, :], Y_vl[:, :ml, :]).item()

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
                    print(f"  Early stopping at epoch {total_epochs + 1}")
                    break

            total_epochs += 1
            stage_epochs += 1

            if total_epochs <= 5 or total_epochs % 5 == 0:
                train_loss = np.mean(epoch_losses)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Ep {total_epochs} (len={seq_len}): "
                      f"train={train_loss:.5f} val={val_loss:.5f} "
                      f"lr={current_lr:.1e} [{epoch_time:.1f}s/ep]")

        patience_counter = max(0, patience_counter - 10)

    elapsed_hours = (time.time() - start_time) / 3600
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final spike correlation
    model.eval()
    with torch.no_grad():
        val_preds_final = []
        for vi in range(0, X_val.shape[0], batch_size):
            vx = X_val[vi:vi + batch_size]
            vo = model(vx)
            vp = vo[0] if isinstance(vo, tuple) else vo
            val_preds_final.append(vp.cpu())
        val_pred = torch.cat(val_preds_final, dim=0).numpy()
        val_true = Y_val.cpu().numpy()

    ml = min(val_pred.shape[1], val_true.shape[1])
    val_pred = val_pred[:, :ml, :]
    val_true = val_true[:, :ml, :]

    corrs = []
    for b in range(val_pred.shape[0]):
        for ch in range(val_pred.shape[2]):
            if np.std(val_true[b, :, ch]) > 1e-8:
                c = np.corrcoef(val_true[b, :, ch], val_pred[b, :, ch])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
    spike_corr = float(np.mean(corrs)) if corrs else 0.0

    print(f"  Training done: {total_epochs} ep in {elapsed_hours:.2f}h")
    print(f"  spike_corr={spike_corr:.4f}, best_val_loss={best_val_loss:.5f}")

    return model, {
        'spike_corr': spike_corr,
        'best_val_loss': float(best_val_loss),
        'total_epochs': total_epochs,
        'best_epoch': best_epoch,
        'hours': round(elapsed_hours, 3),
        'converged': patience_counter < patience,
    }


# ─────────────────────────────────────────────────────────────────────
# GRU-ODE specific: per-gate analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_gru_gates(model: nn.Module, X_val: torch.Tensor,
                      device: str = 'cuda', batch_size: int = 16) -> Dict:
    """
    Analyze GRU-ODE gate activations to understand which latent dims
    are "dedicated" (update gate near 0 = always evolving) vs
    "static" (update gate near 1 = rarely changing).

    This reveals whether GRU-ODE creates biological-variable-like
    slots in its latent space.
    """
    model.eval()
    all_update_gates = []
    all_reset_gates = []

    with torch.no_grad():
        for i in range(0, min(X_val.shape[0], 50), batch_size):  # Sample 50 windows
            batch = X_val[i:i + batch_size].to(device)
            b, seq_len, _ = batch.shape
            z = model.encoder(batch[:, 0, :])

            update_seq = []
            reset_seq = []
            for t in range(seq_len):
                u = batch[:, t, :]
                zu = torch.cat([z, u], dim=-1)
                update = torch.sigmoid(model.gru_ode_cell.W_z(zu))
                reset = torch.sigmoid(model.gru_ode_cell.W_r(zu))
                update_seq.append(update.cpu().numpy())
                reset_seq.append(reset.cpu().numpy())
                if t < seq_len - 1:
                    z = model.gru_ode_cell(z, u)

            all_update_gates.append(np.stack(update_seq, axis=1))  # (batch, T, latent)
            all_reset_gates.append(np.stack(reset_seq, axis=1))

    updates = np.concatenate(all_update_gates, axis=0)  # (N, T, latent)
    resets = np.concatenate(all_reset_gates, axis=0)

    # Mean gate activations per latent dim
    mean_update = np.mean(updates, axis=(0, 1))  # (latent_dim,)
    mean_reset = np.mean(resets, axis=(0, 1))

    # Temporal variability of gates (high = gate responds to input dynamics)
    update_temporal_var = np.mean(np.std(updates, axis=1), axis=0)
    reset_temporal_var = np.mean(np.std(resets, axis=1), axis=0)

    # Classify dims
    n_latent = mean_update.shape[0]
    dedicated_dims = np.sum(mean_update < 0.3)   # Always evolving
    static_dims = np.sum(mean_update > 0.7)       # Rarely changing
    dynamic_dims = n_latent - dedicated_dims - static_dims

    print(f"\n  === GRU-ODE Gate Analysis ===")
    print(f"  Update gate (low = always evolving, high = static):")
    print(f"    Mean: {mean_update.mean():.4f} +/- {mean_update.std():.4f}")
    print(f"    Dedicated (update<0.3): {dedicated_dims}/{n_latent}")
    print(f"    Dynamic (0.3-0.7):      {dynamic_dims}/{n_latent}")
    print(f"    Static (update>0.7):    {static_dims}/{n_latent}")
    print(f"  Reset gate:")
    print(f"    Mean: {mean_reset.mean():.4f} +/- {mean_reset.std():.4f}")
    print(f"  Temporal responsiveness (gate std over time):")
    print(f"    Update gate: {update_temporal_var.mean():.4f}")
    print(f"    Reset gate:  {reset_temporal_var.mean():.4f}")

    # Top 5 most "dedicated" dims (lowest update gate → always evolving)
    dedicated_idx = np.argsort(mean_update)[:5]
    print(f"  Top 5 most dedicated dims (lowest update gate):")
    for idx in dedicated_idx:
        print(f"    dim {idx}: update={mean_update[idx]:.4f}, "
              f"reset={mean_reset[idx]:.4f}, "
              f"update_var={update_temporal_var[idx]:.4f}")

    return {
        'mean_update_gate': float(mean_update.mean()),
        'mean_reset_gate': float(mean_reset.mean()),
        'dedicated_dims': int(dedicated_dims),
        'dynamic_dims': int(dynamic_dims),
        'static_dims': int(static_dims),
        'update_temporal_var': float(update_temporal_var.mean()),
        'reset_temporal_var': float(reset_temporal_var.mean()),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run_analysis(data_dir: str, device: str = 'cuda',
                 max_hours: float = 2.0, latent_dim: int = 32):
    print("=" * 70)
    print("GRU-ODE BIOLOGICAL ALIGNMENT ANALYSIS")
    print("=" * 70)
    print(f"  Latent dim: {latent_dim}")
    print(f"  Device: {device}")
    print(f"  Budget: {max_hours}h")
    print(f"  Data: {data_dir}")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    train_data, val_data, bio_ground_truth = load_ar2_data(data_dir)
    print(f"  Train: {train_data['X_train'].shape}")
    print(f"  Val:   {val_data['X_val'].shape}")

    # Build registry
    registry = build_biovar_registry()
    print(f"  Registry: {len(registry)} biological variables")

    target_T = val_data['X_val'].shape[1]
    bio_matrix = align_bio_matrix(bio_ground_truth, registry, target_T)

    # Create model
    print(f"\n[2] Creating GRU-ODE model...")
    model = GRUODEModel(
        n_input=21, n_output=20,
        latent_dim=latent_dim, hidden_dim=128
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print(f"\n[3] Training GRU-ODE...")
    model, train_info = train_gru_ode(
        model, train_data, val_data,
        device=device, max_hours=max_hours
    )

    # Extract latents
    print(f"\n[4] Extracting latent states...")
    X_val_tensor = torch.tensor(val_data['X_val'])
    latents = extract_latents(model, X_val_tensor, device=device)
    if latents is None:
        print("  FAILED. Aborting.")
        return

    # ── Pearson biovar recovery ──
    print(f"\n[5] Pearson biovar recovery...")
    recovery = score_biovar_recovery(latents, bio_ground_truth, registry, 0.5)
    print(f"  Pearson recovered: {recovery.n_recovered}/160 (|r|>0.5)")
    print(f"  CCA score: {recovery.cca_score:.4f}")
    print(f"  By category: {recovery.recovered_by_category}")
    print(f"  By timescale: {recovery.recovered_by_timescale}")

    # Top 10 + Bottom 5 Pearson
    corr_vec = recovery.correlation_vector
    top10 = np.argsort(corr_vec)[-10:][::-1]
    print(f"\n  Top 10 Pearson correlations:")
    for idx in top10:
        print(f"    {registry[idx].name}: |r|={corr_vec[idx]:.4f} "
              f"({registry[idx].category}/{registry[idx].timescale})")

    near_miss = np.sum((corr_vec > 0.3) & (corr_vec < 0.5))
    print(f"\n  Near-misses (0.3<|r|<0.5): {near_miss}")
    print(f"  Non-zero (|r|>0.01): {np.sum(corr_vec > 0.01)}")

    # ── CKA ──
    print(f"\n[6] Computing CKA...")
    T = latents.shape[1]
    if T > 2000:
        sub = np.random.RandomState(42).choice(T, 2000, replace=False)
        L_sub = latents[:, sub].T
        B_sub = bio_matrix[:, sub].T
    else:
        L_sub = latents.T
        B_sub = bio_matrix.T
    cka_score = compute_cka(L_sub, B_sub)
    print(f"  CKA = {cka_score:.4f}")

    # ── Mutual Information ──
    print(f"\n[7] Computing Mutual Information (top 20 bio vars)...")
    mi_result = compute_mutual_info(latents, bio_matrix, top_k=20)
    print(f"  MI: mean={mi_result['mean_mi']:.4f}, max={mi_result['max_mi']:.4f}")

    # ── Ridge R² ──
    print(f"\n[8] Computing Ridge Regression R² (5-fold CV)...")
    print(f"  >>> THIS IS THE KEY METRIC <<<")
    ridge_result = compute_ridge_r2(latents, bio_matrix, alpha=1.0)
    print(f"  Ridge R² mean: {ridge_result['mean_r2']:.4f}")
    print(f"  Decodable (R²>0.25): {ridge_result['n_decodable_r2_gt_025']}/160")
    print(f"  Decodable (R²>0.50): {ridge_result['n_decodable_r2_gt_050']}/160")

    # Category breakdown
    cat_r2 = {}
    for var in registry:
        r2 = ridge_result['per_var'].get(var.id, 0.0)
        cat_r2.setdefault(var.category, []).append(r2)
    print(f"\n  Ridge R² by category:")
    for cat, vals in sorted(cat_r2.items()):
        print(f"    {cat}: mean R²={np.mean(vals):.4f}, "
              f"decodable={sum(1 for v in vals if v > 0.25)}/{len(vals)}")

    # Timescale breakdown
    ts_r2 = {}
    for var in registry:
        r2 = ridge_result['per_var'].get(var.id, 0.0)
        ts_r2.setdefault(var.timescale, []).append(r2)
    print(f"\n  Ridge R² by timescale:")
    for ts, vals in sorted(ts_r2.items()):
        print(f"    {ts}: mean R²={np.mean(vals):.4f}, "
              f"decodable={sum(1 for v in vals if v > 0.25)}/{len(vals)}")

    # ── GRU Gate Analysis ──
    print(f"\n[9] GRU-ODE gate analysis...")
    X_val_tensor_dev = torch.tensor(val_data['X_val'])
    gate_info = analyze_gru_gates(model, X_val_tensor_dev, device=device)

    # ── Comparison with LTC ──
    print("\n" + "=" * 70)
    print("  VERDICT: GRU-ODE vs LTC")
    print("=" * 70)
    print(f"  GRU-ODE (dim={latent_dim}):")
    print(f"    Spike corr:     {train_info['spike_corr']:.4f}")
    print(f"    Pearson:        {recovery.n_recovered}/160")
    print(f"    CKA:            {cka_score:.4f}")
    print(f"    MI (mean):      {mi_result['mean_mi']:.4f}")
    print(f"    Ridge R² mean:  {ridge_result['mean_r2']:.4f}")
    print(f"    Ridge decode25: {ridge_result['n_decodable_r2_gt_025']}/160")
    print(f"    Ridge decode50: {ridge_result['n_decodable_r2_gt_050']}/160")
    print(f"    Gate dedicated: {gate_info['dedicated_dims']}/{latent_dim}")
    print("-" * 70)
    print(f"  LTC (dim=32):     (from sweep: spike_corr~0.407, R²~0)")
    print(f"  LTC (dim=128):    (from sweep: spike_corr~0.407, R²~0)")
    print("-" * 70)

    if ridge_result['mean_r2'] > 0.1:
        print("  CONCLUSION: GRU-ODE genuinely encodes biological variables.")
        print("  The high Pearson is NOT just shared trends.")
    elif ridge_result['mean_r2'] < 0.05 and recovery.n_recovered > 5:
        print("  CONCLUSION: GRU-ODE Pearson reflects SHARED TRENDS, not encoding.")
        print("  Similar to LTC — high corr but alien representation.")
    else:
        print("  CONCLUSION: Mixed result. Check per-category R² for details.")

    print("=" * 70)

    # Save results
    output = {
        'experiment': 'GRU_ODE_biological_alignment_analysis',
        'latent_dim': latent_dim,
        'n_params': n_params,
        'train_info': train_info,
        'spike_corr': train_info['spike_corr'],
        'pearson_recovered': recovery.n_recovered,
        'pearson_by_category': recovery.recovered_by_category,
        'pearson_by_timescale': recovery.recovered_by_timescale,
        'cca_score': float(recovery.cca_score),
        'cka_score': cka_score,
        'mi_mean': mi_result['mean_mi'],
        'mi_max': mi_result['max_mi'],
        'ridge_mean_r2': ridge_result['mean_r2'],
        'ridge_decodable_025': ridge_result['n_decodable_r2_gt_025'],
        'ridge_decodable_050': ridge_result['n_decodable_r2_gt_050'],
        'gate_analysis': gate_info,
        'top10_pearson': [
            {'name': registry[idx].name, 'r': round(float(corr_vec[idx]), 4)}
            for idx in top10
        ],
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'gru_ode_analysis_results.json'
    )
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRU-ODE Biological Alignment Analysis')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_hours', type=float, default=2.0)
    parser.add_argument('--latent_dim', type=int, default=32)
    args = parser.parse_args()
    run_analysis(args.data_dir, args.device, args.max_hours, args.latent_dim)
