"""
LTC Latent Dimension Sweep Experiment
======================================
Standalone experiment: Does LTC with more latent capacity spontaneously
dedicate latent dimensions to biological variables?

Trains 3 LTC models (latent_dim = 32, 128, 256) on identical data with
identical training protocol, then compares biological variable recovery
using 4 alignment metrics:
  1. Pearson |r| (1-to-1 alignment)
  2. CKA (Centered Kernel Alignment — representation similarity)
  3. Mutual Information (sklearn MI regression for top bio vars)
  4. Ridge Regression R² (can you linearly decode bio vars from latents?)

The Ridge R² is the most important: if R² is high but Pearson is low,
biology is encoded but distributed. If R² is also low, the representation
is genuinely alien.

Usage:
  python ltc_dimension_sweep.py --data_dir /path/to/rung3_data --device cuda
  python ltc_dimension_sweep.py --data_dir /root/rung3_data --device cuda  # Vast.ai
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
from typing import Dict, List, Tuple, Optional

# Add parent dir to path so we can import from the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ar3_data_loader import load_ar2_data
from architectures.ltc_network import LTCModel
from core.biovar_recovery_space import (
    build_biovar_registry,
    score_biovar_recovery,
)

# ─────────────────────────────────────────────────────────────────────
# Training constants (identical to main DESCARTES pipeline)
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
# Advanced alignment metrics
# ─────────────────────────────────────────────────────────────────────

def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Centered Kernel Alignment (linear CKA).

    Measures representational similarity between two sets of features.
    CKA is invariant to orthogonal transformations and isotropic scaling,
    making it ideal for comparing neural network representations.

    Args:
        X: (n_samples, d1) — latent representations
        Y: (n_samples, d2) — biological variable matrix

    Returns:
        CKA score in [0, 1]. Higher = more similar representations.
    """
    # Center the matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Linear kernel: K = X @ X.T, L = Y @ Y.T
    # CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    # For linear kernels, HSIC(K, L) = ||Y.T @ X||_F^2 / (n-1)^2
    n = X.shape[0]

    XtX = X.T @ X  # (d1, d1)
    YtY = Y.T @ Y  # (d2, d2)
    YtX = Y.T @ X  # (d2, d1)

    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_mutual_info(latents: np.ndarray, bio_matrix: np.ndarray,
                        top_k: int = 20) -> Dict:
    """
    Mutual information between latent dims and top bio variables.

    Uses sklearn's mutual_info_regression which estimates MI via
    k-nearest-neighbors (Kraskov estimator).

    Args:
        latents: (latent_dim, T)
        bio_matrix: (n_bio, T) — aligned bio variable matrix
        top_k: number of bio vars to analyze (highest variance)

    Returns:
        Dict with mean_mi, max_mi, per-variable MI scores
    """
    from sklearn.feature_selection import mutual_info_regression

    # Pick top-k bio vars by variance (skip constant ones)
    bio_var = np.var(bio_matrix, axis=1)
    valid_mask = bio_var > 1e-10
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return {'mean_mi': 0.0, 'max_mi': 0.0, 'per_var': {}}

    # Sort by variance, take top_k
    sorted_idx = valid_indices[np.argsort(bio_var[valid_indices])[::-1]]
    top_indices = sorted_idx[:top_k]

    X = latents.T  # (T, latent_dim) — features
    mi_scores = {}

    for bio_idx in top_indices:
        y = bio_matrix[bio_idx]  # (T,)
        if np.std(y) < 1e-10:
            continue
        # MI between all latent dims and this bio var
        mi = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
        mi_scores[int(bio_idx)] = float(np.max(mi))  # Best latent dim's MI

    mean_mi = float(np.mean(list(mi_scores.values()))) if mi_scores else 0.0
    max_mi = float(np.max(list(mi_scores.values()))) if mi_scores else 0.0

    return {'mean_mi': mean_mi, 'max_mi': max_mi, 'per_var': mi_scores}


def compute_ridge_r2(latents: np.ndarray, bio_matrix: np.ndarray,
                     alpha: float = 1.0) -> Dict:
    """
    Ridge regression R² — can we linearly predict each bio var
    from ALL latent dims jointly?

    This is the MOST IMPORTANT metric. It catches distributed
    representations that 1-to-1 Pearson misses.

    If R² is high but Pearson is low → biology is encoded but distributed
    If R² is also low → representation is genuinely alien

    Args:
        latents: (latent_dim, T)
        bio_matrix: (n_bio, T)
        alpha: Ridge regularization strength

    Returns:
        Dict with mean_r2, per-variable R², recovered count (R² > 0.25)
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    X = latents.T  # (T, latent_dim)
    n_bio = bio_matrix.shape[0]

    # Subsample if T is very large (speed up cross-validation)
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
        # 5-fold cross-validated R²
        try:
            scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
            r2 = float(np.mean(scores))
            # Clamp to [0, 1] — negative R² means worse than mean predictor
            r2_scores[i] = max(0.0, r2)
        except Exception:
            r2_scores[i] = 0.0

    r2_values = list(r2_scores.values())
    mean_r2 = float(np.mean(r2_values))
    n_decodable = sum(1 for v in r2_values if v > 0.25)

    return {
        'mean_r2': mean_r2,
        'n_decodable_r2_gt_025': n_decodable,
        'n_decodable_r2_gt_050': sum(1 for v in r2_values if v > 0.50),
        'per_var': {int(k): round(v, 4) for k, v in r2_scores.items()},
    }


# ─────────────────────────────────────────────────────────────────────
# Latent extraction (standalone, same logic as orchestrator)
# ─────────────────────────────────────────────────────────────────────

def extract_latents(model: nn.Module, X_val: torch.Tensor,
                    device: str = 'cuda',
                    batch_size: int = 16) -> np.ndarray:
    """
    Extract latent trajectories from trained LTC model.

    Returns: (latent_dim, T) averaged over validation windows.
    """
    model.eval()
    all_latents = []

    with torch.no_grad():
        for i in range(0, X_val.shape[0], batch_size):
            batch = X_val[i:i + batch_size].to(device)
            output = model(batch, return_latent=True)

            if isinstance(output, tuple) and len(output) >= 2:
                latent = output[1]  # (batch, seq_len, latent_dim)
            else:
                print("  WARNING: Model did not return latent trajectory!")
                return None

            all_latents.append(latent.cpu().numpy())

    # Concatenate: (N, seq_len, latent_dim)
    latents = np.concatenate(all_latents, axis=0)

    # Average across validation windows → (seq_len, latent_dim)
    mean_latent = np.mean(latents, axis=0)

    # Transpose → (latent_dim, T)
    result = mean_latent.T

    # Diagnostics
    print(f"  [latent] Shape: {result.shape}")
    print(f"  [latent] Stats: mean={result.mean():.4f}, std={result.std():.4f}, "
          f"min={result.min():.4f}, max={result.max():.4f}")
    per_dim_std = np.std(result, axis=1)
    n_constant = np.sum(per_dim_std < 1e-6)
    if n_constant > 0:
        print(f"  [latent] WARNING: {n_constant}/{result.shape[0]} dims are near-constant")
    print(f"  [latent] Active dims: {result.shape[0] - n_constant}/{result.shape[0]}")

    return result


# ─────────────────────────────────────────────────────────────────────
# Bio variable alignment and stacking (standalone, matches fixed version)
# ─────────────────────────────────────────────────────────────────────

def align_bio_matrix(bio_ground_truth: Dict, registry, target_T: int) -> np.ndarray:
    """
    Build (n_bio, T) matrix from ground truth, aligned to target_T.

    Uses colon separator convention: var.name = "tc_m_T:5"
    Splits on ':' to get gt_key="tc_m_T", neuron_idx=5.
    """
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
            # Legacy fallback
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

        signal = gt_array[neuron_idx]  # (T_original,)

        # Resample to target_T if needed
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
# Training loop (standalone, mirrors FullTrainingPipeline exactly)
# ─────────────────────────────────────────────────────────────────────

def train_ltc(model: nn.Module, train_data: Dict, val_data: Dict,
              device: str = 'cuda', max_hours: float = 2.0,
              lr: float = 5e-4, batch_size: int = 32,
              max_epochs: int = 300, patience: int = 30) -> Tuple[nn.Module, Dict]:
    """
    Train LTC with progressive sequence length schedule.
    Identical protocol to FullTrainingPipeline.
    """
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
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
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

    full_seq_len = X_train.shape[1]
    n_train = X_train.shape[0]

    # Build progressive schedule
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

            # --- Train ---
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
                    elapsed_h = (time.time() - start_time) / 3600
                    return model, {
                        'spike_corr': 0.0, 'best_val_loss': float('inf'),
                        'total_epochs': total_epochs, 'hours': elapsed_h,
                        'converged': False
                    }

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            # --- Validate (batched) ---
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

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final spike correlation on full-length sequences
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
        'converged': patience_counter < patience
    }


# ─────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────

def run_experiment(data_dir: str, device: str = 'cuda',
                   max_hours: float = 2.0,
                   dims: List[int] = None):
    """Run the full LTC dimension sweep experiment."""

    if dims is None:
        dims = [32, 128, 256]

    print("=" * 70)
    print("LTC LATENT DIMENSION SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"Dimensions: {dims}")
    print(f"Device: {device}")
    print(f"Budget: {max_hours}h per model")
    print(f"Data: {data_dir}")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/2] Loading data...")
    train_data, val_data, bio_ground_truth = load_ar2_data(data_dir)
    print(f"  Train: {train_data['X_train'].shape}")
    print(f"  Val:   {val_data['X_val'].shape}")
    print(f"  Bio GT keys: {list(bio_ground_truth.keys())[:10]}...")

    # ── Build bio registry ──
    registry = build_biovar_registry()
    print(f"  Registry: {len(registry)} biological variables")

    # ── Align bio ground truth ──
    target_T = val_data['X_val'].shape[1]  # 2000
    bio_matrix = align_bio_matrix(bio_ground_truth, registry, target_T)

    # ── Run sweep ──
    results = {}
    experiment_start = time.time()

    for latent_dim in dims:
        print("\n" + "=" * 70)
        print(f"  TRAINING LTC latent_dim={latent_dim}")
        print("=" * 70)

        model = LTCModel(
            n_input=21,    # 20 retinal + 1 GABA
            n_output=20,   # 20 TC neurons
            latent_dim=latent_dim,
            hidden_dim=128
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # ── Train ──
        model, train_info = train_ltc(
            model, train_data, val_data,
            device=device, max_hours=max_hours
        )

        # ── Extract latents ──
        print(f"\n  Extracting latent states...")
        X_val_tensor = torch.tensor(val_data['X_val'])
        latents = extract_latents(model, X_val_tensor, device=device)

        if latents is None:
            print("  FAILED to extract latents. Skipping.")
            results[latent_dim] = {'error': 'latent extraction failed'}
            continue

        # ── Pearson biovar recovery (standard pipeline) ──
        print(f"\n  Computing Pearson biovar recovery...")
        recovery = score_biovar_recovery(
            latents, bio_ground_truth, registry, recovery_threshold=0.5
        )

        print(f"  Pearson: {recovery.n_recovered}/160 recovered (|r|>0.5)")
        print(f"  CCA score: {recovery.cca_score:.4f}")

        # Top 10 correlations
        corr_vec = recovery.correlation_vector
        top10_idx = np.argsort(corr_vec)[-10:][::-1]
        print(f"  Top 10 Pearson correlations:")
        for idx in top10_idx:
            print(f"    {registry[idx].name}: |r|={corr_vec[idx]:.4f} "
                  f"({registry[idx].category}/{registry[idx].timescale})")

        # ── CKA ──
        print(f"\n  Computing CKA...")
        # Subsample for speed
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
        print(f"\n  Computing Mutual Information (top 20 bio vars)...")
        mi_result = compute_mutual_info(latents, bio_matrix, top_k=20)
        print(f"  MI: mean={mi_result['mean_mi']:.4f}, max={mi_result['max_mi']:.4f}")

        # ── Ridge Regression R² ──
        print(f"\n  Computing Ridge Regression R² (5-fold CV)...")
        print(f"  (This is the key metric — tests distributed encoding)")
        ridge_result = compute_ridge_r2(latents, bio_matrix, alpha=1.0)
        print(f"  Ridge R²: mean={ridge_result['mean_r2']:.4f}")
        print(f"  Decodable (R²>0.25): {ridge_result['n_decodable_r2_gt_025']}/160")
        print(f"  Decodable (R²>0.50): {ridge_result['n_decodable_r2_gt_050']}/160")

        # ── Category breakdown for Ridge R² ──
        cat_r2 = {}
        for var in registry:
            cat = var.category
            r2 = ridge_result['per_var'].get(var.id, 0.0)
            cat_r2.setdefault(cat, []).append(r2)
        print(f"\n  Ridge R² by category:")
        for cat, vals in sorted(cat_r2.items()):
            print(f"    {cat}: mean R²={np.mean(vals):.4f}, "
                  f"decodable={sum(1 for v in vals if v > 0.25)}/{len(vals)}")

        # ── Store results ──
        results[latent_dim] = {
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
            'top10_pearson': [
                {'name': registry[idx].name, 'r': round(float(corr_vec[idx]), 4)}
                for idx in top10_idx
            ],
        }

        # Free GPU memory between runs
        del model
        torch.cuda.empty_cache()

    total_time = (time.time() - experiment_start) / 3600

    # ─────────────────────────────────────────────────────────────────
    # Print comparison table
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  COMPARISON TABLE: LTC Latent Dimension Sweep")
    print("=" * 80)
    header = (f"{'Dim':>6} | {'Params':>10} | {'Spike':>6} | "
              f"{'Pearson':>8} | {'CKA':>6} | {'MI':>6} | "
              f"{'Ridge R²':>9} | {'Decode25':>9} | {'Decode50':>9}")
    print(header)
    print("-" * 80)

    for dim in dims:
        if dim not in results or 'error' in results[dim]:
            print(f"{dim:>6} | {'ERROR':>10} |")
            continue
        r = results[dim]
        print(f"{dim:>6} | {r['n_params']:>10,} | "
              f"{r['spike_corr']:>6.3f} | "
              f"{r['pearson_recovered']:>4}/160 | "
              f"{r['cka_score']:>6.4f} | "
              f"{r['mi_mean']:>6.3f} | "
              f"{r['ridge_mean_r2']:>9.4f} | "
              f"{r['ridge_decodable_025']:>5}/160 | "
              f"{r['ridge_decodable_050']:>5}/160")

    print("=" * 80)

    # ── Interpretation ──
    print("\n  INTERPRETATION GUIDE:")
    print("  " + "-" * 50)
    print("  High Ridge R² + Low Pearson = DISTRIBUTED encoding")
    print("    (Biology IS there, spread across latent dims)")
    print("  High Ridge R² + High Pearson = ALIGNED encoding")
    print("    (Biology IS there, one-to-one mapping)")
    print("  Low Ridge R² + Low Pearson = ALIEN encoding")
    print("    (Biology is NOT linearly decodable)")
    print("  CKA measures overall representational similarity")
    print("  " + "-" * 50)

    print(f"\n  Total experiment time: {total_time:.2f}h")

    # ── Save results ──
    output = {
        'experiment': 'LTC_latent_dimension_sweep',
        'dims': dims,
        'max_hours_per_model': max_hours,
        'total_hours': round(total_time, 3),
        'results': {str(k): v for k, v in results.items()},
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ltc_dimension_sweep_results.json'
    )
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LTC Latent Dimension Sweep Experiment'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to rung3_data directory with .h5 files')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--max_hours', type=float, default=2.0,
                        help='Training budget per model in hours')
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 128, 256],
                        help='Latent dimensions to test')
    args = parser.parse_args()

    run_experiment(
        data_dir=args.data_dir,
        device=args.device,
        max_hours=args.max_hours,
        dims=args.dims,
    )
