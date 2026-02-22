"""
Comprehensive evaluation suite for A-R3b.

Measures BOTH spike prediction quality AND biological variable encoding:
  1. Spike correlation (same as A-R3, for comparability)
  2. Ridge R-squared with 5-fold CV (THE definitive encoding metric)
  3. CKA - Centered Kernel Alignment (representational similarity)
  4. Mutual Information (nonlinear encoding check)
  5. Per-category breakdown (tc_gating vs nrt_state vs synaptic)
  6. GRU-ODE gate analysis (which dims became "dedicated" to biology)

The gate analysis is particularly interesting: in A-R3 (spike-only),
25/32 dims had update gate > 0.7 (static). If the bio loss "recruits"
these static dims for biological encoding, we should see more dims with
update gate < 0.3 (dedicated to actively tracking bio variables).

Thresholds for biological encoding:
  - R-squared > 0.25 = "genuinely encoded" (above noise)
  - R-squared > 0.50 = "strongly encoded" (clear signal)
  - CKA > 0.3 = meaningful representational overlap
"""
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────
# Variable organization (matches DESCARTES 160-dim recovery space)
# ─────────────────────────────────────────────────────────────────────

VAR_GROUPS = [
    'tc_m_T', 'tc_h_T', 'tc_m_h',        # tc_gating: 60 vars
    'nrt_m_Ts', 'nrt_h_Ts', 'V_nrt',      # nrt_state: 60 vars
    'gabaa_per_tc', 'gabab_per_tc',        # synaptic:  40 vars
]

VAR_TO_CATEGORY = {
    'tc_m_T': 'tc_gating', 'tc_h_T': 'tc_gating', 'tc_m_h': 'tc_gating',
    'nrt_m_Ts': 'nrt_state', 'nrt_h_Ts': 'nrt_state', 'V_nrt': 'nrt_state',
    'gabaa_per_tc': 'synaptic', 'gabab_per_tc': 'synaptic',
}


# ─────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────

def evaluate_bio_recovery(model, val_loader, bio_ground_truth, device='cuda',
                          n_windows_eval=10):
    """
    Full evaluation suite for A-R3b experiment.

    Extracts latent trajectories from validation data, then runs all
    alignment metrics against biological ground truth.

    Args:
        model: trained GRUODEBio instance
        val_loader: DataLoader yielding (x, y, y_binary, bio_targets)
        bio_ground_truth: dict from load_ar2_data() — raw bio variable arrays
        device: compute device
        n_windows_eval: number of validation windows to use for eval
                        (more = more accurate but slower)

    Returns:
        results: dict with all metrics and per-category breakdowns
    """
    model.eval()
    model = model.to(device)

    # 1. Extract latent trajectories and spike predictions
    all_latents = []
    all_spike_preds = []
    all_spike_targets = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_windows_eval:
                break
            x, y, y_binary, bio = batch
            x = x.to(device)
            spike_pred, bio_preds, z_traj = model(x, return_latents=True)

            all_latents.append(z_traj.cpu().numpy())
            all_spike_preds.append(torch.sigmoid(spike_pred).cpu().numpy())
            all_spike_targets.append(y.numpy())

    latents = np.concatenate(all_latents, axis=0)      # (N, T, latent_dim)
    spike_preds = np.concatenate(all_spike_preds, axis=0)
    spike_targets = np.concatenate(all_spike_targets, axis=0)

    # Use first window's latents for detailed analysis
    # Shape: (T, latent_dim) from first validation window
    z_flat = latents[0]

    # 2. Spike correlation
    spike_corr = compute_spike_correlation(spike_preds, spike_targets)

    # 3. Prepare biological variable matrix
    T_eval = z_flat.shape[0]
    bio_matrix, bio_names = flatten_bio_variables(bio_ground_truth, T=T_eval)

    if bio_matrix is None or bio_matrix.shape[1] == 0:
        print("  WARNING: No bio variables found for evaluation")
        return {
            'spike_corr': spike_corr,
            'pearson': {'n_above_05': 0, 'n_above_08': 0, 'max_r': 0.0},
            'ridge': {'mean_r2': 0.0, 'decodable_025': 0, 'decodable_050': 0,
                      'total': 0, 'by_category': {}},
            'cka': 0.0,
            'mi': {'mean_mi': 0.0, 'max_mi': 0.0},
            'gates': {},
        }

    # 4. Pearson correlations (for comparison with A-R3)
    pearson_results = compute_pearson_recovery(z_flat, bio_matrix, bio_names)

    # 5. Ridge R-squared with 5-fold CV (THE KEY METRIC)
    ridge_results = compute_ridge_r2(z_flat, bio_matrix, bio_names)

    # 6. CKA
    cka_score = compute_cka(z_flat, bio_matrix)

    # 7. Mutual Information
    mi_results = compute_mutual_info(z_flat, bio_matrix)

    # 8. Gate analysis (GRU-ODE specific)
    gate_results = analyze_gru_gates(model, val_loader, device)

    results = {
        'spike_corr': spike_corr,
        'pearson': pearson_results,
        'ridge': ridge_results,
        'cka': cka_score,
        'mi': mi_results,
        'gates': gate_results,
    }

    return results


# ─────────────────────────────────────────────────────────────────────
# Spike prediction metrics
# ─────────────────────────────────────────────────────────────────────

def compute_spike_correlation(preds, targets):
    """
    Mean Pearson correlation between predicted and actual spike trains.

    Averages across windows first, then computes per-neuron correlation.
    This matches the A-R3 evaluation methodology.
    """
    # Average over windows: (T, n_neurons)
    pred_mean = preds.mean(axis=0)
    target_mean = targets.mean(axis=0)

    corrs = []
    for n in range(pred_mean.shape[1]):
        p = pred_mean[:, n]
        t = target_mean[:, n]
        if np.std(p) < 1e-10 or np.std(t) < 1e-10:
            continue
        r, _ = pearsonr(p, t)
        if not np.isnan(r):
            corrs.append(abs(r))

    return float(np.mean(corrs)) if corrs else 0.0


# ─────────────────────────────────────────────────────────────────────
# Ridge R-squared (THE definitive metric)
# ─────────────────────────────────────────────────────────────────────

def compute_ridge_r2(latents, bio_matrix, bio_names, n_folds=5):
    """
    5-fold cross-validated Ridge R-squared for each biological variable.

    THIS IS THE DEFINITIVE METRIC. A-R3 showed that Pearson correlations
    are unreliable (shared trends create mirages). Ridge R-squared with
    cross-validation tests whether the relationship generalizes to
    held-out data — genuine encoding vs coincidence.

    Uses time-based folds (not random) to preserve temporal structure.
    Floor at 0 — negative R-squared means worse than predicting the mean.
    """
    n_timesteps, n_bio = bio_matrix.shape

    fold_size = n_timesteps // n_folds
    r2_scores = np.zeros(n_bio)

    for i in range(n_bio):
        y = bio_matrix[:, i]
        X = latents

        # Skip constant variables
        if np.std(y) < 1e-8:
            r2_scores[i] = 0.0
            continue

        fold_r2s = []
        for fold in range(n_folds):
            # Time-based split (not random — preserves temporal structure)
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size

            X_train = np.vstack([X[:val_start], X[val_end:]])
            y_train = np.concatenate([y[:val_start], y[val_end:]])
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            # Standardize (fit on train, transform both)
            scaler_X = StandardScaler().fit(X_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

            X_train_s = scaler_X.transform(X_train)
            X_val_s = scaler_X.transform(X_val)
            y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

            # Fit Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_s, y_train_s)
            y_pred = ridge.predict(X_val_s)

            # R-squared
            ss_res = np.sum((y_val_s - y_pred) ** 2)
            ss_tot = np.sum((y_val_s - y_val_s.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            fold_r2s.append(max(r2, 0))  # Floor at 0

        r2_scores[i] = np.mean(fold_r2s) if fold_r2s else 0.0

    # Categorize results
    categories = categorize_bio_vars(bio_names)
    results = {
        'mean_r2': float(np.mean(r2_scores)),
        'median_r2': float(np.median(r2_scores)),
        'max_r2': float(np.max(r2_scores)),
        'decodable_025': int(np.sum(r2_scores > 0.25)),
        'decodable_050': int(np.sum(r2_scores > 0.50)),
        'total': int(n_bio),
        'by_category': {},
        'per_variable': {},  # Full per-variable breakdown
    }

    for cat_name, indices in categories.items():
        if not indices:
            continue
        cat_r2 = r2_scores[indices]
        results['by_category'][cat_name] = {
            'mean_r2': float(np.mean(cat_r2)),
            'max_r2': float(np.max(cat_r2)),
            'decodable_025': int(np.sum(cat_r2 > 0.25)),
            'decodable_050': int(np.sum(cat_r2 > 0.50)),
            'total': len(indices),
        }

    # Per-variable results (top 20 by R-squared)
    top_indices = np.argsort(r2_scores)[::-1][:20]
    for idx in top_indices:
        results['per_variable'][bio_names[idx]] = float(r2_scores[idx])

    return results


# ─────────────────────────────────────────────────────────────────────
# Pearson correlations (for A-R3 comparability)
# ─────────────────────────────────────────────────────────────────────

def compute_pearson_recovery(latents, bio_matrix, bio_names):
    """
    1-to-1 Pearson |r| between each latent dim and each bio variable.

    NOTE: This metric is UNRELIABLE for slow variables (as proven by A-R3).
    High Pearson can arise from shared temporal trends, not genuine encoding.
    Included only for comparison with A-R3 results.
    """
    n_latent = latents.shape[1]
    n_bio = bio_matrix.shape[1]

    # Compute full correlation matrix
    max_r_per_bio = np.zeros(n_bio)
    best_dim_per_bio = np.zeros(n_bio, dtype=int)

    for j in range(n_bio):
        bio_var = bio_matrix[:, j]
        if np.std(bio_var) < 1e-10:
            continue

        best_r = 0
        best_d = 0
        for d in range(n_latent):
            lat_var = latents[:, d]
            if np.std(lat_var) < 1e-10:
                continue
            r, _ = pearsonr(lat_var, bio_var)
            if abs(r) > best_r:
                best_r = abs(r)
                best_d = d

        max_r_per_bio[j] = best_r
        best_dim_per_bio[j] = best_d

    return {
        'mean_max_r': float(np.mean(max_r_per_bio)),
        'median_max_r': float(np.median(max_r_per_bio)),
        'n_above_05': int(np.sum(max_r_per_bio > 0.5)),
        'n_above_08': int(np.sum(max_r_per_bio > 0.8)),
        'n_above_09': int(np.sum(max_r_per_bio > 0.9)),
        'max_r': float(np.max(max_r_per_bio)),
        'total': int(n_bio),
    }


# ─────────────────────────────────────────────────────────────────────
# CKA — Centered Kernel Alignment
# ─────────────────────────────────────────────────────────────────────

def compute_cka(X, Y):
    """
    Linear CKA between latent space and bio variable space.

    CKA is invariant to orthogonal transformations and isotropic scaling,
    making it a more robust measure of representational similarity than
    raw correlation. A CKA > 0.3 indicates meaningful structural overlap
    between the latent representation and biological variable space.
    """
    # Center both matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices (via inner products for efficiency)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    # HSIC estimates
    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0

    return float(hsic_xy / denom)


# ─────────────────────────────────────────────────────────────────────
# Mutual Information
# ─────────────────────────────────────────────────────────────────────

def compute_mutual_info(latents, bio_matrix, top_k=20):
    """
    Mutual information between latent dims and top bio variables.

    Uses sklearn's mutual_info_regression with k-NN estimator.
    Only evaluates top-k bio variables (by variance) for speed.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        return {'mean_mi': 0.0, 'max_mi': 0.0, 'error': 'sklearn not available'}

    n_bio = bio_matrix.shape[1]
    bio_var = np.var(bio_matrix, axis=0)

    # Filter to non-constant variables
    valid_mask = bio_var > 1e-10
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return {'mean_mi': 0.0, 'max_mi': 0.0}

    # Top-k by variance
    sorted_idx = valid_indices[np.argsort(bio_var[valid_indices])[::-1]]
    top_indices = sorted_idx[:top_k]

    X = latents  # (T, latent_dim)
    mi_scores = {}

    for bio_idx in top_indices:
        y = bio_matrix[:, bio_idx]
        if np.std(y) < 1e-10:
            continue
        mi = mutual_info_regression(X, y, n_neighbors=5, random_state=42)
        mi_scores[int(bio_idx)] = float(np.max(mi))

    mean_mi = float(np.mean(list(mi_scores.values()))) if mi_scores else 0.0
    max_mi = float(np.max(list(mi_scores.values()))) if mi_scores else 0.0

    return {
        'mean_mi': mean_mi,
        'max_mi': max_mi,
        'n_evaluated': len(mi_scores),
        'per_var': mi_scores,
    }


# ─────────────────────────────────────────────────────────────────────
# GRU-ODE Gate Analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_gru_gates(model, val_loader, device):
    """
    Analyze GRU-ODE gate behavior to detect bio loss recruitment.

    In A-R3 (spike-only): 25/32 dims had update > 0.7 (static).
    With bio loss: do more dims become "dedicated" (update < 0.3)?

    The hypothesis is that bio loss forces the network to use its
    static dims (which were essentially wasted capacity in A-R3)
    to track biological variables.

    Classification:
      - Dedicated (update < 0.3): dim actively tracks dynamics
      - Dynamic (0.3 <= update <= 0.7): dim partially evolves
      - Static (update > 0.7): dim held nearly constant
    """
    model.eval()

    # Hook to capture gate activations
    update_gates_buffer = []
    reset_gates_buffer = []

    def hook_update(module, input, output):
        update_gates_buffer.append(torch.sigmoid(output).detach().cpu())

    def hook_reset(module, input, output):
        reset_gates_buffer.append(torch.sigmoid(output).detach().cpu())

    # Register hooks on the GRU-ODE gate layers
    try:
        h_update = model.W_z.register_forward_hook(hook_update)
        h_reset = model.W_r.register_forward_hook(hook_reset)
    except AttributeError:
        return {'error': 'Model does not have W_z/W_r attributes'}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x, y, y_bin, bio = batch
            x = x.to(device)
            model(x)
            if i >= 1:  # One batch is enough for gate statistics
                break

    h_update.remove()
    h_reset.remove()

    if not update_gates_buffer:
        return {'error': 'No gate data captured'}

    updates = torch.cat(update_gates_buffer, dim=0)
    resets = torch.cat(reset_gates_buffer, dim=0)

    # Mean gate value per dimension (averaged over time and batch)
    update_mean = updates.mean(dim=0).numpy()  # (latent_dim,)
    reset_mean = resets.mean(dim=0).numpy()

    # Temporal variability (how much gates change over time)
    update_std = updates.std(dim=0).numpy()

    # Classification
    dedicated = int(np.sum(update_mean < 0.3))
    dynamic = int(np.sum((update_mean >= 0.3) & (update_mean <= 0.7)))
    static = int(np.sum(update_mean > 0.7))

    return {
        'update_mean': update_mean.tolist(),
        'reset_mean': reset_mean.tolist(),
        'update_std': update_std.tolist(),
        'dedicated_dims': dedicated,
        'dynamic_dims': dynamic,
        'static_dims': static,
        'total_dims': len(update_mean),
        'mean_update': float(np.mean(update_mean)),
        'mean_reset': float(np.mean(reset_mean)),
    }


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

def flatten_bio_variables(bio_gt, T):
    """
    Flatten all bio variable groups into a single (T, N_vars) matrix.

    Uses bare-name variables (canonical trial) when available.
    Falls back to first GABA-specific version if bare name missing.

    Returns:
        bio_matrix: (T, n_vars) numpy array, or None if no vars found
        bio_names: list of 'var_name:neuron_idx' strings
    """
    bio_matrix = []
    bio_names = []

    for var_name in VAR_GROUPS:
        # Prefer bare name (canonical trial)
        if var_name in bio_gt:
            data = bio_gt[var_name]
        else:
            # Try GABA-specific fallback
            found = False
            for key in sorted(bio_gt.keys()):
                if key.startswith(var_name + '_gaba'):
                    data = bio_gt[key]
                    found = True
                    break
            if not found:
                continue

        # data shape: (20_neurons, T_full)
        for neuron_idx in range(data.shape[0]):
            trace = data[neuron_idx]
            if len(trace) >= T:
                trace = trace[:T]
            else:
                # Pad with last value if too short
                pad = np.full(T - len(trace), trace[-1] if len(trace) > 0 else 0)
                trace = np.concatenate([trace, pad])

            bio_matrix.append(trace)
            bio_names.append(f"{var_name}:{neuron_idx}")

    if not bio_matrix:
        return None, []

    bio_matrix = np.array(bio_matrix).T  # (T, n_vars)
    return bio_matrix, bio_names


def categorize_bio_vars(bio_names):
    """Map variable names to category index lists."""
    categories = {'tc_gating': [], 'nrt_state': [], 'synaptic': []}

    for i, name in enumerate(bio_names):
        var_name = name.split(':')[0]  # Remove neuron index
        if var_name in VAR_TO_CATEGORY:
            categories[VAR_TO_CATEGORY[var_name]].append(i)

    return categories


def print_evaluation_summary(results, alpha):
    """Pretty-print the evaluation results."""
    print(f"\n  {'='*60}")
    print(f"  EVALUATION RESULTS (alpha={alpha:.2f})")
    print(f"  {'='*60}")

    print(f"  Spike correlation:    {results['spike_corr']:.4f}")

    r = results['ridge']
    print(f"\n  Ridge R-squared (THE KEY METRIC):")
    print(f"    Mean R2:            {r['mean_r2']:.4f}")
    print(f"    Max R2:             {r['max_r2']:.4f}")
    print(f"    Decodable (>0.25):  {r['decodable_025']}/{r['total']}")
    print(f"    Strongly (>0.50):   {r['decodable_050']}/{r['total']}")

    if r['by_category']:
        print(f"\n    By category:")
        for cat, cr in r['by_category'].items():
            print(f"      {cat:15s}: R2={cr['mean_r2']:.4f}, "
                  f"decodable={cr['decodable_025']}/{cr['total']}")

    p = results['pearson']
    print(f"\n  Pearson (unreliable, for A-R3 comparison):")
    print(f"    Mean max |r|:       {p['mean_max_r']:.4f}")
    print(f"    |r| > 0.8:          {p['n_above_08']}/{p['total']}")

    print(f"\n  CKA:                  {results['cka']:.4f}")

    mi = results.get('mi', {})
    print(f"  MI mean:              {mi.get('mean_mi', 0):.4f}")

    g = results.get('gates', {})
    if 'dedicated_dims' in g:
        total = g.get('total_dims', '?')
        print(f"\n  Gate analysis:")
        print(f"    Dedicated (<0.3):   {g['dedicated_dims']}/{total}")
        print(f"    Dynamic (0.3-0.7):  {g['dynamic_dims']}/{total}")
        print(f"    Static (>0.7):      {g['static_dims']}/{total}")

    print(f"  {'='*60}")
