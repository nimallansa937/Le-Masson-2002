"""
Latent Comparison — THE NOVEL CONTRIBUTION.

Compares model internal representations against biological ground truth
intermediates using three complementary methods:

1. Canonical Correlation Analysis (CCA)
   - Finds linear projections that maximize correlation between
     model latent space and biological variables
   - Permutation test for significance (block bootstrap for autocorrelation)

2. Representational Similarity Analysis (RSA)
   - Compares the geometry of representational spaces
   - Spearman correlation between model and biological RDMs

3. Individual Variable Recovery
   - Ridge regression readout: can each biological variable be linearly
     decoded from model latents?
   - Bonferroni-corrected significance testing

Together, these test whether functional equivalence (same input→output
mapping) implies mechanistic equivalence (similar internal representations).
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

try:
    from sklearn.cross_decomposition import CCA as SklearnCCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from rung3.config import (
    CCA_N_PERMUTATIONS, CCA_BLOCK_SIZE, CCA_MAX_COMPONENTS,
    RSA_N_TIMEPOINTS, RSA_DISTANCE_METRIC,
    VAR_RECOVERY_ALPHA, VAR_RECOVERY_CV, VAR_RECOVERY_BONFERRONI,
)


# =========================================================================
# 1. Canonical Correlation Analysis
# =========================================================================

def compute_cca(model_latent, bio_intermediate, n_components=None):
    """Compute CCA between model latent states and biological intermediates.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
        Model's internal representation.
    bio_intermediate : ndarray (n_timepoints, n_bio_dims)
        Biological ground truth variables.
    n_components : int or None
        Number of canonical components. Default: min(dims)/2.

    Returns
    -------
    result : dict
        'canonical_correlations': ndarray (n_components,)
        'model_loadings': ndarray (latent_dim, n_components)
        'bio_loadings': ndarray (n_bio_dims, n_components)
        'mean_corr': float — mean canonical correlation
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    n_t, d_model = model_latent.shape
    n_t2, d_bio = bio_intermediate.shape
    assert n_t == n_t2, f"Time dimension mismatch: {n_t} vs {n_t2}"

    if n_components is None:
        n_components = min(min(d_model, d_bio) // 2, CCA_MAX_COMPONENTS)
    n_components = max(1, min(n_components, min(d_model, d_bio)))

    cca = SklearnCCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(model_latent, bio_intermediate)

    # Canonical correlations
    cc = np.array([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                   for i in range(n_components)])

    return {
        'canonical_correlations': cc,
        'model_loadings': cca.x_weights_,
        'bio_loadings': cca.y_weights_,
        'mean_corr': float(np.mean(cc)),
        'n_components': n_components,
    }


def cca_permutation_test(model_latent, bio_intermediate,
                          n_permutations=CCA_N_PERMUTATIONS,
                          block_size=CCA_BLOCK_SIZE, seed=42):
    """CCA with block-bootstrap permutation test for significance.

    Block bootstrap preserves temporal autocorrelation structure.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_intermediate : ndarray (n_timepoints, n_bio_dims)
    n_permutations : int
    block_size : int
        Block size for block bootstrap.

    Returns
    -------
    result : dict
        'observed': CCA result dict
        'null_distribution': ndarray (n_permutations,) of mean canonical corrs
        'p_value': float
    """
    rng = np.random.default_rng(seed)
    n_t = model_latent.shape[0]

    # Observed CCA
    observed = compute_cca(model_latent, bio_intermediate)

    # Null distribution via block bootstrap permutation
    null_corrs = []
    n_blocks = max(1, n_t // block_size)

    for perm in range(n_permutations):
        # Generate block-permuted time indices
        block_starts = rng.choice(n_t - block_size,
                                   size=n_blocks, replace=True)
        perm_idx = np.concatenate([
            np.arange(s, s + block_size) for s in block_starts
        ])[:n_t]

        # Permute biological data (keep model fixed)
        bio_perm = bio_intermediate[perm_idx]

        try:
            null_result = compute_cca(model_latent, bio_perm)
            null_corrs.append(null_result['mean_corr'])
        except Exception:
            null_corrs.append(0.0)

    null_corrs = np.array(null_corrs)
    p_value = float(np.mean(null_corrs >= observed['mean_corr']))

    return {
        'observed': observed,
        'null_distribution': null_corrs,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


# =========================================================================
# 2. Representational Similarity Analysis
# =========================================================================

def compute_rdm(X, metric=RSA_DISTANCE_METRIC, subsample=RSA_N_TIMEPOINTS,
                seed=42):
    """Compute Representational Dissimilarity Matrix.

    Parameters
    ----------
    X : ndarray (n_timepoints, n_dims)
    metric : str
        Distance metric for pdist.
    subsample : int
        Subsample timepoints to manage memory.

    Returns
    -------
    rdm : ndarray (n_samples, n_samples)
    indices : ndarray — which timepoints were used
    """
    n_t = X.shape[0]
    rng = np.random.default_rng(seed)

    if n_t > subsample:
        indices = rng.choice(n_t, size=subsample, replace=False)
        indices.sort()
        X_sub = X[indices]
    else:
        indices = np.arange(n_t)
        X_sub = X

    rdm = squareform(pdist(X_sub, metric=metric))
    return rdm, indices


def rsa_comparison(model_latent, bio_intermediate, metric=RSA_DISTANCE_METRIC,
                    subsample=RSA_N_TIMEPOINTS, seed=42):
    """Representational Similarity Analysis.

    Computes Spearman correlation between model and biological RDMs.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_intermediate : ndarray (n_timepoints, n_bio_dims)

    Returns
    -------
    result : dict
        'rsa_correlation': float — Spearman correlation between RDMs
        'p_value': float
        'model_rdm': ndarray
        'bio_rdm': ndarray
    """
    # Compute RDMs (same timepoints for both)
    model_rdm, idx = compute_rdm(model_latent, metric, subsample, seed)
    bio_rdm, _ = compute_rdm(bio_intermediate[idx], metric, subsample, seed)

    # Extract upper triangle
    triu_idx = np.triu_indices_from(model_rdm, k=1)
    model_vec = model_rdm[triu_idx]
    bio_vec = bio_rdm[triu_idx]

    # Spearman correlation
    rho, p_value = stats.spearmanr(model_vec, bio_vec)

    return {
        'rsa_correlation': float(rho),
        'p_value': float(p_value),
        'model_rdm': model_rdm,
        'bio_rdm': bio_rdm,
    }


# =========================================================================
# 3. Individual Variable Recovery
# =========================================================================

def variable_recovery(model_latent, bio_variables, variable_names=None,
                       alpha=VAR_RECOVERY_ALPHA, cv=VAR_RECOVERY_CV,
                       bonferroni=VAR_RECOVERY_BONFERRONI):
    """Test which biological variables can be decoded from model latents.

    For each biological variable (column of bio_variables), fit a ridge
    regression from model latents and evaluate via cross-validation.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_variables : ndarray (n_timepoints, n_bio_dims)
    variable_names : list of str, optional
    alpha : float
        Ridge regularization.
    cv : int
        Cross-validation folds.
    bonferroni : bool
        Apply Bonferroni correction.

    Returns
    -------
    result : dict
        'r2_scores': ndarray (n_bio_dims,) — cross-validated R² per variable
        'variable_names': list of str
        'significant': ndarray (n_bio_dims,) of bool
        'mean_r2': float
        'n_significant': int
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    n_t, n_bio = bio_variables.shape

    if variable_names is None:
        variable_names = [f'var_{i}' for i in range(n_bio)]

    r2_scores = np.zeros(n_bio)
    p_values = np.zeros(n_bio)

    ridge = Ridge(alpha=alpha)

    for i in range(n_bio):
        y = bio_variables[:, i]

        # Skip constant variables
        if np.std(y) < 1e-10:
            r2_scores[i] = 0.0
            p_values[i] = 1.0
            continue

        # Cross-validated R²
        scores = cross_val_score(ridge, model_latent, y,
                                  cv=cv, scoring='r2')
        r2_scores[i] = float(np.mean(scores))

        # Simple permutation test for p-value
        # Approximate p-value from R² distribution
        # For rigorous testing, compare to null of permuted y
        n_perm = 100
        null_r2 = []
        rng = np.random.default_rng(42 + i)
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            null_scores = cross_val_score(ridge, model_latent, y_perm,
                                           cv=cv, scoring='r2')
            null_r2.append(np.mean(null_scores))
        null_r2 = np.array(null_r2)
        p_values[i] = float(np.mean(null_r2 >= r2_scores[i]))

    # Bonferroni correction
    if bonferroni:
        significance_threshold = 0.05 / n_bio
    else:
        significance_threshold = 0.05

    significant = p_values < significance_threshold

    return {
        'r2_scores': r2_scores,
        'p_values': p_values,
        'variable_names': variable_names,
        'significant': significant,
        'mean_r2': float(np.mean(r2_scores)),
        'n_significant': int(np.sum(significant)),
        'n_total': n_bio,
        'significance_threshold': significance_threshold,
    }


# =========================================================================
# Combined Latent Evaluation
# =========================================================================

def extract_model_latent(model, X_input, device=None):
    """Extract latent representations from a model.

    Parameters
    ----------
    model : object
        Model with forward(x, return_latent=True).
    X_input : ndarray (seq_len, input_dim) or (batch, seq_len, input_dim)

    Returns
    -------
    latent : ndarray (seq_len, latent_dim)
    """
    if X_input.ndim == 2:
        X_input = X_input[np.newaxis]

    try:
        import torch
        if hasattr(model, 'parameters'):
            if device is None:
                device = next(model.parameters()).device
            x_t = torch.from_numpy(X_input).to(device)
            model.eval()
            with torch.no_grad():
                _, latent_dict = model(x_t, return_latent=True)
                latent = latent_dict['hidden'].cpu().numpy()
        else:
            _, latent_dict = model.forward(X_input, return_latent=True)
            latent = latent_dict['hidden']
    except ImportError:
        _, latent_dict = model.forward(X_input, return_latent=True)
        latent = latent_dict['hidden']

    if latent.ndim == 3:
        latent = latent[0]  # Remove batch dimension

    return latent


def prepare_bio_intermediates(intermediates_window):
    """Flatten biological intermediate variables into a single matrix.

    Parameters
    ----------
    intermediates_window : dict
        Keys like 'tc_m_T', values (window_bins, n_neurons).

    Returns
    -------
    bio_matrix : ndarray (window_bins, total_bio_dims)
    variable_names : list of str
    """
    arrays = []
    names = []

    ordered_keys = ['tc_m_T', 'tc_h_T', 'tc_m_h',
                     'nrt_m_Ts', 'nrt_h_Ts',
                     'gabaa_per_tc', 'gabab_per_tc', 'ampa_per_nrt']

    for key in ordered_keys:
        if key in intermediates_window:
            arr = intermediates_window[key]  # (window_bins, n_neurons)
            n_neurons = arr.shape[1]
            arrays.append(arr)
            for j in range(n_neurons):
                names.append(f'{key}_{j}')

    if not arrays:
        return None, []

    bio_matrix = np.concatenate(arrays, axis=1)
    return bio_matrix, names


def full_latent_comparison(model, model_name, X_input, intermediates,
                            verbose=True):
    """Run complete latent comparison suite.

    Parameters
    ----------
    model : object
    model_name : str
    X_input : ndarray (seq_len, input_dim)
    intermediates : dict
        Biological intermediate variables for this window.

    Returns
    -------
    results : dict with CCA, RSA, and variable recovery results.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Latent Comparison: {model_name}")
        print(f"{'='*60}")

    # Extract model latent
    latent = extract_model_latent(model, X_input)
    if verbose:
        print(f"  Model latent: {latent.shape}")

    # Prepare biological intermediates
    bio_matrix, var_names = prepare_bio_intermediates(intermediates)
    if bio_matrix is None:
        print("  WARNING: No biological intermediates available")
        return {'error': 'No intermediates'}

    if verbose:
        print(f"  Bio intermediates: {bio_matrix.shape} ({len(var_names)} vars)")

    # Ensure same length
    n_t = min(latent.shape[0], bio_matrix.shape[0])
    latent = latent[:n_t]
    bio_matrix = bio_matrix[:n_t]

    results = {'model_name': model_name}

    # 1. CCA with permutation test
    if verbose:
        print("\n  1. Canonical Correlation Analysis...")
    try:
        cca_result = cca_permutation_test(latent, bio_matrix)
        results['cca'] = {
            'mean_correlation': cca_result['observed']['mean_corr'],
            'canonical_correlations': cca_result['observed'][
                'canonical_correlations'].tolist(),
            'p_value': cca_result['p_value'],
            'significant': cca_result['significant'],
        }
        if verbose:
            cc = cca_result['observed']['canonical_correlations']
            print(f"     Mean CCA: {cca_result['observed']['mean_corr']:.4f}")
            print(f"     Top-3 CC: {cc[:3]}")
            print(f"     p-value: {cca_result['p_value']:.4f}")
    except Exception as e:
        results['cca'] = {'error': str(e)}
        if verbose:
            print(f"     ERROR: {e}")

    # 2. RSA
    if verbose:
        print("\n  2. Representational Similarity Analysis...")
    try:
        rsa_result = rsa_comparison(latent, bio_matrix)
        results['rsa'] = {
            'correlation': rsa_result['rsa_correlation'],
            'p_value': rsa_result['p_value'],
        }
        if verbose:
            print(f"     RSA correlation: {rsa_result['rsa_correlation']:.4f}")
            print(f"     p-value: {rsa_result['p_value']:.4e}")
    except Exception as e:
        results['rsa'] = {'error': str(e)}
        if verbose:
            print(f"     ERROR: {e}")

    # 3. Variable recovery
    if verbose:
        print("\n  3. Individual Variable Recovery...")
    try:
        vr_result = variable_recovery(latent, bio_matrix, var_names)
        results['variable_recovery'] = {
            'mean_r2': vr_result['mean_r2'],
            'n_significant': vr_result['n_significant'],
            'n_total': vr_result['n_total'],
            'top_variables': [],
        }
        # Top 10 recoverable variables
        top_idx = np.argsort(vr_result['r2_scores'])[::-1][:10]
        for idx in top_idx:
            results['variable_recovery']['top_variables'].append({
                'name': vr_result['variable_names'][idx],
                'r2': float(vr_result['r2_scores'][idx]),
                'significant': bool(vr_result['significant'][idx]),
            })

        if verbose:
            print(f"     Mean R²: {vr_result['mean_r2']:.4f}")
            print(f"     Significant: {vr_result['n_significant']}/"
                  f"{vr_result['n_total']}")
            print(f"     Top 5 variables:")
            for idx in top_idx[:5]:
                sig = '*' if vr_result['significant'][idx] else ' '
                print(f"       {sig} {vr_result['variable_names'][idx]}: "
                      f"R²={vr_result['r2_scores'][idx]:.4f}")
    except Exception as e:
        results['variable_recovery'] = {'error': str(e)}
        if verbose:
            print(f"     ERROR: {e}")

    return results
