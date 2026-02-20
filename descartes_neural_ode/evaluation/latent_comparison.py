"""
Latent Space Comparison for DESCARTES-NeuralODE.

Compares model internal representations against biological ground truth
intermediate variables using three complementary methods:

1. Canonical Correlation Analysis (CCA)
   - Finds linear projections that maximize correlation between
     model latent space and biological variables
   - Block bootstrap permutation test for significance

2. Representational Similarity Analysis (RSA)
   - Compares the geometry of representational spaces
   - Spearman correlation between model and biological RDMs

3. Individual Variable Recovery
   - Ridge regression readout: can each biological variable be linearly
     decoded from model latents?
   - Bonferroni-corrected significance testing

These measure MECHANISTIC equivalence — does the model develop
similar internal representations to the biological circuit?

Adapted from rung3/evaluation/latent_comparison.py but using the
DESCARTES 160-dimensional biovar registry for structured analysis.
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional

try:
    from sklearn.cross_decomposition import CCA as SklearnCCA
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Defaults
CCA_MAX_COMPONENTS = 20
CCA_PCA_COMPONENTS = 30
CCA_MAX_ITER = 500
CCA_N_PERMUTATIONS = 200
CCA_BLOCK_SIZE = 50
RSA_N_TIMEPOINTS = 500
RSA_DISTANCE_METRIC = 'correlation'
VAR_RECOVERY_ALPHA = 1.0
VAR_RECOVERY_CV = 5


# =========================================================================
# 1. Canonical Correlation Analysis
# =========================================================================

def compute_cca(
    model_latent: np.ndarray,
    bio_matrix: np.ndarray,
    n_components: Optional[int] = None,
) -> Dict:
    """CCA between model latent states and biological variables.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_matrix : ndarray (n_timepoints, n_bio_dims)
    n_components : int or None

    Returns
    -------
    result : dict
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required")

    n_t, d_model = model_latent.shape
    _, d_bio = bio_matrix.shape

    if n_components is None:
        n_components = min(min(d_model, d_bio) // 2, CCA_MAX_COMPONENTS)
    n_components = max(1, min(n_components, min(d_model, d_bio)))

    cca = SklearnCCA(n_components=n_components, max_iter=CCA_MAX_ITER)
    X_c, Y_c = cca.fit_transform(model_latent, bio_matrix)

    cc = np.array([
        np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        for i in range(n_components)
    ])

    return {
        'canonical_correlations': cc,
        'mean_corr': float(np.mean(cc)),
        'n_components': n_components,
    }


def cca_permutation_test(
    model_latent: np.ndarray,
    bio_matrix: np.ndarray,
    n_permutations: int = CCA_N_PERMUTATIONS,
    block_size: int = CCA_BLOCK_SIZE,
    seed: int = 42,
) -> Dict:
    """CCA with block-bootstrap permutation test.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_matrix : ndarray (n_timepoints, n_bio_dims)
    n_permutations : int
    block_size : int
    seed : int

    Returns
    -------
    result : dict
    """
    rng = np.random.default_rng(seed)
    n_t = model_latent.shape[0]

    # PCA reduction for speed
    d_model = model_latent.shape[1]
    d_bio = bio_matrix.shape[1]
    n_pca = CCA_PCA_COMPONENTS

    if d_model > n_pca:
        model_latent = PCA(n_components=n_pca).fit_transform(model_latent)
    if d_bio > n_pca:
        bio_matrix = PCA(n_components=n_pca).fit_transform(bio_matrix)

    observed = compute_cca(model_latent, bio_matrix)

    null_corrs = []
    n_blocks = max(1, n_t // block_size)

    for _ in range(n_permutations):
        block_starts = rng.choice(n_t - block_size, size=n_blocks, replace=True)
        perm_idx = np.concatenate([
            np.arange(s, s + block_size) for s in block_starts
        ])[:n_t]
        bio_perm = bio_matrix[perm_idx]
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

def compute_rdm(
    X: np.ndarray,
    metric: str = RSA_DISTANCE_METRIC,
    subsample: int = RSA_N_TIMEPOINTS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Representational Dissimilarity Matrix.

    Parameters
    ----------
    X : ndarray (n_timepoints, n_dims)
    metric : str
    subsample : int
    seed : int

    Returns
    -------
    rdm : ndarray (n_samples, n_samples)
    indices : ndarray
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


def rsa_comparison(
    model_latent: np.ndarray,
    bio_matrix: np.ndarray,
    metric: str = RSA_DISTANCE_METRIC,
    subsample: int = RSA_N_TIMEPOINTS,
    seed: int = 42,
) -> Dict:
    """Representational Similarity Analysis.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_matrix : ndarray (n_timepoints, n_bio_dims)

    Returns
    -------
    result : dict
    """
    model_rdm, idx = compute_rdm(model_latent, metric, subsample, seed)
    bio_rdm, _ = compute_rdm(bio_matrix[idx], metric, subsample, seed)

    triu_idx = np.triu_indices_from(model_rdm, k=1)
    model_vec = model_rdm[triu_idx]
    bio_vec = bio_rdm[triu_idx]

    rho, p_value = stats.spearmanr(model_vec, bio_vec)

    return {
        'rsa_correlation': float(rho),
        'p_value': float(p_value),
    }


# =========================================================================
# 3. Individual Variable Recovery
# =========================================================================

def variable_recovery(
    model_latent: np.ndarray,
    bio_matrix: np.ndarray,
    variable_names: Optional[List[str]] = None,
    alpha: float = VAR_RECOVERY_ALPHA,
    cv: int = VAR_RECOVERY_CV,
    bonferroni: bool = True,
    n_perm: int = 100,
) -> Dict:
    """Test which biological variables can be decoded from model latents.

    For each biological variable, fits ridge regression from model latents
    and evaluates via cross-validation.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_matrix : ndarray (n_timepoints, n_bio_dims)
    variable_names : list of str, optional
    alpha : float
    cv : int
    bonferroni : bool
    n_perm : int

    Returns
    -------
    result : dict
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required")

    n_t, n_bio = bio_matrix.shape

    if variable_names is None:
        variable_names = [f'var_{i}' for i in range(n_bio)]

    r2_scores = np.zeros(n_bio)
    p_values = np.ones(n_bio)

    ridge = Ridge(alpha=alpha)
    rng = np.random.default_rng(42)

    for i in range(n_bio):
        y = bio_matrix[:, i]

        if np.std(y) < 1e-10:
            r2_scores[i] = 0.0
            p_values[i] = 1.0
            continue

        scores = cross_val_score(ridge, model_latent, y, cv=cv, scoring='r2')
        r2_scores[i] = float(np.mean(scores))

        # Permutation test
        null_r2 = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            null_scores = cross_val_score(ridge, model_latent, y_perm, cv=cv, scoring='r2')
            null_r2.append(np.mean(null_scores))
        null_r2 = np.array(null_r2)
        p_values[i] = float(np.mean(null_r2 >= r2_scores[i]))

    significance_threshold = 0.05 / n_bio if bonferroni else 0.05
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
# Per-Variable Pearson Correlation (fast alternative to ridge regression)
# =========================================================================

def per_variable_correlation(
    model_latent: np.ndarray,
    bio_matrix: np.ndarray,
    variable_names: Optional[List[str]] = None,
) -> Dict:
    """Find best Pearson correlation per bio variable across all latent dims.

    This is a fast alternative to full variable_recovery() that computes
    the maximum absolute Pearson r between each bio variable and any
    latent dimension.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_matrix : ndarray (n_timepoints, n_bio_dims)
    variable_names : list of str, optional

    Returns
    -------
    result : dict
    """
    n_t, n_latent = model_latent.shape
    _, n_bio = bio_matrix.shape

    if variable_names is None:
        variable_names = [f'var_{i}' for i in range(n_bio)]

    best_r = np.zeros(n_bio)
    best_latent_idx = np.zeros(n_bio, dtype=int)

    for i in range(n_bio):
        y = bio_matrix[:, i]
        if np.std(y) < 1e-10:
            continue
        for j in range(n_latent):
            r, _ = pearsonr(model_latent[:, j], y)
            if abs(r) > abs(best_r[i]):
                best_r[i] = r
                best_latent_idx[i] = j

    return {
        'best_correlation': np.abs(best_r),
        'best_latent_mapping': best_latent_idx,
        'variable_names': variable_names,
        'mean_abs_r': float(np.mean(np.abs(best_r))),
        'n_above_05': int(np.sum(np.abs(best_r) > 0.5)),
        'n_above_03': int(np.sum(np.abs(best_r) > 0.3)),
        'n_total': n_bio,
    }


# =========================================================================
# Combined Latent Evaluation
# =========================================================================

def full_latent_comparison(
    model_latent: np.ndarray,
    bio_matrix: np.ndarray,
    variable_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """Run complete latent comparison suite.

    Parameters
    ----------
    model_latent : ndarray (n_timepoints, latent_dim)
    bio_matrix : ndarray (n_timepoints, n_bio_dims)
    variable_names : list of str, optional
    verbose : bool

    Returns
    -------
    results : dict
    """
    n_t = min(model_latent.shape[0], bio_matrix.shape[0])
    model_latent = model_latent[:n_t]
    bio_matrix = bio_matrix[:n_t]

    results = {}

    # 1. CCA
    if verbose:
        print("  CCA analysis...")
    try:
        cca_result = cca_permutation_test(model_latent, bio_matrix)
        results['cca'] = {
            'mean_correlation': cca_result['observed']['mean_corr'],
            'p_value': cca_result['p_value'],
            'significant': cca_result['significant'],
        }
    except Exception as e:
        results['cca'] = {'error': str(e)}

    # 2. RSA
    if verbose:
        print("  RSA analysis...")
    try:
        rsa_result = rsa_comparison(model_latent, bio_matrix)
        results['rsa'] = {
            'correlation': rsa_result['rsa_correlation'],
            'p_value': rsa_result['p_value'],
        }
    except Exception as e:
        results['rsa'] = {'error': str(e)}

    # 3. Per-variable correlation (fast)
    if verbose:
        print("  Per-variable correlation...")
    try:
        pvc = per_variable_correlation(model_latent, bio_matrix, variable_names)
        results['per_variable'] = {
            'mean_abs_r': pvc['mean_abs_r'],
            'n_above_05': pvc['n_above_05'],
            'n_above_03': pvc['n_above_03'],
            'n_total': pvc['n_total'],
        }
    except Exception as e:
        results['per_variable'] = {'error': str(e)}

    # 4. Variable recovery (slow but rigorous)
    if verbose:
        print("  Ridge regression variable recovery...")
    try:
        vr = variable_recovery(model_latent, bio_matrix, variable_names)
        results['variable_recovery'] = {
            'mean_r2': vr['mean_r2'],
            'n_significant': vr['n_significant'],
            'n_total': vr['n_total'],
        }
    except Exception as e:
        results['variable_recovery'] = {'error': str(e)}

    return results
