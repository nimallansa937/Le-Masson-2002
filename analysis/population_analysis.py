"""
Population-level analysis for Rung 2 bifurcation surface.

Extends the Rung 1 EC50 threshold detection to population context
and provides surface/heatmap plotting utilities.
"""

import numpy as np
from analysis.oscillation import find_bifurcation_threshold


def population_bifurcation_threshold(results_by_gaba, metric='mean_pause_rate'):
    """Find population bifurcation threshold using EC50.

    Parameters
    ----------
    results_by_gaba : list of dict
        Sorted by gaba_gmax. Each dict must have 'gaba_gmax' and the metric.
    metric : str
        Metric to use for threshold detection.

    Returns
    -------
    threshold : float
        EC50 threshold in nS.
    """
    return find_bifurcation_threshold(results_by_gaba, metric=metric)


def build_bifurcation_surface(all_results, gaba_values, fractions):
    """Build 2D bifurcation surface from sweep results.

    Parameters
    ----------
    all_results : dict
        Keys: (fraction, gaba_gmax) -> dict with metrics.
    gaba_values : array
    fractions : array

    Returns
    -------
    surface : dict of ndarray
        Keys are metric names, values are (n_fractions, n_gaba) arrays.
    """
    n_frac = len(fractions)
    n_gaba = len(gaba_values)

    surface = {
        'coherence': np.full((n_frac, n_gaba), np.nan),
        'mean_pause_rate': np.full((n_frac, n_gaba), np.nan),
        'mean_burst_fraction': np.full((n_frac, n_gaba), np.nan),
        'participation': np.full((n_frac, n_gaba), np.nan),
        'pop_frequency': np.full((n_frac, n_gaba), np.nan),
        'freq_std': np.full((n_frac, n_gaba), np.nan),
    }

    for fi, frac in enumerate(fractions):
        for gi, gaba in enumerate(gaba_values):
            key = (frac, gaba)
            if key in all_results:
                r = all_results[key]
                for metric in surface:
                    if metric in r:
                        surface[metric][fi, gi] = r[metric]

    return surface


def extract_threshold_curve(surface_results, gaba_values, fractions):
    """Extract bifurcation threshold at each replacement fraction.

    Parameters
    ----------
    surface_results : list of dict
        Each dict has 'fraction', 'gaba_gmax', 'mean_pause_rate', etc.
    gaba_values : array
    fractions : array

    Returns
    -------
    thresholds : dict
        fraction -> threshold_nS.
    """
    thresholds = {}

    for frac in fractions:
        frac_results = sorted(
            [r for r in surface_results if abs(r['fraction'] - frac) < 0.001],
            key=lambda r: r['gaba_gmax']
        )
        if len(frac_results) >= 5:
            thresh = find_bifurcation_threshold(
                frac_results, metric='mean_pause_rate')
            thresholds[frac] = thresh
        else:
            thresholds[frac] = float('nan')

    return thresholds
