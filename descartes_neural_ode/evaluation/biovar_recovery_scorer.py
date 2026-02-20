"""
Biological Variable Recovery Scorer for DESCARTES-NeuralODE.

This is the CORE evaluation module — the one that produces the
160-dimensional recovery vector used by the DESCARTES gap analysis.

For each of the 160 biological ground truth variables:
  1. Find the model latent dimension with highest |r|
  2. If |r| > threshold (default 0.5), mark as "recovered"
  3. Record the mapping and correlation strength

The recovery vector drives:
  - Gap computation (which variables remain unrecovered)
  - Gap direction (which categories/timescales to target)
  - DreamCoder pattern analysis
  - Balloon expansion decisions

Also provides structured breakdowns:
  - By category (tc_gating, nrt_state, synaptic)
  - By timescale (fast, medium, slow)
  - By dynamics type (monotonic, oscillatory, switching)
  - By subcategory (mT, hT, mH, V, gaba_a, gaba_b)
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.biovar_recovery_space import (
    BiologicalVariable, build_biovar_registry,
    RecoveryResult, score_biovar_recovery
)


@dataclass
class DetailedRecoveryReport:
    """Extended recovery report with per-variable details."""
    recovery: RecoveryResult
    per_variable_details: List[Dict]
    by_subcategory: Dict[str, Dict]
    by_dynamics_type: Dict[str, Dict]
    summary: str


def compute_detailed_recovery(
    model_latents: np.ndarray,
    bio_ground_truth: Dict[str, np.ndarray],
    registry: Optional[List[BiologicalVariable]] = None,
    recovery_threshold: float = 0.5,
) -> DetailedRecoveryReport:
    """Compute detailed 160-dimensional recovery analysis.

    Parameters
    ----------
    model_latents : ndarray (latent_dim, T) or (T, latent_dim)
        Model's internal state trajectories.
    bio_ground_truth : dict
        Maps variable prefix (e.g., "tc_mT") -> ndarray (n_neurons, T_bio).
    registry : list of BiologicalVariable, optional
        If None, builds default 160-variable registry.
    recovery_threshold : float
        Correlation threshold for "recovered".

    Returns
    -------
    report : DetailedRecoveryReport
    """
    if registry is None:
        registry = build_biovar_registry()

    # Ensure latents are (latent_dim, T)
    if model_latents.ndim == 2 and model_latents.shape[0] > model_latents.shape[1]:
        model_latents = model_latents.T

    # Core recovery scoring
    recovery = score_biovar_recovery(
        model_latents, bio_ground_truth, registry, recovery_threshold
    )

    # Per-variable details
    per_variable = []
    for var in registry:
        per_variable.append({
            'id': var.id,
            'name': var.name,
            'category': var.category,
            'subcategory': var.subcategory,
            'timescale': var.timescale,
            'dynamics_type': var.dynamics_type,
            'neuron_index': var.neuron_index,
            'recovered': bool(recovery.recovery_vector[var.id] == 1.0),
            'best_correlation': float(recovery.correlation_vector[var.id]),
            'best_latent_dim': recovery.best_latent_mapping.get(var.id, -1),
        })

    # By subcategory
    by_subcategory = {}
    for var in registry:
        sc = var.subcategory
        if sc not in by_subcategory:
            by_subcategory[sc] = {'total': 0, 'recovered': 0, 'mean_corr': []}
        by_subcategory[sc]['total'] += 1
        by_subcategory[sc]['mean_corr'].append(recovery.correlation_vector[var.id])
        if recovery.recovery_vector[var.id] == 1.0:
            by_subcategory[sc]['recovered'] += 1

    for sc in by_subcategory:
        corrs = by_subcategory[sc]['mean_corr']
        by_subcategory[sc]['mean_corr'] = float(np.mean(corrs))
        by_subcategory[sc]['fraction_recovered'] = (
            by_subcategory[sc]['recovered'] / by_subcategory[sc]['total']
        )

    # By dynamics type
    by_dynamics = {}
    for var in registry:
        dt = var.dynamics_type
        if dt not in by_dynamics:
            by_dynamics[dt] = {'total': 0, 'recovered': 0, 'mean_corr': []}
        by_dynamics[dt]['total'] += 1
        by_dynamics[dt]['mean_corr'].append(recovery.correlation_vector[var.id])
        if recovery.recovery_vector[var.id] == 1.0:
            by_dynamics[dt]['recovered'] += 1

    for dt in by_dynamics:
        corrs = by_dynamics[dt]['mean_corr']
        by_dynamics[dt]['mean_corr'] = float(np.mean(corrs))
        by_dynamics[dt]['fraction_recovered'] = (
            by_dynamics[dt]['recovered'] / by_dynamics[dt]['total']
        )

    # Summary text
    summary_lines = [
        f"Recovery: {recovery.n_recovered}/160 biological variables (|r| > {recovery_threshold})",
        f"CCA score: {recovery.cca_score:.3f}",
        f"By category: {recovery.recovered_by_category}",
        f"By timescale: {recovery.recovered_by_timescale}",
    ]

    # Highlight best and worst subcategories
    best_sc = max(by_subcategory, key=lambda k: by_subcategory[k]['fraction_recovered'])
    worst_sc = min(by_subcategory, key=lambda k: by_subcategory[k]['fraction_recovered'])
    summary_lines.append(
        f"Best subcategory: {best_sc} ({by_subcategory[best_sc]['fraction_recovered']:.0%})"
    )
    summary_lines.append(
        f"Worst subcategory: {worst_sc} ({by_subcategory[worst_sc]['fraction_recovered']:.0%})"
    )

    return DetailedRecoveryReport(
        recovery=recovery,
        per_variable_details=per_variable,
        by_subcategory=by_subcategory,
        by_dynamics_type=by_dynamics,
        summary='\n'.join(summary_lines),
    )


def compare_architectures_recovery(
    results: Dict[str, DetailedRecoveryReport],
) -> Dict:
    """Compare recovery across multiple architectures.

    Parameters
    ----------
    results : dict
        Maps architecture_id -> DetailedRecoveryReport.

    Returns
    -------
    comparison : dict
    """
    comparison = {
        'architectures': {},
        'union_recovery': np.zeros(160),
        'best_per_variable': {},
    }

    for arch_id, report in results.items():
        comparison['architectures'][arch_id] = {
            'n_recovered': report.recovery.n_recovered,
            'cca_score': report.recovery.cca_score,
            'by_category': report.recovery.recovered_by_category,
            'by_timescale': report.recovery.recovered_by_timescale,
            'by_subcategory': {
                sc: info['fraction_recovered']
                for sc, info in report.by_subcategory.items()
            },
        }
        comparison['union_recovery'] = np.maximum(
            comparison['union_recovery'],
            report.recovery.recovery_vector,
        )

    comparison['union_n_recovered'] = int(comparison['union_recovery'].sum())
    comparison['union_fraction'] = float(comparison['union_recovery'].mean())

    # Find best architecture per variable
    for i in range(160):
        best_arch = None
        best_r = 0.0
        for arch_id, report in results.items():
            r = report.recovery.correlation_vector[i]
            if r > best_r:
                best_r = r
                best_arch = arch_id
        comparison['best_per_variable'][i] = {
            'best_architecture': best_arch,
            'best_correlation': float(best_r),
        }

    return comparison


def recovery_to_json(report: DetailedRecoveryReport) -> Dict:
    """Serialize recovery report to JSON-compatible dict.

    Parameters
    ----------
    report : DetailedRecoveryReport

    Returns
    -------
    data : dict
    """
    return {
        'n_recovered': report.recovery.n_recovered,
        'cca_score': report.recovery.cca_score,
        'recovery_vector': report.recovery.recovery_vector.tolist(),
        'correlation_vector': report.recovery.correlation_vector.tolist(),
        'by_category': report.recovery.recovered_by_category,
        'by_timescale': report.recovery.recovered_by_timescale,
        'by_subcategory': {
            sc: {
                'total': info['total'],
                'recovered': info['recovered'],
                'fraction': info['fraction_recovered'],
                'mean_corr': info['mean_corr'],
            }
            for sc, info in report.by_subcategory.items()
        },
        'by_dynamics_type': {
            dt: {
                'total': info['total'],
                'recovered': info['recovered'],
                'fraction': info['fraction_recovered'],
                'mean_corr': info['mean_corr'],
            }
            for dt, info in report.by_dynamics_type.items()
        },
        'summary': report.summary,
    }
