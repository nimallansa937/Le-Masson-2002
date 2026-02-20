"""
DreamCoder History Accumulator.

Analyzes iteration results to extract structured patterns:
1. Co-recovery clusters: variables always recovered/missed together
2. Property correlations: architecture features -> variable recovery
3. Near-miss tracking: variables close to threshold across attempts
4. Family performance: aggregate stats per architecture family
5. Timescale alignment: which solvers/time-handling recover which timescales

These patterns are fed to the LLM to guide architecture suggestion.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DreamCoderHistory:
    """Accumulates and analyzes results across DESCARTES iterations."""

    def __init__(self, bio_var_names: List[str],
                 bio_var_categories: Dict[str, List[int]],
                 bio_var_timescales: Dict[str, List[int]]):
        """
        Args:
            bio_var_names: List of 160 biological variable names
            bio_var_categories: Maps category name -> list of variable indices
                e.g. {'tc_gating': [0..59], 'nrt_state': [60..119], 'synaptic': [120..159]}
            bio_var_timescales: Maps timescale -> list of variable indices
                e.g. {'fast': [...], 'slow': [...], 'medium': [...]}
        """
        self.bio_var_names = bio_var_names
        self.bio_var_categories = bio_var_categories
        self.bio_var_timescales = bio_var_timescales
        self.n_vars = len(bio_var_names)

        # Accumulated data
        self.recovery_matrix = []     # List of 160-dim binary vectors
        self.correlation_matrix = []  # List of 160-dim correlation vectors (raw r values)
        self.architecture_configs = []  # List of config dicts
        self.iteration_names = []

    def add_result(self, name: str, config: Dict[str, str],
                   recovery_vector: np.ndarray, correlation_vector: np.ndarray):
        """
        Record one iteration's results.

        Args:
            name: Architecture template name
            config: Dict with keys time_handling, gradient_strategy, etc.
            recovery_vector: 160-dim binary (1=recovered, 0=not)
            correlation_vector: 160-dim float (raw Pearson r values)
        """
        self.iteration_names.append(name)
        self.architecture_configs.append(config)
        self.recovery_matrix.append(recovery_vector.copy())
        self.correlation_matrix.append(correlation_vector.copy())

        logger.info(f"DreamCoder recorded: {name} -> {int(recovery_vector.sum())}/160")

    def extract_patterns(self) -> List[Dict[str, Any]]:
        """
        Run DreamCoder sleep phase -- analyze all results to find patterns.

        Returns list of pattern dicts, each with:
            type: str (co_recovery, property_correlation, near_miss,
                       family_exhaustion, timescale_alignment)
            description: str (human-readable)
            evidence: str (supporting data)
            confidence: float (0-1)
        """
        patterns = []

        if len(self.recovery_matrix) < 2:
            logger.info("Need at least 2 iterations for pattern extraction")
            return patterns

        R = np.array(self.recovery_matrix)      # (n_iters, 160)
        C = np.array(self.correlation_matrix)    # (n_iters, 160)

        # Pattern 1: Co-recovery clusters
        patterns.extend(self._find_co_recovery_clusters(R))

        # Pattern 2: Property correlations
        patterns.extend(self._find_property_correlations(R))

        # Pattern 3: Near-misses
        patterns.extend(self._find_near_misses(C))

        # Pattern 4: Timescale alignment
        patterns.extend(self._find_timescale_alignment(R))

        # Pattern 5: Category-level trends
        patterns.extend(self._find_category_trends(R))

        logger.info(f"Extracted {len(patterns)} patterns from "
                     f"{len(self.recovery_matrix)} iterations")
        return patterns

    def _find_co_recovery_clusters(self, R: np.ndarray) -> List[Dict]:
        """Find variables that are always recovered together or missed together."""
        patterns = []
        n_iters = R.shape[0]

        if n_iters < 3:
            return patterns

        # Variables recovered in exact same iterations
        var_signatures = {}
        for i in range(self.n_vars):
            sig = tuple(R[:, i].astype(int))
            if sig not in var_signatures:
                var_signatures[sig] = []
            var_signatures[sig].append(i)

        for sig, var_indices in var_signatures.items():
            if len(var_indices) >= 5 and sum(sig) > 0:  # cluster of 5+, not all-zero
                var_names = [self.bio_var_names[i] for i in var_indices[:8]]
                categories = set()
                for idx in var_indices:
                    for cat, indices in self.bio_var_categories.items():
                        if idx in indices:
                            categories.add(cat)

                patterns.append({
                    'type': 'co_recovery',
                    'description': (f"{len(var_indices)} variables always "
                                    f"recovered/missed together: {var_names}"),
                    'evidence': (f"Signature across {n_iters} iterations: {sig}. "
                                 f"Categories: {categories}"),
                    'confidence': min(1.0, n_iters / 5),
                    'variable_indices': var_indices
                })

        return patterns

    def _find_property_correlations(self, R: np.ndarray) -> List[Dict]:
        """Find architecture properties that correlate with recovering specific variable types."""
        patterns = []

        # For each property, compare recovery rates when property is on vs off
        properties_to_check = ['time_handling', 'gradient_strategy',
                               'latent_structure', 'input_coupling', 'solver']

        for prop in properties_to_check:
            values = [c.get(prop, 'unknown') for c in self.architecture_configs]
            unique_values = set(values)

            if len(unique_values) < 2:
                continue

            for val in unique_values:
                mask = np.array([v == val for v in values])
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue

                for cat_name, cat_indices in self.bio_var_categories.items():
                    # Recovery rate for this category when property=val vs not
                    with_val = (R[mask][:, cat_indices].mean()
                                if mask.sum() > 0 else 0)
                    without_val = (R[~mask][:, cat_indices].mean()
                                   if (~mask).sum() > 0 else 0)

                    diff = with_val - without_val
                    if abs(diff) > 0.15:  # meaningful difference
                        direction = "helps" if diff > 0 else "hurts"
                        patterns.append({
                            'type': 'property_correlation',
                            'description': (f"{prop}={val} {direction} {cat_name} "
                                            f"recovery ({with_val:.0%} vs {without_val:.0%})"),
                            'evidence': (f"Diff={diff:+.0%} across {mask.sum()} "
                                         f"iterations with, {(~mask).sum()} without"),
                            'confidence': min(1.0, (mask.sum() + (~mask).sum()) / 6),
                            'property': prop,
                            'value': val,
                            'category': cat_name,
                            'effect': diff
                        })

        return patterns

    def _find_near_misses(self, C: np.ndarray) -> List[Dict]:
        """Find variables that almost reached recovery threshold across attempts."""
        patterns = []

        # For each variable, find max correlation across all attempts
        max_corr = np.nanmax(C, axis=0)  # (160,)

        near_miss_indices = np.where((max_corr > 0.3) & (max_corr < 0.5))[0]

        if len(near_miss_indices) > 0:
            near_miss_names = [self.bio_var_names[i] for i in near_miss_indices[:15]]
            near_miss_corrs = [float(max_corr[i]) for i in near_miss_indices[:15]]

            # Check which categories these near-misses belong to
            cat_counts = defaultdict(int)
            for idx in near_miss_indices:
                for cat, indices in self.bio_var_categories.items():
                    if idx in indices:
                        cat_counts[cat] += 1

            patterns.append({
                'type': 'near_miss',
                'description': (f"{len(near_miss_indices)} variables have "
                                f"near-miss correlations (0.3-0.5)"),
                'evidence': (f"Top near-misses: "
                             f"{list(zip(near_miss_names, near_miss_corrs))}. "
                             f"By category: {dict(cat_counts)}"),
                'confidence': 0.8,
                'variable_names': near_miss_names,
                'max_correlations': near_miss_corrs,
                'category_counts': dict(cat_counts)
            })

        return patterns

    def _find_timescale_alignment(self, R: np.ndarray) -> List[Dict]:
        """Find which architecture features align with which timescales."""
        patterns = []

        for ts_name, ts_indices in self.bio_var_timescales.items():
            for i, config in enumerate(self.architecture_configs):
                recovery_rate = R[i, ts_indices].mean()
                if recovery_rate > 0.3:
                    patterns.append({
                        'type': 'timescale_alignment',
                        'description': (f"{self.iteration_names[i]} recovers "
                                        f"{recovery_rate:.0%} of "
                                        f"{ts_name}-timescale variables"),
                        'evidence': (f"Config: time={config.get('time_handling')}, "
                                     f"solver={config.get('solver')}"),
                        'confidence': 0.7,
                        'timescale': ts_name,
                        'architecture': self.iteration_names[i],
                        'recovery_rate': float(recovery_rate)
                    })

        return patterns

    def _find_category_trends(self, R: np.ndarray) -> List[Dict]:
        """Find categories that are consistently missed or recovered."""
        patterns = []

        for cat_name, cat_indices in self.bio_var_categories.items():
            cat_recovery = R[:, cat_indices]  # (n_iters, n_cat_vars)

            # Category never recovered by any architecture
            if cat_recovery.sum() == 0:
                patterns.append({
                    'type': 'category_never_recovered',
                    'description': (f"Category '{cat_name}' ({len(cat_indices)} vars) "
                                    f"NEVER recovered by ANY architecture"),
                    'evidence': (f"0/{len(self.recovery_matrix)} architectures "
                                 f"recovered any {cat_name} variable"),
                    'confidence': min(1.0, len(self.recovery_matrix) / 4),
                    'category': cat_name,
                    'severity': 'ontological_gap'
                })

            # Category improving over iterations
            elif len(self.recovery_matrix) >= 3:
                per_iter = [cat_recovery[i].mean()
                            for i in range(len(self.recovery_matrix))]
                if per_iter[-1] > per_iter[0] + 0.1:
                    patterns.append({
                        'type': 'category_improving',
                        'description': (f"Category '{cat_name}' recovery improving: "
                                        f"{per_iter[0]:.0%} -> {per_iter[-1]:.0%}"),
                        'evidence': f"Trajectory: {[f'{p:.0%}' for p in per_iter]}",
                        'confidence': 0.6,
                        'category': cat_name
                    })

        return patterns
