"""
Layer 4: BioVar Pattern Extractor (DreamCoder Analog)

Wake phase: Record each architecture's recovery result.
Sleep phase: Analyze patterns across all attempts to discover:
  1. Variable clusters that are always co-recovered or co-missed
  2. Architecture properties correlated with recovering specific variable types
  3. Family exhaustion — when all variants of an architecture family fail
  4. Near-miss patterns — architectures that almost recover a variable

These patterns guide the next architecture to try, just as DreamCoder's
patterns guide the next C1 template in Collatz ordering search.

THE KEY DIFFERENCE FROM STANDARD NEURAL ARCHITECTURE SEARCH:
NAS optimizes a scalar metric (accuracy, loss). This system tracks
a 160-dimensional recovery vector and learns WHICH architectural
features correspond to WHICH biological variables. The gap has
structure, not just magnitude.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import numpy as np

from .architecture_templates import ArchitectureTemplate
from .biovar_recovery_space import RecoveryResult, BiologicalVariable


@dataclass
class AttemptRecord:
    """One architecture attempt with full diagnostics."""
    template: ArchitectureTemplate
    recovery: RecoveryResult
    spike_correlation: float
    bifurcation_error: float     # nS error from biological threshold
    training_time_hours: float
    failed_early: bool           # Did short-segment verifier reject it?
    failure_reason: Optional[str] = None


@dataclass
class DiscoveredPattern:
    """A pattern extracted during sleep phase."""
    pattern_type: str            # "co_recovery", "property_correlation", "family_exhausted", "near_miss"
    description: str
    affected_variables: List[int]
    suggested_property: Optional[str] = None  # Architecture property to try
    suggested_value: Optional[str] = None     # Specific value
    confidence: float = 0.0      # 0-1


class BioVarPatternExtractor:
    """
    DreamCoder-style pattern extraction for Neural ODE architecture search.

    Wake phase: Record architecture attempts with recovery results.
    Sleep phase: Analyze patterns to suggest new architectures.
    """

    def __init__(self, registry: List[BiologicalVariable]):
        self.registry = registry
        self.attempts: List[AttemptRecord] = []
        self.patterns: List[DiscoveredPattern] = []
        self.library: Dict[str, Dict] = {}  # Learned primitives

    # ========== WAKE PHASE ==========

    def record_attempt(self, template: ArchitectureTemplate,
                       recovery: RecoveryResult,
                       spike_corr: float,
                       bifurcation_error: float,
                       training_hours: float,
                       failed_early: bool = False,
                       failure_reason: str = None):
        """Record an architecture attempt for later analysis."""
        self.attempts.append(AttemptRecord(
            template=template,
            recovery=recovery,
            spike_correlation=spike_corr,
            bifurcation_error=bifurcation_error,
            training_time_hours=training_hours,
            failed_early=failed_early,
            failure_reason=failure_reason
        ))

    # ========== SLEEP PHASE ==========

    def sleep_phase_analyze(self) -> List[DiscoveredPattern]:
        """
        Analyze all attempts to discover patterns.
        Called after each new attempt or batch of attempts.

        Returns list of newly discovered patterns.
        """
        if len(self.attempts) < 2:
            return []

        new_patterns = []

        # Pattern 1: Variable co-recovery clusters
        new_patterns.extend(self._find_co_recovery_clusters())

        # Pattern 2: Architecture property -> variable type correlations
        new_patterns.extend(self._find_property_correlations())

        # Pattern 3: Family exhaustion
        new_patterns.extend(self._find_exhausted_families())

        # Pattern 4: Near-miss variables
        new_patterns.extend(self._find_near_misses())

        # Pattern 5: Timescale-architecture alignment
        new_patterns.extend(self._find_timescale_alignment())

        self.patterns.extend(new_patterns)
        return new_patterns

    def _find_co_recovery_clusters(self) -> List[DiscoveredPattern]:
        """
        Find groups of variables that are always recovered together
        or always missed together across architectures.

        If variables {A, B, C} are always co-recovered, they likely
        share a common mechanism. An architecture that recovers A
        will probably also get B and C — so target them as a group.
        """
        patterns = []
        successful = [a for a in self.attempts if not a.failed_early and a.recovery is not None and a.recovery.n_recovered > 0]
        if len(successful) < 2:
            return patterns

        # Build co-recovery matrix
        n_vars = len(self.registry)
        co_matrix = np.zeros((n_vars, n_vars))

        for attempt in successful:
            rv = attempt.recovery.recovery_vector
            # Outer product: co_matrix[i,j] incremented when both recovered
            co_matrix += np.outer(rv, rv)
            # Also track co-absence
            absent = 1.0 - rv
            co_matrix += np.outer(absent, absent)

        # Normalize
        co_matrix /= len(successful)

        # Find strong clusters (co-recovery > 0.8)
        visited = set()
        for i in range(n_vars):
            if i in visited:
                continue
            cluster = [i]
            for j in range(i + 1, n_vars):
                if co_matrix[i, j] > 0.8:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) > 2:
                # Identify what these variables share
                cats = [self.registry[v].category for v in cluster]
                subcats = [self.registry[v].subcategory for v in cluster]
                common_cat = Counter(cats).most_common(1)[0]
                common_subcat = Counter(subcats).most_common(1)[0]

                patterns.append(DiscoveredPattern(
                    pattern_type="co_recovery",
                    description=(
                        f"Variables {cluster[:5]}... ({len(cluster)} total) are always "
                        f"co-recovered. Common: {common_cat[0]} / {common_subcat[0]}. "
                        f"Target this cluster as a group."
                    ),
                    affected_variables=cluster,
                    confidence=common_cat[1] / len(cluster)
                ))

        return patterns

    def _find_property_correlations(self) -> List[DiscoveredPattern]:
        """
        Find correlations between architecture properties and
        recovery of specific variable types.

        Example discovery: "LTC time_handling recovers 80% of
        slow-timescale variables but only 20% of fast-timescale.
        Standard ODE recovers neither."
        """
        patterns = []
        successful = [a for a in self.attempts if not a.failed_early]
        if len(successful) < 3:
            return patterns

        # For each property dimension, check recovery by category
        property_names = ['time_handling', 'gradient_strategy', 'latent_structure',
                          'input_coupling', 'solver']

        for prop in property_names:
            prop_groups = defaultdict(list)  # property_value -> [recovery_results]
            for attempt in successful:
                if attempt.recovery is None:
                    continue
                val = getattr(attempt.template, prop).value
                prop_groups[val].append(attempt.recovery)

            # Compare recovery by category across property values
            for val, recoveries in prop_groups.items():
                if len(recoveries) < 1:
                    continue
                avg_by_cat = {}
                for cat in ['tc_gating', 'nrt_state', 'synaptic']:
                    avg_by_cat[cat] = np.mean([
                        r.recovered_by_category.get(cat, 0) for r in recoveries
                    ])

                # Find if this property value has a strong category preference
                if avg_by_cat:
                    best_cat = max(avg_by_cat, key=avg_by_cat.get)
                    worst_cat = min(avg_by_cat, key=avg_by_cat.get)
                    if avg_by_cat[best_cat] > 2 * avg_by_cat.get(worst_cat, 0.01):
                        patterns.append(DiscoveredPattern(
                            pattern_type="property_correlation",
                            description=(
                                f"{prop}={val} strongly favors {best_cat} variables "
                                f"(avg {avg_by_cat[best_cat]:.0f} recovered) over "
                                f"{worst_cat} (avg {avg_by_cat.get(worst_cat, 0):.0f}). "
                                f"To recover {worst_cat}, try different {prop}."
                            ),
                            affected_variables=[],
                            suggested_property=prop,
                            suggested_value=f"NOT_{val}",
                            confidence=0.7
                        ))

        return patterns

    def _find_exhausted_families(self) -> List[DiscoveredPattern]:
        """
        Detect when all templates in a family have been tried and failed.
        Triggers Balloon Principle — need fundamentally new architecture.
        """
        patterns = []
        family_attempts = defaultdict(list)
        for attempt in self.attempts:
            family_attempts[attempt.template.family].append(attempt)

        for family, attempts in family_attempts.items():
            if len(attempts) >= 2:  # At least 2 attempts in family
                best_recovery = max(
                    (a.recovery.n_recovered for a in attempts if a.recovery is not None),
                    default=0
                )
                all_failed = all(a.spike_correlation < 0.3 for a in attempts)

                if all_failed:
                    patterns.append(DiscoveredPattern(
                        pattern_type="family_exhausted",
                        description=(
                            f"Architecture family '{family}' exhausted after "
                            f"{len(attempts)} attempts. Best recovery: {best_recovery}/160. "
                            f"Balloon expansion needed — try different family."
                        ),
                        affected_variables=[],
                        confidence=0.9
                    ))

        return patterns

    def _find_near_misses(self) -> List[DiscoveredPattern]:
        """
        Find variables with correlation 0.3-0.5 (near the recovery threshold).
        These are the easiest wins — small architectural changes might push
        them over the threshold.
        """
        patterns = []
        for attempt in self.attempts:
            if attempt.failed_early or attempt.recovery is None:
                continue
            near_misses = []
            for i, corr in enumerate(attempt.recovery.correlation_vector):
                if 0.3 < corr < 0.5:
                    near_misses.append((i, corr))

            if len(near_misses) > 5:
                var_names = [self.registry[i].name for i, _ in near_misses[:5]]
                patterns.append(DiscoveredPattern(
                    pattern_type="near_miss",
                    description=(
                        f"Architecture {attempt.template.id} has {len(near_misses)} "
                        f"near-miss variables (r=0.3-0.5): {var_names}... "
                        f"Small modifications may recover these."
                    ),
                    affected_variables=[i for i, _ in near_misses],
                    confidence=0.6
                ))

        return patterns

    def _find_timescale_alignment(self) -> List[DiscoveredPattern]:
        """
        Check if architectures with specific solver/time properties
        better recover specific timescale variables.
        """
        patterns = []
        for attempt in self.attempts:
            if attempt.failed_early or attempt.recovery is None:
                continue
            by_ts = attempt.recovery.recovered_by_timescale
            fast = by_ts.get('fast', 0)
            slow = by_ts.get('slow', 0)

            if fast > 3 * max(slow, 1):
                patterns.append(DiscoveredPattern(
                    pattern_type="timescale_alignment",
                    description=(
                        f"{attempt.template.id} recovers fast ({fast}) but not slow ({slow}) "
                        f"variables. Solver: {attempt.template.solver.value}. "
                        f"May need adaptive time stepping or multi-scale architecture."
                    ),
                    affected_variables=[],
                    suggested_property="latent_structure",
                    suggested_value="hierarchical",
                    confidence=0.5
                ))

        return patterns

    # ========== LIBRARY COMPRESSION ==========

    def compress_library(self) -> Dict:
        """
        Compress patterns into actionable primitives.
        MDL principle: fewest primitives explaining most patterns.
        """
        primitives = []
        for pattern in self.patterns:
            if pattern.suggested_property:
                primitives.append({
                    'type': pattern.pattern_type,
                    'property': pattern.suggested_property,
                    'suggested_value': pattern.suggested_value,
                    'confidence': pattern.confidence,
                    'description': pattern.description
                })

        return {
            'n_attempts': len(self.attempts),
            'n_patterns': len(self.patterns),
            'n_primitives': len(primitives),
            'primitives': primitives,
            'best_recovery': max(
                (a.recovery.n_recovered for a in self.attempts
                 if not a.failed_early and a.recovery is not None),
                default=0
            ),
            'family_exhaustion': {
                p.description.split("'")[1]: True
                for p in self.patterns if p.pattern_type == "family_exhausted"
            }
        }
