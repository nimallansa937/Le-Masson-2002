"""
Layer 6: Combined Gap Computation

Integrates gap information from:
  - BioVarRecoverySpace: which variables are missing
  - ArchitectureTension: which features conflict
  - BioVarPatternExtractor: which patterns predict recovery

Produces a unified gap direction for Balloon Expansion.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .biovar_recovery_space import BioVarRecoverySpace, BioVarGap
from .architecture_tension import ArchitectureTensionAnalyzer, ArchitectureTension
from .biovar_pattern_extractor import BioVarPatternExtractor, DiscoveredPattern


@dataclass
class CombinedGapAnalysis:
    """Unified gap analysis combining all layers."""
    biovar_gap: BioVarGap
    tensions: List[ArchitectureTension]
    patterns: List[DiscoveredPattern]

    # Derived
    priority_categories: List[str]     # Which variable categories to target
    priority_timescales: List[str]     # Which timescales to target
    suggested_properties: Dict[str, str]  # property -> suggested value
    urgency: float                     # 0-1, how far from target

    def summary(self) -> str:
        """Human-readable gap summary."""
        lines = []
        lines.append(f"Gap: {self.biovar_gap.residual_magnitude:.0%} ({len(self.biovar_gap.missing_variables)}/160 variables missing)")
        lines.append(f"Direction: {self.biovar_gap.gap_direction}")

        if self.tensions:
            worst = max(self.tensions, key=lambda t: t.tension_magnitude)
            lines.append(f"Worst tension: {worst.property_1}={worst.value_1} vs {worst.value_2} (magnitude={worst.tension_magnitude:.2f})")
            lines.append(f"  Resolution: {worst.resolution_hint}")

        if self.patterns:
            lines.append(f"Patterns: {len(self.patterns)}")
            for p in self.patterns[:3]:
                lines.append(f"  - [{p.pattern_type}] {p.description[:60]}...")

        if self.suggested_properties:
            lines.append(f"Suggested properties: {self.suggested_properties}")

        return "\n".join(lines)


class GapAnalyzer:
    """
    Combined gap analyzer — integrates all layers for
    unified gap direction computation.
    """

    def __init__(self, recovery_space: BioVarRecoverySpace,
                 tension_analyzer: ArchitectureTensionAnalyzer,
                 pattern_extractor: BioVarPatternExtractor):
        self.recovery_space = recovery_space
        self.tension_analyzer = tension_analyzer
        self.pattern_extractor = pattern_extractor

    def analyze(self) -> CombinedGapAnalysis:
        """Compute combined gap analysis."""
        # Layer 1: BioVar gap
        biovar_gap = self.recovery_space.compute_gap()

        # Layer 4: Patterns
        patterns = self.pattern_extractor.patterns

        # Layer 5: Tensions
        results = {}
        for arch_id, recovery in self.recovery_space.recovery_results.items():
            # Need to find the template — use arch_id as key
            results[arch_id] = recovery
        tensions = self.tension_analyzer.tensions

        # Priority categories: highest gap fraction
        priority_cats = sorted(
            biovar_gap.gap_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )
        priority_categories = [cat for cat, frac in priority_cats if frac > 0.3]

        # Priority timescales
        priority_timescales = []
        for ts, var_ids in biovar_gap.missing_by_timescale.items():
            if len(var_ids) > 10:
                priority_timescales.append(ts)

        # Suggested properties from patterns
        suggested = {}
        for pattern in patterns:
            if pattern.suggested_property and pattern.confidence > 0.5:
                suggested[pattern.suggested_property] = pattern.suggested_value

        # Add suggestions from tension resolutions
        for tension in tensions:
            if tension.tension_magnitude > 0.5:
                suggested['latent_structure'] = 'hierarchical'

        return CombinedGapAnalysis(
            biovar_gap=biovar_gap,
            tensions=tensions,
            patterns=patterns,
            priority_categories=priority_categories,
            priority_timescales=priority_timescales,
            suggested_properties=suggested,
            urgency=biovar_gap.residual_magnitude
        )
