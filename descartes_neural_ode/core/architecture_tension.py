"""
Layer 5: Architecture Tension Analyzer (Geometric Algebra Analog)

In DESCARTES-Collatz, the GeometricAlgebraGapDetector finds bivector
tensions between constraint hyperplanes. Here, we find tensions between
architecture feature requirements — features that conflict in recovering
different variable types.

Example tension: "EULER solver helps gradient flow (recovers fast variables)
but hurts accuracy for stiff dynamics (misses slow variables)."
This tension suggests a MULTI-SCALE architecture with different solvers
for different latent subspaces.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

from .architecture_templates import ArchitectureTemplate
from .biovar_recovery_space import RecoveryResult, BiologicalVariable


@dataclass
class ArchitectureTension:
    """A tension between two architecture properties."""
    property_1: str           # e.g., "solver"
    value_1: str              # e.g., "euler"
    helps_category: str       # e.g., "tc_gating"
    property_2: str           # (same or different property)
    value_2: str              # e.g., "dopri5"
    helps_other_category: str # e.g., "nrt_state"
    tension_magnitude: float  # 0-1
    resolution_hint: str      # Suggested way to resolve


class ArchitectureTensionAnalyzer:
    """
    Geometric algebra analog: Find conflicting architecture requirements.

    When recovering variable type A requires property X=a, but recovering
    variable type B requires X=b, we have a tension. Resolving tensions
    requires hybrid architectures that can satisfy both requirements.
    """

    def __init__(self, registry: List[BiologicalVariable]):
        self.registry = registry
        self.tensions: List[ArchitectureTension] = []

    def analyze(self, results: Dict[str, Tuple[ArchitectureTemplate, RecoveryResult]]
                ) -> List[ArchitectureTension]:
        """
        Find tensions across all attempted architectures.

        Args:
            results: arch_id -> (template, recovery_result)

        Returns:
            List of discovered tensions
        """
        self.tensions = []

        # Build property -> category recovery matrix
        property_names = ['time_handling', 'gradient_strategy', 'latent_structure',
                          'input_coupling', 'solver']

        for prop in property_names:
            # Group architectures by property value
            value_recoveries = defaultdict(lambda: defaultdict(list))

            for arch_id, (template, recovery) in results.items():
                val = getattr(template, prop).value
                for cat in ['tc_gating', 'nrt_state', 'synaptic']:
                    n_rec = recovery.recovered_by_category.get(cat, 0)
                    value_recoveries[val][cat].append(n_rec)

            # Find tensions: property value A helps category X but not Y,
            # while property value B helps Y but not X
            values = list(value_recoveries.keys())
            for i, val_a in enumerate(values):
                for val_b in values[i + 1:]:
                    for cat_x in ['tc_gating', 'nrt_state', 'synaptic']:
                        for cat_y in ['tc_gating', 'nrt_state', 'synaptic']:
                            if cat_x == cat_y:
                                continue

                            avg_a_x = np.mean(value_recoveries[val_a][cat_x]) if value_recoveries[val_a][cat_x] else 0
                            avg_a_y = np.mean(value_recoveries[val_a][cat_y]) if value_recoveries[val_a][cat_y] else 0
                            avg_b_x = np.mean(value_recoveries[val_b][cat_x]) if value_recoveries[val_b][cat_x] else 0
                            avg_b_y = np.mean(value_recoveries[val_b][cat_y]) if value_recoveries[val_b][cat_y] else 0

                            # Tension: val_a good for cat_x, bad for cat_y
                            #          val_b good for cat_y, bad for cat_x
                            if (avg_a_x > avg_a_y + 2 and avg_b_y > avg_b_x + 2):
                                magnitude = min(
                                    (avg_a_x - avg_a_y) / max(avg_a_x, 1),
                                    (avg_b_y - avg_b_x) / max(avg_b_y, 1)
                                )

                                self.tensions.append(ArchitectureTension(
                                    property_1=prop,
                                    value_1=val_a,
                                    helps_category=cat_x,
                                    property_2=prop,
                                    value_2=val_b,
                                    helps_other_category=cat_y,
                                    tension_magnitude=float(magnitude),
                                    resolution_hint=(
                                        f"Hybrid architecture needed: use {val_a} "
                                        f"for {cat_x} subspace and {val_b} for "
                                        f"{cat_y} subspace. Or use hierarchical "
                                        f"latent structure with separate dynamics."
                                    )
                                ))

        return self.tensions

    def get_resolution_suggestions(self) -> List[str]:
        """Get unique resolution suggestions from all tensions."""
        return list(set(t.resolution_hint for t in self.tensions))

    def get_worst_tension(self) -> Optional[ArchitectureTension]:
        """Get the tension with highest magnitude."""
        if not self.tensions:
            return None
        return max(self.tensions, key=lambda t: t.tension_magnitude)
