"""
Tests for architecture templates and search space.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.architecture_templates import (
    ArchitectureTemplate, get_initial_templates,
    TimeHandling, GradientStrategy, LatentStructure,
    InputCoupling, SolverChoice,
)


class TestArchitectureProperties:
    """Test that architecture property enums are well-formed."""

    def test_time_handling_values(self):
        assert len(TimeHandling) == 6
        assert TimeHandling.STANDARD_ODE.value == "standard_ode"
        assert TimeHandling.LTC.value == "ltc"

    def test_gradient_strategy_values(self):
        assert len(GradientStrategy) == 5
        assert GradientStrategy.ADJOINT.value == "adjoint"
        assert GradientStrategy.DISTILLATION.value == "distillation"

    def test_latent_structure_values(self):
        assert len(LatentStructure) >= 4
        assert LatentStructure.BIOPHYSICAL.value == "biophysical"

    def test_solver_choice_values(self):
        assert SolverChoice.DOPRI5.value == "dopri5"
        assert SolverChoice.EULER.value == "euler"
        assert SolverChoice.MIDPOINT.value == "midpoint"


class TestInitialTemplates:
    """Test the 9 initial architecture templates."""

    def test_returns_9_templates(self):
        templates = get_initial_templates()
        assert len(templates) == 9

    def test_unique_ids(self):
        templates = get_initial_templates()
        ids = [t.id for t in templates]
        assert len(ids) == len(set(ids)), "Template IDs must be unique"

    def test_unique_names(self):
        templates = get_initial_templates()
        names = [t.name for t in templates]
        assert len(names) == len(set(names)), "Template names must be unique"

    def test_t0_is_exhausted(self):
        """Standard ODE baseline should be pre-marked as failed from A-R3."""
        templates = get_initial_templates()
        t0 = [t for t in templates if t.id == "standard_ode_baseline"][0]
        assert t0.exhausted is True
        assert t0.best_spike_corr == pytest.approx(0.012, abs=0.01)

    def test_all_have_valid_properties(self):
        templates = get_initial_templates()
        for t in templates:
            assert isinstance(t.time_handling, TimeHandling)
            assert isinstance(t.gradient_strategy, GradientStrategy)
            assert isinstance(t.latent_structure, LatentStructure)
            assert isinstance(t.input_coupling, InputCoupling)
            assert isinstance(t.solver, SolverChoice)

    def test_complexity_ordering(self):
        """Templates should have assigned complexity values."""
        templates = get_initial_templates()
        for t in templates:
            assert t.complexity >= 1
            assert t.complexity <= 5

    def test_latent_dim_ranges(self):
        templates = get_initial_templates()
        for t in templates:
            assert len(t.latent_dim_range) == 2
            assert t.latent_dim_range[0] <= t.latent_dim_range[1]
            assert t.latent_dim_range[0] >= 16  # Minimum useful latent dim


class TestPropertyVector:
    """Test the property vector encoding."""

    def test_vector_is_6d(self):
        templates = get_initial_templates()
        for t in templates:
            vec = t.to_property_vector()
            assert vec.shape == (6,), f"Property vector should be 6D, got {vec.shape}"

    def test_vector_values_are_finite(self):
        templates = get_initial_templates()
        for t in templates:
            vec = t.to_property_vector()
            assert np.all(np.isfinite(vec))

    def test_different_templates_have_different_vectors(self):
        templates = get_initial_templates()
        vectors = [t.to_property_vector() for t in templates]
        # At least some templates should have different vectors
        unique_count = len(set(tuple(v) for v in vectors))
        assert unique_count >= 5, "Most templates should have unique property vectors"


class TestTemplateManipulation:
    """Test template state tracking."""

    def test_attempt_counting(self):
        templates = get_initial_templates()
        t = templates[1]  # Not T0
        assert t.attempts == 0
        t.attempts += 1
        assert t.attempts == 1

    def test_best_score_tracking(self):
        templates = get_initial_templates()
        t = templates[1]
        assert t.best_spike_corr == 0.0
        t.best_spike_corr = 0.45
        assert t.best_spike_corr == 0.45

    def test_biovar_recovery_tracking(self):
        templates = get_initial_templates()
        t = templates[1]
        assert t.best_biovar_recovery == 0
        t.best_biovar_recovery = 42
        assert t.best_biovar_recovery == 42
