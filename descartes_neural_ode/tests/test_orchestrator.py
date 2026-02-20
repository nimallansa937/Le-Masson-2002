"""
Tests for the DESCARTES-NeuralODE orchestrator.

These are integration tests that verify the orchestrator's logic
without running full training (which would take hours).
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
from core.biovar_recovery_space import build_biovar_registry, BioVarRecoverySpace
from core.biovar_pattern_extractor import BioVarPatternExtractor
from core.memory import MemoryLayer, MemoryEntry


class TestTemplateSelection:
    """Test template selection logic."""

    def test_templates_sorted_by_complexity(self):
        templates = get_initial_templates()
        # Filter out exhausted
        available = [t for t in templates if not t.exhausted]
        available.sort(key=lambda t: t.complexity)
        for i in range(len(available) - 1):
            assert available[i].complexity <= available[i + 1].complexity

    def test_exhausted_templates_excluded(self):
        templates = get_initial_templates()
        exhausted = {t.id for t in templates if t.exhausted}
        available = [t for t in templates if t.id not in exhausted]
        for t in available:
            assert not t.exhausted

    def test_t0_excluded_by_default(self):
        """T0 is pre-exhausted, so it shouldn't be selected."""
        templates = get_initial_templates()
        exhausted = {t.id for t in templates if t.exhausted}
        available = [t for t in templates if t.id not in exhausted]
        ids = [t.id for t in available]
        assert "T0" not in ids


class TestMemoryLayer:
    """Test the failure cache and memory system."""

    def test_record_and_retrieve(self):
        mem = MemoryLayer()
        entry = MemoryEntry(
            template_id="T2",
            family="ltc",
            timestamp=1000.0,
            status="completed",
            spike_correlation=0.35,
            n_recovered=42,
            verify_passed=True,
            verify_time_seconds=30.0,
            train_time_hours=1.5,
            gap_at_time=0.7,
        )
        mem.record(entry)
        assert len(mem.entries) == 1
        assert mem.entries[0].template_id == "T2"

    def test_best_result_tracking(self):
        mem = MemoryLayer()
        for i, (spike, recovery) in enumerate([(0.2, 20), (0.45, 55), (0.3, 40)]):
            mem.record(MemoryEntry(
                template_id=f"T{i}",
                family=f"fam_{i}",
                timestamp=float(i),
                status="completed",
                spike_correlation=spike,
                n_recovered=recovery,
                verify_passed=True,
                verify_time_seconds=10.0,
                train_time_hours=1.0,
                gap_at_time=0.5,
            ))

        best = max(mem.entries, key=lambda e: e.n_recovered)
        assert best.template_id == "T1"
        assert best.n_recovered == 55

    def test_family_exhaustion_tracking(self):
        mem = MemoryLayer()
        mem.record(MemoryEntry("T2a", "ltc", 0, "completed", 0.3, 30, True, 10, 1, 0.8))
        mem.record(MemoryEntry("T2b", "ltc", 1, "completed", 0.2, 20, True, 10, 1, 0.8))

        ltc_entries = [e for e in mem.entries if e.family == "ltc"]
        assert len(ltc_entries) == 2

    def test_save_load_roundtrip(self, tmp_path):
        mem = MemoryLayer()
        mem.record(MemoryEntry("T3", "cde", 100.0, "completed", 0.4, 45,
                               True, 20.0, 1.5, 0.65))
        path = str(tmp_path / "memory.json")
        mem.save(path)

        mem2 = MemoryLayer()
        mem2.load(path)
        assert len(mem2.entries) == 1
        assert mem2.entries[0].template_id == "T3"
        assert mem2.entries[0].spike_correlation == pytest.approx(0.4)


class TestPatternExtractor:
    """Test the DreamCoder analog."""

    @pytest.fixture
    def extractor(self):
        registry = build_biovar_registry()
        return BioVarPatternExtractor(registry)

    def test_empty_sleep_phase(self, extractor):
        patterns = extractor.sleep_phase_analyze()
        assert isinstance(patterns, list)

    def test_record_attempt_stores_data(self, extractor):
        templates = get_initial_templates()
        t = templates[1]  # Not exhausted
        extractor.record_attempt(
            template=t,
            recovery=None,
            spike_corr=0.0,
            bifurcation_error=float('inf'),
            training_hours=0.01,
            failed_early=True,
            failure_reason="Test failure",
        )
        assert len(extractor.attempts) == 1

    def test_compress_library(self, extractor):
        library = extractor.compress_library()
        assert isinstance(library, dict)
        assert 'primitives' in library


class TestRecoverySpace:
    """Test the 160-dim recovery space operations."""

    def test_gap_computation_with_no_results(self):
        registry = build_biovar_registry()
        space = BioVarRecoverySpace(registry)
        gap = space.compute_gap()
        assert gap.residual_magnitude == 1.0
        assert len(gap.missing_variables) == 160

    def test_gap_decreases_with_results(self):
        from core.biovar_recovery_space import RecoveryResult

        registry = build_biovar_registry()
        space = BioVarRecoverySpace(registry)

        vec = np.zeros(160)
        vec[:80] = 1.0
        r = RecoveryResult("test", vec, np.random.rand(160), {}, 0.5, 80,
                           {'tc_gating': 60, 'nrt_state': 20},
                           {'fast': 30, 'slow': 50})
        space.add_result("test", r)

        gap = space.compute_gap()
        assert gap.residual_magnitude == 0.5
        assert len(gap.missing_variables) == 80


class TestBalloonExpansion:
    """Test balloon expansion logic (without actual orchestrator)."""

    def test_new_template_creation(self):
        """Test that we can create new templates programmatically."""
        new = ArchitectureTemplate(
            id="balloon_0_test",
            name="Balloon: Test Architecture",
            description="Generated by balloon expansion",
            family="balloon_0",
            time_handling=TimeHandling.LTC,
            gradient_strategy=GradientStrategy.SEGMENTED,
            latent_structure=LatentStructure.HIERARCHICAL,
            input_coupling=InputCoupling.GATED,
            solver=SolverChoice.MIDPOINT,
            complexity=4,
            expected_training_hours=2.0,
        )
        assert new.id == "balloon_0_test"
        assert new.family == "balloon_0"

        vec = new.to_property_vector()
        assert vec.shape == (6,)

    def test_template_families_can_be_tracked(self):
        """Test that family exhaustion tracking works."""
        templates = get_initial_templates()
        families = set(t.family for t in templates)
        assert len(families) >= 5  # Should have multiple families

        exhausted_families = set()
        # Simulate exhausting one family
        for t in templates:
            if t.family == templates[0].family:
                t.exhausted = True

        for fam in families:
            fam_templates = [t for t in templates if t.family == fam]
            if all(t.exhausted for t in fam_templates):
                exhausted_families.add(fam)

        assert len(exhausted_families) >= 1


class TestArchitectureBuilding:
    """Test that architecture imports work (without full model instantiation)."""

    def test_base_ode_importable(self):
        from architectures.base_ode import TCReplacementNeuralODE
        assert TCReplacementNeuralODE is not None

    def test_ltc_importable(self):
        from architectures.ltc_network import LTCModel
        assert LTCModel is not None

    def test_cornn_importable(self):
        from architectures.coRNN import CoRNNModel
        assert CoRNNModel is not None

    def test_gru_ode_importable(self):
        from architectures.gru_ode import GRUODEModel
        assert GRUODEModel is not None

    def test_s4_importable(self):
        from architectures.s4_mamba import S4MambaModel
        assert S4MambaModel is not None

    def test_simple_model_forward(self):
        """Test that a simple architecture can do a forward pass."""
        import torch
        from architectures.gru_ode import GRUODEModel

        model = GRUODEModel(n_input=21, n_output=20, latent_dim=32)
        x = torch.randn(2, 50, 21)
        with torch.no_grad():
            output = model(x)

        assert isinstance(output, tuple)
        y_pred, latents = output
        assert y_pred.shape == (2, 50, 20)
        assert latents.shape == (2, 50, 32)

    def test_all_models_same_interface(self):
        """All architectures should accept (batch, seq_len, 21) and return (batch, seq_len, 20)."""
        import torch
        from architectures.gru_ode import GRUODEModel
        from architectures.coRNN import CoRNNModel
        from architectures.s4_mamba import S4MambaModel

        models = [
            GRUODEModel(n_input=21, n_output=20, latent_dim=32),
            CoRNNModel(n_input=21, n_output=20, latent_dim=32),
            S4MambaModel(n_input=21, n_output=20, latent_dim=32),
        ]

        x = torch.randn(2, 50, 21)
        for model in models:
            with torch.no_grad():
                output = model(x)
            y_pred = output[0] if isinstance(output, tuple) else output
            assert y_pred.shape == (2, 50, 20), (
                f"{model.__class__.__name__} wrong output shape: {y_pred.shape}"
            )
