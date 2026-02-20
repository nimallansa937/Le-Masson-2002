"""
Tests for biological variable recovery scoring.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.biovar_recovery_space import (
    BiologicalVariable, build_biovar_registry,
    RecoveryResult, score_biovar_recovery,
    BioVarRecoverySpace, BioVarGap,
)


class TestBiovarRegistry:
    """Test the 160-variable biological registry."""

    def test_registry_has_160_variables(self):
        registry = build_biovar_registry()
        assert len(registry) == 160

    def test_ids_are_sequential(self):
        registry = build_biovar_registry()
        ids = [v.id for v in registry]
        assert ids == list(range(160))

    def test_category_counts(self):
        registry = build_biovar_registry()
        categories = {}
        for v in registry:
            categories[v.category] = categories.get(v.category, 0) + 1
        assert categories['tc_gating'] == 60
        assert categories['nrt_state'] == 60
        assert categories['synaptic'] == 40

    def test_timescales_present(self):
        registry = build_biovar_registry()
        timescales = set(v.timescale for v in registry)
        assert 'fast' in timescales
        assert 'slow' in timescales

    def test_subcategories_present(self):
        registry = build_biovar_registry()
        subcats = set(v.subcategory for v in registry)
        expected = {'mT', 'hT', 'mH', 'V', 'gaba_a', 'gaba_b'}
        assert expected == subcats

    def test_neuron_indices_valid(self):
        registry = build_biovar_registry()
        for v in registry:
            assert 0 <= v.neuron_index < 20

    def test_variable_names_are_unique(self):
        registry = build_biovar_registry()
        names = [v.name for v in registry]
        assert len(names) == len(set(names))

    def test_dynamics_types_present(self):
        registry = build_biovar_registry()
        types = set(v.dynamics_type for v in registry)
        assert 'monotonic' in types
        assert 'oscillatory' in types
        assert 'switching' in types


class TestRecoveryScoring:
    """Test the score_biovar_recovery function."""

    @pytest.fixture
    def registry(self):
        return build_biovar_registry()

    @pytest.fixture
    def synthetic_data(self, registry):
        """Create synthetic latents and ground truth."""
        T = 500
        latent_dim = 64
        model_latents = np.random.randn(latent_dim, T)

        # Create bio ground truth that partially correlates with latents
        bio_gt = {}
        for var in registry:
            parts = var.name.rsplit('_', 1)
            prefix = parts[0]
            if prefix not in bio_gt:
                bio_gt[prefix] = np.random.randn(20, T)

        return model_latents, bio_gt

    def test_returns_recovery_result(self, registry, synthetic_data):
        latents, bio_gt = synthetic_data
        result = score_biovar_recovery(latents, bio_gt, registry)
        assert isinstance(result, RecoveryResult)

    def test_recovery_vector_shape(self, registry, synthetic_data):
        latents, bio_gt = synthetic_data
        result = score_biovar_recovery(latents, bio_gt, registry)
        assert result.recovery_vector.shape == (160,)

    def test_recovery_vector_is_binary(self, registry, synthetic_data):
        latents, bio_gt = synthetic_data
        result = score_biovar_recovery(latents, bio_gt, registry)
        assert set(np.unique(result.recovery_vector)).issubset({0.0, 1.0})

    def test_correlation_vector_shape(self, registry, synthetic_data):
        latents, bio_gt = synthetic_data
        result = score_biovar_recovery(latents, bio_gt, registry)
        assert result.correlation_vector.shape == (160,)

    def test_correlation_vector_bounds(self, registry, synthetic_data):
        latents, bio_gt = synthetic_data
        result = score_biovar_recovery(latents, bio_gt, registry)
        assert np.all(result.correlation_vector >= 0)
        assert np.all(result.correlation_vector <= 1.0 + 1e-6)

    def test_n_recovered_consistent(self, registry, synthetic_data):
        latents, bio_gt = synthetic_data
        result = score_biovar_recovery(latents, bio_gt, registry)
        assert result.n_recovered == int(result.recovery_vector.sum())

    def test_category_breakdown_sums(self, registry, synthetic_data):
        latents, bio_gt = synthetic_data
        result = score_biovar_recovery(latents, bio_gt, registry)
        total_from_cats = sum(result.recovered_by_category.values())
        assert total_from_cats == result.n_recovered

    def test_perfect_correlation_recovers_all(self, registry):
        """If latents perfectly match bio vars, all should be recovered."""
        T = 200
        # Use 160 latent dims, each perfectly correlated with one bio var
        latents = np.random.randn(160, T)
        bio_gt = {}
        for var in registry:
            parts = var.name.rsplit('_', 1)
            prefix = parts[0]
            neuron = int(parts[1])
            if prefix not in bio_gt:
                bio_gt[prefix] = np.zeros((20, T))
            bio_gt[prefix][neuron] = latents[var.id]

        result = score_biovar_recovery(latents, bio_gt, registry)
        assert result.n_recovered == 160

    def test_zero_latents_recovers_none(self, registry):
        """Zero latents should recover nothing."""
        T = 200
        latents = np.zeros((64, T))
        bio_gt = {}
        for var in registry:
            parts = var.name.rsplit('_', 1)
            prefix = parts[0]
            if prefix not in bio_gt:
                bio_gt[prefix] = np.random.randn(20, T)

        result = score_biovar_recovery(latents, bio_gt, registry)
        assert result.n_recovered == 0


class TestBioVarRecoverySpace:
    """Test the recovery space and gap analysis."""

    @pytest.fixture
    def space(self):
        registry = build_biovar_registry()
        return BioVarRecoverySpace(registry)

    def test_empty_gap_is_full(self, space):
        gap = space.compute_gap()
        assert gap.residual_magnitude == 1.0
        assert len(gap.missing_variables) == 160

    def test_add_result_updates_union(self, space):
        recovery_vec = np.zeros(160)
        recovery_vec[:30] = 1.0  # Recover first 30

        result = RecoveryResult(
            architecture_id="test",
            recovery_vector=recovery_vec,
            correlation_vector=np.random.rand(160),
            best_latent_mapping={i: i for i in range(30)},
            cca_score=0.5,
            n_recovered=30,
            recovered_by_category={'tc_gating': 30},
            recovered_by_timescale={'fast': 10, 'slow': 20},
        )
        space.add_result("test", result)

        gap = space.compute_gap()
        assert gap.residual_magnitude < 1.0
        assert len(gap.missing_variables) == 130

    def test_gap_profile_has_all_categories(self, space):
        gap = space.compute_gap()
        assert 'tc_gating' in gap.gap_profile
        assert 'nrt_state' in gap.gap_profile
        assert 'synaptic' in gap.gap_profile

    def test_gap_direction_is_string(self, space):
        gap = space.compute_gap()
        assert isinstance(gap.gap_direction, str)
        assert len(gap.gap_direction) > 10

    def test_multiple_results_reduce_gap(self, space):
        # First result: recover TC gating
        vec1 = np.zeros(160)
        vec1[:60] = 1.0
        r1 = RecoveryResult("a", vec1, np.random.rand(160), {}, 0.5, 60,
                            {'tc_gating': 60}, {'fast': 20, 'slow': 40})
        space.add_result("a", r1)

        gap1 = space.compute_gap()

        # Second result: recover nRt state
        vec2 = np.zeros(160)
        vec2[60:120] = 1.0
        r2 = RecoveryResult("b", vec2, np.random.rand(160), {}, 0.5, 60,
                            {'nrt_state': 60}, {'fast': 40, 'slow': 20})
        space.add_result("b", r2)

        gap2 = space.compute_gap()
        assert gap2.residual_magnitude < gap1.residual_magnitude

    def test_architecture_comparison(self, space):
        vec = np.zeros(160)
        vec[:10] = 1.0
        r = RecoveryResult("test", vec, np.random.rand(160), {}, 0.3, 10,
                           {'tc_gating': 10}, {'fast': 5, 'slow': 5})
        space.add_result("test", r)

        comp = space.get_architecture_comparison()
        assert "test" in comp
        assert comp["test"]['n_recovered'] == 10
