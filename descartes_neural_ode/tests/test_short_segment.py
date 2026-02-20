"""
Tests for the Short-Segment Verifier (Z3-C1 analog).
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.short_segment_verifier import ShortSegmentVerifier, VerificationResult


class SimpleModel(nn.Module):
    """A simple model that should pass verification."""

    def __init__(self, n_input=21, n_output=20, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden), nn.Tanh(),
            nn.Linear(hidden, n_output), nn.Sigmoid(),
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape
        outputs = []
        for t in range(seq_len):
            outputs.append(self.net(x[:, t, :]))
        return torch.stack(outputs, dim=1)


class BrokenModel(nn.Module):
    """A model that produces NaN — should fail verification."""

    def forward(self, x):
        return x[:, :, :20] * float('nan')


class ConstantModel(nn.Module):
    """A model that always outputs 0.5 — may fail on correlation."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, seq_len, _ = x.shape
        return torch.ones(batch, seq_len, 20) * 0.5 + self.dummy * 0


def make_synthetic_data(n_samples=16, seq_len=50, n_input=21, n_output=20):
    """Create synthetic training data for verifier."""
    X = np.random.rand(n_samples, seq_len, n_input).astype(np.float32)
    Y = np.random.rand(n_samples, seq_len, n_output).astype(np.float32)
    return {'X_short': X, 'Y_short': Y}


class TestVerificationResult:
    """Test the VerificationResult dataclass."""

    def test_passed_result(self):
        r = VerificationResult(
            passed=True,
            spike_correlation_50step=0.35,
            loss_converged=True,
            final_loss=0.02,
            training_time_seconds=5.0,
        )
        assert r.passed
        assert r.failure_reason is None

    def test_failed_result(self):
        r = VerificationResult(
            passed=False,
            spike_correlation_50step=0.05,
            loss_converged=False,
            final_loss=0.5,
            training_time_seconds=3.0,
            failure_reason="Loss did not converge",
        )
        assert not r.passed
        assert "converge" in r.failure_reason


class TestShortSegmentVerifier:
    """Test the verifier with different model types."""

    @pytest.fixture
    def verifier(self):
        data = make_synthetic_data()
        return ShortSegmentVerifier(data, device='cpu')

    def test_simple_model_runs(self, verifier):
        model = SimpleModel()
        result = verifier.verify(model, n_epochs=5)
        assert isinstance(result, VerificationResult)
        assert result.training_time_seconds > 0

    def test_nan_model_fails(self, verifier):
        model = BrokenModel()
        result = verifier.verify(model, n_epochs=5)
        assert not result.passed
        assert result.failure_reason is not None

    def test_result_has_required_fields(self, verifier):
        model = SimpleModel()
        result = verifier.verify(model, n_epochs=5)
        assert hasattr(result, 'passed')
        assert hasattr(result, 'spike_correlation_50step')
        assert hasattr(result, 'loss_converged')
        assert hasattr(result, 'final_loss')
        assert hasattr(result, 'training_time_seconds')

    def test_loss_is_finite(self, verifier):
        model = SimpleModel()
        result = verifier.verify(model, n_epochs=10)
        if result.passed:
            assert np.isfinite(result.final_loss)

    def test_tuple_output_model(self, verifier):
        """Models that return (pred, latent) tuples should work."""

        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(21, 64), nn.Tanh(),
                    nn.Linear(64, 20), nn.Sigmoid(),
                )

            def forward(self, x):
                batch, seq_len, _ = x.shape
                outputs = []
                for t in range(seq_len):
                    outputs.append(self.net(x[:, t, :]))
                pred = torch.stack(outputs, dim=1)
                latent = torch.zeros(batch, seq_len, 64)
                return pred, latent

        model = TupleModel()
        result = verifier.verify(model, n_epochs=5)
        assert isinstance(result, VerificationResult)


class TestVerifierEdgeCases:
    """Edge cases for the verifier."""

    def test_single_sample(self):
        data = make_synthetic_data(n_samples=1)
        verifier = ShortSegmentVerifier(data, device='cpu')
        model = SimpleModel()
        result = verifier.verify(model, n_epochs=3)
        assert isinstance(result, VerificationResult)

    def test_very_short_training(self):
        data = make_synthetic_data()
        verifier = ShortSegmentVerifier(data, device='cpu')
        model = SimpleModel()
        result = verifier.verify(model, n_epochs=1)
        assert isinstance(result, VerificationResult)
