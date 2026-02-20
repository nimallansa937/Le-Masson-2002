"""
Layer 2: Short-Segment Verifier (Z3-C1 Analog)

Quick filter: Can this architecture learn TC-nRt dynamics on
short windows? If not, skip it — don't waste hours on full training.

This is analogous to Z3 checking an ordering on [1, 10000] before
attempting a full proof. Cheap, bounded, catches obvious failures.

IMPORTANT: The standard Neural ODE PASSED this test in A-R3!
It can learn short-segment dynamics fine. The failure is at 2000 steps.
So this filter catches architectures that are worse than the baseline,
not the baseline itself. Its real value is after Balloon Expansion
generates novel architectures that might have implementation bugs.

The verifier checks:
  - Can the model's forward pass run without errors?
  - Does loss decrease (model is learning something)?
  - No NaN/Inf in gradients?
It does NOT check spike correlation — that's too strict for short
windows of real biological data and would filter out good architectures.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of short-segment verification."""
    passed: bool
    spike_correlation_50step: float
    loss_converged: bool
    final_loss: float
    training_time_seconds: float
    failure_reason: Optional[str] = None


class ShortSegmentVerifier:
    """
    Z3-C1 analog: Quick architecture validity check.

    Train on short windows for 30 epochs.
    Pass criteria:
      - Loss decreases (any meaningful decrease = model is learning)
      - No NaN/Inf in loss or gradients
      - Forward/backward pass completes without errors

    Spike correlation is measured but NOT used as pass criterion —
    short windows of real biological data can have near-zero
    correlation even for architectures that work well on full sequences.

    Estimated time: 30-120 seconds per architecture on CPU.
    """

    def __init__(self, train_data: Dict, device: str = 'cuda'):
        """
        Args:
            train_data: dict with 'X_short' (batch, T_short, 21) and 'Y_short' (batch, T_short, 20)
                        Pre-sliced short windows from A-R2 data.
        """
        self.X = torch.tensor(train_data['X_short'], device=device)
        self.Y = torch.tensor(train_data['Y_short'], device=device)
        self.device = device

    def verify(self, model: nn.Module, lr: float = 1e-3,
               n_epochs: int = 20) -> VerificationResult:
        """
        Quick training test on short segments.

        Returns VerificationResult with pass/fail and diagnostics.
        """
        import time
        start = time.time()
        model = model.to(self.device)

        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        except ValueError:
            return VerificationResult(
                passed=False,
                spike_correlation_50step=0.0,
                loss_converged=False,
                final_loss=float('inf'),
                training_time_seconds=time.time() - start,
                failure_reason="Model has no trainable parameters"
            )

        losses = []
        try:
            for epoch in range(n_epochs):
                model.train()
                optimizer.zero_grad()

                # Forward pass — architecture must accept (batch, time, 21)
                # and return at minimum a prediction tensor (batch, time, 20)
                output = model(self.X)
                if isinstance(output, tuple):
                    y_pred = output[0]  # First element is always prediction
                else:
                    y_pred = output

                loss = nn.functional.mse_loss(y_pred, self.Y)

                if torch.isnan(loss) or torch.isinf(loss):
                    return VerificationResult(
                        passed=False,
                        spike_correlation_50step=0.0,
                        loss_converged=False,
                        final_loss=float('inf'),
                        training_time_seconds=time.time() - start,
                        failure_reason="NaN/Inf loss"
                    )

                loss.backward()

                # Check for gradient pathology
                max_grad = max(
                    p.grad.abs().max().item()
                    for p in model.parameters()
                    if p.grad is not None
                )
                if max_grad == 0.0 and epoch > 5:
                    return VerificationResult(
                        passed=False,
                        spike_correlation_50step=0.0,
                        loss_converged=False,
                        final_loss=loss.item(),
                        training_time_seconds=time.time() - start,
                        failure_reason="Zero gradients (vanishing)"
                    )

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

        except RuntimeError as e:
            return VerificationResult(
                passed=False,
                spike_correlation_50step=0.0,
                loss_converged=False,
                final_loss=float('inf'),
                training_time_seconds=time.time() - start,
                failure_reason=f"Runtime error: {str(e)[:200]}"
            )

        # Evaluate convergence
        loss_ratio = losses[-1] / (losses[0] + 1e-10)
        loss_decrease = losses[0] - losses[-1]
        # Converged if: 50% relative decrease OR meaningful absolute decrease
        # (when initial loss is already low, ratio check is too strict)
        converged = (loss_ratio < 0.5) or (loss_decrease > 0.001 and losses[-1] < losses[0])

        # Spike correlation on short segment
        model.eval()
        with torch.no_grad():
            output = model(self.X)
            y_pred = output[0] if isinstance(output, tuple) else output
            y_np = y_pred.cpu().numpy().flatten()
            y_true = self.Y.cpu().numpy().flatten()
            corr_matrix = np.corrcoef(y_np, y_true)
            spike_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        # Pass criterion: convergence only (loss decreased).
        # Spike correlation is measured but NOT required — short windows
        # of real biological data can have near-zero correlation even for
        # architectures that work well on full 2000-step sequences.
        passed = converged

        return VerificationResult(
            passed=passed,
            spike_correlation_50step=spike_corr,
            loss_converged=converged,
            final_loss=losses[-1],
            training_time_seconds=time.time() - start,
            failure_reason=None if passed else (
                "Loss did not converge" if not converged else
                f"Spike correlation too low: {spike_corr:.3f}"
            )
        )
