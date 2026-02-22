"""
Combined spike + biological auxiliary loss for A-R3b.

The loss function balances two objectives:
  1. Spike prediction accuracy (functional equivalence)
  2. Biological variable recovery (mechanistic equivalence)

The mixing parameter alpha controls the trade-off.
alpha=1.0 recovers the A-R3 baseline (spike-only).
alpha=0.0 is the bio-only extreme (no spike objective).

Timescale-aware smoothing: fast gating variables (tc_gating) are
evaluated at full 1ms resolution, while slow synaptic variables
get temporal smoothing to avoid penalizing high-frequency noise
in slowly-varying quantities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedBioLoss(nn.Module):
    """
    L_total = alpha * L_spike + (1-alpha) * L_bio

    L_spike = 0.7 * BCE(binary_spikes) + 0.3 * MSE(smoothed_rates)
    L_bio = (1/N_cats) * sum_category  w_cat * MSE(bio_pred, bio_target)

    Category weights compensate for different numbers of variables
    and different timescales. Temporal smoothing removes high-frequency
    noise from slow-variable loss computation.
    """

    def __init__(self, alpha=0.5, category_weights=None,
                 temporal_smooth_ms=None):
        """
        Args:
            alpha: Balance between spike and bio loss. 1.0 = spike-only.
            category_weights: Dict of weights per bio category.
                If None, weights are 1.0 (uniform).
            temporal_smooth_ms: Dict of smoothing kernel widths per category.
                If None, no temporal smoothing.
                Recommended: {'tc_gating': 0, 'nrt_state': 5, 'synaptic': 20}
                (fast variables get no smoothing, slow get heavy smoothing)
        """
        super().__init__()
        self.alpha = alpha
        self.category_weights = category_weights or {}
        self.temporal_smooth_ms = temporal_smooth_ms or {}
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def spike_loss(self, spike_pred, y_binary, y_rates):
        """
        Standard A-R3 spike loss (unchanged from DESCARTES).

        0.7 * BCE on binary spikes + 0.3 * MSE on smoothed rates.
        The spike_pred are raw logits — sigmoid is applied internally
        by BCEWithLogitsLoss and explicitly for the MSE term.
        """
        l_bce = self.bce(spike_pred, y_binary)
        l_mse = self.mse(torch.sigmoid(spike_pred), y_rates)
        return 0.7 * l_bce + 0.3 * l_mse

    def bio_loss(self, bio_preds, bio_targets):
        """
        Biological variable reconstruction loss with timescale-aware smoothing.

        Args:
            bio_preds: dict of (batch, time, bio_dim) predictions per category
            bio_targets: dict of (batch, time, bio_dim) ground truth per category

        Returns:
            total_bio_loss: weighted sum of per-category MSE losses
            per_category_losses: dict of individual category losses for logging
        """
        total = 0.0
        per_category = {}

        for cat_name, pred in bio_preds.items():
            if cat_name not in bio_targets:
                continue

            target = bio_targets[cat_name]

            # Temporal smoothing for slow variables:
            # Applying a low-pass filter before computing loss prevents the
            # network from wasting capacity trying to match high-frequency
            # noise in slowly-varying biological variables.
            smooth_ms = self.temporal_smooth_ms.get(cat_name, 0)
            if smooth_ms > 0:
                # avg_pool1d approximation of Gaussian smoothing
                # smooth_ms / dt_bin gives kernel size in timesteps
                # For dt_bin = 1ms, smooth_ms = 20 -> kernel_size = 20
                kernel_size = max(1, int(smooth_ms))
                if kernel_size > 1 and pred.shape[1] > kernel_size:
                    # Smooth both pred and target identically
                    # avg_pool1d expects (batch, channels, time)
                    pred_s = F.avg_pool1d(
                        pred.transpose(1, 2), kernel_size, stride=1,
                        padding=kernel_size // 2
                    ).transpose(1, 2)
                    target_s = F.avg_pool1d(
                        target.transpose(1, 2), kernel_size, stride=1,
                        padding=kernel_size // 2
                    ).transpose(1, 2)
                    # Trim to same length (pooling can change length by 1)
                    min_len = min(pred_s.shape[1], target_s.shape[1])
                    pred_s = pred_s[:, :min_len]
                    target_s = target_s[:, :min_len]
                    cat_loss = self.mse(pred_s, target_s)
                else:
                    cat_loss = self.mse(pred, target)
            else:
                cat_loss = self.mse(pred, target)

            # Apply category weight (default: uniform 1.0)
            w = self.category_weights.get(cat_name, 1.0)
            per_category[cat_name] = cat_loss.item()
            total += w * cat_loss

        # Normalize by number of active categories
        n_cats = len(per_category)
        if n_cats > 0:
            total = total / n_cats

        return total, per_category

    def forward(self, spike_pred, y_binary, y_rates, bio_preds, bio_targets):
        """
        Combined loss: alpha * L_spike + (1 - alpha) * L_bio.

        Args:
            spike_pred: (batch, time, output_dim) — raw logits
            y_binary: (batch, time, output_dim) — binary spike targets
            y_rates: (batch, time, output_dim) — smoothed rate targets
            bio_preds: dict of category -> (batch, time, bio_dim) predictions
            bio_targets: dict of category -> (batch, time, bio_dim) targets

        Returns:
            total_loss: scalar loss for backprop
            loss_components: dict with individual loss values for logging
        """
        l_spike = self.spike_loss(spike_pred, y_binary, y_rates)

        if self.alpha >= 1.0 or len(bio_preds) == 0:
            # Pure spike loss (A-R3 baseline)
            return l_spike, {'spike': l_spike.item(), 'bio_total': 0.0,
                             'total': l_spike.item()}

        l_bio, per_cat = self.bio_loss(bio_preds, bio_targets)

        total = self.alpha * l_spike + (1 - self.alpha) * l_bio

        components = {
            'spike': l_spike.item(),
            'bio_total': l_bio.item() if isinstance(l_bio, torch.Tensor) else l_bio,
            'total': total.item(),
            **{f'bio_{k}': v for k, v in per_cat.items()}
        }
        return total, components
