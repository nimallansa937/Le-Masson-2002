"""
Output evaluation metrics for Rung 3 models.

Compares model predictions against biological ground truth using:
  1. Spike train correlation (Pearson on smoothed rates)
  2. Victor-Purpura distance (temporal precision)
  3. Bifurcation threshold test (functional equivalence)
  4. Population coherence test
"""

import sys
import os
import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from rung3.config import (
    CORRELATION_SMOOTH_MS, VP_COST,
    BIF_EC_FRAC, BIF_N_BASELINE, BIF_N_CEILING,
    BIF_TARGET_NS, BIF_TARGET_SD, BIN_DT_MS,
)


def spike_train_correlation(y_true, y_pred, smooth_ms=CORRELATION_SMOOTH_MS):
    """Pearson correlation between smoothed firing rates.

    Parameters
    ----------
    y_true : ndarray (seq_len, n_neurons) or (n_neurons, seq_len)
        Ground truth smoothed rates.
    y_pred : ndarray same shape
        Model predictions.
    smooth_ms : float
        Additional smoothing kernel (ms).

    Returns
    -------
    mean_corr : float
        Mean correlation across neurons.
    per_neuron : list of float
    """
    if y_true.shape[0] < y_true.shape[1]:
        # Assume (n_neurons, seq_len) → transpose
        y_true = y_true.T
        y_pred = y_pred.T

    n_neurons = y_true.shape[1]
    per_neuron = []

    for i in range(n_neurons):
        t = y_true[:, i]
        p = y_pred[:, i]

        # Additional smoothing
        if smooth_ms > 0:
            kernel_size = int(smooth_ms / BIN_DT_MS)
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                t = np.convolve(t, kernel, mode='same')
                p = np.convolve(p, kernel, mode='same')

        if np.std(t) < 1e-10 or np.std(p) < 1e-10:
            per_neuron.append(0.0)
        else:
            c = np.corrcoef(t, p)[0, 1]
            per_neuron.append(float(c) if not np.isnan(c) else 0.0)

    return float(np.mean(per_neuron)), per_neuron


def victor_purpura_distance(spike_times_a, spike_times_b, cost=VP_COST):
    """Victor-Purpura distance between two spike trains.

    The VP distance is the minimum cost of transforming one spike train
    into another via insertions (cost 1), deletions (cost 1), and
    shifts (cost = q * |dt| where dt is the time shift).

    Parameters
    ----------
    spike_times_a : ndarray
        Spike times in seconds.
    spike_times_b : ndarray
        Spike times in seconds.
    cost : float
        Cost parameter q (1/s). Higher = more sensitive to timing.

    Returns
    -------
    distance : float
    """
    n_a = len(spike_times_a)
    n_b = len(spike_times_b)

    if n_a == 0:
        return float(n_b)
    if n_b == 0:
        return float(n_a)

    # Dynamic programming
    D = np.zeros((n_a + 1, n_b + 1))
    D[:, 0] = np.arange(n_a + 1)
    D[0, :] = np.arange(n_b + 1)

    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            shift_cost = cost * abs(spike_times_a[i-1] - spike_times_b[j-1])
            D[i, j] = min(
                D[i-1, j] + 1,          # Delete from A
                D[i, j-1] + 1,          # Insert into A
                D[i-1, j-1] + shift_cost  # Shift
            )

    return float(D[n_a, n_b])


def mean_vp_distance(tc_spikes_true, tc_spikes_pred, cost=VP_COST):
    """Mean VP distance across all TC neurons.

    Parameters
    ----------
    tc_spikes_true : list of ndarray
        Per-neuron spike times (biological).
    tc_spikes_pred : list of ndarray
        Per-neuron spike times (model).

    Returns
    -------
    mean_dist : float
    per_neuron : list of float
    """
    assert len(tc_spikes_true) == len(tc_spikes_pred)
    per_neuron = []
    for s_true, s_pred in zip(tc_spikes_true, tc_spikes_pred):
        d = victor_purpura_distance(
            np.asarray(s_true), np.asarray(s_pred), cost)
        per_neuron.append(d)

    return float(np.mean(per_neuron)), per_neuron


def bifurcation_test(model, gaba_values, seeds, data_dir,
                      from_predictions=True):
    """Test whether model preserves the GABA bifurcation threshold.

    Run model predictions across GABA sweep and compute bifurcation
    threshold using EC50 method. Compare to biological target (29 nS).

    Parameters
    ----------
    model : object
        Model with forward(x) method.
    gaba_values : ndarray
    seeds : list of int
    data_dir : str
    from_predictions : bool
        If True, compute pause rates from model predictions.

    Returns
    -------
    threshold : float
        Model's bifurcation threshold (nS).
    within_1sd : bool
    results : list of dict
    """
    from rung3.phase0_recording import load_trial_hdf5, trial_filename
    from rung3.preprocessing import (
        build_input_features, bin_spike_trains, smooth_rates
    )
    from analysis.oscillation import find_bifurcation_threshold

    results = []

    for gaba in gaba_values:
        pause_rates = []

        for seed in seeds:
            fpath = trial_filename(gaba, seed, data_dir)
            if not os.path.exists(fpath):
                continue

            data = load_trial_hdf5(fpath)

            # Build input
            features = build_input_features(
                data['retinal_spike_times'],
                data['gaba_gmax'],
                data['duration_s'])

            # Run model
            x = features[np.newaxis]  # (1, seq_len, 21)

            try:
                import torch
                if hasattr(model, 'parameters'):
                    # PyTorch model
                    device = next(model.parameters()).device
                    x_t = torch.from_numpy(x).to(device)
                    with torch.no_grad():
                        pred = model(x_t).cpu().numpy()[0]
                else:
                    pred = model.forward(x)[0]
            except ImportError:
                pred = model.forward(x)[0]

            # Convert predicted rates to spike-like pause rates
            # Threshold predictions to get "spikes"
            threshold = np.mean(pred) + np.std(pred)
            n_bins = pred.shape[0]
            duration_s = data['duration_s']

            for neuron_idx in range(pred.shape[1]):
                rate = pred[:, neuron_idx]
                # Detect "pauses" — periods where predicted rate is low
                low_rate = rate < threshold * 0.3
                # Count transitions from high to low (pause onsets)
                transitions = np.diff(low_rate.astype(int))
                n_pauses = np.sum(transitions == 1)
                pr = n_pauses / duration_s if duration_s > 0 else 0
                pause_rates.append(pr)

        if pause_rates:
            results.append({
                'gaba_gmax': float(gaba),
                'mean_pause_rate': float(np.mean(pause_rates)),
            })

    if len(results) < 5:
        return float('nan'), False, results

    threshold = find_bifurcation_threshold(
        results, metric='mean_pause_rate',
        n_baseline=BIF_N_BASELINE, n_ceiling=BIF_N_CEILING,
        ec_frac=BIF_EC_FRAC)

    within_1sd = abs(threshold - BIF_TARGET_NS) <= BIF_TARGET_SD

    return threshold, within_1sd, results


def population_coherence_test(model, gaba_value, seeds, data_dir):
    """Test model's population coherence at a given GABA level.

    Returns
    -------
    coherence : float
    """
    from rung3.phase0_recording import load_trial_hdf5, trial_filename
    from rung3.preprocessing import build_input_features
    from population.population_metrics import population_spindle_coherence

    coherences = []

    for seed in seeds:
        fpath = trial_filename(gaba_value, seed, data_dir)
        if not os.path.exists(fpath):
            continue

        data = load_trial_hdf5(fpath)
        features = build_input_features(
            data['retinal_spike_times'],
            data['gaba_gmax'],
            data['duration_s'])

        x = features[np.newaxis]

        try:
            import torch
            if hasattr(model, 'parameters'):
                device = next(model.parameters()).device
                x_t = torch.from_numpy(x).to(device)
                with torch.no_grad():
                    pred = model(x_t).cpu().numpy()[0]
            else:
                pred = model.forward(x)[0]
        except ImportError:
            pred = model.forward(x)[0]

        # Treat predicted rates as "voltage-like" for coherence
        # Transpose to (n_neurons, n_timepoints)
        V_pred = pred.T  # (20, seq_len)
        t = np.arange(V_pred.shape[1]) * BIN_DT_MS / 1000.0

        coh = population_spindle_coherence(V_pred, t)
        coherences.append(coh)

    return float(np.mean(coherences)) if coherences else 0.0


def evaluate_output_quality(model, model_name, data_dir, seeds,
                             gaba_values=None, verbose=True):
    """Complete output quality evaluation.

    Returns
    -------
    metrics : dict
    """
    from rung3.phase0_recording import load_trial_hdf5, list_trials
    from rung3.preprocessing import preprocess_trial
    from rung3.config import GABA_VALUES, VAL_SEEDS

    if gaba_values is None:
        gaba_values = GABA_VALUES
    if seeds is None:
        seeds = VAL_SEEDS

    if verbose:
        print(f"\n{'='*60}")
        print(f"Output Quality Evaluation: {model_name}")
        print(f"{'='*60}")

    # 1. Spike train correlation across all validation trials
    all_corrs = []
    all_trials = list_trials(data_dir)
    val_trials = [t for t in all_trials if t['seed'] in seeds]

    for trial_info in val_trials:
        data = load_trial_hdf5(trial_info['filepath'])
        X, Y_rate, _, _ = preprocess_trial(data)

        for w in range(min(X.shape[0], 5)):  # Sample 5 windows per trial
            try:
                import torch
                if hasattr(model, 'parameters'):
                    device = next(model.parameters()).device
                    x_t = torch.from_numpy(X[w:w+1]).to(device)
                    with torch.no_grad():
                        pred = model(x_t).cpu().numpy()[0]
                else:
                    pred = model.forward(X[w:w+1])[0]
            except ImportError:
                pred = model.forward(X[w:w+1])[0]

            corr, _ = spike_train_correlation(Y_rate[w], pred)
            all_corrs.append(corr)

    mean_corr = float(np.mean(all_corrs))
    std_corr = float(np.std(all_corrs))

    if verbose:
        print(f"  Spike correlation: {mean_corr:.4f} +/- {std_corr:.4f}")

    # 2. Bifurcation threshold
    threshold, within_1sd, bif_results = bifurcation_test(
        model, gaba_values, seeds, data_dir)

    if verbose:
        print(f"  Bifurcation threshold: {threshold:.1f} nS "
              f"(target: {BIF_TARGET_NS} +/- {BIF_TARGET_SD})")
        print(f"  Within 1 SD: {within_1sd}")

    metrics = {
        'model_name': model_name,
        'spike_correlation_mean': mean_corr,
        'spike_correlation_std': std_corr,
        'bifurcation_threshold': threshold,
        'bifurcation_within_1sd': within_1sd,
        'n_trials_evaluated': len(val_trials),
    }

    return metrics
