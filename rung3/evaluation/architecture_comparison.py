"""
Architecture comparison — head-to-head analysis of all three models.

Produces:
  - Comparison table: output quality + latent alignment for all 3 models
  - Publication figures: bifurcation curves, CCA spectrum, RSA heatmaps,
    variable recovery matrix
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rung3.config import (
    DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR,
    GABA_VALUES, VAL_SEEDS, BIF_TARGET_NS, BIF_TARGET_SD,
)


def load_all_results(checkpoint_dir=CHECKPOINT_DIR):
    """Load evaluation results for all three models.

    Returns
    -------
    results : dict
        Keys: 'volterra', 'lstm', 'neural_ode'
        Values: combined output + latent metrics dicts
    """
    results = {}

    for model_name in ['volterra', 'lstm', 'neural_ode']:
        metrics_path = os.path.join(checkpoint_dir,
                                      f'{model_name}_evaluation.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                results[model_name] = json.load(f)
        else:
            results[model_name] = None

    return results


def print_comparison_table(results):
    """Print head-to-head comparison table."""
    print(f"\n{'='*80}")
    print(f"{'ARCHITECTURE COMPARISON':^80}")
    print(f"{'='*80}")

    header = f"{'Metric':<30} {'Volterra':<16} {'LSTM':<16} {'Neural ODE':<16}"
    print(header)
    print('-' * 80)

    metrics_to_show = [
        ('Output Quality', None),
        ('  Spike correlation', 'spike_correlation_mean'),
        ('  Bifurcation (nS)', 'bifurcation_threshold'),
        ('  Within 1 SD?', 'bifurcation_within_1sd'),
        ('Latent Alignment', None),
        ('  CCA mean corr', ('cca', 'mean_correlation')),
        ('  CCA p-value', ('cca', 'p_value')),
        ('  RSA correlation', ('rsa', 'correlation')),
        ('  Var recovery R²', ('variable_recovery', 'mean_r2')),
        ('  Vars significant', ('variable_recovery', 'n_significant')),
    ]

    model_names = ['volterra', 'lstm', 'neural_ode']

    for label, key in metrics_to_show:
        if key is None:
            print(f"\n{label}")
            continue

        vals = []
        for mn in model_names:
            r = results.get(mn)
            if r is None:
                vals.append('N/A')
                continue

            if isinstance(key, tuple):
                # Nested key
                v = r
                for k in key:
                    if isinstance(v, dict):
                        v = v.get(k, 'N/A')
                    else:
                        v = 'N/A'
                        break
            else:
                v = r.get(key, 'N/A')

            if isinstance(v, float):
                if 'p_value' in str(key):
                    vals.append(f'{v:.4e}')
                elif abs(v) > 100:
                    vals.append(f'{v:.1f}')
                else:
                    vals.append(f'{v:.4f}')
            elif isinstance(v, bool):
                vals.append('Yes' if v else 'No')
            elif isinstance(v, int):
                vals.append(str(v))
            else:
                vals.append(str(v))

        row = f"{label:<30} {vals[0]:<16} {vals[1]:<16} {vals[2]:<16}"
        print(row)

    print(f"\n{'='*80}")


def plot_bifurcation_comparison(results, figure_dir=FIGURE_DIR):
    """Plot bifurcation curves for all models overlaid."""
    os.makedirs(figure_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'volterra': '#2196F3', 'lstm': '#F44336', 'neural_ode': '#4CAF50'}
    labels = {'volterra': 'Volterra-Laguerre', 'lstm': 'LSTM',
              'neural_ode': 'Neural ODE'}

    for model_name in ['volterra', 'lstm', 'neural_ode']:
        r = results.get(model_name)
        if r is None or 'bifurcation_results' not in r:
            continue
        bif = r['bifurcation_results']
        gaba = [b['gaba_gmax'] for b in bif]
        pr = [b['mean_pause_rate'] for b in bif]
        ax.plot(gaba, pr, 'o-', color=colors[model_name],
                label=labels[model_name], linewidth=2, markersize=4)

    # Target threshold
    ax.axvline(BIF_TARGET_NS, color='gray', linestyle='--', linewidth=1.5,
               label=f'Target: {BIF_TARGET_NS} nS')
    ax.axvspan(BIF_TARGET_NS - BIF_TARGET_SD, BIF_TARGET_NS + BIF_TARGET_SD,
               alpha=0.15, color='gray')

    ax.set_xlabel('GABA Conductance (nS)', fontsize=12)
    ax.set_ylabel('Mean Pause Rate (Hz)', fontsize=12)
    ax.set_title('Bifurcation Comparison: Biological vs Model Predictions',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, 'bifurcation_comparison.png'),
                dpi=150)
    plt.close(fig)
    print(f"Saved: {figure_dir}/bifurcation_comparison.png")


def plot_cca_spectrum(results, figure_dir=FIGURE_DIR):
    """Plot CCA canonical correlation spectrum for all models."""
    os.makedirs(figure_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'volterra': '#2196F3', 'lstm': '#F44336', 'neural_ode': '#4CAF50'}
    labels = {'volterra': 'Volterra-Laguerre', 'lstm': 'LSTM',
              'neural_ode': 'Neural ODE'}

    for model_name in ['volterra', 'lstm', 'neural_ode']:
        r = results.get(model_name)
        if r is None or 'cca' not in r:
            continue
        cca = r['cca']
        if 'canonical_correlations' not in cca:
            continue
        cc = cca['canonical_correlations']
        ax.plot(range(1, len(cc)+1), cc, 'o-', color=colors[model_name],
                label=labels[model_name], linewidth=2, markersize=5)

    ax.set_xlabel('CCA Component', fontsize=12)
    ax.set_ylabel('Canonical Correlation', fontsize=12)
    ax.set_title('CCA Spectrum: Model Latents vs Biological Variables',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, 'cca_spectrum.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {figure_dir}/cca_spectrum.png")


def plot_variable_recovery_matrix(results, figure_dir=FIGURE_DIR):
    """Plot heatmap of variable recovery R² for all models."""
    os.makedirs(figure_dir, exist_ok=True)

    model_names = ['volterra', 'lstm', 'neural_ode']
    model_labels = ['Volterra', 'LSTM', 'Neural ODE']

    # Collect variable names and R² from first available model
    var_categories = {
        'TC gating': ['tc_m_T', 'tc_h_T', 'tc_m_h'],
        'nRt gating': ['nrt_m_Ts', 'nrt_h_Ts'],
        'Synaptic': ['gabaa_per_tc', 'gabab_per_tc', 'ampa_per_nrt'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, (model_name, model_label) in enumerate(
            zip(model_names, model_labels)):
        r = results.get(model_name)
        if r is None or 'variable_recovery' not in r:
            axes[ax_idx].set_title(f'{model_label}\n(No data)')
            continue

        vr = r['variable_recovery']
        if 'top_variables' not in vr:
            continue

        # Group by variable category
        cat_r2 = {}
        for top_var in vr.get('top_variables', []):
            name = top_var['name']
            r2 = top_var['r2']
            # Extract category from name
            base = '_'.join(name.split('_')[:-1])
            if base not in cat_r2:
                cat_r2[base] = []
            cat_r2[base].append(r2)

        # Plot bar chart of mean R² per category
        cats = list(cat_r2.keys())
        means = [np.mean(v) for v in cat_r2.values()]

        ax = axes[ax_idx]
        bars = ax.barh(range(len(cats)), means, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats)
        ax.set_xlabel('Mean R²')
        ax.set_title(f'{model_label}\n(Mean R²={vr["mean_r2"]:.3f})')
        ax.set_xlim(0, 1)

    fig.suptitle('Variable Recovery: Decoding Biological Variables from Model Latents',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, 'variable_recovery.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {figure_dir}/variable_recovery.png")


def run_comparison(checkpoint_dir=CHECKPOINT_DIR, figure_dir=FIGURE_DIR):
    """Run full architecture comparison."""
    results = load_all_results(checkpoint_dir)

    n_available = sum(1 for v in results.values() if v is not None)
    print(f"\nLoaded results for {n_available}/3 models")

    if n_available == 0:
        print("No evaluation results found. Run evaluation first.")
        return results

    print_comparison_table(results)

    try:
        plot_bifurcation_comparison(results, figure_dir)
    except Exception as e:
        print(f"Bifurcation plot error: {e}")

    try:
        plot_cca_spectrum(results, figure_dir)
    except Exception as e:
        print(f"CCA spectrum plot error: {e}")

    try:
        plot_variable_recovery_matrix(results, figure_dir)
    except Exception as e:
        print(f"Variable recovery plot error: {e}")

    # Save combined results
    combined_path = os.path.join(checkpoint_dir, 'architecture_comparison.json')
    serializable = {}
    for k, v in results.items():
        if v is not None:
            serializable[k] = {
                key: val for key, val in v.items()
                if not isinstance(val, np.ndarray)
            }
    with open(combined_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nSaved: {combined_path}")

    return results


if __name__ == '__main__':
    run_comparison()
