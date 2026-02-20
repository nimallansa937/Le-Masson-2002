"""
Publication Figure Generation for DESCARTES-NeuralODE.

Generates all figures from Guide Task 7:
  1. Bifurcation curves: biological vs each architecture
  2. Latent-biological correlation heatmap
  3. CCA spectrum comparison across architectures
  4. RSA matrices side by side
  5. Hidden dimension sweep: output accuracy vs dim AND latent correlation vs dim
  6. GABA interpolation performance comparison
  7. Architecture comparison bar chart (Paper 1 main figure)

All figures use publication-quality formatting with matplotlib.
"""
import sys
import os
import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Publication style
STYLE_PARAMS = {
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

# Architecture display names and colors
ARCH_LABELS = {
    'standard_ode_baseline': 'Std Neural ODE',
    'segmented_ode': 'Segmented ODE',
    'ltc_network': 'LTC Network',
    'neural_cde': 'Neural CDE',
    'coupled_oscillatory': 'coRNN',
    'gru_ode': 'GRU-ODE',
    'hybrid_lstm_ode': 'Hybrid LSTM→ODE',
    'volterra_distilled_ode': 'Volterra-Distilled',
    's4_mamba': 'S4/Mamba',
}

ARCH_COLORS = {
    'standard_ode_baseline': '#d62728',  # Red (known failure)
    'segmented_ode': '#ff7f0e',
    'ltc_network': '#2ca02c',
    'neural_cde': '#1f77b4',
    'coupled_oscillatory': '#9467bd',
    'gru_ode': '#8c564b',
    'hybrid_lstm_ode': '#e377c2',
    'volterra_distilled_ode': '#7f7f7f',
    's4_mamba': '#bcbd22',
}


def _setup_style():
    """Apply publication style to matplotlib."""
    if HAS_MATPLOTLIB:
        plt.rcParams.update(STYLE_PARAMS)
        if HAS_SEABORN:
            sns.set_style("whitegrid")


def plot_architecture_comparison(
    comparison_data: Dict,
    output_path: str = 'fig_architecture_comparison.png',
):
    """Bar chart comparing all architectures on key metrics (Paper 1 main figure).

    Parameters
    ----------
    comparison_data : dict
        From full_comparison.py output.
    output_path : str
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    _setup_style()

    archs = []
    spike_corrs = []
    n_recovered = []
    coherence = []

    for arch_id, entry in comparison_data.get('architectures', {}).items():
        if 'error' in entry:
            continue
        archs.append(ARCH_LABELS.get(arch_id, arch_id))
        spike_corrs.append(entry.get('spike_correlation', 0))
        n_recovered.append(entry.get('n_recovered_05', 0))
        coherence.append(entry.get('coherence_ratio', 0))

    if not archs:
        print("No valid architecture data for comparison plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(archs))
    width = 0.6
    colors = [ARCH_COLORS.get(aid, '#333333')
              for aid in comparison_data.get('architectures', {}).keys()
              if 'error' not in comparison_data['architectures'].get(aid, {})]

    # Spike correlation
    axes[0].bar(x, spike_corrs, width, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Target')
    axes[0].set_ylabel('Spike Correlation (Pearson r)')
    axes[0].set_title('Output Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(archs, rotation=45, ha='right')
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Biovar recovery
    axes[1].bar(x, n_recovered, width, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].axhline(y=120, color='red', linestyle='--', alpha=0.5, label='Target (120/160)')
    axes[1].set_ylabel('Biological Variables Recovered')
    axes[1].set_title('Mechanistic Equivalence (|r| > 0.5)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(archs, rotation=45, ha='right')
    axes[1].set_ylim(0, 160)
    axes[1].legend()

    # Coherence ratio
    axes[2].bar(x, coherence, width, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Preservation (70%)')
    axes[2].set_ylabel('Coherence Ratio (model/bio)')
    axes[2].set_title('Population Synchronization')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(archs, rotation=45, ha='right')
    axes[2].set_ylim(0, 1.5)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_bifurcation_curves(
    bifurcation_data: Dict,
    output_path: str = 'fig_bifurcation_curves.png',
):
    """Plot bifurcation curves for each architecture vs biological.

    Parameters
    ----------
    bifurcation_data : dict
        Maps arch_id -> {'gaba_levels': [...], 'pause_rates': [...], 'threshold': float}
    output_path : str
    """
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Biological reference
    bio_threshold = 29.0
    bio_threshold_sd = 4.0

    ax.axvspan(bio_threshold - bio_threshold_sd, bio_threshold + bio_threshold_sd,
               alpha=0.2, color='green', label='Bio threshold ±1 SD')
    ax.axvline(x=bio_threshold, color='green', linestyle='-', linewidth=2, label='Bio threshold (29 nS)')

    for arch_id, data in bifurcation_data.items():
        gaba = data.get('gaba_levels', [])
        pr = data.get('pause_rates', [])
        threshold = data.get('threshold', None)

        if not gaba or not pr:
            continue

        color = ARCH_COLORS.get(arch_id, '#333333')
        label = ARCH_LABELS.get(arch_id, arch_id)

        ax.plot(gaba, pr, 'o-', color=color, label=label, markersize=4)
        if threshold is not None:
            ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.5)

    ax.set_xlabel('GABA Conductance (nS)')
    ax.set_ylabel('Pause Rate (fraction of time in pause)')
    ax.set_title('Bifurcation Threshold: Tonic → Oscillatory Transition')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_latent_heatmap(
    correlation_matrix: np.ndarray,
    latent_labels: Optional[List[str]] = None,
    bio_labels: Optional[List[str]] = None,
    arch_id: str = 'model',
    output_path: str = 'fig_latent_heatmap.png',
):
    """Heatmap of latent-bio correlations (which bio vars recovered by which dims).

    Parameters
    ----------
    correlation_matrix : ndarray (n_latent, n_bio)
    latent_labels, bio_labels : lists
    arch_id : str
    output_path : str
    """
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    n_lat, n_bio = correlation_matrix.shape

    if HAS_SEABORN:
        sns.heatmap(
            np.abs(correlation_matrix).T,
            ax=ax,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            cbar_kws={'label': '|Pearson r|'},
        )
    else:
        im = ax.imshow(
            np.abs(correlation_matrix).T,
            aspect='auto', cmap='YlOrRd',
            vmin=0, vmax=1,
        )
        plt.colorbar(im, ax=ax, label='|Pearson r|')

    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Biological Variable')
    ax.set_title(f'Latent ↔ Biological Correlation: {ARCH_LABELS.get(arch_id, arch_id)}')

    if bio_labels and n_bio <= 40:
        ax.set_yticks(range(n_bio))
        ax.set_yticklabels(bio_labels, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hidden_sweep(
    sweep_data: Dict,
    output_path: str = 'fig_hidden_sweep.png',
):
    """Plot output accuracy and latent correlation vs hidden dimension.

    Parameters
    ----------
    sweep_data : dict
        From hidden_sweep.py output.
    output_path : str
    """
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    summary = sweep_data.get('summary', {})

    for arch_id, s in summary.items():
        dims = s.get('dims', [])
        spike_corrs = s.get('spike_corr', [])
        n_recovered = s.get('n_recovered', [])

        if not dims:
            continue

        color = ARCH_COLORS.get(arch_id, '#333333')
        label = ARCH_LABELS.get(arch_id, arch_id)

        ax1.plot(dims, spike_corrs, 'o-', color=color, label=label, markersize=5)
        ax2.plot(dims, n_recovered, 's-', color=color, label=label, markersize=5)

    # Bio dimensionality reference line
    ax1.axvline(x=240, color='green', linestyle='--', alpha=0.5, label='Bio dims (240)')
    ax2.axvline(x=240, color='green', linestyle='--', alpha=0.5, label='Bio dims (240)')

    ax1.set_xlabel('Hidden / Latent Dimension')
    ax1.set_ylabel('Spike Correlation')
    ax1.set_title('Output Accuracy vs Model Capacity')
    ax1.legend(fontsize=7)
    ax1.set_xscale('log', base=2)

    ax2.set_xlabel('Hidden / Latent Dimension')
    ax2.set_ylabel('Bio Variables Recovered (|r| > 0.5)')
    ax2.set_title('Mechanistic Recovery vs Model Capacity')
    ax2.legend(fontsize=7)
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_gaba_interpolation(
    interpolation_data: Dict,
    output_path: str = 'fig_gaba_interpolation.png',
):
    """Plot GABA interpolation results.

    Parameters
    ----------
    interpolation_data : dict
        From gaba_interpolation.py output.
    output_path : str
    """
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    archs = []
    seen_corrs = []
    holdout_corrs = []

    for arch_id, res in interpolation_data.items():
        if 'error' in res:
            continue
        archs.append(ARCH_LABELS.get(arch_id, arch_id))
        seen_corrs.append(res.get('val_seen_gaba', {}).get('spike_correlation', 0))
        holdout_corrs.append(res.get('holdout_interpolation', {}).get('spike_correlation_mean', 0))

    if not archs:
        print("No data for GABA interpolation plot.")
        return

    x = np.arange(len(archs))
    width = 0.35

    ax.bar(x - width/2, seen_corrs, width, label='Seen GABA levels',
           color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, holdout_corrs, width, label='Holdout GABA=20 nS',
           color='coral', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Spike Correlation')
    ax.set_title('GABA Interpolation Test')
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_rsa_matrices(
    rsa_data: Dict,
    output_path: str = 'fig_rsa_matrices.png',
):
    """Plot RSA matrices side by side for each architecture.

    Parameters
    ----------
    rsa_data : dict
        Maps arch_id -> {'model_rdm': ndarray, 'bio_rdm': ndarray}
    output_path : str
    """
    if not HAS_MATPLOTLIB:
        return

    _setup_style()

    n_arch = len(rsa_data) + 1  # +1 for bio
    fig, axes = plt.subplots(1, n_arch, figsize=(4 * n_arch, 4))
    if n_arch == 1:
        axes = [axes]

    # Biological RDM (use first available)
    first_arch = list(rsa_data.keys())[0] if rsa_data else None
    if first_arch and 'bio_rdm' in rsa_data[first_arch]:
        bio_rdm = rsa_data[first_arch]['bio_rdm']
        axes[0].imshow(bio_rdm, cmap='viridis', aspect='equal')
        axes[0].set_title('Biological')
        axes[0].set_xlabel('Timepoint')
        axes[0].set_ylabel('Timepoint')

    for idx, (arch_id, data) in enumerate(rsa_data.items()):
        if 'model_rdm' not in data:
            continue
        ax = axes[idx + 1]
        ax.imshow(data['model_rdm'], cmap='viridis', aspect='equal')
        rho = data.get('rsa_correlation', 0)
        ax.set_title(f'{ARCH_LABELS.get(arch_id, arch_id)}\nρ={rho:.3f}')
        ax.set_xlabel('Timepoint')

    plt.suptitle('Representational Dissimilarity Matrices', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(
    results_dir: str,
    figure_dir: str,
    verbose: bool = True,
):
    """Generate all publication figures from saved results.

    Parameters
    ----------
    results_dir : str
        Directory containing JSON result files.
    figure_dir : str
        Output directory for figures.
    verbose : bool
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — cannot generate figures.")
        return

    os.makedirs(figure_dir, exist_ok=True)
    results_path = Path(results_dir)

    # 1. Architecture comparison
    comp_file = results_path / 'architecture_comparison.json'
    if comp_file.exists():
        with open(comp_file) as f:
            comp_data = json.load(f)
        plot_architecture_comparison(
            comp_data,
            os.path.join(figure_dir, 'fig_architecture_comparison.png')
        )

    # 2. Hidden dimension sweep
    sweep_file = results_path / 'hidden_sweep_results.json'
    if sweep_file.exists():
        with open(sweep_file) as f:
            sweep_data = json.load(f)
        plot_hidden_sweep(
            sweep_data,
            os.path.join(figure_dir, 'fig_hidden_sweep.png')
        )

    # 3. GABA interpolation
    interp_file = results_path / 'gaba_interpolation_results.json'
    if interp_file.exists():
        with open(interp_file) as f:
            interp_data = json.load(f)
        plot_gaba_interpolation(
            interp_data,
            os.path.join(figure_dir, 'fig_gaba_interpolation.png')
        )

    if verbose:
        print(f"\nAll available figures generated in {figure_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--results-dir', type=str, default='.',
                       help='Directory containing JSON result files')
    parser.add_argument('--figure-dir', type=str, default='figures_output',
                       help='Output directory for figures')
    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.figure_dir)
