"""
Phase 6: Visualization — R²(tau) timescale plots + probe ladder bar charts.

Two main figures:
  1. R²(tau): X=timescale (log), Y=delta_R², color=probe_type, panels=architecture
     Shows whether encoding quality varies with the timescale of the
     biological variable — fast gating (1ms) vs slow currents (150ms).

  2. Probe Ladder: Grouped bar chart per architecture showing Ridge vs MLP-1
     vs MLP-2 for each target tier, revealing nonlinear encoding.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from a_r3b_reanalysis.config import TARGET_REGISTRY, OUTPUT_DIR


# ============================================================
# Color scheme
# ============================================================

PROBE_COLORS = {
    'ridge': '#2196F3',   # blue
    'mlp_1': '#FF9800',   # orange
    'mlp_2': '#E91E63',   # pink
}

DIAGNOSIS_COLORS = {
    'LINEAR_ENCODING': '#4CAF50',
    'NONLINEAR_ENCODING': '#FF9800',
    'GENUINE_ZOMBIE': '#F44336',
    'EXPECTED_NULL': '#9E9E9E',
    'SPURIOUS': '#FFEB3B',
    'STRUCTURAL': '#9C27B0',
    'SCALE_AMBIGUITY': '#00BCD4',
    'AMBIGUOUS': '#795548',
}


# ============================================================
# Figure 1: R² vs Timescale
# ============================================================

def plot_r2_vs_timescale(results_df, save_dir=None, verbose=True):
    """Plot delta_R² vs target timescale, faceted by architecture.

    Parameters
    ----------
    results_df : pd.DataFrame
        From phase4/5 with columns: architecture, target, timescale_ms,
        probe_type, delta_r2, r2_trained, neuron_idx
    save_dir : Path, optional
    verbose : bool

    Returns
    -------
    fig : matplotlib.Figure
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.copy()

    # Aggregate across neurons: mean delta_r2 per (arch, target, probe)
    agg = df.groupby(['architecture', 'target', 'probe_type', 'timescale_ms']).agg(
        delta_r2_mean=('delta_r2', 'mean'),
        delta_r2_std=('delta_r2', 'std'),
        r2_trained_mean=('r2_trained', 'mean'),
        selectivity_mean=('selectivity', 'mean'),
    ).reset_index()

    architectures = sorted(agg['architecture'].unique())
    n_arch = len(architectures)

    fig, axes = plt.subplots(1, n_arch, figsize=(5 * n_arch, 4.5),
                              sharey=True, squeeze=False)

    for i, arch in enumerate(architectures):
        ax = axes[0, i]
        arch_data = agg[agg['architecture'] == arch]

        for probe_type in ['ridge', 'mlp_1', 'mlp_2']:
            pdata = arch_data[arch_data['probe_type'] == probe_type]
            if pdata.empty:
                continue

            color = PROBE_COLORS.get(probe_type, 'gray')
            label = probe_type.replace('_', '-').upper()

            ax.errorbar(
                pdata['timescale_ms'], pdata['delta_r2_mean'],
                yerr=pdata['delta_r2_std'],
                fmt='o-', color=color, label=label,
                markersize=5, capsize=3, alpha=0.8,
            )

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Target Timescale (ms)')
        ax.set_title(arch.upper())
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel(r'$\Delta R^2$ (trained $-$ untrained)')
    fig.suptitle(r'A-R3b: $\Delta R^2$ vs Biological Timescale', fontsize=13)
    fig.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        path = save_dir / f'a_r3b_r2_vs_timescale.{ext}'
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved: {path}")

    return fig


# ============================================================
# Figure 2: Probe Ladder Bar Charts
# ============================================================

def plot_probe_ladder(results_df, save_dir=None, verbose=True):
    """Grouped bar chart: Ridge vs MLP for each target, per architecture.

    Parameters
    ----------
    results_df : pd.DataFrame
    save_dir : Path, optional
    verbose : bool

    Returns
    -------
    fig : matplotlib.Figure
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.copy()

    # Aggregate across neurons
    agg = df.groupby(['architecture', 'target', 'probe_type', 'tier']).agg(
        r2_trained_mean=('r2_trained', 'mean'),
        delta_r2_mean=('delta_r2', 'mean'),
        selectivity_mean=('selectivity', 'mean'),
    ).reset_index()

    architectures = sorted(agg['architecture'].unique())
    n_arch = len(architectures)

    # Sort targets by tier then timescale
    target_order = sorted(
        TARGET_REGISTRY.keys(),
        key=lambda t: (TARGET_REGISTRY[t]['tier'],
                       TARGET_REGISTRY[t]['timescale_ms'])
    )

    fig, axes = plt.subplots(n_arch, 1, figsize=(12, 3.5 * n_arch),
                              squeeze=False)

    probe_types = ['ridge', 'mlp_1', 'mlp_2']
    n_probes = len(probe_types)
    bar_width = 0.25

    for i, arch in enumerate(architectures):
        ax = axes[i, 0]
        arch_data = agg[agg['architecture'] == arch]

        # Filter to targets that appear in data
        present_targets = [t for t in target_order
                          if t in arch_data['target'].values]

        x = np.arange(len(present_targets))

        for j, probe_type in enumerate(probe_types):
            pdata = arch_data[arch_data['probe_type'] == probe_type]
            vals = []
            for t in present_targets:
                trow = pdata[pdata['target'] == t]
                vals.append(trow['delta_r2_mean'].values[0]
                           if len(trow) > 0 else 0)

            color = PROBE_COLORS.get(probe_type, 'gray')
            label = probe_type.replace('_', '-').upper()
            ax.bar(x + j * bar_width, vals, bar_width,
                   color=color, label=label, alpha=0.85)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(present_targets, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(r'$\Delta R^2$')
        ax.set_title(f'{arch.upper()} — Probe Ladder')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=8, loc='upper right')

        # Add tier separators
        tier_boundaries = []
        prev_tier = None
        for k, t in enumerate(present_targets):
            tier = TARGET_REGISTRY[t]['tier']
            if prev_tier is not None and tier != prev_tier:
                tier_boundaries.append(k - 0.5)
            prev_tier = tier
        for b in tier_boundaries:
            ax.axvline(x=b, color='black', linestyle=':', linewidth=0.5, alpha=0.5)

    fig.suptitle('A-R3b: Probe Complexity Ladder', fontsize=13)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = save_dir / f'a_r3b_probe_ladder.{ext}'
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved: {path}")

    return fig


# ============================================================
# Figure 3: Diagnosis Heatmap
# ============================================================

def plot_diagnosis_heatmap(results_df, save_dir=None, verbose=True):
    """Heatmap of diagnoses: rows=targets, columns=architectures.

    Uses the best probe type (highest delta_R²) for each cell.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have 'diagnosis' column.
    save_dir : Path, optional
    verbose : bool

    Returns
    -------
    fig : matplotlib.Figure
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.copy()
    if 'diagnosis' not in df.columns:
        from a_r3b_reanalysis.phase5_diagnostics import classify_all_results
        df = classify_all_results(df)

    # For each (arch, target), pick the best probe result per neuron,
    # then take the most common diagnosis across neurons
    best_per_neuron = (
        df.sort_values('delta_r2', ascending=False)
        .groupby(['architecture', 'target', 'neuron_idx'])
        .first()
        .reset_index()
    )

    pivot_data = (
        best_per_neuron.groupby(['architecture', 'target'])['diagnosis']
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    architectures = sorted(pivot_data['architecture'].unique())
    targets = sorted(
        pivot_data['target'].unique(),
        key=lambda t: (TARGET_REGISTRY.get(t, {}).get('tier', 99),
                       TARGET_REGISTRY.get(t, {}).get('timescale_ms', 0))
    )

    # Build matrix
    diag_to_num = {d: i for i, d in enumerate(DIAGNOSIS_COLORS.keys())}
    matrix = np.full((len(targets), len(architectures)), np.nan)

    for _, row in pivot_data.iterrows():
        if row['target'] in targets and row['architecture'] in architectures:
            ti = targets.index(row['target'])
            ai = architectures.index(row['architecture'])
            matrix[ti, ai] = diag_to_num.get(row['diagnosis'], -1)

    # Custom colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = list(DIAGNOSIS_COLORS.values())
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(3 + 1.5 * len(architectures),
                                     1 + 0.4 * len(targets)))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(architectures)))
    ax.set_xticklabels([a.upper() for a in architectures], fontsize=9)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=7)
    ax.set_title('A-R3b: Diagnostic Classification per Target')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=name)
        for name, color in DIAGNOSIS_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=7)

    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = save_dir / f'a_r3b_diagnosis_heatmap.{ext}'
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved: {path}")

    return fig


def generate_all_plots(results_df, save_dir=None, verbose=True):
    """Generate all A-R3b figures."""
    figs = {}
    figs['r2_timescale'] = plot_r2_vs_timescale(results_df, save_dir, verbose)
    figs['probe_ladder'] = plot_probe_ladder(results_df, save_dir, verbose)
    figs['diagnosis_heatmap'] = plot_diagnosis_heatmap(results_df, save_dir, verbose)
    plt.close('all')
    return figs
