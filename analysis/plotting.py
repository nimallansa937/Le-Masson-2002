"""
Plotting utilities for Le Masson replication figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_voltage_traces(results, duration_s=None, title='', save_name=None):
    """Plot TC and nRt voltage traces with retinal spike raster.

    Parameters
    ----------
    results : dict
        Output from ThalamicCircuit.simulate().
    duration_s : float, optional
        Only plot first N seconds.
    """
    ensure_figures_dir()
    t = results['t']
    V_tc = results['V_tc']
    V_nrt = results['V_nrt']
    ret_spikes = results['retinal_spike_times']

    if duration_s is not None:
        mask = t <= duration_s
        t = t[mask]
        V_tc = V_tc[mask]
        V_nrt = V_nrt[mask]
        ret_spikes = ret_spikes[ret_spikes <= duration_s]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Retinal input raster
    axes[0].eventplot([ret_spikes], lineoffsets=0, linelengths=1, color='green')
    axes[0].set_ylabel('Retinal')
    axes[0].set_yticks([])
    axes[0].set_title(title if title else 'Thalamic Circuit Simulation')

    # TC neuron
    axes[1].plot(t, V_tc, 'b-', linewidth=0.5)
    axes[1].set_ylabel('TC (mV)')
    axes[1].set_ylim([-100, 40])

    # nRt neuron
    axes[2].plot(t, V_nrt, 'r-', linewidth=0.5)
    axes[2].set_ylabel('nRt (mV)')
    axes[2].set_ylim([-100, 40])
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_bifurcation(gaba_values, metric_values, metric_name='Oscillation Power',
                     threshold_line=None, save_name=None):
    """Plot bifurcation diagram.

    Parameters
    ----------
    gaba_values : array-like
        GABA G_max values (nS).
    metric_values : array-like
        Corresponding circuit metric (power, CI, binary, etc.).
    threshold_line : float, optional
        Draw vertical line at this GABA value.
    """
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gaba_values, metric_values, 'ko-', markersize=4)
    ax.set_xlabel('GABA G_max (nS)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title('Bifurcation Diagram — Retinothalamic Circuit', fontsize=14)

    if threshold_line is not None:
        ax.axvline(threshold_line, color='red', linestyle='--', linewidth=1.5,
                   label=f'Le Masson threshold = {threshold_line} nS')
        ax.legend()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_cross_correlogram(bins_ms, counts, title='', save_name=None):
    """Plot a cross-correlogram."""
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bins_ms, counts, width=bins_ms[1]-bins_ms[0] if len(bins_ms) > 1 else 2,
           color='steelblue', edgecolor='navy', linewidth=0.3)
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_ci_cc_vs_gaba(gaba_values, ci_values, cc_values, save_name=None):
    """Plot CI and CC vs GABA G_max (Fig 3d,e equivalent)."""
    ensure_figures_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(gaba_values, ci_values, 'bo-', markersize=6)
    ax1.set_xlabel('GABA G_max (nS)')
    ax1.set_ylabel('Contribution Index (CI)')
    ax1.set_title('CI vs Inhibition')
    ax1.grid(True, alpha=0.3)

    ax2.plot(gaba_values, cc_values, 'ro-', markersize=6)
    ax2.set_xlabel('GABA G_max (nS)')
    ax2.set_ylabel('Correlation Index (CC)')
    ax2.set_title('CC vs Inhibition')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
