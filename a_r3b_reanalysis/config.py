"""
A-R3b Configuration — Target Registry, paths, and probe parameters.

Simulation parameters are loaded from default_params.json (NOT hardcoded)
to ensure identifiable targets match the actual A-R2 simulations.
"""

import json
import os
import numpy as np
from pathlib import Path

# ============================================================
# Paths
# ============================================================

# Project root: le_masson_replication/
_THIS_DIR = Path(__file__).resolve().parent
_REPLICATION_DIR = _THIS_DIR.parent
_PROJECT_DIR = _REPLICATION_DIR.parent

# Data from A-R2 simulations (HDF5 trial files)
DATA_DIR = _PROJECT_DIR / "la massion check" / "rung3_data"

# Model checkpoints from A-R3 training
CHECKPOINT_DIR = _PROJECT_DIR / "la massion check" / "rung3_checkpoints"

# A-R3b output directories
OUTPUT_DIR = _THIS_DIR / "results"
TARGETS_DIR = _THIS_DIR / "data" / "a_r3b_targets"
HIDDEN_STATES_DIR = _THIS_DIR / "data" / "a_r3b_hidden_states"

# ============================================================
# Load Simulation Parameters from default_params.json
# ============================================================

_PARAMS_PATH = _REPLICATION_DIR / "params" / "default_params.json"

with open(_PARAMS_PATH, 'r') as f:
    _PARAMS = json.load(f)

# TC neuron ionic conductances (mS/cm²)
g_T_bar = _PARAMS['tc_neuron']['g_T']       # 2.0 mS/cm² (T-type calcium)
g_h_bar = _PARAMS['tc_neuron']['g_h']       # 0.05 mS/cm² (HCN)

# Reversal potentials (mV)
E_Ca = _PARAMS['tc_neuron']['E_Ca']         # 120.0 mV
E_h = _PARAMS['tc_neuron']['E_h']           # -40.0 mV

# Synaptic reversal potentials (mV)
E_GABA_A = _PARAMS['synapses']['gabaa_nrt_tc']['E_rev_mV']  # -90.0 mV
E_GABA_B = _PARAMS['synapses']['gabab_nrt_tc']['E_rev_mV']  # -110.0 mV

# ============================================================
# Data / Windowing (matching rung3 config)
# ============================================================

N_TC = 20
N_NRT = 20
BIN_DT_MS = 1.0
WINDOW_SIZE_MS = 2000
WINDOW_STRIDE_MS = 500
WINDOW_BINS = int(WINDOW_SIZE_MS / BIN_DT_MS)  # 2000

# Trial split (matching rung3)
TRAIN_SEEDS = [42, 43, 44]
VAL_SEEDS = [45, 46]

# ============================================================
# Probe Configuration
# ============================================================

# Ridge probe
RIDGE_ALPHAS = np.logspace(-3, 5, 20)

# MLP probe
MLP_HIDDEN_DIM = 64
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_EPOCHS = 200
MLP_PATIENCE = 20

# Cross-validation
CV_N_SPLITS = 5

# Temporal downsampling for probes: take every Nth bin before fitting.
# 1ms bins × 100 = 100ms resolution → 20 timepoints per 2000ms window.
# Shrinks probe matrix 100x. Scientifically valid: we test *whether*
# encoding exists, not waveform fidelity. Even the fastest target (m_T, τ~1ms)
# has its envelope captured at 100ms over 2s windows.
TEMPORAL_SUBSAMPLE = 100

# Selectivity (block permutation)
SELECTIVITY_N_PERMS_QUICK = 50
SELECTIVITY_N_PERMS_FINAL = 200

# ============================================================
# TARGET_REGISTRY
# ============================================================
# All 17 target variables organized by tier:
#   Tier 1: Identifiable combinations (calibration baseline)
#   Tier 2: Individual gates with nonlinear probes (original A-R3)
#   Tier 3: Shape-normalized gates (scale-ambiguity fix)
#
# 'hdf5_key' maps to the key inside the HDF5 intermediates group.
# 'compute' is 'direct' (already in HDF5) or 'derived' (needs computation).

TARGET_REGISTRY = {
    # --- Tier 1: Identifiable combinations ---
    'G_T': {
        'tier': 1,
        'group': 'tier1_identifiable',
        'timescale_ms': 5.0,
        'compute': 'derived',
        'formula': 'g_T_bar * m_T^2 * h_T',
        'description': 'Effective T-current conductance',
        'zombie_interpretation': 'Fundamental zombie test — failure here means '
                                 'no biophysical encoding at any level',
    },
    'G_h': {
        'tier': 1,
        'group': 'tier1_identifiable',
        'timescale_ms': 100.0,
        'compute': 'derived',
        'formula': 'g_h_bar * m_H',
        'description': 'Effective H-current conductance',
        'zombie_interpretation': 'Slowest identifiable target — should be most recoverable',
    },
    'I_T': {
        'tier': 1,
        'group': 'tier1_identifiable',
        'timescale_ms': 5.0,
        'compute': 'derived',
        'formula': 'G_T * (V - E_Ca)',
        'description': 'T-type calcium ionic current',
    },
    'I_h': {
        'tier': 1,
        'group': 'tier1_identifiable',
        'timescale_ms': 100.0,
        'compute': 'derived',
        'formula': 'G_h * (V - E_h)',
        'description': 'HCN ionic current',
    },
    'I_GABA_A': {
        'tier': 1,
        'group': 'tier1_identifiable',
        'timescale_ms': 5.0,
        'compute': 'derived',
        'formula': 'gaba_a * (V - E_GABA_A)',
        'description': 'GABA_A inhibitory current',
    },
    'I_GABA_B': {
        'tier': 1,
        'group': 'tier1_identifiable',
        'timescale_ms': 150.0,
        'compute': 'derived',
        'formula': 'gaba_b * (V - E_GABA_B)',
        'description': 'GABA_B inhibitory current (slowest synaptic target)',
    },

    # --- Tier 2: Individual gates (original A-R3 targets, now with MLP probes) ---
    'tc_m_T': {
        'tier': 2,
        'group': 'tier2_individual_gates',
        'timescale_ms': 1.0,
        'compute': 'direct',
        'hdf5_key': 'tc_m_T',
        'description': 'T-current activation gate (raw scale)',
    },
    'tc_h_T': {
        'tier': 2,
        'group': 'tier2_individual_gates',
        'timescale_ms': 20.0,
        'compute': 'direct',
        'hdf5_key': 'tc_h_T',
        'description': 'T-current inactivation gate (raw scale)',
    },
    'tc_m_h': {
        'tier': 2,
        'group': 'tier2_individual_gates',
        'timescale_ms': 100.0,
        'compute': 'direct',
        'hdf5_key': 'tc_m_h',
        'description': 'HCN activation gate (raw scale)',
    },

    # --- Tier 3: Shape-normalized gates (removes α-scaling ambiguity) ---
    'tc_m_T_zscore': {
        'tier': 3,
        'group': 'tier3_shape_normalized',
        'timescale_ms': 1.0,
        'compute': 'derived',
        'source_key': 'tc_m_T',
        'normalization': 'zscore',
        'description': 'T-current activation (z-scored — shape only)',
    },
    'tc_h_T_zscore': {
        'tier': 3,
        'group': 'tier3_shape_normalized',
        'timescale_ms': 20.0,
        'compute': 'derived',
        'source_key': 'tc_h_T',
        'normalization': 'zscore',
        'description': 'T-current inactivation (z-scored — shape only)',
    },
    'tc_m_h_zscore': {
        'tier': 3,
        'group': 'tier3_shape_normalized',
        'timescale_ms': 100.0,
        'compute': 'derived',
        'source_key': 'tc_m_h',
        'normalization': 'zscore',
        'description': 'HCN activation (z-scored — shape only)',
    },
    'tc_m_T_rank': {
        'tier': 3,
        'group': 'tier3_shape_normalized',
        'timescale_ms': 1.0,
        'compute': 'derived',
        'source_key': 'tc_m_T',
        'normalization': 'rank',
        'description': 'T-current activation (rank-transformed)',
    },
    'tc_h_T_rank': {
        'tier': 3,
        'group': 'tier3_shape_normalized',
        'timescale_ms': 20.0,
        'compute': 'derived',
        'source_key': 'tc_h_T',
        'normalization': 'rank',
        'description': 'T-current inactivation (rank-transformed)',
    },
    'tc_m_h_rank': {
        'tier': 3,
        'group': 'tier3_shape_normalized',
        'timescale_ms': 100.0,
        'compute': 'derived',
        'source_key': 'tc_m_h',
        'normalization': 'rank',
        'description': 'HCN activation (rank-transformed)',
    },

    # --- Existing synaptic conductances (already Tier 1 identifiable) ---
    'gabaa_per_tc': {
        'tier': 1,
        'group': 'existing_synaptic',
        'timescale_ms': 5.0,
        'compute': 'direct',
        'hdf5_key': 'gabaa_per_tc',
        'description': 'GABA_A synaptic conductance (original A-R3 target)',
    },
    'gabab_per_tc': {
        'tier': 1,
        'group': 'existing_synaptic',
        'timescale_ms': 150.0,
        'compute': 'direct',
        'hdf5_key': 'gabab_per_tc',
        'description': 'GABA_B synaptic conductance (original A-R3 target)',
    },
}

# Architecture names → checkpoint filenames
ARCHITECTURE_CHECKPOINTS = {
    'lstm': 'lstm_best.pt',
    'neural_ode': 'neural_ode_best.pt',
    'volterra': 'volterra_model.npz',
}

PROBE_TYPES = ['ridge', 'mlp_1', 'mlp_2']
