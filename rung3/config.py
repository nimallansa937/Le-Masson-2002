"""
Rung 3 Configuration — All hyperparameters in one place.

Organized by phase:
  - Data generation (Phase 0)
  - Preprocessing
  - Model architectures
  - Training
  - Evaluation
"""

import numpy as np

# =============================================================================
# Phase 0: Data Generation
# =============================================================================

DATA_DIR = "rung3_data"          # HDF5 output directory
FIGURE_DIR = "rung3_figures"     # Figure output directory
CHECKPOINT_DIR = "rung3_checkpoints"  # Model checkpoint directory

# Circuit parameters
N_TC = 20
N_NRT = 20
RETINAL_RATE = 42.0          # Hz
GAMMA_ORDER = 1.5
DT = 0.025e-3                # 25 µs integration step
RECORD_DT = 0.001            # 1 ms recording step (1 kHz)
DURATION_S = 10.0            # 10s per trial

# GABA sweep: 0 to 74 nS in steps of 2 nS
GABA_VALUES = np.arange(0, 75, 2).astype(float)  # 38 values

# Seeds: train on 42-44, validate on 45-46
TRAIN_SEEDS = [42, 43, 44]
VAL_SEEDS = [45, 46]
ALL_SEEDS = TRAIN_SEEDS + VAL_SEEDS

# Total trials: 38 GABA values × 5 seeds = 190 trials
# Each trial ≈ 107 MB → ~20 GB total

# =============================================================================
# Preprocessing
# =============================================================================

BIN_DT_MS = 1.0             # 1 ms bins for spike trains
SMOOTH_SIGMA_MS = 5.0       # Gaussian smoothing σ
WINDOW_SIZE_MS = 2000        # 2s windows
WINDOW_STRIDE_MS = 500       # 500ms stride
INPUT_DIM = 21               # 20 retinal channels + 1 GABA scalar
OUTPUT_DIM = 20              # 20 TC smoothed rates

# =============================================================================
# Model Architectures
# =============================================================================

# --- Volterra-Laguerre ---
VOLTERRA_N_BASES = 12        # Number of Laguerre basis functions
VOLTERRA_ALPHA = 0.85        # Laguerre parameter (memory decay)
VOLTERRA_MEMORY_MS = 200     # Memory depth in ms
VOLTERRA_ORDER = 2           # Max nonlinear order (diagonal 2nd order)
VOLTERRA_RIDGE_ALPHA = 100.0  # Ridge regularization (high for 625-feature space)
VOLTERRA_OUTPUT_FEEDBACK = True  # Include output feedback
VOLTERRA_FB_N_BASES = 6      # Feedback Laguerre bases (fewer than forward)

# --- LSTM ---
LSTM_HIDDEN_SIZE = 128       # Hidden state size per layer
LSTM_NUM_LAYERS = 2          # Number of stacked LSTM layers
LSTM_DROPOUT = 0.1           # Dropout between layers
LSTM_BIDIRECTIONAL = False

# --- Neural ODE ---
NODE_LATENT_DIM = 64         # ODE latent state dimension
NODE_HIDDEN_DIM = 128        # Hidden layer size in ODE function
NODE_N_HIDDEN = 2            # Number of hidden layers in f(z,u)
NODE_SOLVER = 'dopri5'       # ODE solver
NODE_RTOL = 1e-3             # Relative tolerance
NODE_ATOL = 1e-4             # Absolute tolerance
NODE_ADJOINT = True          # Use adjoint method for memory efficiency

# =============================================================================
# Training
# =============================================================================

# Loss weights
LOSS_BCE_WEIGHT = 0.7        # Binary cross-entropy on spike bins
LOSS_MSE_WEIGHT = 0.3        # MSE on smoothed rates

# Optimizer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4          # AdamW weight decay
GRAD_CLIP_NORM = 1.0         # Gradient clipping max norm

# Scheduler
SCHEDULER_PATIENCE = 5       # ReduceLROnPlateau patience
SCHEDULER_FACTOR = 0.5       # LR reduction factor
MIN_LR = 1e-6

# Early stopping
EARLY_STOP_PATIENCE = 15
MIN_DELTA = 1e-4

# Training loop
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 4              # DataLoader workers

# =============================================================================
# Evaluation
# =============================================================================

# Spike correlation
CORRELATION_SMOOTH_MS = 10.0  # Smoothing for rate correlation

# Victor-Purpura distance
VP_COST = 0.1                # Cost per unit time (1/ms) — controls sensitivity

# Bifurcation test
BIF_EC_FRAC = 0.5            # EC50 for threshold detection
BIF_N_BASELINE = 3           # Baseline points
BIF_N_CEILING = 5            # Ceiling points
BIF_TARGET_NS = 29.0         # Target threshold from Le Masson 2002
BIF_TARGET_SD = 4.2          # Target 1 SD

# =============================================================================
# Latent Comparison
# =============================================================================

# CCA
CCA_N_PERMUTATIONS = 1000    # Permutation test iterations
CCA_BLOCK_SIZE = 100         # Block bootstrap size (for autocorrelation)
CCA_MAX_COMPONENTS = 20      # Max CCA components to compute
CCA_PCA_COMPONENTS = 30      # PCA reduction before CCA (standard in neuro CCA)
CCA_MAX_ITER = 1000          # sklearn CCA NIPALS iterations (default 500 too low)

# RSA
RSA_N_TIMEPOINTS = 500       # Subsample timepoints for RSA (memory)
RSA_DISTANCE_METRIC = 'correlation'  # RDM distance metric

# Variable recovery
VAR_RECOVERY_ALPHA = 1.0     # Ridge regression alpha
VAR_RECOVERY_CV = 5          # Cross-validation folds
VAR_RECOVERY_BONFERRONI = True  # Bonferroni correction

# Biological intermediate dimensions
# TC: m_T (20), h_T (20), m_h (20) = 60
# nRt: m_Ts (20), h_Ts (20) = 40
# Synaptic: GABA_A_per_tc (20), GABA_B_per_tc (20), AMPA_per_nrt (20) = 60
# nRt voltages (20), TC voltages (20) = 40
# Total: 60 + 40 + 60 + 40 = 200 dimensions
N_BIO_DIMS = 200
