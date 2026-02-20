"""
Default configuration for DESCARTES-NeuralODE.
"""

# ============================================================
# DATA
# ============================================================
INPUT_DIM = 21     # 20 retinal + 1 GABA
OUTPUT_DIM = 20    # 20 TC neurons
N_NEURONS = 20
T_FULL = 2000      # Full window length (1ms bins)
T_SHORT = 50       # Short segment for verifier

# ============================================================
# SEARCH
# ============================================================
MAX_ITERATIONS = 20
TARGET_RECOVERY = 120       # out of 160 bio vars
TARGET_SPIKE_CORR = 0.5
MAX_HOURS_PER_MODEL = 2.0

# ============================================================
# TRAINING (per architecture)
# ============================================================
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 30
SCHEDULER_PATIENCE = 15
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
MIN_LR = 1e-6

# ============================================================
# VERIFIER
# ============================================================
VERIFY_EPOCHS = 20
VERIFY_LR = 1e-3
VERIFY_LOSS_THRESHOLD = 0.5    # Loss must decrease by 50%
VERIFY_CORR_THRESHOLD = 0.1    # Spike corr > 0.1

# ============================================================
# RECOVERY
# ============================================================
RECOVERY_THRESHOLD = 0.5       # |r| > 0.5 = "recovered"
CCA_N_COMPONENTS = 10

# ============================================================
# BALLOON
# ============================================================
MAX_BALLOON_EXPANSIONS = 5
