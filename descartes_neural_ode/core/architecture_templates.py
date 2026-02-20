"""
Layer 0: Architecture Template Space

Each template defines a family of Neural ODE variants.
Properties are orthogonal dimensions that can be combined.
Total search space = product of all property values.

The key insight from DESCARTES: we don't search randomly.
We use gap analysis to identify WHICH properties matter for
recovering WHICH biological variables.
"""
from dataclasses import dataclass, field
from typing import Literal, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


# ============================================================
# ARCHITECTURE PROPERTIES (orthogonal dimensions)
# ============================================================

class TimeHandling(Enum):
    """How the model handles continuous time — the core ODE choice."""
    STANDARD_ODE = "standard_ode"           # dz/dt = f(z,t) via torchdiffeq
    LTC = "ltc"                              # Liquid Time-Constant: τ(z)·dz/dt = f(z) - z
    NEURAL_CDE = "neural_cde"               # dz/dt = f(z)·dX/dt (controlled by input path)
    GRU_ODE = "gru_ode"                      # GRU gates applied to ODE dynamics
    COUPLED_OSCILLATOR = "coupled_oscillator" # coRNN: coupled second-order system
    STATE_SPACE = "state_space"              # S4/Mamba structured state space


class GradientStrategy(Enum):
    """How gradients flow through time — the reason standard ODE fails."""
    ADJOINT = "adjoint"                 # Standard adjoint method (memory efficient, gradient issues)
    DIRECT_BACKPROP = "direct"          # Direct backpropagation (memory heavy, better gradients)
    SEGMENTED = "segmented"             # Break into segments, reset gradient graph between
    DISTILLATION = "distillation"       # No backprop through ODE; use teacher signal
    SHOOTING = "shooting"               # Multiple shooting: optimize segment boundaries jointly


class LatentStructure(Enum):
    """Inductive bias on latent space structure."""
    UNCONSTRAINED = "unconstrained"     # Free latent dimensions
    BIOPHYSICAL = "biophysical"         # Constrained to mimic ion channel gating (0-1 range, time constants)
    OSCILLATORY = "oscillatory"         # Built-in oscillatory modes (relevant for spindle dynamics)
    HIERARCHICAL = "hierarchical"       # Factored into fast/slow subsystems
    SPARSE = "sparse"                   # L1-regularized for interpretability


class InputCoupling(Enum):
    """How external input (retinal spikes + GABA) enters the ODE."""
    ADDITIVE = "additive"               # dz/dt = f(z) + g(u)
    MULTIPLICATIVE = "multiplicative"   # dz/dt = f(z) * g(u)  (like synaptic conductance)
    CONTROLLED = "controlled"           # dz/dt = f(z) · dX/dt  (Neural CDE style)
    GATED = "gated"                     # dz/dt = σ(u) * f(z) + (1-σ(u)) * h(z,u)
    CONCATENATED = "concatenated"       # dz/dt = f([z; u])  (standard, used in failed A-R3)


class SolverChoice(Enum):
    """ODE solver — affects both accuracy and gradient flow."""
    DOPRI5 = "dopri5"                   # Adaptive RK45 (default, failed in A-R3)
    EULER = "euler"                     # Fixed-step Euler (crude but gradient-friendly)
    MIDPOINT = "midpoint"               # Fixed-step midpoint (better than Euler, still gradient-friendly)
    IMPLICIT_ADAMS = "implicit_adams"   # For stiff systems (TC neuron dynamics are stiff)
    TSIT5 = "tsit5"                     # Tsitouras 5th order (better error estimates)


# ============================================================
# TEMPLATE DEFINITION
# ============================================================

@dataclass
class ArchitectureTemplate:
    """
    A C1 template in DESCARTES terms.
    Defines a FAMILY of architectures (the C2 parameters
    are the specific hyperparameters and random seed).
    """
    id: str
    name: str
    description: str
    family: str  # For exhaustion tracking: "standard_ode", "ltc", "hybrid", etc.

    # Properties
    time_handling: TimeHandling
    gradient_strategy: GradientStrategy
    latent_structure: LatentStructure
    input_coupling: InputCoupling
    solver: SolverChoice

    # Hyperparameter ranges (C2 search space)
    latent_dim_range: Tuple[int, int] = (32, 256)
    hidden_dim_range: Tuple[int, int] = (64, 256)
    n_layers_range: Tuple[int, int] = (2, 4)
    lr_range: Tuple[float, float] = (1e-4, 1e-2)

    # Metadata
    complexity: int = 1  # 1=simplest, 5=most complex
    expected_training_hours: float = 1.0
    vector: Optional[np.ndarray] = None  # For concept space embedding

    # Tracking
    exhausted: bool = False
    attempts: int = 0
    best_spike_corr: float = 0.0
    best_biovar_recovery: int = 0  # out of 160

    def to_property_vector(self) -> np.ndarray:
        """
        Encode properties as a 6-dimensional vector.
        Used by ConceptVectorSpace for gap analysis.
        """
        return np.array([
            list(TimeHandling).index(self.time_handling),
            list(GradientStrategy).index(self.gradient_strategy),
            list(LatentStructure).index(self.latent_structure),
            list(InputCoupling).index(self.input_coupling),
            list(SolverChoice).index(self.solver),
            self.complexity
        ], dtype=np.float32)


# ============================================================
# PRE-DEFINED TEMPLATE LIBRARY
# ============================================================

def get_initial_templates() -> List[ArchitectureTemplate]:
    """
    Initial C1 library. These are the architectures to try BEFORE
    DreamCoder learns any patterns. Ordered by complexity.

    The standard Neural ODE (template 0) is included as baseline
    even though we know it fails — its failure pattern seeds
    DreamCoder's learning.
    """
    templates = [
        # T0: Known failure — standard Neural ODE from A-R3
        ArchitectureTemplate(
            id="standard_ode_baseline",
            name="Standard Neural ODE (A-R3 baseline)",
            description="dz/dt = f([z;u]), dopri5 solver, adjoint gradients. "
                        "Known failure: 0.012 spike correlation. Included to seed "
                        "DreamCoder with failure patterns.",
            family="standard_ode",
            time_handling=TimeHandling.STANDARD_ODE,
            gradient_strategy=GradientStrategy.ADJOINT,
            latent_structure=LatentStructure.UNCONSTRAINED,
            input_coupling=InputCoupling.CONCATENATED,
            solver=SolverChoice.DOPRI5,
            complexity=1,
            expected_training_hours=2.0,
            # Pre-fill with known A-R3 results
            exhausted=True,
            attempts=1,
            best_spike_corr=0.012,
            best_biovar_recovery=0
        ),

        # T1: Fix gradient strategy — segmented backprop
        ArchitectureTemplate(
            id="segmented_ode",
            name="Segmented Neural ODE",
            description="Same ODE but break 2000 steps into 50-step segments. "
                        "Reset gradient graph between segments. Gradients only need "
                        "to survive 50 steps instead of 2000.",
            family="segmented_ode",
            time_handling=TimeHandling.STANDARD_ODE,
            gradient_strategy=GradientStrategy.SEGMENTED,
            latent_structure=LatentStructure.UNCONSTRAINED,
            input_coupling=InputCoupling.CONCATENATED,
            solver=SolverChoice.DOPRI5,
            complexity=2,
            expected_training_hours=1.5
        ),

        # T2: Liquid Time-Constant Network
        ArchitectureTemplate(
            id="ltc_network",
            name="Liquid Time-Constant Network",
            description="tau(z)*dz/dt = f(z) - z. The time constant tau is itself a "
                        "learned function of state. This creates state-dependent "
                        "dynamics where the system can have fast responses in some "
                        "regions and slow in others — matching biological T-current "
                        "activation/inactivation time constants.",
            family="ltc",
            time_handling=TimeHandling.LTC,
            gradient_strategy=GradientStrategy.DIRECT_BACKPROP,
            latent_structure=LatentStructure.UNCONSTRAINED,
            input_coupling=InputCoupling.GATED,
            solver=SolverChoice.EULER,  # Fixed step for gradient flow
            complexity=2,
            expected_training_hours=1.0
        ),

        # T3: Neural CDE — input drives dynamics
        ArchitectureTemplate(
            id="neural_cde",
            name="Neural Controlled Differential Equation",
            description="dz/dt = f(z) * dX/dt. The dynamics are CONTROLLED by the "
                        "input derivative, not just modulated. This means the ODE "
                        "evolves only when input changes — natural for spike-driven "
                        "systems where input is sparse binary events.",
            family="neural_cde",
            time_handling=TimeHandling.NEURAL_CDE,
            gradient_strategy=GradientStrategy.ADJOINT,
            latent_structure=LatentStructure.UNCONSTRAINED,
            input_coupling=InputCoupling.CONTROLLED,
            solver=SolverChoice.DOPRI5,
            complexity=3,
            expected_training_hours=2.5
        ),

        # T4: Coupled Oscillatory RNN (coRNN)
        ArchitectureTemplate(
            id="coupled_oscillatory",
            name="Coupled Oscillatory RNN (coRNN)",
            description="Second-order system: d2z/dt2 + gamma*dz/dt + omega2*z = f(u). "
                        "Built-in oscillatory modes at frequency omega. The TC-nRt "
                        "circuit produces 7-10 Hz spindle oscillations — an "
                        "oscillatory inductive bias may help recover this.",
            family="oscillatory",
            time_handling=TimeHandling.COUPLED_OSCILLATOR,
            gradient_strategy=GradientStrategy.DIRECT_BACKPROP,
            latent_structure=LatentStructure.OSCILLATORY,
            input_coupling=InputCoupling.ADDITIVE,
            solver=SolverChoice.MIDPOINT,
            complexity=2,
            expected_training_hours=1.0
        ),

        # T5: GRU-ODE — discrete gates + continuous dynamics
        ArchitectureTemplate(
            id="gru_ode",
            name="GRU-ODE-Bayes",
            description="GRU gating mechanism applied to continuous ODE dynamics. "
                        "Reset and update gates control which latent dimensions "
                        "evolve and which are held. May recover ion channel gating "
                        "behavior where gates open/close on different timescales.",
            family="gru_ode",
            time_handling=TimeHandling.GRU_ODE,
            gradient_strategy=GradientStrategy.DIRECT_BACKPROP,
            latent_structure=LatentStructure.UNCONSTRAINED,
            input_coupling=InputCoupling.GATED,
            solver=SolverChoice.EULER,
            complexity=3,
            expected_training_hours=1.5
        ),

        # T6: Hybrid LSTM encoder + ODE dynamics
        ArchitectureTemplate(
            id="hybrid_lstm_ode",
            name="Hybrid LSTM->ODE",
            description="LSTM encodes input into latent state (gradient highways "
                        "solve vanishing gradient). ODE evolves latent dynamics "
                        "between observations. Combines LSTM's training stability "
                        "with ODE's continuous-time inductive bias.",
            family="hybrid",
            time_handling=TimeHandling.STANDARD_ODE,
            gradient_strategy=GradientStrategy.SEGMENTED,
            latent_structure=LatentStructure.HIERARCHICAL,
            input_coupling=InputCoupling.GATED,
            solver=SolverChoice.MIDPOINT,
            complexity=3,
            expected_training_hours=2.0
        ),

        # T7: Knowledge-distilled ODE (uses Volterra's 89 recovered vars as teacher)
        ArchitectureTemplate(
            id="volterra_distilled_ode",
            name="Volterra-Distilled Neural ODE",
            description="Train ODE not via gradient through time, but by matching "
                        "its latent dimensions to Volterra's Laguerre coefficients "
                        "that correspond to recovered biological variables. The "
                        "Volterra already found 89/160 — use those as teaching "
                        "signal. Bypasses gradient problem entirely.",
            family="distilled",
            time_handling=TimeHandling.STANDARD_ODE,
            gradient_strategy=GradientStrategy.DISTILLATION,
            latent_structure=LatentStructure.BIOPHYSICAL,
            input_coupling=InputCoupling.CONCATENATED,
            solver=SolverChoice.MIDPOINT,
            complexity=2,
            expected_training_hours=1.0
        ),

        # T8: S4/Mamba structured state space
        ArchitectureTemplate(
            id="s4_mamba",
            name="Structured State Space (S4/Mamba)",
            description="dx/dt = Ax + Bu, y = Cx + Du with structured A matrix "
                        "(HiPPO initialization). Linear recurrence enables O(n log n) "
                        "computation over long sequences. The A matrix's eigenvalues "
                        "may align with biological time constants.",
            family="state_space",
            time_handling=TimeHandling.STATE_SPACE,
            gradient_strategy=GradientStrategy.DIRECT_BACKPROP,
            latent_structure=LatentStructure.HIERARCHICAL,
            input_coupling=InputCoupling.ADDITIVE,
            solver=SolverChoice.EULER,  # Discretized SSM
            complexity=3,
            expected_training_hours=1.5
        ),
    ]

    # Assign concept space vectors
    for t in templates:
        t.vector = t.to_property_vector()

    return templates
