"""
Layer 1: Biological Variable Recovery Space

The ground truth from A-R2 provides 160 biological intermediate variables:
- 20 TC neurons x (m_T, h_T, m_H) = 60 gating variables
- 20 nRt neurons x (V, m_T, h_T) = 60 nRt state variables
- 20 TC synapses x (GABA_A, GABA_B) = 40 synaptic conductances

Each architecture attempt produces a 160-dimensional binary recovery vector:
  recovery[i] = 1 if variable i was recovered (r > threshold), 0 otherwise

The GAP is the set of unrecovered variables. The gap DIRECTION tells us
which TYPES of variables are missed — and therefore which architectural
features might help recover them.

KEY INSIGHT: The Volterra already recovered 89/160. The gap is the
remaining 71. Any new architecture needs to target those 71 specifically.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform


# ============================================================
# BIOLOGICAL VARIABLE REGISTRY
# ============================================================

@dataclass
class BiologicalVariable:
    """One of the 160 ground truth variables."""
    id: int
    name: str                    # e.g., "tc_mT_0" (TC neuron 0, T-current activation)
    category: str                # "tc_gating", "nrt_state", "synaptic"
    subcategory: str             # "mT", "hT", "mH", "V", "gaba_a", "gaba_b"
    neuron_index: int            # Which neuron (0-19)
    timescale: str               # "fast" (<5ms), "medium" (5-50ms), "slow" (>50ms)
    dynamics_type: str           # "monotonic", "oscillatory", "switching"


def build_biovar_registry() -> List[BiologicalVariable]:
    """
    Build the complete registry of 160 biological variables.

    These correspond to the ground truth intermediates recorded
    in A-R2 Phase 0 (transformation_replacement_guide.md, Phase 0).

    IMPORTANT: Variable names use the EXACT keys from the HDF5 data loader:
      tc_m_T, tc_h_T, tc_m_h    (TC gating, stored in bio_gt as (20, T))
      V_nrt, nrt_m_Ts, nrt_h_Ts (nRt state, stored in bio_gt as (20, T))
      gabaa_per_tc, gabab_per_tc (synaptic, stored in bio_gt as (20, T))

    Each variable is named as "{gt_key}_{neuron_index}" so that
    _align_and_stack can split on the LAST underscore to recover the
    ground truth key and neuron index.
    """
    registry = []
    idx = 0

    # TC gating variables (60 total)
    # gt_key -> (subcategory, timescale, dynamics_type)
    tc_gating_vars = [
        ("tc_m_T", "mT", "fast", "switching"),      # T-current activation: fast, binary-like
        ("tc_h_T", "hT", "slow", "oscillatory"),     # T-current inactivation: slow, oscillates in spindles
        ("tc_m_h", "mH", "slow", "monotonic"),       # H-current: slow, monotonic ramp
    ]
    for neuron in range(20):
        for gt_key, subcat, ts, dyn in tc_gating_vars:
            registry.append(BiologicalVariable(
                id=idx, name=f"{gt_key}:{neuron}",
                category="tc_gating", subcategory=subcat,
                neuron_index=neuron, timescale=ts, dynamics_type=dyn
            ))
            idx += 1

    # nRt state variables (60 total)
    nrt_state_vars = [
        ("V_nrt", "V", "fast", "oscillatory"),       # nRt voltage: fast, oscillatory
        ("nrt_m_Ts", "mT", "fast", "switching"),     # nRt T-current activation
        ("nrt_h_Ts", "hT", "slow", "oscillatory"),   # nRt T-current inactivation
    ]
    for neuron in range(20):
        for gt_key, subcat, ts, dyn in nrt_state_vars:
            registry.append(BiologicalVariable(
                id=idx, name=f"{gt_key}:{neuron}",
                category="nrt_state", subcategory=subcat,
                neuron_index=neuron, timescale=ts, dynamics_type=dyn
            ))
            idx += 1

    # Synaptic conductances (40 total)
    synaptic_vars = [
        ("gabaa_per_tc", "gaba_a", "fast", "switching"),   # GABA_A: fast inhibition
        ("gabab_per_tc", "gaba_b", "slow", "monotonic"),   # GABA_B: slow inhibition
    ]
    for synapse in range(20):
        for gt_key, subcat, ts, dyn in synaptic_vars:
            registry.append(BiologicalVariable(
                id=idx, name=f"{gt_key}:{synapse}",
                category="synaptic", subcategory=subcat,
                neuron_index=synapse, timescale=ts, dynamics_type=dyn
            ))
            idx += 1

    assert len(registry) == 160, f"Expected 160 variables, got {len(registry)}"
    return registry


# ============================================================
# RECOVERY SCORING
# ============================================================

@dataclass
class RecoveryResult:
    """Result of checking one architecture against all 160 bio variables."""
    architecture_id: str
    recovery_vector: np.ndarray          # (160,) binary: 1=recovered, 0=not
    correlation_vector: np.ndarray       # (160,) continuous: best |r| per variable
    best_latent_mapping: Dict[int, int]  # bio_var_id -> latent_dim_id
    cca_score: float                     # Aggregate CCA score
    n_recovered: int                     # Count of recovered variables
    recovered_by_category: Dict[str, int]  # Category -> count
    recovered_by_timescale: Dict[str, int] # Timescale -> count


def score_biovar_recovery(
    model_latents: np.ndarray,       # (latent_dim, T)
    bio_ground_truth: Dict[str, np.ndarray],  # name -> (n_neurons, T_sub)
    registry: List[BiologicalVariable],
    recovery_threshold: float = 0.5  # |r| > 0.5 = "recovered"
) -> RecoveryResult:
    """
    Score how many of 160 biological variables are recovered by model latents.

    For each biological variable, find the model latent dimension with
    highest absolute Pearson correlation. If |r| > threshold, the variable
    is "recovered" — the model has spontaneously learned a representation
    that tracks this biological quantity.

    This is the core measurement from A-R3 Phase 7
    (transformation_replacement_guide.md, Phase 7).
    """
    n_bio = len(registry)
    n_latent = model_latents.shape[0]

    # Align temporal resolution (bio may be at 0.1ms, model at 1.0ms)
    # Downsample bio to match model timesteps
    bio_matrix = _align_and_stack(bio_ground_truth, registry, model_latents.shape[1])

    recovery_vector = np.zeros(n_bio)
    correlation_vector = np.zeros(n_bio)
    best_mapping = {}

    # Pre-check: skip bio variables that are constant (zero-variance)
    bio_std = np.std(bio_matrix, axis=1)
    latent_std = np.std(model_latents, axis=1)
    n_const_bio = np.sum(bio_std < 1e-10)
    n_const_latent = np.sum(latent_std < 1e-10)
    if n_const_bio > 0:
        print(f"  [biovar] WARNING: {n_const_bio}/160 bio variables are constant "
              f"(not matched to ground truth)")
    if n_const_latent > 0:
        print(f"  [biovar] WARNING: {n_const_latent}/{n_latent} latent dims are constant")

    for i in range(n_bio):
        if bio_std[i] < 1e-10:
            # Bio variable not matched to ground truth — skip
            continue
        best_r = 0.0
        best_j = -1
        for j in range(n_latent):
            if latent_std[j] < 1e-10:
                continue  # Skip constant latent dims
            r, _ = pearsonr(bio_matrix[i], model_latents[j])
            if np.isnan(r):
                continue
            if abs(r) > abs(best_r):
                best_r = r
                best_j = j

        correlation_vector[i] = abs(best_r)
        recovery_vector[i] = 1.0 if abs(best_r) > recovery_threshold else 0.0
        if abs(best_r) > recovery_threshold:
            best_mapping[i] = best_j

    n_recovered = int(recovery_vector.sum())

    # Category breakdown
    by_category = {}
    by_timescale = {}
    for var in registry:
        cat = var.category
        ts = var.timescale
        if cat not in by_category:
            by_category[cat] = 0
        if ts not in by_timescale:
            by_timescale[ts] = 0
        if recovery_vector[var.id] == 1.0:
            by_category[cat] += 1
            by_timescale[ts] += 1

    # Aggregate CCA
    cca_score = _compute_aggregate_cca(model_latents, bio_matrix)

    return RecoveryResult(
        architecture_id="",  # Set by caller
        recovery_vector=recovery_vector,
        correlation_vector=correlation_vector,
        best_latent_mapping=best_mapping,
        cca_score=cca_score,
        n_recovered=n_recovered,
        recovered_by_category=by_category,
        recovered_by_timescale=by_timescale
    )


# ============================================================
# GAP ANALYSIS
# ============================================================

@dataclass
class BioVarGap:
    """The gap between current recovery and full biological ground truth."""
    missing_variables: List[int]         # IDs of unrecovered variables
    missing_by_category: Dict[str, List[int]]  # Category -> missing var IDs
    missing_by_timescale: Dict[str, List[int]]  # Timescale -> missing var IDs
    gap_profile: Dict[str, float]        # Summary: category -> fraction missing
    gap_direction: str                   # Human-readable: what to target next
    residual_magnitude: float            # 0 = all recovered, 1 = none recovered


class BioVarRecoverySpace:
    """
    The concept vector space for Neural ODE architecture search.

    Each architecture's recovery result is a 160-dim binary vector.
    The UNION of all recovered variables across architectures = coverage.
    The COMPLEMENT of the union = the gap.
    The gap's COMPOSITION (which categories, timescales) = the direction.
    """

    def __init__(self, registry: List[BiologicalVariable]):
        self.registry = registry
        self.n_vars = len(registry)
        self.recovery_results: Dict[str, RecoveryResult] = {}
        self.union_recovery = np.zeros(self.n_vars)  # Union across all architectures

    def add_result(self, arch_id: str, result: RecoveryResult):
        """Record a recovery result and update union coverage."""
        result.architecture_id = arch_id
        self.recovery_results[arch_id] = result
        self.union_recovery = np.maximum(self.union_recovery, result.recovery_vector)

    def compute_gap(self) -> BioVarGap:
        """
        Compute the current gap: which biological variables has
        NO architecture recovered yet?

        This is the DESCARTES gap residual — the component of the
        target (all 160 variables) that lies outside the span of
        all attempted architectures.
        """
        missing = [i for i in range(self.n_vars) if self.union_recovery[i] == 0]

        by_cat = {}
        by_ts = {}
        cat_totals = {}
        ts_totals = {}

        for var in self.registry:
            cat = var.category
            ts = var.timescale
            cat_totals[cat] = cat_totals.get(cat, 0) + 1
            ts_totals[ts] = ts_totals.get(ts, 0) + 1

            if var.id in missing:
                by_cat.setdefault(cat, []).append(var.id)
                by_ts.setdefault(ts, []).append(var.id)

        # Gap profile: fraction missing per category
        gap_profile = {
            cat: len(by_cat.get(cat, [])) / cat_totals[cat]
            for cat in cat_totals
        }

        # Gap direction: which category has highest fraction missing?
        worst_category = max(gap_profile, key=gap_profile.get) if gap_profile else "none"
        worst_timescale = max(
            {ts: len(by_ts.get(ts, [])) / ts_totals[ts] for ts in ts_totals},
            key=lambda ts: len(by_ts.get(ts, [])) / ts_totals[ts]
        ) if ts_totals else "none"

        direction = (
            f"Highest gap in {worst_category} variables "
            f"({gap_profile.get(worst_category, 0):.0%} missing) "
            f"especially {worst_timescale}-timescale dynamics. "
            f"Need architecture that captures {worst_timescale} "
            f"{worst_category} behavior."
        )

        return BioVarGap(
            missing_variables=missing,
            missing_by_category=by_cat,
            missing_by_timescale=by_ts,
            gap_profile=gap_profile,
            gap_direction=direction,
            residual_magnitude=len(missing) / self.n_vars
        )

    def get_architecture_comparison(self) -> Dict[str, Dict]:
        """
        Compare all attempted architectures side by side.
        This is the data that DreamCoder analyzes for patterns.
        """
        comparison = {}
        for arch_id, result in self.recovery_results.items():
            comparison[arch_id] = {
                'n_recovered': result.n_recovered,
                'cca_score': result.cca_score,
                'by_category': result.recovered_by_category,
                'by_timescale': result.recovered_by_timescale,
                'correlation_vector': result.correlation_vector,
            }
        return comparison


def _align_and_stack(bio_gt, registry, target_T):
    """
    Align biological ground truth to model temporal resolution.

    Registry names use format "gt_key:neuron_idx" (e.g., "tc_m_T:5").
    The gt_key maps directly to bio_gt dict keys.
    """
    from scipy.interpolate import interp1d

    bio_matrix = np.zeros((len(registry), target_T))
    matched = 0
    unmatched_keys = set()

    for var in registry:
        # Split on ':' separator — gt_key:neuron_idx
        if ':' in var.name:
            gt_key, neuron_str = var.name.rsplit(':', 1)
            neuron_idx = int(neuron_str)
        else:
            # Legacy fallback: split on last underscore
            parts = var.name.rsplit('_', 1)
            gt_key = parts[0]
            neuron_idx = int(parts[1])

        if gt_key in bio_gt:
            raw = bio_gt[gt_key][neuron_idx]
            # Resample to target_T
            x_old = np.linspace(0, 1, len(raw))
            x_new = np.linspace(0, 1, target_T)
            bio_matrix[var.id] = interp1d(x_old, raw, kind='linear')(x_new)
            matched += 1
        else:
            unmatched_keys.add(gt_key)

    if unmatched_keys:
        print(f"  [biovar] WARNING: {len(unmatched_keys)} GT keys not found: "
              f"{sorted(unmatched_keys)[:5]}")
        print(f"  [biovar] Available GT keys: {sorted(bio_gt.keys())[:10]}...")
    print(f"  [biovar] Matched {matched}/160 variables to ground truth "
          f"(target_T={target_T})")

    return bio_matrix


def _compute_aggregate_cca(latents, bio_matrix, n_components=10):
    """Aggregate CCA score between model latents and bio variables."""
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(latents.T)  # (T, latent_dim)
    Y = StandardScaler().fit_transform(bio_matrix.T)  # (T, 160)
    n_comp = min(n_components, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_comp)
    try:
        X_c, Y_c = cca.fit_transform(X, Y)
        correlations = [pearsonr(X_c[:, i], Y_c[:, i])[0] for i in range(n_comp)]
        return float(np.mean(correlations))
    except Exception:
        return 0.0
