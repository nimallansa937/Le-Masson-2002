"""
DESCARTES-NeuralODE Orchestrator

Main loop:
  1. Select next architecture template (simplest untried)
  2. Short-segment verify (Z3-C1) — reject if 50-step dynamics unlearnable
  3. Full training + evaluation (Z3-C2) — train to convergence, score recovery
  4. Record attempt in DreamCoder (wake phase)
  5. Compute biovar recovery and gap analysis
  6. If all templates exhausted -> DreamCoder sleep phase -> Balloon Expansion
  7. Gap direction guides next template selection or generation
  8. Repeat until target recovery achieved or budget exhausted

Differs from standard NAS in three ways:
  - Tracks 160-dimensional recovery vector, not scalar metric
  - Learns FROM failures (DreamCoder patterns), not just best result
  - Gap DIRECTION guides search, not random/evolutionary selection
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import time

from core.architecture_templates import (
    ArchitectureTemplate, get_initial_templates,
    TimeHandling, GradientStrategy, LatentStructure,
    InputCoupling, SolverChoice
)
from core.biovar_recovery_space import (
    BioVarRecoverySpace, build_biovar_registry,
    score_biovar_recovery
)
from core.short_segment_verifier import ShortSegmentVerifier
from core.biovar_pattern_extractor import BioVarPatternExtractor
from core.full_training_pipeline import FullTrainingPipeline
from core.memory import MemoryLayer, MemoryEntry

# LLM integration (optional)
try:
    from core.llm_architect import LLMArchitect, IterationRecord, LLMSuggestion
    from core.dreamcoder_history import DreamCoderHistory
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class DescartesPipelineResult:
    """Final result of the DESCARTES search."""
    best_architecture: str
    best_spike_correlation: float
    best_biovar_recovery: int      # out of 160
    total_iterations: int
    balloon_expansions: int
    gap_remaining: float           # 0 = all recovered
    patterns_discovered: int
    iteration_log: List[Dict]


class DescartesNeuralODEOrchestrator:
    """
    Main DESCARTES loop for Neural ODE architecture search.

    Adapts the Balloon Principle + DreamCoder + Gap Analysis
    framework from Collatz ordering search to Neural ODE
    architecture search for biological variable recovery.
    """

    def __init__(
        self,
        train_data: Dict,           # A-R2 training data
        val_data: Dict,             # A-R2 validation data
        bio_ground_truth: Dict,     # Ground truth bio variables
        device: str = 'cuda',
        max_training_hours: float = 2.0,
        verbose: bool = True,
        use_llm: bool = False,
        api_key: str = None,
        max_llm_expansions: int = 10
    ):
        # Data
        self.train_data = train_data
        self.val_data = val_data
        self.bio_gt = bio_ground_truth
        self.device = device
        self.max_hours = max_training_hours
        self.verbose = verbose

        # Core components (DESCARTES layers)
        self.registry = build_biovar_registry()
        self.recovery_space = BioVarRecoverySpace(self.registry)
        self.verifier = ShortSegmentVerifier(train_data, device)
        self.dreamcoder = BioVarPatternExtractor(self.registry)
        self.trainer = FullTrainingPipeline(train_data, val_data, device, max_training_hours)
        self.memory = MemoryLayer()

        # Templates
        self.templates = get_initial_templates()
        self.exhausted_templates = set()
        self.exhausted_families = set()

        # State
        self.balloon_expansions = 0
        self.iteration_log = []

        # LLM integration
        self.use_llm = use_llm and LLM_AVAILABLE
        self.max_llm_expansions = max_llm_expansions

        if self.use_llm:
            self.llm_architect = LLMArchitect(api_key=api_key)

            # Build category and timescale index maps for DreamCoderHistory
            bio_var_names = [v.name for v in self.registry]
            bio_var_categories = {}
            bio_var_timescales = {}
            for v in self.registry:
                bio_var_categories.setdefault(v.category, []).append(v.id)
                bio_var_timescales.setdefault(v.timescale, []).append(v.id)

            self.dreamcoder_history = DreamCoderHistory(
                bio_var_names=bio_var_names,
                bio_var_categories=bio_var_categories,
                bio_var_timescales=bio_var_timescales
            )

            if self.verbose:
                print(f"  LLM-guided search ENABLED (max {max_llm_expansions} expansions)")
        else:
            self.llm_architect = None
            self.dreamcoder_history = None
            if use_llm and not LLM_AVAILABLE:
                print("  WARNING: --use-llm requested but anthropic package not installed")
                print("  Install with: pip install anthropic")

    def run(
        self,
        max_iterations: int = 20,
        target_recovery: int = 120,   # out of 160 — exceed Volterra's 89
        target_spike_corr: float = 0.5
    ) -> DescartesPipelineResult:
        """
        Main DESCARTES loop.

        Stops when:
          - Target recovery achieved (>= 120/160 bio vars)
          - Max iterations reached
          - All architectures exhausted and Balloon can't expand
        """
        if self.verbose:
            print("=" * 70)
            print("DESCARTES-NeuralODE: Gap-Guided Architecture Search")
            print("=" * 70)
            print(f"Target: {target_recovery}/160 bio vars, spike corr >= {target_spike_corr}")
            print(f"Templates: {len(self.templates)}")
            print(f"Budget: {max_iterations} iterations, {self.max_hours}h per training")
            if self.use_llm:
                print(f"LLM-guided: YES (max {self.max_llm_expansions} expansions)")
            print()

        best_arch = None
        best_spike = 0.0
        best_recovery = 0

        for iteration in range(max_iterations):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Iteration {iteration + 1}/{max_iterations}")
                print(f"{'=' * 60}")

            # -------- SELECT NEXT TEMPLATE --------
            available = [t for t in self.templates
                         if t.id not in self.exhausted_templates
                         and t.family not in self.exhausted_families]

            if not available:
                if self.verbose:
                    print("\nALL TEMPLATES EXHAUSTED — BALLOON EXPANSION")

                new_templates = []

                # Try LLM-guided expansion first (if enabled)
                if self.use_llm and self.llm_architect and self.llm_architect.available:
                    gap = self.recovery_space.compute_gap()
                    new_templates = self._balloon_expand_llm(gap)

                # Fall back to hardcoded expansion if LLM not available or failed
                if not new_templates:
                    # DreamCoder sleep phase
                    patterns = self.dreamcoder.sleep_phase_analyze()
                    library = self.dreamcoder.compress_library()

                    if self.verbose:
                        print(f"  Patterns discovered: {len(patterns)}")
                        for p in patterns:
                            print(f"    - [{p.pattern_type}] {p.description[:80]}...")

                    # Generate new templates from patterns
                    new_templates = self._balloon_expand(patterns, library)

                if not new_templates:
                    if self.verbose:
                        print("  Cannot expand further. Stopping.")
                    break

                self.templates.extend(new_templates)
                self.balloon_expansions += 1
                available = new_templates

                if self.verbose:
                    print(f"  Generated {len(new_templates)} new templates")
                    for t in new_templates:
                        print(f"    - {t.name}")

            # Select simplest available template
            available.sort(key=lambda t: t.complexity)
            template = available[0]

            if self.verbose:
                print(f"\nTrying: {template.name}")
                print(f"  Family: {template.family}")
                print(f"  Time: {template.time_handling.value}")
                print(f"  Gradient: {template.gradient_strategy.value}")

            # -------- SHORT-SEGMENT VERIFY (Z3-C1) --------
            model = self._build_model(template)
            if model is None:
                if self.verbose:
                    print(f"  X Failed to build model: implementation not found")
                self.exhausted_templates.add(template.id)
                self._record_iteration(iteration, template, None, None, "build_failed")
                continue

            verify_result = self.verifier.verify(model)

            if not verify_result.passed:
                if self.verbose:
                    print(f"  X Short-segment FAILED: {verify_result.failure_reason}")

                # LLM Decision Point 1: Ask LLM for a fix and retry ONCE
                retried_ok = False
                if (self.use_llm and self.llm_architect
                        and self.llm_architect.available):
                    retried_ok = self._llm_retry_verify(
                        model, template, verify_result
                    )

                if retried_ok:
                    # Retry succeeded — re-run verification to get updated result
                    verify_result = self._llm_last_verify_result
                    if self.verbose:
                        print(f"  OK LLM fix worked! Continuing to full training.")
                else:
                    self.exhausted_templates.add(template.id)
                    self.dreamcoder.record_attempt(
                        template, None, 0.0, float('inf'),
                        verify_result.training_time_seconds / 3600,
                        failed_early=True,
                        failure_reason=verify_result.failure_reason
                    )
                    self._record_for_llm(
                        template, None, None, verify_result,
                        failed=True,
                        failure_reason=f"verify_failed: {verify_result.failure_reason}"
                    )
                    self._record_iteration(iteration, template, verify_result, None,
                                           f"verify_failed: {verify_result.failure_reason}")
                    continue

            if self.verbose:
                print(f"  OK Short-segment passed (50-step corr: {verify_result.spike_correlation_50step:.3f})")

            # -------- FULL TRAINING (Z3-C2) --------
            if self.verbose:
                print(f"  Training full model (budget: {self.max_hours}h)...")

            model, train_result = self.trainer.train(model, template, verbose=self.verbose)
            spike_corr = train_result.spike_correlation
            bif_error = train_result.bifurcation_error

            if self.verbose:
                print(f"  Spike correlation: {spike_corr:.3f}")
                print(f"  Bifurcation error: {bif_error:.1f} nS")

            # -------- BIO VARIABLE RECOVERY --------
            latents = self._extract_latents(model, template)
            recovery = score_biovar_recovery(
                latents, self.bio_gt, self.registry
            )
            self.recovery_space.add_result(template.id, recovery)

            if self.verbose:
                print(f"  Bio vars recovered: {recovery.n_recovered}/160")
                print(f"  By category: {recovery.recovered_by_category}")
                print(f"  CCA score: {recovery.cca_score:.3f}")

                # Diagnostic: show top 5 correlations and 5 lowest
                corr = recovery.correlation_vector
                top5_idx = np.argsort(corr)[-5:][::-1]
                print(f"  Top 5 correlations:")
                for idx in top5_idx:
                    print(f"    {self.registry[idx].name}: r={corr[idx]:.4f}")
                bot5_idx = np.argsort(corr)[:5]
                print(f"  Bottom 5 correlations:")
                for idx in bot5_idx:
                    print(f"    {self.registry[idx].name}: r={corr[idx]:.4f}")
                n_nonzero = np.sum(corr > 0.01)
                n_near_miss = np.sum((corr > 0.3) & (corr < 0.5))
                print(f"  Non-zero correlations: {n_nonzero}/160, "
                      f"Near-miss (0.3-0.5): {n_near_miss}")

            # -------- LLM DECISION POINT 2: C2 HYPERPARAMETER TWEAK --------
            # If LLM enabled, ask if a C2 tweak could improve recovery. Retry ONCE.
            if (self.use_llm and self.llm_architect
                    and self.llm_architect.available
                    and recovery.n_recovered < target_recovery
                    and recovery.n_recovered > 0):
                improved = self._llm_retry_c2_tweak(
                    model, template, verify_result,
                    spike_corr, recovery, train_result
                )
                if improved:
                    # Use improved results
                    spike_corr = improved['spike_corr']
                    recovery = improved['recovery']
                    train_result = improved['train_result']
                    model = improved['model']
                    if self.verbose:
                        print(f"  * C2 tweak improved recovery: "
                              f"{recovery.n_recovered}/160")

            # -------- RECORD IN DREAMCODER --------
            self.dreamcoder.record_attempt(
                template, recovery, spike_corr, bif_error,
                train_result.training_hours
            )

            # -------- RECORD FOR LLM (if enabled) --------
            self._record_for_llm(template, recovery, train_result, verify_result)

            # -------- RECORD IN MEMORY --------
            self.memory.record(MemoryEntry(
                template_id=template.id,
                family=template.family,
                timestamp=time.time(),
                status="completed",
                spike_correlation=spike_corr,
                n_recovered=recovery.n_recovered,
                verify_passed=True,
                verify_time_seconds=verify_result.training_time_seconds,
                train_time_hours=train_result.training_hours,
                gap_at_time=self.recovery_space.compute_gap().residual_magnitude
            ))

            # -------- GAP ANALYSIS --------
            gap = self.recovery_space.compute_gap()

            if self.verbose:
                print(f"\n  Gap analysis:")
                print(f"    Remaining gap: {gap.residual_magnitude:.1%}")
                print(f"    Direction: {gap.gap_direction[:80]}...")
                print(f"    Profile: {gap.gap_profile}")

            # -------- LLM DECISION POINT 3: NEAR-MISS INTERPRETATION --------
            if (self.use_llm and self.llm_architect
                    and self.llm_architect.available
                    and hasattr(recovery, 'correlation_vector')):
                self._llm_interpret_near_misses(
                    template, recovery, gap
                )

            # -------- UPDATE BEST --------
            template.attempts += 1
            template.best_spike_corr = max(template.best_spike_corr, spike_corr)
            template.best_biovar_recovery = max(template.best_biovar_recovery,
                                                 recovery.n_recovered)

            if recovery.n_recovered > best_recovery:
                best_recovery = recovery.n_recovered
                best_arch = template.id
                best_spike = spike_corr
                if self.verbose:
                    print(f"\n  * NEW BEST: {template.name} ({recovery.n_recovered}/160)")

            # Mark exhausted
            self.exhausted_templates.add(template.id)

            # Check family exhaustion
            family_attempts = [t for t in self.templates
                              if t.family == template.family
                              and t.id in self.exhausted_templates]
            family_total = [t for t in self.templates if t.family == template.family]
            if len(family_attempts) == len(family_total):
                self.exhausted_families.add(template.family)
                if self.verbose:
                    print(f"  Family '{template.family}' now exhausted")

            # Log
            self._record_iteration(iteration, template, verify_result,
                                    recovery, "completed")

            # -------- CHECK STOPPING --------
            if recovery.n_recovered >= target_recovery and spike_corr >= target_spike_corr:
                if self.verbose:
                    print(f"\nTARGET ACHIEVED at iteration {iteration + 1}")
                break

        # Final DreamCoder analysis
        final_patterns = self.dreamcoder.sleep_phase_analyze()

        # Final LLM DreamCoder analysis (if enabled)
        if self.use_llm and self.dreamcoder_history:
            llm_patterns = self.dreamcoder_history.extract_patterns()
            if self.verbose and llm_patterns:
                print(f"\nFinal DreamCoder-LLM patterns ({len(llm_patterns)}):")
                for p in llm_patterns[:5]:
                    print(f"  [{p['type']}] {p['description'][:80]}...")

        return DescartesPipelineResult(
            best_architecture=best_arch or "none",
            best_spike_correlation=best_spike,
            best_biovar_recovery=best_recovery,
            total_iterations=len(self.iteration_log),
            balloon_expansions=self.balloon_expansions,
            gap_remaining=self.recovery_space.compute_gap().residual_magnitude,
            patterns_discovered=len(self.dreamcoder.patterns),
            iteration_log=self.iteration_log
        )

    def _build_model(self, template: ArchitectureTemplate) -> Optional[torch.nn.Module]:
        """
        Build a PyTorch model from an architecture template.

        Maps template properties to actual implementations in architectures/.
        """
        try:
            if template.time_handling == TimeHandling.STANDARD_ODE:
                from architectures.base_ode import TCReplacementNeuralODE
                return TCReplacementNeuralODE(
                    n_input=21, n_output=20,
                    latent_dim=template.latent_dim_range[0],
                    solver=template.solver.value
                )
            elif template.time_handling == TimeHandling.LTC:
                from architectures.ltc_network import LTCModel
                return LTCModel(n_input=21, n_output=20,
                               latent_dim=template.latent_dim_range[0])
            elif template.time_handling == TimeHandling.NEURAL_CDE:
                from architectures.neural_cde import NeuralCDEModel
                return NeuralCDEModel(n_input=21, n_output=20,
                                     latent_dim=template.latent_dim_range[0])
            elif template.time_handling == TimeHandling.COUPLED_OSCILLATOR:
                from architectures.coRNN import CoRNNModel
                return CoRNNModel(n_input=21, n_output=20,
                                 latent_dim=template.latent_dim_range[0])
            elif template.time_handling == TimeHandling.GRU_ODE:
                from architectures.gru_ode import GRUODEModel
                return GRUODEModel(n_input=21, n_output=20,
                                  latent_dim=template.latent_dim_range[0])
            elif template.time_handling == TimeHandling.STATE_SPACE:
                from architectures.s4_mamba import S4MambaModel
                return S4MambaModel(n_input=21, n_output=20,
                                   latent_dim=template.latent_dim_range[0])
        except ImportError as e:
            if self.verbose:
                print(f"  Architecture not implemented: {e}")
            return None
        return None

    def _extract_latents(self, model, template) -> np.ndarray:
        """
        Extract latent trajectories for bio variable comparison.

        Runs model on validation data and extracts internal states.
        Shape: (latent_dim, T)
        """
        model.eval()
        with torch.no_grad():
            # Use first validation window
            x_val_np = (self.val_data['X_val'][:1]
                        if isinstance(self.val_data['X_val'], np.ndarray)
                        else self.val_data['X_val'][:1].cpu().numpy())
            x_val = torch.tensor(x_val_np, device=self.device, dtype=torch.float32)

            output = model(x_val)

            if isinstance(output, tuple) and len(output) >= 2:
                latents = output[1]
                if isinstance(latents, torch.Tensor):
                    latents_np = latents.cpu().numpy()
                    # Shape: (1, T, latent_dim) -> (latent_dim, T)
                    result = latents_np[0].T
                elif isinstance(latents, dict) and 'hidden' in latents:
                    h = latents['hidden']
                    result = (h[0].T if isinstance(h, np.ndarray)
                              else h[0].cpu().numpy().T)
                else:
                    print(f"  [latent] WARNING: Unexpected latent type: {type(latents)}")
                    result = None
            elif isinstance(output, torch.Tensor):
                # Model only returned predictions, no latents
                print(f"  [latent] WARNING: Model returned only predictions "
                      f"(shape {output.shape}), no latent states")
                result = None
            else:
                print(f"  [latent] WARNING: Unexpected output type: {type(output)}")
                result = None

        if result is not None:
            # Diagnostics
            if self.verbose:
                print(f"  [latent] Extracted shape: {result.shape}")
                print(f"  [latent] Stats: mean={result.mean():.4f}, "
                      f"std={result.std():.4f}, "
                      f"min={result.min():.4f}, max={result.max():.4f}")
                # Check for constant/degenerate states
                per_dim_std = np.std(result, axis=1)
                n_constant = np.sum(per_dim_std < 1e-6)
                if n_constant > 0:
                    print(f"  [latent] WARNING: {n_constant}/{result.shape[0]} "
                          f"latent dims are near-constant (std < 1e-6)")
            return result

        # FAIL LOUD instead of silent random fallback
        latent_dim = template.latent_dim_range[0]
        T = self.val_data.get('T', 2000)
        print(f"  [latent] ERROR: Could not extract latents! "
              f"Using zeros ({latent_dim}, {T}) — bio recovery will be 0/160")
        return np.zeros((latent_dim, T))

    def _balloon_expand(self, patterns, library) -> List[ArchitectureTemplate]:
        """
        BALLOON PRINCIPLE: Generate new architecture templates
        from DreamCoder's failure patterns.

        This is where the DESCARTES innovation lives. Instead of
        random architecture search, we use structured failure analysis
        to generate targeted new architectures.
        """
        new_templates = []
        gap = self.recovery_space.compute_gap()

        for primitive in library.get('primitives', []):
            if primitive['type'] == 'property_correlation':
                prop = primitive.get('property')
                if prop == 'latent_structure' and 'hierarchical' in str(primitive.get('suggested_value', '')):
                    new_templates.append(ArchitectureTemplate(
                        id=f"balloon_{self.balloon_expansions}_hierarchical",
                        name=f"Balloon Expansion: Hierarchical Latent ODE",
                        description=(
                            f"Generated by DreamCoder pattern: {primitive['description'][:100]}. "
                            f"Uses hierarchical latent space with fast/slow subsystems."
                        ),
                        family=f"balloon_{self.balloon_expansions}",
                        time_handling=TimeHandling.LTC,
                        gradient_strategy=GradientStrategy.SEGMENTED,
                        latent_structure=LatentStructure.HIERARCHICAL,
                        input_coupling=InputCoupling.GATED,
                        solver=SolverChoice.MIDPOINT,
                        complexity=4,
                        expected_training_hours=2.0
                    ))

            elif primitive['type'] == 'near_miss':
                new_templates.append(ArchitectureTemplate(
                    id=f"balloon_{self.balloon_expansions}_nearmiss",
                    name=f"Balloon Expansion: Near-Miss Targeted",
                    description=(
                        f"Targets {len(primitive.get('affected_variables', []))} "
                        f"near-miss variables via auxiliary loss on their recovery."
                    ),
                    family=f"balloon_{self.balloon_expansions}",
                    time_handling=TimeHandling.GRU_ODE,
                    gradient_strategy=GradientStrategy.DISTILLATION,
                    latent_structure=LatentStructure.BIOPHYSICAL,
                    input_coupling=InputCoupling.MULTIPLICATIVE,
                    solver=SolverChoice.MIDPOINT,
                    complexity=4,
                    expected_training_hours=1.5
                ))

        # If gap analysis shows specific category dominance
        if gap.gap_profile:
            worst = max(gap.gap_profile, key=gap.gap_profile.get)
            if worst == 'nrt_state' and gap.gap_profile[worst] > 0.5:
                new_templates.append(ArchitectureTemplate(
                    id=f"balloon_{self.balloon_expansions}_nrt_targeted",
                    name=f"Balloon: nRt-Targeted Feedback ODE",
                    description=(
                        f"Gap analysis shows {gap.gap_profile[worst]:.0%} of nRt state "
                        f"variables unrecovered. Adding explicit feedback pathway "
                        f"in ODE to mimic TC->nRt->TC loop."
                    ),
                    family=f"balloon_{self.balloon_expansions}",
                    time_handling=TimeHandling.COUPLED_OSCILLATOR,
                    gradient_strategy=GradientStrategy.SEGMENTED,
                    latent_structure=LatentStructure.OSCILLATORY,
                    input_coupling=InputCoupling.MULTIPLICATIVE,
                    solver=SolverChoice.MIDPOINT,
                    complexity=4,
                    expected_training_hours=2.5
                ))

        # Assign vectors
        for t in new_templates:
            t.vector = t.to_property_vector()

        return new_templates

    # ================================================================
    # LLM INTEGRATION METHODS
    # ================================================================

    def _llm_retry_verify(self, model, template, verify_result) -> bool:
        """
        LLM Decision Point 1: After short-segment failure, ask LLM for a fix
        and retry verification ONCE.

        Returns True if retry succeeded (sets self._llm_last_verify_result).
        Returns False if LLM failed, fix didn't work, or retry also failed.
        """
        self._llm_last_verify_result = None

        template_config = {
            'time_handling': template.time_handling.value,
            'gradient_strategy': template.gradient_strategy.value,
            'latent_structure': template.latent_structure.value,
            'input_coupling': template.input_coupling.value,
            'solver': template.solver.value
        }

        if self.verbose:
            print(f"  -> Asking LLM for verification fix...")

        fix = self.llm_architect.suggest_verify_fix(
            template_config=template_config,
            failure_reason=verify_result.failure_reason or "unknown",
            short_segment_corr=(verify_result.spike_correlation_50step
                                if hasattr(verify_result, 'spike_correlation_50step')
                                else None),
            final_loss=verify_result.final_loss if hasattr(verify_result, 'final_loss') else None
        )

        if fix is None:
            if self.verbose:
                print(f"  -> LLM fix unavailable, skipping retry")
            return False

        if self.verbose:
            print(f"  -> LLM fix: [{fix.get('fix_type')}] {fix.get('description', '')}")

        # Apply the fix overrides to verification retry
        overrides = fix.get('overrides', {})
        retry_lr = overrides.get('lr', 1e-3)
        retry_epochs = overrides.get('n_epochs', 20)

        # Rebuild model if solver changed
        if 'solver' in overrides:
            # Temporarily modify template solver for model rebuild
            from core.architecture_templates import SolverChoice
            original_solver = template.solver
            try:
                template.solver = SolverChoice(overrides['solver'])
                model = self._build_model(template)
                template.solver = original_solver  # restore
            except (ValueError, KeyError):
                template.solver = original_solver
                model = self._build_model(template)

            if model is None:
                if self.verbose:
                    print(f"  -> Failed to rebuild model with new solver")
                return False
        else:
            # Rebuild fresh model (reset weights)
            model = self._build_model(template)
            if model is None:
                return False

        # Apply gradient clipping if suggested
        if 'grad_clip' in overrides:
            # Store for use during verify — the verifier handles this internally
            # We pass it through lr parameter (verifier's default grad clip is 1.0)
            pass  # grad_clip handled by verifier internally

        # Retry verification with LLM-suggested parameters
        try:
            retry_result = self.verifier.verify(
                model, lr=retry_lr, n_epochs=retry_epochs
            )
        except Exception as e:
            if self.verbose:
                print(f"  -> Retry verification raised exception: {e}")
            return False

        if retry_result.passed:
            self._llm_last_verify_result = retry_result
            return True
        else:
            if self.verbose:
                print(f"  -> LLM retry also failed: {retry_result.failure_reason}")
            return False

    def _llm_retry_c2_tweak(self, model, template, verify_result,
                            spike_corr, recovery, train_result) -> Optional[Dict]:
        """
        LLM Decision Point 2: After full training, ask LLM for a C2
        hyperparameter tweak and retrain ONCE.

        Returns dict with improved results if retry beat original,
        or None if no improvement.
        """
        template_config = {
            'time_handling': template.time_handling.value,
            'gradient_strategy': template.gradient_strategy.value,
            'latent_structure': template.latent_structure.value,
            'input_coupling': template.input_coupling.value,
            'solver': template.solver.value
        }

        # Get near-miss variable names
        near_misses = []
        if hasattr(recovery, 'correlation_vector'):
            near_miss_idx = np.where(
                (recovery.correlation_vector > 0.3) &
                (recovery.correlation_vector < 0.5)
            )[0]
            near_misses = [self.registry[i].name for i in near_miss_idx[:15]]

        if not near_misses:
            # No near-misses to optimize for
            return None

        if self.verbose:
            print(f"\n  -> Asking LLM for C2 tweak ({len(near_misses)} near-misses)...")

        tweak = self.llm_architect.suggest_c2_tweak(
            template_config=template_config,
            spike_correlation=spike_corr,
            bio_vars_recovered=recovery.n_recovered,
            recovery_by_category=recovery.recovered_by_category,
            near_misses=near_misses,
            training_hours=train_result.training_hours,
            epochs_completed=train_result.total_epochs,
            best_val_loss=train_result.best_val_loss,
            converged=train_result.converged
        )

        if tweak is None:
            if self.verbose:
                print(f"  -> LLM C2 tweak unavailable, keeping original result")
            return None

        if self.verbose:
            print(f"  -> LLM C2 tweak: [{tweak.get('tweak_type')}] "
                  f"{tweak.get('description', '')}")

        overrides = tweak.get('overrides', {})
        if not overrides:
            return None

        # Rebuild model if latent_dim changed
        retry_model = model
        if 'latent_dim' in overrides:
            new_dim = overrides['latent_dim']
            original_range = template.latent_dim_range
            template.latent_dim_range = (new_dim, new_dim * 4)
            retry_model = self._build_model(template)
            template.latent_dim_range = original_range  # restore
            if retry_model is None:
                if self.verbose:
                    print(f"  -> Failed to rebuild model with latent_dim={new_dim}")
                return None
        else:
            # Rebuild fresh model (new random init)
            retry_model = self._build_model(template)
            if retry_model is None:
                return None

        # Retrain with tweaked hyperparameters
        retry_lr = overrides.get('lr', 5e-4)
        retry_batch = overrides.get('batch_size', 32)
        retry_patience = overrides.get('patience', 30)
        retry_epochs = overrides.get('max_epochs', 300)

        if self.verbose:
            print(f"  -> Retraining with tweaked C2: lr={retry_lr}, "
                  f"batch={retry_batch}, patience={retry_patience}...")

        try:
            retry_model, retry_train = self.trainer.train(
                retry_model, template,
                lr=retry_lr,
                batch_size=retry_batch,
                max_epochs=retry_epochs,
                patience=retry_patience,
                verbose=self.verbose
            )
        except Exception as e:
            if self.verbose:
                print(f"  -> C2 retry training failed: {e}")
            return None

        # Score retry recovery
        try:
            retry_latents = self._extract_latents(retry_model, template)
            retry_recovery = score_biovar_recovery(
                retry_latents, self.bio_gt, self.registry
            )
        except Exception as e:
            if self.verbose:
                print(f"  -> C2 retry recovery scoring failed: {e}")
            return None

        retry_spike = retry_train.spike_correlation

        if self.verbose:
            print(f"  -> C2 retry: spike_corr={retry_spike:.3f}, "
                  f"bio_vars={retry_recovery.n_recovered}/160 "
                  f"(original: {spike_corr:.3f}, {recovery.n_recovered}/160)")

        # Keep retry only if it improved
        if retry_recovery.n_recovered > recovery.n_recovered:
            self.recovery_space.add_result(
                f"{template.id}_c2_retry", retry_recovery
            )
            return {
                'spike_corr': retry_spike,
                'recovery': retry_recovery,
                'train_result': retry_train,
                'model': retry_model
            }
        else:
            if self.verbose:
                print(f"  -> C2 retry did not improve, keeping original")
            return None

    def _llm_interpret_near_misses(self, template, recovery, gap):
        """
        LLM Decision Point 3: After gap analysis, ask LLM to interpret
        WHY near-miss variables didn't cross the threshold.

        Results are logged and stored for future balloon expansions.
        """
        # Build near-miss variable info
        near_miss_vars = []
        if hasattr(recovery, 'correlation_vector'):
            for i in range(len(recovery.correlation_vector)):
                r = recovery.correlation_vector[i]
                if 0.3 < r < 0.5:
                    var = self.registry[i]
                    near_miss_vars.append({
                        'name': var.name,
                        'correlation': float(r),
                        'category': var.category,
                        'timescale': var.timescale,
                        'dynamics_type': getattr(var, 'dynamics_type', 'unknown')
                    })

        if not near_miss_vars:
            return

        # Get DreamCoder patterns
        dc_patterns = []
        if self.dreamcoder_history:
            dc_patterns = self.dreamcoder_history.extract_patterns()

        best_config = {
            'time_handling': template.time_handling.value,
            'gradient_strategy': template.gradient_strategy.value,
            'latent_structure': template.latent_structure.value,
            'input_coupling': template.input_coupling.value,
            'solver': template.solver.value
        }

        if self.verbose:
            print(f"\n  -> Asking LLM to interpret {len(near_miss_vars)} near-misses...")

        interpretation = self.llm_architect.interpret_near_misses(
            near_miss_vars=near_miss_vars,
            best_architecture=best_config,
            gap_profile=gap.gap_profile if hasattr(gap, 'gap_profile') else {},
            patterns=dc_patterns
        )

        if interpretation is None:
            return

        if self.verbose:
            print(f"  -> LLM interpretation: {interpretation.get('interpretation', '')[:120]}...")
            mods = interpretation.get('suggested_modifications', [])
            if mods:
                for m in mods[:3]:
                    print(f"     Suggestion: {m.get('modification', '')[:80]}")
                    print(f"       Targets: {m.get('targets', '')}")

        # Store interpretation in the LLM architect's patterns for future use
        self.llm_architect.record_patterns(
            self.llm_architect.patterns + [{
                'type': 'llm_near_miss_interpretation',
                'description': interpretation.get('interpretation', ''),
                'evidence': json.dumps(interpretation.get('suggested_modifications', [])),
                'confidence': 0.8,
                'priority_variables': interpretation.get('priority_variables', [])
            }]
        )

    def _record_for_llm(self, template, recovery, train_result=None,
                        verify_result=None, failed=False, failure_reason=None):
        """Record iteration results for LLM architect and DreamCoder history."""
        if not self.use_llm:
            return

        # Build IterationRecord for LLM history
        record = IterationRecord(
            iteration=len(self.llm_architect.history) + 1,
            template_name=template.name,
            family=template.family,
            time_handling=template.time_handling.value,
            gradient_strategy=template.gradient_strategy.value,
            latent_structure=template.latent_structure.value,
            input_coupling=template.input_coupling.value,
            solver=template.solver.value,
            spike_correlation=train_result.spike_correlation if train_result else 0.0,
            bio_vars_recovered=recovery.n_recovered if recovery else 0,
            recovery_by_category=recovery.recovered_by_category if recovery else {},
            near_misses=[],  # populated below if available
            cca_score=recovery.cca_score if recovery else None,
            epochs_completed=getattr(train_result, 'epochs_completed', 0) if train_result else 0,
            final_train_loss=getattr(train_result, 'final_train_loss', None) if train_result else None,
            final_val_loss=getattr(train_result, 'final_val_loss', None) if train_result else None,
            training_time_hours=train_result.training_hours if train_result else None,
            short_segment_corr=(verify_result.spike_correlation_50step
                                if verify_result and hasattr(verify_result, 'spike_correlation_50step')
                                else None),
            failed=failed,
            failure_reason=failure_reason
        )

        # Extract near-miss variable names (0.3 < r < 0.5)
        if recovery and hasattr(recovery, 'correlation_vector'):
            near_miss_idx = np.where(
                (recovery.correlation_vector > 0.3) &
                (recovery.correlation_vector < 0.5)
            )[0]
            record.near_misses = [self.registry[i].name for i in near_miss_idx[:15]]

        self.llm_architect.record_iteration(record)

        # Record in DreamCoder history (needs recovery + correlation vectors)
        if recovery and hasattr(recovery, 'recovery_vector') and hasattr(recovery, 'correlation_vector'):
            self.dreamcoder_history.add_result(
                name=template.name,
                config={
                    'time_handling': template.time_handling.value,
                    'gradient_strategy': template.gradient_strategy.value,
                    'latent_structure': template.latent_structure.value,
                    'input_coupling': template.input_coupling.value,
                    'solver': template.solver.value
                },
                recovery_vector=recovery.recovery_vector,
                correlation_vector=recovery.correlation_vector
            )

    def _balloon_expand_llm(self, gap) -> List[ArchitectureTemplate]:
        """
        LLM-powered balloon expansion.

        Uses DreamCoder patterns + full history to ask the LLM for
        a new architecture template targeting the specific gap.
        """
        if self.llm_architect.balloon_count >= self.max_llm_expansions:
            if self.verbose:
                print(f"  Max LLM expansions ({self.max_llm_expansions}) reached")
            return []

        # Run DreamCoder sleep phase on accumulated history
        dc_patterns = self.dreamcoder_history.extract_patterns()
        self.llm_architect.record_patterns(dc_patterns)

        if self.verbose:
            print(f"  DreamCoder extracted {len(dc_patterns)} patterns:")
            for p in dc_patterns[:5]:
                print(f"    [{p['type']}] {p['description'][:80]}...")

        # Record exhausted families
        for fam in self.exhausted_families:
            self.llm_architect.record_exhausted_family(fam)

        # Format gap for LLM
        gap_dict = {
            'remaining_gap': getattr(gap, 'residual_magnitude', 1.0),
            'direction': getattr(gap, 'gap_direction', 'unknown'),
            'profile': getattr(gap, 'gap_profile', {}),
        }
        if hasattr(gap, 'timescale_profile'):
            gap_dict['timescale_profile'] = gap.timescale_profile
        if hasattr(gap, 'dynamics_profile'):
            gap_dict['dynamics_profile'] = gap.dynamics_profile

        # Ask LLM for suggestion
        if self.verbose:
            print(f"  Calling LLM for architecture suggestion...")

        suggestion = self.llm_architect.suggest_architecture(gap_dict)

        if suggestion is None:
            if self.verbose:
                print(f"  LLM suggestion failed, falling back to hardcoded expansion")
            return []

        # Convert LLM suggestion to ArchitectureTemplate
        template = self._suggestion_to_template(suggestion)

        if self.verbose:
            print(f"  LLM BALLOON EXPANSION #{self.llm_architect.balloon_count}")
            print(f"    Name: {suggestion.name}")
            print(f"    Reasoning: {suggestion.reasoning}")
            print(f"    Config: time={suggestion.time_handling}, "
                  f"gradient={suggestion.gradient_strategy}, "
                  f"latent={suggestion.latent_structure}")
            if suggestion.custom_notes:
                print(f"    Notes: {suggestion.custom_notes}")

        return [template]

    def _suggestion_to_template(self, suggestion: 'LLMSuggestion') -> ArchitectureTemplate:
        """Convert LLM suggestion to ArchitectureTemplate."""

        def safe_enum(enum_class, value, default):
            """Map string value to enum, with fallback."""
            try:
                return enum_class(value)
            except (ValueError, KeyError):
                if self.verbose:
                    print(f"    Warning: Unknown {enum_class.__name__} "
                          f"value '{value}', using {default}")
                return default

        balloon_id = self.llm_architect.balloon_count

        template = ArchitectureTemplate(
            id=f"llm_{balloon_id}_{suggestion.name[:30]}",
            name=f"LLM-{balloon_id}: {suggestion.name}",
            description=suggestion.reasoning,
            family=f"llm_generated_{balloon_id}",
            time_handling=safe_enum(
                TimeHandling, suggestion.time_handling, TimeHandling.LTC),
            gradient_strategy=safe_enum(
                GradientStrategy, suggestion.gradient_strategy, GradientStrategy.SEGMENTED),
            latent_structure=safe_enum(
                LatentStructure, suggestion.latent_structure, LatentStructure.UNCONSTRAINED),
            input_coupling=safe_enum(
                InputCoupling, suggestion.input_coupling, InputCoupling.ADDITIVE),
            solver=safe_enum(
                SolverChoice, suggestion.solver, SolverChoice.EULER),
            latent_dim_range=(suggestion.latent_dim, suggestion.latent_dim * 4),
            complexity=4,
            expected_training_hours=2.0
        )
        template.vector = template.to_property_vector()
        return template

    # ================================================================
    # LOGGING
    # ================================================================

    def _record_iteration(self, iteration, template, verify_result,
                          recovery, status):
        """Log iteration for analysis."""
        self.iteration_log.append({
            'iteration': iteration + 1,
            'template_id': template.id,
            'family': template.family,
            'status': status,
            'verify_passed': verify_result.passed if verify_result else False,
            'n_recovered': recovery.n_recovered if recovery else 0,
            'gap_remaining': self.recovery_space.compute_gap().residual_magnitude
                            if recovery else 1.0,
            'balloon_expansions': self.balloon_expansions
        })
