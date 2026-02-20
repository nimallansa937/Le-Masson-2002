"""
LLM-powered architecture suggestion engine for DESCARTES Neural ODE.

Replaces hardcoded gap->architecture lookup with LLM reasoning.
The LLM receives:
  1. Full history of all attempts (architecture, recovery, patterns)
  2. Current gap analysis (what's missing, by category and timescale)
  3. DreamCoder patterns (co-recovery clusters, property correlations)
  4. Exhausted families (what NOT to suggest)

Returns: A new ArchitectureTemplate targeting the gap.

Cost: ~$0.01 per call. One call per iteration (every 1-4 hours of GPU time).
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Complete record of one DESCARTES iteration."""
    iteration: int
    template_name: str
    family: str
    time_handling: str
    gradient_strategy: str
    latent_structure: str
    input_coupling: str
    solver: str

    # Results
    spike_correlation: float
    bio_vars_recovered: int
    bio_vars_total: int = 160
    recovery_by_category: Dict[str, int] = None
    near_misses: List[str] = None  # variables with 0.3 < r < 0.5
    cca_score: float = None

    # Training diagnostics
    epochs_completed: int = 0
    final_train_loss: float = None
    final_val_loss: float = None
    training_time_hours: float = None
    short_segment_corr: float = None

    # Failure info
    failed: bool = False
    failure_reason: str = None


@dataclass
class LLMSuggestion:
    """Architecture suggestion from LLM."""
    name: str
    reasoning: str
    time_handling: str
    gradient_strategy: str
    latent_structure: str
    input_coupling: str
    solver: str
    latent_dim: int = 64
    custom_notes: str = None


class LLMArchitect:
    """LLM-powered architecture suggestion engine."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: Model to use. Sonnet is sufficient and cheaper than Opus.
        """
        self.model = model
        self.history: List[IterationRecord] = []
        self.patterns: List[Dict[str, Any]] = []
        self.exhausted_families: set = set()
        self.balloon_count: int = 0

        # Initialize client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.available = True
            logger.info(f"LLM Architect initialized with model={model}")
        except ImportError:
            logger.warning("anthropic package not installed. LLM suggestions unavailable.")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")
            self.available = False

    def record_iteration(self, record: IterationRecord):
        """Add an iteration result to history."""
        self.history.append(record)
        logger.info(f"Recorded iteration {record.iteration}: "
                     f"{record.template_name} -> {record.bio_vars_recovered}/160")

    def record_patterns(self, patterns: List[Dict[str, Any]]):
        """Update DreamCoder patterns."""
        self.patterns = patterns

    def record_exhausted_family(self, family: str):
        """Mark a family as exhausted."""
        self.exhausted_families.add(family)

    def _build_system_prompt(self) -> str:
        """System prompt establishing the LLM's role and constraints."""
        return """You are an architecture search agent for Neural ODEs modeling thalamic circuits.

CONTEXT: The DESCARTES framework searches for Neural ODE architectures whose learned hidden states
recover biological intermediate variables (ion channel gating, membrane voltages, synaptic conductances)
from a thalamic TC-nRt circuit. There are 160 biological variables organized as:
- tc_gating (60 vars): T-current activation (m_T), inactivation (h_T), H-current (m_H) for 20 TC neurons
- nrt_state (60 vars): membrane voltage (V), T-current m_T and h_T for 20 nRt neurons
- synaptic (40 vars): GABA_A and GABA_B conductances for 20 TC synapses

The circuit exhibits bifurcation between relay mode and oscillatory mode depending on inhibitory gain.
Key dynamics: rebound bursting (driven by T-current de-inactivation), spindle oscillations (8-14 Hz),
feedback inhibition loop (TC->nRt->TC).

KNOWN RESULTS FROM PRIOR WORK (A-R3 baselines):
- Volterra-Laguerre: spike_corr=0.549, recovered 89/160 bio vars (kernel model, not ODE)
- LSTM: spike_corr=0.451, recovered ~20/160 (discrete gates, poor individual variable recovery)
- Standard Neural ODE: spike_corr=0.012 (vanishing gradients through 2000 timesteps)

YOUR TASK: Given the history of architecture attempts and their recovery profiles, suggest ONE new
architecture template that specifically targets the unrecovered biological variables.

CONSTRAINTS:
- Must be implementable as a PyTorch nn.Module
- Must accept input shape (batch, time, 21) and output shape (batch, time, 20)
- Must expose latent states for BioVar comparison
- Must be trainable within 2-4 hours on a single GPU
- Do NOT suggest architectures from exhausted families

VALID PROPERTY VALUES:
- TimeHandling: standard_ode, ltc, neural_cde, gru_ode, coupled_oscillator, state_space
- GradientStrategy: adjoint, direct, segmented, distillation, shooting
- LatentStructure: unconstrained, biophysical, oscillatory, hierarchical, sparse
- InputCoupling: additive, multiplicative, controlled, gated, concatenated
- SolverChoice: dopri5, euler, midpoint, implicit_adams, tsit5

You may also suggest COMBINATIONS not in the predefined templates (that's the point).
Respond ONLY with valid JSON, no markdown fences, no explanation outside the JSON."""

    def _build_user_prompt(self, current_gap: Dict[str, Any]) -> str:
        """Build the user prompt with full history and gap analysis."""

        # Format history
        history_text = "## ITERATION HISTORY\n\n"
        if not self.history:
            history_text += "No iterations completed yet.\n"
        else:
            # What worked
            working = [r for r in self.history if r.bio_vars_recovered > 0]
            failed = [r for r in self.history if r.bio_vars_recovered == 0]

            if working:
                history_text += "### WHAT WORKED (keep these features):\n"
                for r in sorted(working, key=lambda x: x.bio_vars_recovered, reverse=True):
                    history_text += f"- {r.template_name}: {r.bio_vars_recovered}/160 recovered\n"
                    history_text += (f"  Properties: time={r.time_handling}, "
                                     f"gradient={r.gradient_strategy}, "
                                     f"latent={r.latent_structure}, "
                                     f"input={r.input_coupling}, solver={r.solver}\n")
                    if r.recovery_by_category:
                        history_text += f"  By category: {r.recovery_by_category}\n"
                    if r.near_misses:
                        history_text += f"  Near-misses (r=0.3-0.5): {r.near_misses[:10]}\n"

            if failed:
                history_text += "\n### WHAT FAILED (avoid these patterns):\n"
                for r in failed:
                    reason = r.failure_reason or "zero recovery"
                    history_text += f"- {r.template_name} ({r.family}): {reason}\n"
                    history_text += (f"  Properties: time={r.time_handling}, "
                                     f"gradient={r.gradient_strategy}\n")
                    if r.short_segment_corr is not None:
                        history_text += f"  Short-segment corr: {r.short_segment_corr:.4f}\n"
                    if r.epochs_completed > 0:
                        history_text += (f"  Completed {r.epochs_completed} epochs, "
                                         f"final loss: {r.final_train_loss}\n")

        # Format gap analysis
        gap_text = "\n## CURRENT GAP ANALYSIS\n\n"
        gap_text += f"Remaining gap: {current_gap.get('remaining_gap', 'unknown')}%\n"
        gap_text += f"Gap direction: {current_gap.get('direction', 'unknown')}\n"
        gap_text += f"Gap profile by category: {current_gap.get('profile', {})}\n"
        if 'timescale_profile' in current_gap:
            gap_text += f"Gap by timescale: {current_gap['timescale_profile']}\n"
        if 'dynamics_profile' in current_gap:
            gap_text += f"Gap by dynamics type: {current_gap['dynamics_profile']}\n"

        # Format DreamCoder patterns
        pattern_text = "\n## DREAMCODER PATTERNS\n\n"
        if not self.patterns:
            pattern_text += "No patterns extracted yet (too few iterations).\n"
        else:
            for i, p in enumerate(self.patterns):
                pattern_text += (f"Pattern {i+1}: {p.get('type', 'unknown')} "
                                  f"-- {p.get('description', '')}\n")
                if 'evidence' in p:
                    pattern_text += f"  Evidence: {p['evidence']}\n"

        # Exhausted families
        exhaust_text = (f"\n## EXHAUSTED FAMILIES (do NOT suggest these): "
                        f"{list(self.exhausted_families)}\n")

        # Best so far
        best_text = "\n## BEST RESULT SO FAR\n"
        if self.history:
            best = max(self.history, key=lambda r: r.bio_vars_recovered)
            best_text += f"{best.template_name}: {best.bio_vars_recovered}/160\n"
            if best.recovery_by_category:
                best_text += f"Recovery: {best.recovery_by_category}\n"
        else:
            best_text += "None yet.\n"

        # Final instruction
        instruction = """
## YOUR TASK

Based on the above history, gap analysis, and patterns, suggest ONE new architecture
that targets the unrecovered variables. Explain your reasoning.

Respond with JSON:
{
    "name": "descriptive_name",
    "reasoning": "2-3 sentences explaining why this targets the gap",
    "time_handling": "one of the valid values or 'custom'",
    "gradient_strategy": "one of the valid values",
    "latent_structure": "one of the valid values",
    "input_coupling": "one of the valid values",
    "solver": "one of the valid values",
    "latent_dim": 64,
    "custom_notes": "any implementation details the builder should know"
}
"""

        return history_text + gap_text + pattern_text + exhaust_text + best_text + instruction

    def suggest_architecture(self, current_gap: Dict[str, Any]) -> Optional[LLMSuggestion]:
        """
        Call LLM to suggest a new architecture template.

        Args:
            current_gap: Dict with keys 'remaining_gap', 'direction', 'profile',
                        optionally 'timescale_profile', 'dynamics_profile'

        Returns:
            LLMSuggestion or None if LLM unavailable/fails
        """
        if not self.available:
            logger.warning("LLM not available, returning None")
            return None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(current_gap)

        try:
            logger.info("Calling LLM for architecture suggestion...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            raw_text = response.content[0].text.strip()

            # Clean up potential markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

            config = json.loads(raw_text)

            suggestion = LLMSuggestion(
                name=config["name"],
                reasoning=config["reasoning"],
                time_handling=config["time_handling"],
                gradient_strategy=config["gradient_strategy"],
                latent_structure=config["latent_structure"],
                input_coupling=config["input_coupling"],
                solver=config["solver"],
                latent_dim=config.get("latent_dim", 64),
                custom_notes=config.get("custom_notes", None)
            )

            self.balloon_count += 1
            logger.info(f"LLM suggested: {suggestion.name}")
            logger.info(f"Reasoning: {suggestion.reasoning}")

            return suggestion

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {raw_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    # ================================================================
    # DECISION POINT 2: SHORT-SEGMENT FAILURE RECOVERY
    # ================================================================

    def suggest_verify_fix(
        self,
        template_config: Dict[str, str],
        failure_reason: str,
        short_segment_corr: Optional[float] = None,
        final_loss: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        After short-segment verification fails, ask LLM for a fix.

        The LLM sees the error type (NaN, divergence, zero gradients, non-convergence)
        and the architecture config, then suggests ONE targeted fix:
        lr change, gradient clipping adjustment, normalization, or solver swap.

        Returns dict with keys: fix_type, description, and parameter overrides,
        or None if LLM unavailable/fails.
        """
        if not self.available:
            return None

        system = """You are a Neural ODE training debugger for thalamic circuit models.

A short-segment verification test (50 timesteps, 20 epochs) has FAILED for a Neural ODE architecture.
Your job: diagnose the failure and suggest ONE targeted fix that can be applied as a parameter override
on a retry. The retry gets ONE attempt — make it count.

COMMON FAILURE MODES AND FIXES:
- "NaN/Inf loss": Usually exploding gradients. Fix: lower lr (0.1x), add gradient clipping, or switch solver.
- "Zero gradients (vanishing)": Gradients die. Fix: raise lr (10x), use skip connections, or switch to euler solver.
- "Loss did not converge": Learning too slow or stuck. Fix: raise lr (3-5x), increase epochs, or change optimizer.
- "Runtime error": ODE solver diverged. Fix: switch to euler/midpoint (fixed-step), lower rtol/atol.

Respond ONLY with valid JSON (no markdown fences):
{
    "fix_type": "lr_change|grad_clip|normalization|solver_swap|other",
    "description": "1 sentence explaining the fix",
    "overrides": {
        "lr": 0.001,
        "n_epochs": 30,
        "grad_clip": 0.5
    }
}

The "overrides" dict can contain: lr (float), n_epochs (int), grad_clip (float),
solver (string: euler/midpoint/rk4), batch_size (int).
Only include keys you want to change."""

        user = f"""FAILED ARCHITECTURE:
- time_handling: {template_config.get('time_handling', 'unknown')}
- gradient_strategy: {template_config.get('gradient_strategy', 'unknown')}
- latent_structure: {template_config.get('latent_structure', 'unknown')}
- input_coupling: {template_config.get('input_coupling', 'unknown')}
- solver: {template_config.get('solver', 'unknown')}

FAILURE: {failure_reason}
Short-segment correlation: {short_segment_corr if short_segment_corr is not None else 'N/A'}
Final loss: {final_loss if final_loss is not None else 'N/A'}

Suggest ONE fix to retry verification."""

        try:
            logger.info(f"Calling LLM for verify fix (failure: {failure_reason})...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            fix = json.loads(raw)
            logger.info(f"LLM verify fix: {fix.get('fix_type')} — {fix.get('description')}")
            return fix

        except Exception as e:
            logger.error(f"LLM verify fix call failed: {e}")
            return None

    # ================================================================
    # DECISION POINT 3: POST-TRAINING C2 HYPERPARAMETER TWEAK
    # ================================================================

    def suggest_c2_tweak(
        self,
        template_config: Dict[str, str],
        spike_correlation: float,
        bio_vars_recovered: int,
        recovery_by_category: Dict[str, int],
        near_misses: List[str],
        training_hours: float,
        epochs_completed: int,
        best_val_loss: float,
        converged: bool
    ) -> Optional[Dict[str, Any]]:
        """
        After full training, ask LLM to suggest a C2 hyperparameter tweak.

        The LLM sees the full recovery profile and training diagnostics,
        then suggests ONE tweak: latent_dim, lr, patience, batch_size, etc.
        The architecture gets ONE retry with the tweaked C2 before moving on.

        Returns dict with keys: tweak_type, description, overrides,
        or None if LLM unavailable/fails.
        """
        if not self.available:
            return None

        system = """You are a hyperparameter tuning agent for Neural ODE models of thalamic circuits.

A Neural ODE architecture has completed full training. You see the results:
spike correlation, bio variable recovery (out of 160), and near-miss variables.

Your job: suggest ONE targeted hyperparameter tweak for a retry. The model gets
ONE more training run — focus on the single change most likely to push near-miss
variables (r=0.3-0.5) over the recovery threshold (r>0.5).

TUNING PRINCIPLES:
- Low bio recovery + decent spike corr → latent_dim too small (increase 2x)
- Many near-misses in one category → add category-specific loss weight
- Training didn't converge → increase patience or lower lr
- Fast convergence (few epochs) → lr may be too high, model underfitting
- Slow convergence (many epochs, high loss) → lr too low or model capacity insufficient

Respond ONLY with valid JSON (no markdown fences):
{
    "tweak_type": "latent_dim|lr|patience|batch_size|loss_weight|other",
    "description": "1 sentence explaining why this tweak targets the gap",
    "overrides": {
        "lr": 0.0003,
        "latent_dim": 128
    }
}

The "overrides" dict can contain: lr (float), latent_dim (int), batch_size (int),
patience (int), max_epochs (int). Only include keys you want to change."""

        user = f"""ARCHITECTURE:
- time_handling: {template_config.get('time_handling')}
- gradient_strategy: {template_config.get('gradient_strategy')}
- latent_structure: {template_config.get('latent_structure')}
- input_coupling: {template_config.get('input_coupling')}
- solver: {template_config.get('solver')}

TRAINING RESULTS:
- Spike correlation: {spike_correlation:.4f}
- Bio vars recovered: {bio_vars_recovered}/160
- Recovery by category: {recovery_by_category}
- Near-miss variables (r=0.3-0.5): {near_misses[:15]}
- Training time: {training_hours:.2f}h
- Epochs completed: {epochs_completed}
- Best val loss: {best_val_loss:.6f}
- Converged: {converged}

Suggest ONE C2 hyperparameter tweak to improve recovery on retry."""

        try:
            logger.info(f"Calling LLM for C2 tweak (recovery={bio_vars_recovered}/160)...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            tweak = json.loads(raw)
            logger.info(f"LLM C2 tweak: {tweak.get('tweak_type')} — {tweak.get('description')}")
            return tweak

        except Exception as e:
            logger.error(f"LLM C2 tweak call failed: {e}")
            return None

    # ================================================================
    # DECISION POINT 4: POST-GAP NEAR-MISS INTERPRETATION
    # ================================================================

    def interpret_near_misses(
        self,
        near_miss_vars: List[Dict[str, Any]],
        best_architecture: Dict[str, str],
        gap_profile: Dict[str, float],
        patterns: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        After gap analysis, ask LLM to interpret WHY near-miss variables
        (r=0.3-0.5) didn't cross the recovery threshold.

        Returns dict with keys: interpretation, suggested_modifications, priority_variables,
        or None if LLM unavailable/fails.
        """
        if not self.available:
            return None

        system = """You are a neuroscience-informed architecture analyst for Neural ODE thalamic models.

After each training iteration, some biological variables have near-miss correlations
(Pearson r between 0.3 and 0.5 with the best-matching latent dimension). These are
tantalizingly close to the recovery threshold (r>0.5) but didn't make it.

Your job: interpret WHY these specific variables are near-misses and suggest targeted
modifications to push them over the threshold.

BIOLOGICAL CONTEXT:
- tc_gating variables: T-current activation (m_T, fast ~1ms), inactivation (h_T, slow ~100ms),
  H-current (m_H, very slow ~1s). These have VERY different timescales.
- nrt_state variables: membrane voltage (V, fast), T-current gates in nRt neurons.
  nRt neurons receive TC excitation and provide feedback inhibition.
- synaptic variables: GABA_A (fast, ~10ms decay) and GABA_B (slow, ~150ms decay).
  These are the TC->nRt->TC feedback loop conductances.

COMMON REASONS FOR NEAR-MISS:
- Timescale mismatch: model captures fast dynamics but slow variables need longer memory
- Coupling structure: feedback variables need explicit recurrent pathways
- Dimension pressure: too few latent dims for the biological complexity
- Loss imbalance: spike prediction loss doesn't reward individual variable recovery

Respond ONLY with valid JSON (no markdown fences):
{
    "interpretation": "2-3 sentences explaining the biological/architectural reason",
    "priority_variables": ["var1", "var2"],
    "suggested_modifications": [
        {"modification": "description", "targets": "which variables this helps"}
    ]
}"""

        # Format near-miss variables for LLM
        nm_text = "NEAR-MISS VARIABLES (r=0.3-0.5):\n"
        for v in near_miss_vars[:20]:
            nm_text += f"- {v.get('name', '?')}: r={v.get('correlation', 0):.3f}, "
            nm_text += f"category={v.get('category', '?')}, timescale={v.get('timescale', '?')}\n"

        user = f"""{nm_text}
BEST ARCHITECTURE SO FAR:
- time_handling: {best_architecture.get('time_handling')}
- gradient_strategy: {best_architecture.get('gradient_strategy')}
- latent_structure: {best_architecture.get('latent_structure')}
- input_coupling: {best_architecture.get('input_coupling')}
- solver: {best_architecture.get('solver')}

GAP PROFILE (fraction missing per category): {gap_profile}

DREAMCODER PATTERNS:
{chr(10).join(f"- [{p.get('type')}] {p.get('description', '')[:80]}" for p in patterns[:8])}

Interpret why these variables are near-misses and suggest targeted fixes."""

        try:
            logger.info(f"Calling LLM for near-miss interpretation ({len(near_miss_vars)} vars)...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            result = json.loads(raw)
            logger.info(f"LLM near-miss interpretation: {result.get('interpretation', '')[:100]}...")
            return result

        except Exception as e:
            logger.error(f"LLM near-miss interpretation failed: {e}")
            return None

    def get_prompt_for_logging(self, current_gap: Dict[str, Any]) -> str:
        """Return the full prompt that would be sent, for debugging/logging."""
        return self._build_system_prompt() + "\n\n---\n\n" + self._build_user_prompt(current_gap)

    def get_history_summary(self) -> str:
        """Return human-readable summary of all iterations."""
        lines = ["DESCARTES LLM-Guided Search History", "=" * 40]
        for r in self.history:
            status = (f"{r.bio_vars_recovered}/160"
                      if not r.failed
                      else f"FAILED: {r.failure_reason}")
            lines.append(f"  {r.iteration}: {r.template_name} -> {status}")
        lines.append(f"Exhausted families: {list(self.exhausted_families)}")
        lines.append(f"LLM expansions: {self.balloon_count}")
        return "\n".join(lines)
