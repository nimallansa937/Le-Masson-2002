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
