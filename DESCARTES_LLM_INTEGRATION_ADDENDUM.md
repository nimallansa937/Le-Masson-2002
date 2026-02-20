# DESCARTES Neural ODE — LLM Integration Addendum
## Claude Code Implementation Guide

### Purpose

Replace the hardcoded gap-to-architecture lookup table in the DESCARTES orchestrator with an LLM-powered architecture suggestion engine. The LLM receives DreamCoder patterns, gap analysis, and full history of all previous attempts, then generates new architecture templates targeting the specific biological variables that remain unrecovered.

This converts DESCARTES from a fixed 9-template search into an open-ended LLM-guided architecture search with accumulating knowledge.

---

## Architecture Change

### Before (Fixed Search)
```
Gap compass (lookup table) → hardcoded mapping → next template from T0-T8
```

### After (LLM-Guided Search)
```
Gap compass + DreamCoder history → LLM prompt → new architecture template → train → 
results feed back into next prompt
```

### What Changes
- `orchestrator.py` — add LLM balloon expansion mode
- New file: `llm_architect.py` — LLM interface + prompt construction
- New file: `dreamcoder_history.py` — accumulates structured results across iterations
- `run_descartes_neural_ode.py` — add `--use-llm` and `--api-key` flags
- Stopping conditions change from "9 iterations" to result-driven

### What Stays The Same
- All architecture templates (T0-T8) still run first
- Short-segment verifier unchanged
- Full training pipeline unchanged
- BioVar recovery scoring unchanged
- Gap analysis computation unchanged

---

## File 1: `llm_architect.py`

Location: `descartes_neural_ode/core/llm_architect.py`

```python
"""
LLM-powered architecture suggestion engine for DESCARTES Neural ODE.

Replaces hardcoded gap→architecture lookup with LLM reasoning.
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
        logger.info(f"Recorded iteration {record.iteration}: {record.template_name} → {record.bio_vars_recovered}/160")
    
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
feedback inhibition loop (TC→nRt→TC).

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
                    history_text += f"  Properties: time={r.time_handling}, gradient={r.gradient_strategy}, "
                    history_text += f"latent={r.latent_structure}, input={r.input_coupling}, solver={r.solver}\n"
                    if r.recovery_by_category:
                        history_text += f"  By category: {r.recovery_by_category}\n"
                    if r.near_misses:
                        history_text += f"  Near-misses (r=0.3-0.5): {r.near_misses[:10]}\n"
            
            if failed:
                history_text += "\n### WHAT FAILED (avoid these patterns):\n"
                for r in failed:
                    reason = r.failure_reason or "zero recovery"
                    history_text += f"- {r.template_name} ({r.family}): {reason}\n"
                    history_text += f"  Properties: time={r.time_handling}, gradient={r.gradient_strategy}\n"
                    if r.short_segment_corr is not None:
                        history_text += f"  Short-segment corr: {r.short_segment_corr:.4f}\n"
                    if r.epochs_completed > 0:
                        history_text += f"  Completed {r.epochs_completed} epochs, final loss: {r.final_train_loss}\n"
        
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
                pattern_text += f"Pattern {i+1}: {p.get('type', 'unknown')} — {p.get('description', '')}\n"
                if 'evidence' in p:
                    pattern_text += f"  Evidence: {p['evidence']}\n"
        
        # Exhausted families
        exhaust_text = f"\n## EXHAUSTED FAMILIES (do NOT suggest these): {list(self.exhausted_families)}\n"
        
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
            status = f"{r.bio_vars_recovered}/160" if not r.failed else f"FAILED: {r.failure_reason}"
            lines.append(f"  {r.iteration}: {r.template_name} → {status}")
        lines.append(f"Exhausted families: {list(self.exhausted_families)}")
        lines.append(f"LLM expansions: {self.balloon_count}")
        return "\n".join(lines)
```

---

## File 2: `dreamcoder_history.py`

Location: `descartes_neural_ode/core/dreamcoder_history.py`

```python
"""
DreamCoder History Accumulator.

Analyzes iteration results to extract structured patterns:
1. Co-recovery clusters: variables always recovered/missed together
2. Property correlations: architecture features → variable recovery
3. Near-miss tracking: variables close to threshold across attempts
4. Family performance: aggregate stats per architecture family
5. Timescale alignment: which solvers/time-handling recover which timescales

These patterns are fed to the LLM to guide architecture suggestion.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DreamCoderHistory:
    """Accumulates and analyzes results across DESCARTES iterations."""
    
    def __init__(self, bio_var_names: List[str], bio_var_categories: Dict[str, List[int]],
                 bio_var_timescales: Dict[str, List[int]]):
        """
        Args:
            bio_var_names: List of 160 biological variable names
            bio_var_categories: Maps category name → list of variable indices
                e.g. {'tc_gating': [0..59], 'nrt_state': [60..119], 'synaptic': [120..159]}
            bio_var_timescales: Maps timescale → list of variable indices
                e.g. {'fast': [...], 'slow': [...], 'mixed': [...]}
        """
        self.bio_var_names = bio_var_names
        self.bio_var_categories = bio_var_categories
        self.bio_var_timescales = bio_var_timescales
        self.n_vars = len(bio_var_names)
        
        # Accumulated data
        self.recovery_matrix = []     # List of 160-dim binary vectors
        self.correlation_matrix = []  # List of 160-dim correlation vectors (raw r values)
        self.architecture_configs = [] # List of config dicts
        self.iteration_names = []
        
    def add_result(self, name: str, config: Dict[str, str], 
                   recovery_vector: np.ndarray, correlation_vector: np.ndarray):
        """
        Record one iteration's results.
        
        Args:
            name: Architecture template name
            config: Dict with keys time_handling, gradient_strategy, etc.
            recovery_vector: 160-dim binary (1=recovered, 0=not)
            correlation_vector: 160-dim float (raw Pearson r values)
        """
        self.iteration_names.append(name)
        self.architecture_configs.append(config)
        self.recovery_matrix.append(recovery_vector.copy())
        self.correlation_matrix.append(correlation_vector.copy())
        
        logger.info(f"DreamCoder recorded: {name} → {int(recovery_vector.sum())}/160")
    
    def extract_patterns(self) -> List[Dict[str, Any]]:
        """
        Run DreamCoder sleep phase — analyze all results to find patterns.
        
        Returns list of pattern dicts, each with:
            type: str (co_recovery, property_correlation, near_miss, 
                       family_exhaustion, timescale_alignment)
            description: str (human-readable)
            evidence: str (supporting data)
            confidence: float (0-1)
        """
        patterns = []
        
        if len(self.recovery_matrix) < 2:
            logger.info("Need at least 2 iterations for pattern extraction")
            return patterns
        
        R = np.array(self.recovery_matrix)      # (n_iters, 160)
        C = np.array(self.correlation_matrix)    # (n_iters, 160)
        
        # Pattern 1: Co-recovery clusters
        patterns.extend(self._find_co_recovery_clusters(R))
        
        # Pattern 2: Property correlations
        patterns.extend(self._find_property_correlations(R))
        
        # Pattern 3: Near-misses
        patterns.extend(self._find_near_misses(C))
        
        # Pattern 4: Timescale alignment
        patterns.extend(self._find_timescale_alignment(R))
        
        # Pattern 5: Category-level trends
        patterns.extend(self._find_category_trends(R))
        
        logger.info(f"Extracted {len(patterns)} patterns from {len(self.recovery_matrix)} iterations")
        return patterns
    
    def _find_co_recovery_clusters(self, R: np.ndarray) -> List[Dict]:
        """Find variables that are always recovered together or missed together."""
        patterns = []
        n_iters = R.shape[0]
        
        if n_iters < 3:
            return patterns
        
        # Variables recovered in exact same iterations
        var_signatures = {}
        for i in range(self.n_vars):
            sig = tuple(R[:, i].astype(int))
            if sig not in var_signatures:
                var_signatures[sig] = []
            var_signatures[sig].append(i)
        
        for sig, var_indices in var_signatures.items():
            if len(var_indices) >= 5 and sum(sig) > 0:  # cluster of 5+, not all-zero
                var_names = [self.bio_var_names[i] for i in var_indices[:8]]
                categories = set()
                for idx in var_indices:
                    for cat, indices in self.bio_var_categories.items():
                        if idx in indices:
                            categories.add(cat)
                
                patterns.append({
                    'type': 'co_recovery',
                    'description': f"{len(var_indices)} variables always recovered/missed together: {var_names}",
                    'evidence': f"Signature across {n_iters} iterations: {sig}. Categories: {categories}",
                    'confidence': min(1.0, n_iters / 5),
                    'variable_indices': var_indices
                })
        
        return patterns
    
    def _find_property_correlations(self, R: np.ndarray) -> List[Dict]:
        """Find architecture properties that correlate with recovering specific variable types."""
        patterns = []
        
        # For each property, compare recovery rates when property is on vs off
        properties_to_check = ['time_handling', 'gradient_strategy', 'latent_structure', 
                               'input_coupling', 'solver']
        
        for prop in properties_to_check:
            values = [c.get(prop, 'unknown') for c in self.architecture_configs]
            unique_values = set(values)
            
            if len(unique_values) < 2:
                continue
            
            for val in unique_values:
                mask = np.array([v == val for v in values])
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue
                
                for cat_name, cat_indices in self.bio_var_categories.items():
                    # Recovery rate for this category when property=val vs not
                    with_val = R[mask][:, cat_indices].mean() if mask.sum() > 0 else 0
                    without_val = R[~mask][:, cat_indices].mean() if (~mask).sum() > 0 else 0
                    
                    diff = with_val - without_val
                    if abs(diff) > 0.15:  # meaningful difference
                        direction = "helps" if diff > 0 else "hurts"
                        patterns.append({
                            'type': 'property_correlation',
                            'description': f"{prop}={val} {direction} {cat_name} recovery ({with_val:.0%} vs {without_val:.0%})",
                            'evidence': f"Diff={diff:+.0%} across {mask.sum()} iterations with, {(~mask).sum()} without",
                            'confidence': min(1.0, (mask.sum() + (~mask).sum()) / 6),
                            'property': prop,
                            'value': val,
                            'category': cat_name,
                            'effect': diff
                        })
        
        return patterns
    
    def _find_near_misses(self, C: np.ndarray) -> List[Dict]:
        """Find variables that almost reached recovery threshold across attempts."""
        patterns = []
        
        # For each variable, find max correlation across all attempts
        max_corr = np.nanmax(C, axis=0)  # (160,)
        
        near_miss_indices = np.where((max_corr > 0.3) & (max_corr < 0.5))[0]
        
        if len(near_miss_indices) > 0:
            near_miss_names = [self.bio_var_names[i] for i in near_miss_indices[:15]]
            near_miss_corrs = [float(max_corr[i]) for i in near_miss_indices[:15]]
            
            # Check which categories these near-misses belong to
            cat_counts = defaultdict(int)
            for idx in near_miss_indices:
                for cat, indices in self.bio_var_categories.items():
                    if idx in indices:
                        cat_counts[cat] += 1
            
            patterns.append({
                'type': 'near_miss',
                'description': f"{len(near_miss_indices)} variables have near-miss correlations (0.3-0.5)",
                'evidence': f"Top near-misses: {list(zip(near_miss_names, near_miss_corrs))}. By category: {dict(cat_counts)}",
                'confidence': 0.8,
                'variable_names': near_miss_names,
                'max_correlations': near_miss_corrs,
                'category_counts': dict(cat_counts)
            })
        
        return patterns
    
    def _find_timescale_alignment(self, R: np.ndarray) -> List[Dict]:
        """Find which architecture features align with which timescales."""
        patterns = []
        
        for ts_name, ts_indices in self.bio_var_timescales.items():
            for i, config in enumerate(self.architecture_configs):
                recovery_rate = R[i, ts_indices].mean()
                if recovery_rate > 0.3:
                    patterns.append({
                        'type': 'timescale_alignment',
                        'description': f"{self.iteration_names[i]} recovers {recovery_rate:.0%} of {ts_name}-timescale variables",
                        'evidence': f"Config: time={config.get('time_handling')}, solver={config.get('solver')}",
                        'confidence': 0.7,
                        'timescale': ts_name,
                        'architecture': self.iteration_names[i],
                        'recovery_rate': float(recovery_rate)
                    })
        
        return patterns
    
    def _find_category_trends(self, R: np.ndarray) -> List[Dict]:
        """Find categories that are consistently missed or recovered."""
        patterns = []
        
        for cat_name, cat_indices in self.bio_var_categories.items():
            cat_recovery = R[:, cat_indices]  # (n_iters, n_cat_vars)
            
            # Category never recovered by any architecture
            if cat_recovery.sum() == 0:
                patterns.append({
                    'type': 'category_never_recovered',
                    'description': f"Category '{cat_name}' ({len(cat_indices)} vars) NEVER recovered by ANY architecture",
                    'evidence': f"0/{len(self.recovery_matrix)} architectures recovered any {cat_name} variable",
                    'confidence': min(1.0, len(self.recovery_matrix) / 4),
                    'category': cat_name,
                    'severity': 'ontological_gap'
                })
            
            # Category improving over iterations
            elif len(self.recovery_matrix) >= 3:
                per_iter = [cat_recovery[i].mean() for i in range(len(self.recovery_matrix))]
                if per_iter[-1] > per_iter[0] + 0.1:
                    patterns.append({
                        'type': 'category_improving',
                        'description': f"Category '{cat_name}' recovery improving: {per_iter[0]:.0%} → {per_iter[-1]:.0%}",
                        'evidence': f"Trajectory: {[f'{p:.0%}' for p in per_iter]}",
                        'confidence': 0.6,
                        'category': cat_name
                    })
        
        return patterns
```

---

## File 3: Orchestrator Modifications

Location: Modify existing `descartes_neural_ode/orchestrator.py`

### Add imports at top:
```python
from core.llm_architect import LLMArchitect, IterationRecord, LLMSuggestion
from core.dreamcoder_history import DreamCoderHistory
```

### Add to `__init__`:
```python
def __init__(self, config, use_llm=False, api_key=None, max_llm_expansions=10):
    # ... existing init code ...
    
    self.use_llm = use_llm
    self.max_llm_expansions = max_llm_expansions
    
    if use_llm:
        self.llm_architect = LLMArchitect(api_key=api_key)
        self.dreamcoder_history = DreamCoderHistory(
            bio_var_names=self.recovery_space.bio_var_names,
            bio_var_categories=self.recovery_space.bio_var_categories,
            bio_var_timescales=self.recovery_space.bio_var_timescales
        )
    else:
        self.llm_architect = None
        self.dreamcoder_history = None
```

### Add method to record results for LLM:
```python
def _record_for_llm(self, template, results, training_diagnostics=None):
    """Record iteration results for LLM architect and DreamCoder."""
    if not self.use_llm:
        return
    
    # Record in LLM history
    record = IterationRecord(
        iteration=len(self.llm_architect.history) + 1,
        template_name=template.name,
        family=template.family,
        time_handling=template.time_handling.value,
        gradient_strategy=template.gradient_strategy.value,
        latent_structure=template.latent_structure.value,
        input_coupling=template.input_coupling.value,
        solver=template.solver.value,
        spike_correlation=results.get('spike_correlation', 0.0),
        bio_vars_recovered=results.get('bio_vars_recovered', 0),
        recovery_by_category=results.get('recovery_by_category', {}),
        near_misses=results.get('near_misses', []),
        cca_score=results.get('cca_score', None),
        epochs_completed=training_diagnostics.get('epochs', 0) if training_diagnostics else 0,
        final_train_loss=training_diagnostics.get('train_loss', None) if training_diagnostics else None,
        final_val_loss=training_diagnostics.get('val_loss', None) if training_diagnostics else None,
        training_time_hours=training_diagnostics.get('time_hours', None) if training_diagnostics else None,
        short_segment_corr=results.get('short_segment_corr', None),
        failed=results.get('failed', False),
        failure_reason=results.get('failure_reason', None)
    )
    self.llm_architect.record_iteration(record)
    
    # Record in DreamCoder
    if 'recovery_vector' in results and 'correlation_vector' in results:
        self.dreamcoder_history.add_result(
            name=template.name,
            config={
                'time_handling': template.time_handling.value,
                'gradient_strategy': template.gradient_strategy.value,
                'latent_structure': template.latent_structure.value,
                'input_coupling': template.input_coupling.value,
                'solver': template.solver.value
            },
            recovery_vector=results['recovery_vector'],
            correlation_vector=results['correlation_vector']
        )
```

### Replace `_balloon_expand` method:
```python
def _balloon_expand(self, gap_analysis):
    """
    Expand architecture search space.
    
    If LLM is available: ask LLM for suggestion based on full history.
    Otherwise: fall back to hardcoded lookup table.
    """
    if self.use_llm and self.llm_architect.available:
        return self._balloon_expand_llm(gap_analysis)
    else:
        return self._balloon_expand_hardcoded(gap_analysis)

def _balloon_expand_llm(self, gap_analysis):
    """LLM-powered balloon expansion."""
    
    if self.llm_architect.balloon_count >= self.max_llm_expansions:
        logger.info(f"Max LLM expansions ({self.max_llm_expansions}) reached")
        return None
    
    # Run DreamCoder sleep phase
    patterns = self.dreamcoder_history.extract_patterns()
    self.llm_architect.record_patterns([p for p in patterns])
    
    # Log patterns
    logger.info(f"DreamCoder extracted {len(patterns)} patterns:")
    for p in patterns:
        logger.info(f"  [{p['type']}] {p['description']}")
    
    # Format gap for LLM
    gap_dict = {
        'remaining_gap': gap_analysis.remaining_gap_percent,
        'direction': gap_analysis.direction,
        'profile': gap_analysis.profile,
    }
    if hasattr(gap_analysis, 'timescale_profile'):
        gap_dict['timescale_profile'] = gap_analysis.timescale_profile
    if hasattr(gap_analysis, 'dynamics_profile'):
        gap_dict['dynamics_profile'] = gap_analysis.dynamics_profile
    
    # Ask LLM
    suggestion = self.llm_architect.suggest_architecture(gap_dict)
    
    if suggestion is None:
        logger.warning("LLM suggestion failed, falling back to hardcoded")
        return self._balloon_expand_hardcoded(gap_analysis)
    
    # Convert LLM suggestion to ArchitectureTemplate
    template = self._suggestion_to_template(suggestion)
    
    # Log the suggestion
    logger.info(f"LLM BALLOON EXPANSION #{self.llm_architect.balloon_count}")
    logger.info(f"  Name: {suggestion.name}")
    logger.info(f"  Reasoning: {suggestion.reasoning}")
    logger.info(f"  Config: time={suggestion.time_handling}, gradient={suggestion.gradient_strategy}")
    if suggestion.custom_notes:
        logger.info(f"  Notes: {suggestion.custom_notes}")
    
    return template

def _suggestion_to_template(self, suggestion: LLMSuggestion):
    """Convert LLM suggestion to ArchitectureTemplate."""
    from core.architecture_templates import (
        ArchitectureTemplate, TimeHandling, GradientStrategy,
        LatentStructure, InputCoupling, SolverChoice
    )
    
    # Map string values to enums, with fallback
    def safe_enum(enum_class, value, default):
        try:
            return enum_class(value)
        except (ValueError, KeyError):
            logger.warning(f"Unknown {enum_class.__name__} value '{value}', using {default}")
            return default
    
    return ArchitectureTemplate(
        id=f"llm_{self.llm_architect.balloon_count}",
        name=suggestion.name,
        description=suggestion.reasoning,
        family=f"llm_generated_{self.llm_architect.balloon_count}",
        time_handling=safe_enum(TimeHandling, suggestion.time_handling, TimeHandling.STANDARD_ODE),
        gradient_strategy=safe_enum(GradientStrategy, suggestion.gradient_strategy, GradientStrategy.SEGMENTED),
        latent_structure=safe_enum(LatentStructure, suggestion.latent_structure, LatentStructure.UNCONSTRAINED),
        input_coupling=safe_enum(InputCoupling, suggestion.input_coupling, InputCoupling.ADDITIVE),
        solver=safe_enum(SolverChoice, suggestion.solver, SolverChoice.EULER),
        latent_dim=suggestion.latent_dim,
        custom_notes=suggestion.custom_notes
    )
```

### Modify the main loop's stopping condition:
```python
def run(self):
    """Main DESCARTES loop with dynamic stopping."""
    
    # Phase 1: Run predefined templates (T0-T8)
    for template in self.predefined_templates:
        results = self._evaluate_template(template)
        self._record_for_llm(template, results)
        
        if results['bio_vars_recovered'] >= self.target_recovery:
            logger.info(f"TARGET ACHIEVED: {results['bio_vars_recovered']}/160")
            return results
    
    # Phase 2: LLM-guided expansion (if enabled)
    if not self.use_llm:
        logger.info("LLM not enabled. Search complete with predefined templates.")
        return self._get_best_result()
    
    consecutive_no_improvement = 0
    best_recovery = max(r.bio_vars_recovered for r in self.llm_architect.history)
    
    for i in range(self.max_llm_expansions):
        gap = self._compute_current_gap()
        template = self._balloon_expand_llm(gap)
        
        if template is None:
            logger.info("No more expansions possible")
            break
        
        results = self._evaluate_template(template)
        self._record_for_llm(template, results)
        
        # Check stopping conditions
        if results['bio_vars_recovered'] >= self.target_recovery:
            logger.info(f"TARGET ACHIEVED: {results['bio_vars_recovered']}/160")
            return results
        
        if results['bio_vars_recovered'] > best_recovery:
            best_recovery = results['bio_vars_recovered']
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
        
        if consecutive_no_improvement >= 3:
            logger.info("3 consecutive iterations with no improvement. Stopping.")
            break
    
    return self._get_best_result()
```

---

## File 4: Updated Run Script

Location: Modify `run_descartes_neural_ode.py`

```python
#!/usr/bin/env python3
"""
DESCARTES Neural ODE Architecture Search

Usage:
    # Fixed 9-template search (original)
    python run_descartes_neural_ode.py --data-dir /path/to/rung3_data
    
    # LLM-guided search (new)
    python run_descartes_neural_ode.py --data-dir /path/to/rung3_data --use-llm
    
    # With custom settings
    python run_descartes_neural_ode.py \
        --data-dir /path/to/rung3_data \
        --use-llm \
        --max-llm-expansions 15 \
        --time-budget 4.0 \
        --target-recovery 120
"""

import argparse
import os
from orchestrator import DESCARTESOrchestrator


def main():
    parser = argparse.ArgumentParser(description='DESCARTES Neural ODE Search')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to Rung 3 data directory')
    
    # LLM settings
    parser.add_argument('--use-llm', action='store_true',
                       help='Enable LLM-guided architecture search')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--max-llm-expansions', type=int, default=10,
                       help='Max LLM-generated architectures (default: 10)')
    
    # Training settings
    parser.add_argument('--time-budget', type=float, default=2.0,
                       help='Hours per architecture (default: 2.0)')
    parser.add_argument('--target-recovery', type=int, default=120,
                       help='Target bio vars to recover (default: 120/160)')
    
    # Stopping
    parser.add_argument('--max-total-hours', type=float, default=48.0,
                       help='Total GPU budget in hours (default: 48)')
    
    args = parser.parse_args()
    
    # Validate
    if args.use_llm:
        api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: --use-llm requires API key via --api-key or ANTHROPIC_API_KEY env var")
            print("Get your key at: https://console.anthropic.com/")
            return
    else:
        api_key = None
    
    config = {
        'data_dir': args.data_dir,
        'time_budget_hours': args.time_budget,
        'target_recovery': args.target_recovery,
        'max_total_hours': args.max_total_hours,
    }
    
    orchestrator = DESCARTESOrchestrator(
        config=config,
        use_llm=args.use_llm,
        api_key=api_key,
        max_llm_expansions=args.max_llm_expansions
    )
    
    print("=" * 60)
    print("DESCARTES Neural ODE Architecture Search")
    print("=" * 60)
    print(f"Data: {args.data_dir}")
    print(f"LLM-guided: {args.use_llm}")
    print(f"Time per architecture: {args.time_budget}h")
    print(f"Target: {args.target_recovery}/160 bio vars")
    if args.use_llm:
        print(f"Max LLM expansions: {args.max_llm_expansions}")
    print(f"Total budget: {args.max_total_hours}h")
    print("=" * 60)
    
    results = orchestrator.run()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best architecture: {results.get('best_name', 'none')}")
    print(f"Bio vars recovered: {results.get('bio_vars_recovered', 0)}/160")
    print(f"Spike correlation: {results.get('spike_correlation', 0.0):.3f}")
    
    if args.use_llm and orchestrator.llm_architect:
        print(f"\nLLM expansions used: {orchestrator.llm_architect.balloon_count}")
        print(orchestrator.llm_architect.get_history_summary())


if __name__ == '__main__':
    main()
```

---

## Installation Requirement

Add to existing requirements:
```
anthropic>=0.39.0
```

Or install on Vast.ai:
```bash
pip install anthropic --break-system-packages
```

---

## Stopping Conditions Summary

| Condition | Action |
|---|---|
| Bio vars ≥ target (default 120/160) | **SUCCESS** — stop |
| 3 consecutive LLM suggestions with no improvement | **Diminishing returns** — stop |
| Max LLM expansions reached (default 10) | **Budget limit** — stop |
| Total GPU hours exceeded (default 48h) | **Time limit** — stop |
| LLM API fails 3 times consecutively | **Fallback** to hardcoded, then stop after predefined |

---

## Expected Iteration Flow

```
Iterations 1-9:   Predefined templates T0-T8 (fixed, ~18h GPU)
                   DreamCoder accumulates 9 data points
                   
Iteration 10:     BALLOON EXPANSION #1
                   DreamCoder sleep phase → patterns
                   LLM reads: 9 results + patterns + gap
                   LLM suggests: T_llm_1 (targets specific gap)
                   
Iteration 11:     BALLOON EXPANSION #2
                   DreamCoder now has 10 data points
                   LLM reads: 10 results + updated patterns + new gap
                   Prompt is RICHER than iteration 10
                   LLM suggests: T_llm_2 (informed by T_llm_1 result)
                   
...continues until stopping condition met...
```

---

## Key Design Decisions

1. **LLM is ONLY used for balloon expansion** — not for short-segment verification, training, or scoring. Those remain deterministic.

2. **DreamCoder patterns are STRUCTURED data** fed to LLM — not raw numbers. The LLM receives "nrt_state never recovered by any architecture" not a 160-dim vector.

3. **History accumulates in the prompt** — each iteration makes the LLM's context richer. This is the meta-learning: the LLM's effective knowledge about YOUR specific problem grows with each iteration.

4. **Fallback is always available** — if API fails, hardcoded lookup table still works. The system degrades gracefully.

5. **Sonnet, not Opus** — architecture suggestion doesn't need maximum intelligence. Sonnet is sufficient and 5x cheaper. Can upgrade to Opus if suggestions seem poor.

6. **One JSON response per call** — keeps parsing simple and robust. No multi-turn conversation needed.
