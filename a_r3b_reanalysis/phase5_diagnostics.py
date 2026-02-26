"""
Phase 5: Diagnostic Classification — Zombie Verdict.

Classifies each (architecture, target, neuron, probe) result into one of
seven diagnostic categories based on the probe ladder + selectivity + baseline:

  LINEAR_ENCODING     — Ridge R² high, selective, trained > untrained
  NONLINEAR_ENCODING  — Ridge low, MLP high, selective, trained > untrained
  SCALE_AMBIGUITY     — Low R² but high Pearson r (Walch-Eisenberg scaling)
  GENUINE_ZOMBIE      — All probes fail, selective, Tier 1 target
  EXPECTED_NULL       — Tier 2/3 target that fails (non-identifiable)
  SPURIOUS            — High R² but NOT selective (autocorrelation artifact)
  STRUCTURAL          — High R² for both trained AND untrained (input artifact)

The per-architecture ZOMBIE VERDICT is the final output:
  "ENCODING" if any Tier 1 target shows LINEAR or NONLINEAR encoding
  "ZOMBIE" if all Tier 1 targets are GENUINE_ZOMBIE
  "AMBIGUOUS" otherwise
"""

import numpy as np
import pandas as pd
from pathlib import Path

from a_r3b_reanalysis.config import TARGET_REGISTRY, OUTPUT_DIR


# ============================================================
# Diagnostic Thresholds
# ============================================================

R2_THRESHOLD = 0.05         # Minimum R² to consider "encoding"
DELTA_R2_THRESHOLD = 0.05   # Minimum delta_R² (trained - untrained)
                             # Set to 0.05 (not 0.02) because noisy MLP probes
                             # produce per-neuron |delta_R²| > 0.02 by chance.
SELECTIVITY_THRESHOLD = 0.01  # Minimum selectivity to be "selective"
P_VALUE_THRESHOLD = 0.05    # Significance threshold
PEARSON_R_THRESHOLD = 0.3   # For detecting scale ambiguity


# ============================================================
# Classification
# ============================================================

def classify_result(row):
    """Classify a single result row into a diagnostic category.

    Parameters
    ----------
    row : dict or pd.Series
        Must contain: r2_trained, r2_untrained, delta_r2, selectivity,
                      p_value, pearson_trained, tier, probe_type

    Returns
    -------
    diagnosis : str
        One of the seven diagnostic categories.
    """
    r2_t = row['r2_trained']
    r2_u = row['r2_untrained']
    delta = row['delta_r2']
    sel = row['selectivity']
    p = row['p_value']
    pearson = row['pearson_trained']
    tier = row['tier']
    probe = row['probe_type']

    # Check if both trained and untrained have high R² → structural artifact
    if r2_t > R2_THRESHOLD and r2_u > R2_THRESHOLD and delta < DELTA_R2_THRESHOLD:
        return 'STRUCTURAL'

    # Check selectivity — if not selective, it's spurious
    if r2_t > R2_THRESHOLD and (sel < SELECTIVITY_THRESHOLD or p > P_VALUE_THRESHOLD):
        return 'SPURIOUS'

    # High R², selective, trained > untrained → genuine encoding
    if r2_t > R2_THRESHOLD and delta > DELTA_R2_THRESHOLD and sel > SELECTIVITY_THRESHOLD:
        if probe == 'ridge':
            return 'LINEAR_ENCODING'
        else:
            return 'NONLINEAR_ENCODING'

    # Low R² but high Pearson r → scale ambiguity (Walch-Eisenberg)
    if r2_t < R2_THRESHOLD and abs(pearson) > PEARSON_R_THRESHOLD:
        return 'SCALE_AMBIGUITY'

    # All probes fail on a Tier 1 identifiable target → genuine zombie
    if r2_t < R2_THRESHOLD and tier == 1:
        return 'GENUINE_ZOMBIE'

    # Tier 2 or 3 failure is expected (non-identifiable individual gates)
    if r2_t < R2_THRESHOLD and tier in (2, 3):
        return 'EXPECTED_NULL'

    return 'AMBIGUOUS'


def classify_all_results(results_df):
    """Apply diagnostic classification to all results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from phase4_pipeline.run_pipeline()

    Returns
    -------
    df : pd.DataFrame
        Original dataframe with 'diagnosis' column added.
    """
    df = results_df.copy()
    df['diagnosis'] = df.apply(classify_result, axis=1)
    return df


# ============================================================
# Aggregate Diagnostics
# ============================================================

def generate_diagnostic_summary(results_df, verbose=True):
    """Generate the full diagnostic summary.

    For each (architecture, target), aggregates across neurons and
    reports the dominant diagnosis. For each architecture, determines
    the zombie verdict.

    Parameters
    ----------
    results_df : pd.DataFrame
        With 'diagnosis' column from classify_all_results().

    Returns
    -------
    arch_summary : pd.DataFrame
        Per-architecture summary.
    target_summary : pd.DataFrame
        Per (architecture, target) summary.
    verdicts : dict
        architecture -> zombie verdict string.
    """
    df = results_df.copy()
    if 'diagnosis' not in df.columns:
        df = classify_all_results(df)

    # --- Per (architecture, target, probe) summary ---
    target_rows = []
    for (arch, target, probe), group in df.groupby(
            ['architecture', 'target', 'probe_type']):
        target_info = TARGET_REGISTRY.get(target, {})
        tier = target_info.get('tier', 0)
        timescale = target_info.get('timescale_ms', 0)

        diagnoses = group['diagnosis'].value_counts()
        dominant = diagnoses.index[0] if len(diagnoses) > 0 else 'UNKNOWN'

        target_rows.append({
            'architecture': arch,
            'target': target,
            'probe_type': probe,
            'tier': tier,
            'timescale_ms': timescale,
            'n_neurons': len(group),
            'r2_trained_mean': group['r2_trained'].mean(),
            'r2_trained_max': group['r2_trained'].max(),
            'delta_r2_mean': group['delta_r2'].mean(),
            'selectivity_mean': group['selectivity'].mean(),
            'n_significant': (group['p_value'] < P_VALUE_THRESHOLD).sum(),
            'dominant_diagnosis': dominant,
            'diagnosis_counts': diagnoses.to_dict(),
        })

    target_summary = pd.DataFrame(target_rows)

    # --- Per-architecture zombie verdict ---
    verdicts = {}
    arch_rows = []

    for arch in df['architecture'].unique():
        adf = df[df['architecture'] == arch]

        # VERDICT LOGIC: Ridge-primary.
        # Use Ridge probes (linear, stable) as primary evidence.
        # MLP probes are used only to detect NONLINEAR encoding when
        # Ridge shows nothing but MLP consistently does.
        tier1 = adf[adf['tier'] == 1]
        tier1_ridge = tier1[tier1['probe_type'] == 'ridge']
        tier1_ridge_diag = tier1_ridge['diagnosis'].value_counts()

        n_ridge_encoding = tier1_ridge_diag.get('LINEAR_ENCODING', 0)
        n_ridge_zombie = tier1_ridge_diag.get('GENUINE_ZOMBIE', 0)
        n_ridge_structural = tier1_ridge_diag.get('STRUCTURAL', 0)
        n_ridge_total = len(tier1_ridge)

        # Also check MLP for nonlinear encoding
        tier1_mlp = tier1[tier1['probe_type'].isin(['mlp_1', 'mlp_2'])]
        tier1_mlp_diag = tier1_mlp['diagnosis'].value_counts()
        n_mlp_encoding = tier1_mlp_diag.get('NONLINEAR_ENCODING', 0)

        # All probe types
        all_diag = tier1['diagnosis'].value_counts()
        n_encoding = sum(all_diag.get(d, 0)
                         for d in ['LINEAR_ENCODING', 'NONLINEAR_ENCODING'])
        n_zombie = all_diag.get('GENUINE_ZOMBIE', 0)
        n_total = len(tier1)

        # Best probe type for this architecture
        best_r2 = adf.groupby('probe_type')['r2_trained'].mean()

        # Decision: Ridge-primary verdict
        # ENCODING requires CONSISTENT Ridge evidence across neurons/targets.
        # A few neurons crossing threshold by noise doesn't count.
        # Require > 25% of Tier 1 Ridge results to show LINEAR_ENCODING.
        if n_ridge_encoding > n_ridge_total * 0.25:
            verdict = 'ENCODING'
            confidence = n_ridge_encoding / max(n_ridge_total, 1)
        elif n_ridge_encoding > 0:
            # Some evidence but not consistent — weak / single-neuron effect
            verdict = 'WEAK_ENCODING'
            confidence = n_ridge_encoding / max(n_ridge_total, 1)
        elif n_ridge_structural > n_ridge_total * 0.7:
            # Overwhelmingly structural — trained == untrained
            verdict = 'STRUCTURAL_ZOMBIE'
            confidence = n_ridge_structural / max(n_ridge_total, 1)
        elif n_ridge_zombie > n_ridge_total * 0.5:
            verdict = 'ZOMBIE'
            confidence = n_ridge_zombie / max(n_ridge_total, 1)
        else:
            verdict = 'AMBIGUOUS'
            confidence = 0.0

        verdicts[arch] = verdict

        arch_rows.append({
            'architecture': arch,
            'verdict': verdict,
            'confidence': confidence,
            'n_tier1_ridge_encoding': n_ridge_encoding,
            'n_tier1_ridge_structural': n_ridge_structural,
            'n_tier1_ridge_zombie': n_ridge_zombie,
            'n_tier1_encoding': n_encoding,
            'n_tier1_zombie': n_zombie,
            'n_tier1_total': n_total,
            'best_r2_ridge': float(best_r2.get('ridge', 0)),
            'best_r2_mlp1': float(best_r2.get('mlp_1', 0)),
            'best_r2_mlp2': float(best_r2.get('mlp_2', 0)),
            'n_total_results': len(adf),
        })

    arch_summary = pd.DataFrame(arch_rows)

    if verbose:
        _print_diagnostic_report(arch_summary, target_summary, verdicts)

    return arch_summary, target_summary, verdicts


def _print_diagnostic_report(arch_summary, target_summary, verdicts):
    """Print the diagnostic report to console."""
    print("\n" + "=" * 70)
    print("A-R3b ZOMBIE PROBE RE-ANALYSIS — DIAGNOSTIC REPORT")
    print("=" * 70)

    for _, row in arch_summary.iterrows():
        arch = row['architecture']
        verdict = row['verdict']
        conf = row['confidence']

        # Verdict formatting
        if verdict == 'ENCODING':
            verdict_str = f"ENCODING (confidence: {conf:.0%})"
        elif verdict == 'WEAK_ENCODING':
            verdict_str = f"WEAK ENCODING (confidence: {conf:.0%}) — not consistent across neurons"
        elif verdict == 'ZOMBIE':
            verdict_str = f"ZOMBIE (confidence: {conf:.0%})"
        elif verdict == 'STRUCTURAL_ZOMBIE':
            verdict_str = f"STRUCTURAL ZOMBIE (confidence: {conf:.0%})"
        else:
            verdict_str = f"AMBIGUOUS"

        print(f"\n  {arch.upper()}: {verdict_str}")
        print(f"    Tier 1: {row['n_tier1_encoding']} encoding, "
              f"{row['n_tier1_zombie']} zombie, "
              f"{row['n_tier1_total']} total")
        print(f"    Mean R²: Ridge={row['best_r2_ridge']:.4f}, "
              f"MLP-1={row['best_r2_mlp1']:.4f}, "
              f"MLP-2={row['best_r2_mlp2']:.4f}")

    # Tier breakdown
    print(f"\n--- Diagnosis Distribution by Tier ---")
    for tier in sorted(target_summary['tier'].unique()):
        tdf = target_summary[target_summary['tier'] == tier]
        diagnoses = {}
        for _, row in tdf.iterrows():
            d = row['dominant_diagnosis']
            diagnoses[d] = diagnoses.get(d, 0) + 1
        print(f"  Tier {tier}: {dict(diagnoses)}")

    # Overall
    print(f"\n--- Architecture Verdicts ---")
    for arch, verdict in verdicts.items():
        print(f"  {arch}: {verdict}")


def save_diagnostics(results_df, verbose=True):
    """Run full diagnostics and save to disk.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from phase4_pipeline.

    Returns
    -------
    output_paths : dict
        Paths to saved files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Classify
    df = classify_all_results(results_df)

    # Generate summaries
    arch_summary, target_summary, verdicts = generate_diagnostic_summary(
        df, verbose=verbose)

    # Save
    paths = {}

    p = OUTPUT_DIR / 'a_r3b_diagnoses.csv'
    df.to_csv(str(p), index=False)
    paths['diagnoses'] = p

    p = OUTPUT_DIR / 'a_r3b_arch_summary.csv'
    arch_summary.to_csv(str(p), index=False)
    paths['arch_summary'] = p

    p = OUTPUT_DIR / 'a_r3b_target_summary.csv'
    target_summary.to_csv(str(p), index=False)
    paths['target_summary'] = p

    if verbose:
        print(f"\nDiagnostics saved to {OUTPUT_DIR}")

    return paths
