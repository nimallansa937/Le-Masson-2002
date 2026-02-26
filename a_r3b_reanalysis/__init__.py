"""
A-R3b: Zombie Probe Re-Analysis.

Resolves the A-R3 ambiguity (0/160 gating variables recovered) by adding:
  1. Identifiable combination targets (G_T, G_h, ionic currents)
  2. Nonlinear MLP probes alongside Ridge
  3. Block-permutation selectivity controls
  4. Untrained baseline subtraction (ΔR²)
  5. Temporal block cross-validation (trial-level grouping)
"""
