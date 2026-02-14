"""Rung 3: Thalamic Transformation Replacement.

Replaces the entire TC-nRt circuit with learned input→output mappings
using three architectures:
  1. Volterra-Laguerre (GLVM with ridge regression)
  2. LSTM (2-layer with gate access)
  3. Neural ODE (torchdiffeq)

Novel contribution: comparing each model's internal representations
against biological ground truth intermediates.
"""
