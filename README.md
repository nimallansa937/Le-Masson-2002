# Le Masson 2002 Computational Replication

Computational replication of the hybrid biological-silicon thalamic circuit from **Le Masson et al. (2002)** "Feedback inhibition controls spike transfer in hybrid thalamic circuits" (*Nature* 417:854-858).

This project implements and tests the **DESCARTES ladder** (Distributional Equivalence through Substrate-independent Computational Architecture Replacement Testing and Evaluation System) for evaluating whether computational surrogates genuinely encode biological dynamics or are "computational zombies".

---

## Development Timeline

### Rung 1: Single-Neuron Circuit Replication (Feb 11, 2026)

**Guide**: Le Masson et al. 2002 (*Nature* 417:854-858); Destexhe, Bal, McCormick & Sejnowski 1996 (*J Neurophysiol* 76:2049-2070)

| Date | Commit | What |
|---|---|---|
| Feb 11 | `20c7502` | Implement Le Masson 2002 thalamic circuit replication |
| Feb 11 | `879e11b` | Fix oscillation detection and TC parameters for accurate bifurcation threshold |
| Feb 11 | `7e302af` | Use EC50 baseline-normalised threshold detection for bifurcation |

Full conductance-based TC + nRt/PGN circuit with GABA_A/GABA_B feedback inhibition. Bifurcation diagram validated against Le Masson's hybrid circuit threshold (29 +/- 4.2 nS). TC neuron model from Destexhe et al. 1996; synaptic kinetics from Destexhe, Mainen & Sejnowski 1994.

---

### Rung 2: Population Replacement (Feb 12-13, 2026)

**Guide**: DESCARTES Rung 2 protocol -- progressive replacement of biological neurons with validated computational models in a population circuit

| Date | Commit | What |
|---|---|---|
| Feb 12 | `98b19fb` | Add Rung 2 population replacement framework |
| Feb 12 | `80ca8a3` | Add live progress output to population runner |
| Feb 13 | `f4253a1` | Simplify progress output |
| Feb 13 | `21cfe29` | Add vectorized population circuit for ~12x speedup |

Replaced single TC neuron with a heterogeneous population (20 TC neurons). Progressive replacement strategy: random, hub-first, hub-last, spatial cluster. Vectorized NumPy implementation achieves 12x speedup over single-neuron loop.

---

### Rung 3: Transformation Replacement (Feb 14, 2026)

**Guide**: DESCARTES Rung 3 protocol -- replace biophysical dynamics with learned surrogates. Volterra-Laguerre model based on Marmarelis (2004) *Nonlinear Dynamic Modeling of Physiological Systems*; LSTM from Hochreiter & Schmidhuber 1997; Neural ODE from Chen et al. 2018 (*NeurIPS*)

| Date | Commit | What |
|---|---|---|
| Feb 14 | `a21de99` | Add Rung 3 transformation replacement framework (Volterra, LSTM, Neural ODE) |
| Feb 14 | `34c670d` | Add setup.py and remove sys.path hacks for proper package installation |
| Feb 14 | `67bc3cb` | Fix missing imports and non-ASCII print chars in rung3 scripts |
| Feb 14 | `8ed1662` | Fix Volterra overfitting and CCA performance bottleneck |

Three surrogate architectures trained on circuit trajectory data:

| Architecture | Type | Hidden Dim | Fitting Method |
|---|---|---|---|
| **Volterra-Laguerre** | Nonlinear system identification | 252 features | Ridge regression (closed-form) |
| **LSTM** | Recurrent neural network | 256 hidden units | PyTorch (SGD) |
| **Neural ODE** | Continuous-time dynamics | 64 hidden units | PyTorch + torchdiffeq |

Original A-R3 test (linear probes only): Volterra recovers 4/160, LSTM and Neural ODE recover 0/160. **Reversed by A-R3b** -- see Key Result below.

---

### DESCARTES Neural ODE Architecture Search (Feb 20-21, 2026)

**Guide**: DESCARTES open-ended search protocol with DreamCoder-inspired wake/sleep learning from failures; LLM-guided balloon expansion for architecture generation beyond human-designed templates

| Date | Commit | What |
|---|---|---|
| Feb 20 | `b52a39c` | Add DESCARTES-NeuralODE framework for systematic architecture search |
| Feb 20 | `f4181b9` | Fix validation OOM: batch validation forward pass to match training batch size |
| Feb 20 | `b74924c` | Add Vast.ai setup and run script |
| Feb 20 | `82fd915` | Fix subsample bug: binary targets were not subsampled with rate targets |
| Feb 20 | `c8f9a28` | Add progressive sequence training: 200 to 1000 to 2000 steps |
| Feb 20 | `1929893` | Add LLM-guided balloon expansion for open-ended architecture search |
| Feb 20 | `cb711b5` | Add 3 LLM decision points beyond balloon expansion |
| Feb 21 | `416f9fa` | Fix biovar recovery scoring: name mismatch caused 0/160 for all models |
| Feb 21 | `3bd1536` | Add LTC latent dimension sweep experiment |
| Feb 21 | `5212fec` | Add GRU-ODE biological alignment analysis |
| Feb 21 | `072f6b0` | Fix GRU-ODE training hyperparams: lr_patience=20, min_lr=1e-5, early_stop=40 |

Eight candidate architectures tested:

| Architecture | Reference |
|---|---|
| GRU-ODE | De Brouwer et al. 2019 |
| LTC Network | Hasani et al. 2021 (*Nature Machine Intelligence*) |
| Neural CDE | Kidger et al. 2020 (*NeurIPS*) |
| S4/Mamba | Gu et al. 2022 / Gu & Dao 2023 |
| coRNN | Rusch & Mishra 2021 (*ICLR*) |
| Hybrid LSTM-ODE | Combined discrete + continuous |
| Distilled ODE | Knowledge distillation approach |
| GRU-ODE-Bio | Auxiliary biological loss term |

---

### A-R3b: Zombie Probe Reanalysis (Feb 22-26, 2026)

**Guide**: DESCARTES A-R3b protocol -- extends A-R3 with untrained baseline control (Gate 7), identifiable combination targets, nonlinear MLP probe ladder, and block-permutation selectivity testing. Probe methodology inspired by Alain & Bengio 2017 (*ICLR*) "Understanding intermediate layers using linear classifier probes"

| Date | Commit | What |
|---|---|---|
| Feb 22 | `47638d8` | Add A-R3b experiment: GRU-ODE with auxiliary biological loss |
| Feb 22 | `0c1341e` | Fix critical per-trial bio variable alignment in A-R3b |
| Feb 22 | `5bed696` | Fix PyTorch 2.10 compat: total_mem to total_memory |
| Feb 26 | `76a1695` | Add A-R3b zombie probe reanalysis pipeline and results (765 probes) |

Pipeline phases:

| Phase | Function | Output |
|---|---|---|
| Phase 0 | Compute 17 identifiable targets from 76 HDF5 trials | 76 .npz target files |
| Phase 1 | Extract hidden states (trained + untrained) for 3 architectures | 3 hidden state .npz files |
| Phase 2 | Temporal block CV probes (Ridge, MLP-1, MLP-2) with trial grouping | Per-probe R2 scores |
| Phase 3 | Block permutation selectivity (20 permutations, Ridge only) | Selectivity + p-values |
| Phase 4 | Full orchestrator with incremental checkpointing and resume | 765-row results CSV |
| Phase 5 | 7-category diagnostic classification, Ridge-primary verdicts | Diagnosis CSV + summary |
| Phase 6 | Timescale plot, probe ladder, diagnosis heatmap | PDF + PNG figures |

---

## Key Result: A Dramatic Reversal

The A-R3b reanalysis **reverses the original A-R3 conclusions**. The headline numbers from averaging across probe types were misleading -- the per-target, per-neuron, per-probe breakdown tells a completely different story.

### The Reversal

| | A-R3 Claim | A-R3b Finding |
|---|---|---|
| **LSTM** | Zombie (0/160) | **Nonlinear encoder** (48/120 Tier 1) |
| **Volterra** | Not zombie (4/160) | **Structural zombie** (0/120 genuine) |
| **Architecture matters?** | Yes (Volterra > LSTM) | Yes, but **reversed** (LSTM > Volterra) |

### LSTM: Genuine Nonlinear Encoder -- NOT a Zombie

Ridge finds encoding in only 11 probes (all from neuron 0). But MLP-1 reveals **68 individual probes with Delta R2 > 0.05**, spanning 13 of 17 targets across all 5 neurons, every one passing selectivity (p = 0.048):

| Target | Timescale (ms) | MLP-1 Delta R2 | All 5 neurons? | Selectivity |
|---|---|---|---|---|
| I_GABA_B | 150 | 0.240 | Yes (0.17-0.29) | 0.096 |
| I_T | 5 | 0.155 | Yes (0.10-0.24) | 0.465 |
| tc_h_T | 20 | 0.134 | Yes (0.10-0.16) | 0.587 |
| tc_m_h | 100 | 0.132 | Yes (0.12-0.15) | 0.473 |
| I_h | 100 | 0.125 | Yes (0.10-0.15) | 0.507 |
| tc_m_T | 1 | 0.114 | Yes (0.10-0.12) | 0.509 |
| G_T | 5 | 0.106 | Yes (0.06-0.16) | 0.464 |
| gabaa_per_tc | 5 | 0.071 | 4/5 neurons | 0.088 |

The encoding is genuinely there -- it's just **nonlinear**. The LSTM stores biological variables in nonlinearly distributed representations across 256 hidden dimensions. Ridge can't decode this; a single-layer MLP can.

### Volterra: Structural Zombie -- Confirmed

Delta R2 = 0.000 for **every single Ridge probe** across all 17 targets and 5 neurons. MLP finds encoding in exactly one variable: G_h (4 neurons, Delta R2 = 0.06-0.40). Everything else is structural passthrough from the Laguerre basis. Verdict: STRUCTURAL_ZOMBIE, confidence 1.0. The original A-R3 finding of 4/160 recovered variables was a **false positive** -- the untrained network achieves identical R2.

### Neural ODE: Genuine Zombie -- Confirmed

Nothing found by any probe. The one possible signal (I_GABA_B Ridge Delta R2 ~ 0.10) has selectivity = 0.001, meaning temporal alignment doesn't matter -- it's spurious, correctly flagged. The 64-dimensional hidden state is too compressed to encode anything.

### Why A-R3 Got It Backwards

The original A-R3 used only linear probes (Ridge) and had no untrained baseline. This created two blind spots:

1. **Missed LSTM's nonlinear encoding.** The LSTM genuinely learns biological variables, but stores them in nonlinearly distributed representations. Ridge can't decode this; MLP can. Without the MLP probe ladder, this encoding is invisible.
2. **Mistook Volterra's structural passthrough for learned encoding.** The Laguerre basis functions structurally overlap with biological timescales. High R2 looked like recovery but Delta R2 = 0.000 -- the untrained network does identically. Without the untrained baseline (Gate 7), this artifact is undetectable.

### Three Novel Findings

1. **Encoding is nonlinear, not linear.** The probe complexity ladder works as designed: Ridge sees nothing, MLP-1 finds strong signal, MLP-2 overfits. The encoding exists but is distributed across dimensions in a nonlinear manifold.

2. **The G_h catastrophe.** Both LSTM and Volterra show extreme MLP instability for G_h specifically (Delta R2 = -1.57 for LSTM MLP-1, -1.10 for Volterra MLP-2). Yet I_h (= G_h x driving force) is perfectly recoverable. The network encodes the **current**, not the conductance -- a functional rather than mechanistic encoding.

3. **Neuron 0 is special.** All 11 Ridge-level encodings come from neuron_idx = 0. This TC neuron has a partially linear encoding, while the other 4 require nonlinear readout. Individual neurons in the circuit may be represented differently in the hidden state.

### Caveats

- **Permutation granularity**: With 20 permutations, the minimum p-value is 1/21 = 0.048. Every passing probe shows exactly this value. More permutations (100+) would give resolution, though consistency across 68 probes makes mass false positives unlikely.
- **MLP overfitting risk**: The G_h catastrophe shows MLPs can go wrong. The Volterra control is reassuring -- if MLP-1 were systematically overfitting, it would find false encoding in Volterra too. It doesn't (except G_h).

---

## Architecture

```
Retinal GC --AMPA--> TC Cell --AMPA--> nRt/PGN Cell
                       ^                    |
                       +--GABA_A + GABA_B---+
```

- **TC Neuron**: Conductance-based single-compartment (I_Na, I_K, I_T, I_h, I_L, I_KL)
- **nRt Neuron**: Conductance-based single-compartment (I_Na, I_K, I_Ts, I_L)
- **Retinal Input**: Gamma-distributed ISI renewal process (Troy & Robson 1992)
- **Synapses**: Destexhe et al. 1994 kinetic formalism (AMPA, GABA_A, GABA_B)

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib torch scikit-learn

# Rung 1: Bifurcation diagram (~30 min)
python run_all.py --exp 4

# Rung 3: Train surrogates
python -m rung3.run_rung3

# A-R3b Reanalysis: Full zombie probe pipeline (~10 hours)
cd le_masson_replication
python -m a_r3b_reanalysis.run_reanalysis --phase all --n-neurons 5 --n-perms 20

# Quick validation (~3 min)
python -m a_r3b_reanalysis.run_reanalysis --phase quick

# Generate report from existing results
python -m a_r3b_reanalysis.run_reanalysis --phase report

# Resume interrupted probe run
python -m a_r3b_reanalysis.run_reanalysis --phase probe --n-neurons 5 --n-perms 20 --resume
```

## Parallel Execution (Vast.ai / Multi-Core)

```bash
# Auto-detect cores
python run_parallel.py

# Full run on 32-core machine
python run_parallel.py --full --workers 32
```

| Machine | Full run (38 values x 60s) |
|---|---|
| 4-8 cores (laptop) | ~2-4 hours |
| 32 cores (Vast.ai) | ~15-20 min |
| 64 cores (Vast.ai) | ~8-10 min |

## Experiments

| # | Experiment | Paper Figure | Script |
|---|---|---|---|
| 1 | Spindle wave generation | Fig 1c, 2 | `experiments/exp1_spindle.py` |
| 2 | Input-output correlation vs inhibition | Fig 3 | `experiments/exp2_correlation.py` |
| 3 | Noradrenaline modulation | Fig 4 | `experiments/exp3_noradrenaline.py` |
| 4 | **Bifurcation diagram (primary deliverable)** | -- | `experiments/exp4_bifurcation.py` |

## Validation Targets (from paper)

| Parameter | Target | Source |
|---|---|---|
| Oscillation threshold (GABA G_max) | 29 +/- 4.2 nS (n=9) | Results |
| Spindle frequency | 9.26 +/- 0.87 Hz (n=27) | Results |
| Spindle duration | 1.74 +/- 0.36 s (n=27) | Results |
| TC input resistance | 68.1 +/- 3.1 MOhm (n=19) | Methods |
| TC AP threshold | -44.4 +/- 0.6 mV (n=11) | Methods |
| GABA_A:GABA_B ratio | 96:4 | Methods |

## Project Structure

```
le_masson_replication/
├── models/                        # Rung 1: Biophysical neuron models
│   ├── tc_neuron.py               #   TC relay neuron (Destexhe/McCormick)
│   ├── nrt_neuron.py              #   nRt/PGN interneuron
│   └── retinal_input.py           #   Gamma-ISI spike generator
├── synapses/                      # Kinetic synapses (Destexhe 1994)
│   ├── ampa.py
│   ├── gabaa.py
│   └── gabab.py
├── circuit/
│   └── thalamic_circuit.py        # Full circuit assembly
├── experiments/                   # Rung 1 experiments (Figs 1-4)
├── analysis/                      # Spike detection, oscillation, plotting
├── population/                    # Rung 2: Population replacement
│   ├── population_circuit.py      #   Single-neuron loop version
│   ├── population_circuit_fast.py #   Vectorized 12x speedup
│   ├── replacement.py             #   Population replacement test
│   └── heterogeneity.py           #   Parameter heterogeneity
├── rung3/                         # Rung 3: Transformation surrogates
│   ├── models/
│   │   ├── volterra_laguerre.py   #   Volterra-Laguerre (NumPy)
│   │   ├── lstm_model.py          #   LSTM (PyTorch)
│   │   └── neural_ode_model.py    #   Neural ODE (torchdiffeq)
│   ├── training/                  #   Training pipeline
│   └── evaluation/                #   CCA, output metrics
├── descartes_neural_ode/          # DESCARTES architecture search
│   ├── architectures/             #   8 candidate architectures
│   ├── orchestrator.py            #   LLM-guided search
│   └── run_ar3b.py                #   Auxiliary biological loss
├── a_r3b_reanalysis/              # A-R3b zombie probe reanalysis
│   ├── config.py                  #   17 target variables, thresholds
│   ├── phase0_identifiable_targets.py  # Compute targets from HDF5
│   ├── phase1_extract_hidden.py   #   Hidden state extraction
│   ├── phase2_probes.py           #   Ridge/MLP temporal block CV
│   ├── phase3_selectivity.py      #   Block permutation testing
│   ├── phase4_pipeline.py         #   Full orchestrator with checkpointing
│   ├── phase5_diagnostics.py      #   7-category classification
│   ├── phase6_plots.py            #   Timescale, ladder, heatmap plots
│   ├── run_reanalysis.py          #   CLI entry point
│   └── results/                   #   CSV data + PDF/PNG figures
├── params/
│   └── default_params.json
├── run_all.py                     # Rung 1 sequential runner
├── run_parallel.py                # Rung 1 parallel runner
└── README.md
```

## Key References

| # | Reference | Used For |
|---|---|---|
| 1 | Le Masson et al. 2002 -- *Nature* 417:854-858 | Rung 1: Original hybrid circuit experiment |
| 2 | Destexhe, Mainen & Sejnowski 1994 -- *J Comput Neurosci* 1:195-230 | Rung 1: Kinetic synapse models |
| 3 | Destexhe, Bal, McCormick & Sejnowski 1996 -- *J Neurophysiol* 76:2049-2070 | Rung 1: TC and nRt neuron models |
| 4 | McCormick & Huguenard 1992 -- *J Neurophysiol* 68:1384-1400 | Rung 1: TC ion channel kinetics |
| 5 | Troy & Robson 1992 -- *Visual Neuroscience* 9:535-553 | Rung 1: Retinal ganglion cell model |
| 6 | Marmarelis 2004 -- *Nonlinear Dynamic Modeling of Physiological Systems* | Rung 3: Volterra-Laguerre method |
| 7 | Hochreiter & Schmidhuber 1997 -- *Neural Computation* 9(8):1735-1780 | Rung 3: LSTM architecture |
| 8 | Chen et al. 2018 -- *NeurIPS* | Rung 3: Neural ODE framework |
| 9 | Hasani et al. 2021 -- *Nature Machine Intelligence* | DESCARTES: Liquid Time-Constant networks |
| 10 | De Brouwer et al. 2019 -- *NeurIPS* | DESCARTES: GRU-ODE-Bayes |
| 11 | Kidger et al. 2020 -- *NeurIPS* | DESCARTES: Neural CDE |
| 12 | Gu et al. 2022 / Gu & Dao 2023 | DESCARTES: S4/Mamba structured state spaces |
| 13 | Rusch & Mishra 2021 -- *ICLR* | DESCARTES: coRNN coupled oscillatory RNN |
| 14 | Alain & Bengio 2017 -- *ICLR* | A-R3b: Linear probe methodology |

## Success Criteria

**Rung 1 (Primary)**: Bifurcation threshold of the all-computational circuit matches Le Masson's hybrid result (29 +/- 4.2 nS).

**Rung 3 + A-R3b (Zombie Test)**: Do trained surrogates encode biological dynamics beyond what untrained networks provide structurally?

**Result: It depends on the architecture and the decoder.**

- **LSTM**: Yes. 68 probes across 13/17 targets show genuine learned encoding (Delta R2 > 0.05, all passing selectivity). But the encoding is **nonlinear** -- invisible to the linear probes used in A-R3.
- **Volterra**: No. Delta R2 = 0.000 universally. Structural passthrough only.
- **Neural ODE**: No. 64-dim hidden state too compressed for biological encoding.

**Interpretation**: The LSTM preserves biological information but in a **transformed format** -- a nonlinear manifold that maintains functional correspondence while restructuring internal representation. A linear reader misses it; a nonlinear reader recovers it. The Volterra achieves high R2 through structural passthrough, not learning. This distinction -- between genuinely learned nonlinear encoding and structural artifact -- is precisely what the A-R3b methodology (untrained baselines + probe ladder + selectivity) was designed to detect. The original A-R3 linear-only analysis got the answer exactly backwards.
