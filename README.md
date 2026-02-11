# Le Masson 2002 Computational Replication

Computational replication of the hybrid biological-silicon thalamic circuit from **Le Masson et al. (2002)** "Feedback inhibition controls spike transfer in hybrid thalamic circuits" (*Nature* 417:854-858).

## What This Is

The original experiment connected a **biological TC neuron** (guinea-pig LGNd slice) to a **silicon nRt/PGN neuron** via bidirectional dynamic clamp. We replace the biological TC with a computational TC model, producing an **all-computational circuit**. If the bifurcation diagram matches the hybrid result, the TC model is validated as capturing whatever dynamical properties the real biological neuron contributed — direct evidence for **substrate independence** at the single-neuron level.

## Architecture

```
Retinal GC ──AMPA──> TC Cell ──AMPA──> nRt/PGN Cell
                       ^                    |
                       └──GABA_A + GABA_B───┘
```

- **TC Neuron**: Conductance-based single-compartment (I_Na, I_K, I_T, I_h, I_L, I_KL) — Destexhe et al. 1996
- **nRt Neuron**: Conductance-based single-compartment (I_Na, I_K, I_Ts, I_L) — Destexhe et al. 1996
- **Retinal Input**: Gamma-distributed ISI renewal process (Troy & Robson 1992)
- **Synapses**: Destexhe et al. 1994 kinetic formalism (AMPA, GABA_A, GABA_B)

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Quick smoke test (~15 seconds)
python run_all.py --test

# Run bifurcation diagram (primary deliverable, ~30 min)
python run_all.py --exp 4

# Run all experiments (~30 min with 10s sims)
python run_all.py

# Full publication-quality run (60s sims, several hours)
python run_all.py --full
```

## Parallel Execution (Vast.ai / Multi-Core)

The GABA conductance sweep is embarrassingly parallel. Use `run_parallel.py` for multi-core machines:

```bash
# Auto-detect cores, 10s simulations
python run_parallel.py

# Full run on 32-core machine
python run_parallel.py --full --workers 32

# Custom sweep
python run_parallel.py --full --workers 64 --gaba-min 0 --gaba-max 74 --gaba-step 1
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
| 4 | **Bifurcation diagram (primary deliverable)** | — | `experiments/exp4_bifurcation.py` |

## Validation Targets (from paper)

| Parameter | Target | Source |
|---|---|---|
| Oscillation threshold (GABA G_max) | 29 ± 4.2 nS (n=9) | Results |
| Spindle frequency | 9.26 ± 0.87 Hz (n=27) | Results |
| Spindle duration | 1.74 ± 0.36 s (n=27) | Results |
| TC input resistance | 68.1 ± 3.1 MΩ (n=19) | Methods |
| TC AP threshold | −44.4 ± 0.6 mV (n=11) | Methods |
| GABA_A:GABA_B ratio | 96:4 | Methods |

## Project Structure

```
le_masson_replication/
├── models/
│   ├── tc_neuron.py          # TC relay neuron (Destexhe/McCormick)
│   ├── nrt_neuron.py         # nRt/PGN interneuron
│   └── retinal_input.py      # Gamma-ISI spike generator
├── synapses/
│   ├── ampa.py               # AMPA kinetic synapse
│   ├── gabaa.py              # GABA_A kinetic synapse
│   └── gabab.py              # GABA_B kinetic synapse (metabotropic)
├── circuit/
│   └── thalamic_circuit.py   # Full circuit assembly
├── experiments/
│   ├── exp1_spindle.py       # Spindle wave generation
│   ├── exp2_correlation.py   # CI/CC vs inhibition
│   ├── exp3_noradrenaline.py # NA modulation
│   └── exp4_bifurcation.py   # Bifurcation diagram
├── analysis/
│   ├── spike_analysis.py     # Spike detection, cross-correlation, CI/CC
│   ├── oscillation.py        # Spindle detection, frequency analysis
│   └── plotting.py           # Figure generation
├── params/
│   └── default_params.json   # All parameters
├── figures/                  # Output figures
├── run_all.py                # Sequential runner
├── run_parallel.py           # Parallel runner for multi-core/cloud
└── README.md
```

## Key References

1. Le Masson et al. 2002 — *Nature* 417:854-858
2. Destexhe, Mainen & Sejnowski 1994 — *J Comput Neurosci* 1:195-230
3. Destexhe, Bal, McCormick & Sejnowski 1996 — *J Neurophysiol* 76:2049-2070
4. McCormick & Huguenard 1992 — *J Neurophysiol* 68:1384-1400

## Success Criteria

**Primary**: Bifurcation threshold of the all-computational circuit matches Le Masson's hybrid result (29 ± 4.2 nS).

**Interpretation**: If matched, the computational TC model captures the dynamical properties that the real biological neuron contributed to circuit function — validating substrate independence at the single-neuron dynamical level.
