# Rung 3: Thalamic Transformation Replacement — Claude Code Implementation Guide

## Purpose

Replace the **entire TC-nRt circuit** with a learned input→output mapping. In Rung 1, a single neuron was swapped. In Rung 2, neurons were progressively replaced while the circuit architecture remained intact. Here, the architecture itself is removed — no TC neurons, no nRt neurons, no synapses — and a black-box model directly transforms retinal input spike trains into predicted TC output spike trains.

This is the MIMO paradigm applied to a thalamic relay circuit with full computational ground truth. It answers two questions:

1. **Format compatibility:** Does the learned transformation produce output that a downstream cortical circuit would accept? (spike timing, oscillatory structure, bifurcation behavior)
2. **Mechanistic equivalence (NOVEL):** Do the model's internal representations correspond to the biological intermediate variables (nRt membrane potentials, T-current gating, rebound burst timing) that were never in the training objective?

Question 2 is the core ARIA COGITO contribution. No prior study has answered it.

---

## What This Experiment Tests

### Rung 2 vs Rung 3: The Critical Distinction

**Rung 2** preserves the circuit architecture. Each replaced neuron still receives biological synaptic input, processes it through the validated computational model, and sends biological synaptic output. The circuit's connectivity and feedback loops remain intact. Substrate independence at Rung 2 means individual processing elements are interchangeable.

**Rung 3** destroys the circuit architecture. The entire feedback loop (TC excites nRt, nRt inhibits TC, rebound burst, repeat) is replaced by a feedforward mapping. If the learned model reproduces the output correctly, it means the *input-output transfer function* is learnable — but the internal computation may be completely different.

### The Philosophical Zombie Test

A Rung 3 model that perfectly reproduces TC output spikes but whose internal states bear no resemblance to nRt voltages, T-current activation, or rebound burst timing is a **computational zombie** — functionally equivalent at the boundary, mechanistically alien inside. This is precisely the scenario that the Berger/Marmarelis MIMO hippocampal prosthesis has never tested, despite decades of work and human clinical trials.

### What Success Looks Like

```
                        Spike      Bifurcation   Oscillatory    Latent ↔ Bio
Architecture            Accuracy   Threshold     Coherence      Correlation
────────────────────────────────────────────────────────────────────────────
Volterra-Laguerre       ≥ 80%      ±5 nS         ≥ 0.7×bio     LOW (expected)
LSTM                    ≥ 85%      ±3 nS         ≥ 0.8×bio     MEDIUM (?)
Neural ODE              ≥ 85%      ±3 nS         ≥ 0.8×bio     HIGH (?)
────────────────────────────────────────────────────────────────────────────
```

The latent-biological correlation column is the novel measurement. If Neural ODEs — which learn continuous dynamics — spontaneously recover biophysical states that Volterra kernels do not, this reveals something deep about the relationship between dynamical form and representational content.

### What Failure Modes Mean

1. **All architectures fail on bifurcation threshold** → The TC-nRt feedback loop is essential and cannot be captured by feedforward mapping. The oscillatory regime is an emergent property of recurrent dynamics, not a learnable function.

2. **Output matches but latents diverge across ALL architectures** → Computational zombie confirmed. Format compatibility achieved without mechanistic equivalence. Strong result for Paper 2.

3. **Output matches AND latents align for Neural ODE but not LSTM** → Continuous-time inductive bias recovers biology. Architectural constraints shape internal representations. Strongest possible result.

4. **Output matches AND latents align for ALL architectures** → The input-output mapping uniquely determines internal computation. No room for zombie solutions. Surprising but would suggest a deep identifiability result.

---

## Prerequisites

### From A-R2

The A-R2 population replacement experiment produces all data needed for A-R3. For each trial in the A-R2 sweep (GABA conductance × replacement fraction × seed), the following are available:

**Training inputs:**
- Retinal spike trains for all N_tc neurons (independent gamma-ISI processes), shape: `(N_tc, T_steps)`
- GABA conductance parameter for that trial

**Training targets (output):**
- TC somatic voltage traces, shape: `(N_tc, T_steps)`
- TC spike times extracted from voltage traces

**Ground truth intermediates (for post-hoc comparison only, NOT used in training):**
- nRt somatic voltage traces, shape: `(N_nrt, T_steps)`
- nRt spike times
- T-current gating variable (m_T, h_T) for each TC neuron, shape: `(N_tc, T_steps)`
- H-current gating variable for each TC neuron
- GABA_A and GABA_B synaptic conductance waveforms at each TC synapse
- AMPA synaptic conductance waveforms at each nRt synapse
- Per-synapse current traces

### What Needs to Be Added to A-R2

**The current A-R2 code must be extended to record intermediate variables.** The population simulation loop needs to save — in addition to voltages and spikes — the gating variable trajectories and synaptic conductance waveforms at every timestep.

This is Phase 0 of this guide.

---

## Phase 0: Extend A-R2 Data Recording

### 0.1 Intermediate Variable Recording

Modify `population_circuit.py` to record the following at each timestep (or at a subsampled rate, e.g., every 0.1 ms = every 4th step at dt=0.025 ms):

```python
# In PopulationCircuit class — add to __init__:
self.record_intermediates = True
self.subsample_factor = 4  # record every 4th step → 0.1 ms resolution

# New recording arrays (allocated after knowing T_steps):
if self.record_intermediates:
    n_record = T_steps // self.subsample_factor + 1
    self.tc_mT = np.zeros((self.n_tc, n_record))     # T-current activation gate
    self.tc_hT = np.zeros((self.n_tc, n_record))      # T-current inactivation gate
    self.tc_mH = np.zeros((self.n_tc, n_record))      # H-current activation gate
    self.nrt_V = np.zeros((self.n_nrt, n_record))     # nRt voltages
    self.nrt_mT = np.zeros((self.n_nrt, n_record))    # nRt T-current activation
    self.nrt_hT = np.zeros((self.n_nrt, n_record))    # nRt T-current inactivation
    self.gaba_a_cond = np.zeros((self.n_tc, n_record)) # Total GABA_A onto each TC
    self.gaba_b_cond = np.zeros((self.n_tc, n_record)) # Total GABA_B onto each TC
    self.ampa_cond = np.zeros((self.n_nrt, n_record))  # Total AMPA onto each nRt
```

### 0.2 Recording in the Integration Loop

```python
# Inside the main timestep loop:
if self.record_intermediates and step % self.subsample_factor == 0:
    idx = step // self.subsample_factor
    for i in range(self.n_tc):
        self.tc_mT[i, idx] = self.tc_neurons[i].m_T
        self.tc_hT[i, idx] = self.tc_neurons[i].h_T
        self.tc_mH[i, idx] = self.tc_neurons[i].m_H
        # Sum all GABA_A conductances projecting to TC_i
        self.gaba_a_cond[i, idx] = sum(
            syn.g for j, syn in self.gaba_a_syns[i]
        )
        self.gaba_b_cond[i, idx] = sum(
            syn.g for j, syn in self.gaba_b_syns[i]
        )
    for j in range(self.n_nrt):
        self.nrt_V[j, idx] = self.nrt_neurons[j].V
        self.nrt_mT[j, idx] = self.nrt_neurons[j].m_T
        self.nrt_hT[j, idx] = self.nrt_neurons[j].h_T
        self.ampa_cond[j, idx] = sum(
            syn.g for i, syn in self.ampa_syns[j]
        )
```

### 0.3 Data Export Format

Save each trial as an HDF5 file for efficient random access:

```python
import h5py

def save_trial_data(filepath, circuit, params):
    """Save complete trial data including intermediates."""
    with h5py.File(filepath, 'w') as f:
        # Metadata
        f.attrs['gaba_gmax'] = params['gaba_gmax']
        f.attrs['replacement_fraction'] = params['replacement_fraction']
        f.attrs['seed'] = params['seed']
        f.attrs['n_tc'] = circuit.n_tc
        f.attrs['n_nrt'] = circuit.n_nrt
        f.attrs['dt'] = params['dt']
        f.attrs['duration'] = params['duration']
        f.attrs['subsample_factor'] = circuit.subsample_factor
        
        # Input (retinal spike trains — store as sparse events)
        inp = f.create_group('input')
        for i in range(circuit.n_tc):
            inp.create_dataset(f'retinal_{i}', data=circuit.retinal_spike_times[i])
        
        # Output (TC spikes and voltages)
        out = f.create_group('output')
        out.create_dataset('tc_V', data=circuit.tc_V, compression='gzip')
        for i in range(circuit.n_tc):
            out.create_dataset(f'tc_spikes_{i}', data=circuit.tc_spike_times[i])
        
        # Ground truth intermediates
        gt = f.create_group('ground_truth')
        gt.create_dataset('tc_mT', data=circuit.tc_mT, compression='gzip')
        gt.create_dataset('tc_hT', data=circuit.tc_hT, compression='gzip')
        gt.create_dataset('tc_mH', data=circuit.tc_mH, compression='gzip')
        gt.create_dataset('nrt_V', data=circuit.nrt_V, compression='gzip')
        gt.create_dataset('nrt_mT', data=circuit.nrt_mT, compression='gzip')
        gt.create_dataset('nrt_hT', data=circuit.nrt_hT, compression='gzip')
        gt.create_dataset('gaba_a_cond', data=circuit.gaba_a_cond, compression='gzip')
        gt.create_dataset('gaba_b_cond', data=circuit.gaba_b_cond, compression='gzip')
        gt.create_dataset('ampa_cond', data=circuit.ampa_cond, compression='gzip')
        
        # Network topology (for reference)
        net = f.create_group('network')
        net.create_dataset('tc_to_nrt', data=circuit.tc_to_nrt)
        net.create_dataset('nrt_to_tc', data=circuit.nrt_to_tc)
        net.create_dataset('nrt_to_nrt', data=circuit.nrt_to_nrt)
```

### 0.4 Which A-R2 Trials to Use

**For A-R3 training data, use ONLY the 0% replacement (all-biological) trials.** These represent the ground-truth circuit operating normally.

From the A-R2 sweep design:
- GABA: [0, 5, 10, 15, 20, 25, 30, 35] nS × 3 seeds = **24 trials at 0% replacement**
- Each trial is ~60 seconds of simulation at dt=0.025 ms
- Total training data: ~24 × 60s = ~1440 seconds of input-output pairs

**Use the other replacement fractions (25%, 50%, 75%, 100%) as hold-out verification** — the model should never see data from replaced circuits during training.

---

## Phase 1: Data Preprocessing Pipeline

### 1.1 Spike Train Binning

Transform continuous spike times into discrete time series suitable for model training.

```python
import numpy as np

def bin_spike_trains(spike_times_list, dt_bin, duration):
    """
    Convert list of spike time arrays to binary matrix.
    
    Args:
        spike_times_list: list of arrays, each containing spike times (ms) for one neuron
        dt_bin: bin width in ms (e.g., 1.0 ms)
        duration: total duration in ms
    
    Returns:
        binary_matrix: (n_neurons, n_bins) binary spike matrix
    """
    n_bins = int(duration / dt_bin)
    n_neurons = len(spike_times_list)
    binary = np.zeros((n_neurons, n_bins), dtype=np.float32)
    
    for i, spikes in enumerate(spike_times_list):
        bin_indices = np.floor(spikes / dt_bin).astype(int)
        bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < n_bins)]
        binary[i, bin_indices] = 1.0
    
    return binary


def smooth_spike_trains(binary_matrix, sigma_ms, dt_bin):
    """
    Convolve binary spikes with Gaussian kernel to get firing rate estimate.
    Better training target than raw binary for continuous models.
    
    Args:
        binary_matrix: (n_neurons, n_bins)
        sigma_ms: Gaussian kernel width in ms
        dt_bin: bin width in ms
    
    Returns:
        rate_matrix: (n_neurons, n_bins) smoothed firing rates (Hz)
    """
    from scipy.ndimage import gaussian_filter1d
    sigma_bins = sigma_ms / dt_bin
    rate = gaussian_filter1d(binary_matrix.astype(float), sigma=sigma_bins, axis=1)
    rate *= (1000.0 / dt_bin)  # convert to Hz
    return rate.astype(np.float32)
```

### 1.2 Temporal Windowing

For LSTM and Neural ODE training, slice continuous recordings into overlapping windows:

```python
def create_training_windows(input_matrix, output_matrix, 
                            window_ms, stride_ms, dt_bin):
    """
    Slice (n_neurons, T) matrices into training windows.
    
    Returns:
        X: (n_windows, window_bins, n_input_neurons)
        Y: (n_windows, window_bins, n_output_neurons)
    """
    window_bins = int(window_ms / dt_bin)
    stride_bins = int(stride_ms / dt_bin)
    T = input_matrix.shape[1]
    
    starts = range(0, T - window_bins, stride_bins)
    X = np.stack([input_matrix[:, s:s+window_bins].T for s in starts])
    Y = np.stack([output_matrix[:, s:s+window_bins].T for s in starts])
    
    return X.astype(np.float32), Y.astype(np.float32)
```

### 1.3 Recommended Preprocessing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| dt_bin | 1.0 ms | Preserves spike timing to ±1 ms (sufficient for Victor-Purpura) |
| Smoothing σ | 5.0 ms | Matches typical synaptic integration time constant |
| Window length | 2000 ms (2 sec) | Captures full spindle oscillation cycle (~7-10 Hz = 100-140 ms period, need ≥10 cycles) |
| Window stride | 500 ms | 75% overlap, maximizes training examples |
| GABA as input feature | Yes | Include as static scalar appended to each timestep — the model must learn condition-dependent behavior |

### 1.4 Input/Output Specification

**Input to all models:**
- Retinal spike trains for all 20 TC neurons: `(20, T)` binary or smoothed
- GABA conductance parameter: scalar, broadcast to all timesteps
- **Total input dimensionality per timestep: 21** (20 retinal channels + 1 GABA param)

**Output (training target):**
- TC spike trains for all 20 neurons: `(20, T)` binary or smoothed
- **Total output dimensionality per timestep: 20**

**Ground truth intermediates (NOT training targets — used only for post-hoc latent comparison):**
- nRt voltages: `(20, T_sub)`
- TC T-current gating: `(20, T_sub)` for m_T, `(20, T_sub)` for h_T
- GABA conductance waveforms: `(20, T_sub)`
- Total intermediate dimensions: **100+** channels

---

## Phase 2: Model Architecture — Volterra-Laguerre (GLVM)

### 2.1 Background

The Volterra-Laguerre model is the architecture used by Berger/Marmarelis for the hippocampal MIMO prosthesis. It represents the input-output transformation as a series of Volterra kernels expanded in a Laguerre basis:

$$y(t) = k_0 + \sum_i \int k_1^i(\tau) x_i(t-\tau) d\tau + \sum_{i,j} \iint k_2^{ij}(\tau_1, \tau_2) x_i(t-\tau_1) x_j(t-\tau_2) d\tau_1 d\tau_2 + h \cdot y(t-\cdot)$$

- **k₁**: First-order kernels (linear filter per input channel) — captures basic synaptic integration
- **k₂**: Second-order kernels (pairwise nonlinear interactions) — captures paired-pulse facilitation, temporal summation
- **h**: Feedback kernel — captures refractory/burst dynamics via output history

### 2.2 Laguerre Basis Expansion

```python
def laguerre_basis(n_basis, memory_ms, dt_bin, alpha=0.5):
    """
    Generate Laguerre basis functions.
    
    Args:
        n_basis: number of basis functions (5-10 typically sufficient)
        memory_ms: temporal extent of kernels in ms
        dt_bin: bin width in ms
        alpha: Laguerre parameter (0 < alpha < 1), controls decay rate
               Higher alpha = slower decay = longer memory
    
    Returns:
        basis: (n_basis, memory_bins) Laguerre functions
    """
    memory_bins = int(memory_ms / dt_bin)
    t = np.arange(memory_bins)
    basis = np.zeros((n_basis, memory_bins))
    
    # Discrete Laguerre functions
    for j in range(n_basis):
        for k in range(j + 1):
            binom = np.math.comb(j, k)
            basis[j] += ((-1)**k * binom * 
                        ((1 - alpha)**((j - k) / 2)) * 
                        (alpha**((j + k) / 2)) *
                        (alpha**(t / 2)) * 
                        (1 - alpha)**(0.5) *
                        np.exp(-alpha * t / 2))  # simplified — use scipy.special.eval_laguerre for exact
    
    # Practical implementation: use recursive relation
    # L_0(t) = sqrt(1-alpha) * alpha^(t/2)
    # L_j(t) = alpha^(1/2) * L_{j-1}(t-1) + (1-alpha)^(1/2) * L_{j-1}(t)
    basis[0] = np.sqrt(1 - alpha) * alpha**(t / 2)
    for j in range(1, n_basis):
        basis[j, 0] = np.sqrt(alpha) * basis[j-1, 0]  # boundary
        basis[j, 1:] = (np.sqrt(alpha) * basis[j-1, :-1] + 
                         np.sqrt(1 - alpha) * basis[j-1, 1:])
    
    return basis


def project_to_laguerre(spike_train, basis):
    """
    Project spike train onto Laguerre basis functions.
    
    Args:
        spike_train: (T,) binary or rate
        basis: (n_basis, memory_bins)
    
    Returns:
        coefficients: (n_basis, T) Laguerre coefficients over time
    """
    from scipy.signal import fftconvolve
    n_basis = basis.shape[0]
    T = len(spike_train)
    coeffs = np.zeros((n_basis, T))
    for j in range(n_basis):
        conv = fftconvolve(spike_train, basis[j], mode='full')[:T]
        coeffs[j] = conv
    return coeffs
```

### 2.3 GLVM Model Class

```python
class VoltLagModel:
    """
    Generalized Laguerre-Volterra Model (GLVM).
    
    Second-order Volterra model with Laguerre basis expansion 
    and output feedback, following Marmarelis (2004).
    
    Input: N_in retinal channels + 1 GABA param
    Output: N_out TC spike probability per bin
    """
    
    def __init__(self, n_input=20, n_output=20, n_basis=7, 
                 memory_ms=200, alpha=0.5, dt_bin=1.0,
                 n_feedback_basis=5, feedback_memory_ms=50):
        self.n_input = n_input
        self.n_output = n_output
        self.n_basis = n_basis
        self.n_fb_basis = n_feedback_basis
        
        # Generate basis functions
        self.basis = laguerre_basis(n_basis, memory_ms, dt_bin, alpha)
        self.fb_basis = laguerre_basis(n_feedback_basis, feedback_memory_ms, dt_bin, alpha)
        
        # Parameters to fit (per output neuron):
        # k0: bias (1)
        # k1: first-order (n_input * n_basis)
        # k2: second-order (n_input * n_basis * (n_input * n_basis + 1) / 2) — symmetric
        # h: feedback (n_output * n_fb_basis)
        # GABA modulation: (n_basis) — scales all kernels by GABA level
        
        self.n_k1_params = n_input * n_basis
        self.n_k2_params = self.n_k1_params * (self.n_k1_params + 1) // 2
        
        # For tractability, use ONLY diagonal k2 (self-interactions) + cross-channel linear
        # Full k2 for 20 inputs × 7 basis = 140 dimensions → ~10K k2 params per output → overfitting
        # Diagonal k2: 20 inputs × 7×(7+1)/2 = 560 params per output → tractable
        self.n_k2_diag = n_input * (n_basis * (n_basis + 1) // 2)
        
        # Total params per output neuron
        self.params_per_output = (1 +                        # k0
                                  self.n_k1_params +          # k1
                                  self.n_k2_diag +            # k2 (diagonal)
                                  n_output * n_feedback_basis + # h
                                  1)                           # GABA scaling
        
        # Initialize parameter matrices
        self.k0 = np.zeros(n_output)
        self.k1 = np.zeros((n_output, n_input, n_basis))
        self.k2_diag = np.zeros((n_output, n_input, n_basis, n_basis))
        self.h = np.zeros((n_output, n_output, n_feedback_basis))
        self.gaba_scale = np.zeros(n_output)
    
    def compute_features(self, retinal_spikes, gaba, output_history=None):
        """
        Compute Laguerre coefficient features for all input channels.
        
        Args:
            retinal_spikes: (n_input, T) spike trains
            gaba: scalar GABA conductance
            output_history: (n_output, T) previous output (for feedback)
        
        Returns:
            v: (n_input, n_basis, T) Laguerre projections
            v_fb: (n_output, n_fb_basis, T) feedback projections
        """
        T = retinal_spikes.shape[1]
        v = np.zeros((self.n_input, self.n_basis, T))
        for i in range(self.n_input):
            v[i] = project_to_laguerre(retinal_spikes[i], self.basis)
        
        v_fb = None
        if output_history is not None:
            v_fb = np.zeros((self.n_output, self.n_fb_basis, T))
            for o in range(self.n_output):
                v_fb[o] = project_to_laguerre(output_history[o], self.fb_basis)
        
        return v, v_fb
    
    def predict(self, retinal_spikes, gaba, output_feedback=None):
        """
        Predict TC output given retinal input and GABA level.
        
        Returns:
            y_pred: (n_output, T) predicted TC firing rate
        """
        v, v_fb = self.compute_features(retinal_spikes, gaba, output_feedback)
        T = retinal_spikes.shape[1]
        y = np.zeros((self.n_output, T))
        
        for o in range(self.n_output):
            # k0 (bias)
            y[o] = self.k0[o]
            
            # k1 (linear: sum over inputs and basis functions)
            for i in range(self.n_input):
                for j in range(self.n_basis):
                    y[o] += self.k1[o, i, j] * v[i, j]
            
            # k2 (diagonal second-order: self-interactions per input)
            for i in range(self.n_input):
                for j1 in range(self.n_basis):
                    for j2 in range(j1, self.n_basis):
                        y[o] += self.k2_diag[o, i, j1, j2] * v[i, j1] * v[i, j2]
            
            # h (output feedback)
            if v_fb is not None:
                for o2 in range(self.n_output):
                    for j in range(self.n_fb_basis):
                        y[o] += self.h[o, o2, j] * v_fb[o2, j]
            
            # GABA modulation
            y[o] *= (1 + self.gaba_scale[o] * gaba)
        
        # Sigmoid nonlinearity → spike probability
        y = 1.0 / (1.0 + np.exp(-y))
        return y
    
    def fit(self, retinal_trains, tc_trains, gaba_values, 
            max_iter=1000, lr=0.01, l2_reg=1e-4):
        """
        Fit model parameters using iterative least squares or gradient descent.
        
        For practical implementation, reshape to linear regression problem:
        y = Φ @ w  (where Φ is the feature matrix, w is parameter vector)
        
        Then solve: w* = (Φ^T Φ + λI)^{-1} Φ^T y  (ridge regression)
        """
        # Implementation: construct feature matrix Φ from Laguerre projections
        # and solve per-output-neuron ridge regression
        # See Marmarelis (2004) Chapter 5 for exact procedure
        pass  # Claude Code will implement this
    
    def get_internal_features(self, retinal_spikes, gaba):
        """
        Extract the model's internal representations for latent comparison.
        
        For GLVM, internal features are the Laguerre coefficients — 
        the projected input filtered through the learned kernel structure.
        These are the "hidden states" of the Volterra model.
        
        Returns:
            features: dict with keys:
                'laguerre_coeffs': (n_input, n_basis, T)
                'k1_output': (n_output, T) — linear kernel contribution
                'k2_output': (n_output, T) — nonlinear kernel contribution
                'feedback_signal': (n_output, T) — feedback kernel contribution
        """
        v, v_fb = self.compute_features(retinal_spikes, gaba)
        
        features = {
            'laguerre_coeffs': v,
            'k1_output': np.zeros((self.n_output, v.shape[2])),
            'k2_output': np.zeros((self.n_output, v.shape[2])),
        }
        
        for o in range(self.n_output):
            for i in range(self.n_input):
                for j in range(self.n_basis):
                    features['k1_output'][o] += self.k1[o, i, j] * v[i, j]
                for j1 in range(self.n_basis):
                    for j2 in range(j1, self.n_basis):
                        features['k2_output'][o] += (
                            self.k2_diag[o, i, j1, j2] * v[i, j1] * v[i, j2])
        
        return features
```

### 2.4 Volterra-Laguerre Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_basis | 7 | Marmarelis standard; captures up to ~7th-order temporal features |
| memory_ms | 200 | Covers T-current deinactivation time (~100 ms) with margin |
| alpha | 0.5 | Start value; optimize via cross-validation (0.3–0.8 range) |
| feedback_memory_ms | 50 | Covers refractory period + initial rebound timing |
| n_feedback_basis | 5 | Sufficient for 50 ms memory |
| Regularization λ | 1e-4 | Ridge penalty; tune via CV |

---

## Phase 3: Model Architecture — LSTM

### 3.1 Architecture

```python
import torch
import torch.nn as nn

class TCReplacementLSTM(nn.Module):
    """
    LSTM model replacing the entire TC-nRt circuit.
    
    Input: (batch, time, 21) — 20 retinal channels + 1 GABA param
    Output: (batch, time, 20) — 20 TC spike probabilities
    
    Hidden state is the candidate "latent representation" that
    will be compared against biological ground truth.
    """
    
    def __init__(self, n_input=21, n_output=20, hidden_size=128, 
                 n_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Linear(n_input, hidden_size)
        
        # LSTM stack
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_output),
            nn.Sigmoid()  # spike probability
        )
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, time, 21)
            hidden: optional initial hidden state
        
        Returns:
            y: (batch, time, 20) predicted TC spike probabilities
            hidden: final hidden state
            all_hidden: (batch, time, hidden_size) all hidden states for latent analysis
        """
        # Input projection
        h = self.input_proj(x)  # (batch, time, hidden_size)
        
        # LSTM
        lstm_out, hidden = self.lstm(h, hidden)  # (batch, time, hidden_size)
        
        # Output
        y = self.output_proj(lstm_out)  # (batch, time, 20)
        
        return y, hidden, lstm_out  # lstm_out = all hidden states
    
    def get_internal_states(self, x):
        """
        Extract all internal representations for latent comparison.
        
        Returns:
            dict with:
                'hidden_states': (batch, time, hidden_size) — LSTM hidden activations
                'cell_states': (batch, time, hidden_size) — LSTM cell states
                'gate_values': dict of gate activations (i, f, g, o)
        """
        y, hidden, all_hidden = self.forward(x)
        
        # For gate extraction, need to hook into LSTM internals
        # Register hooks during forward pass
        return {
            'hidden_states': all_hidden.detach().cpu().numpy(),
            'predictions': y.detach().cpu().numpy()
        }
```

### 3.2 Gate Value Extraction (for Latent Comparison)

```python
class LSTMWithGateAccess(TCReplacementLSTM):
    """Extended LSTM that records gate activations for analysis."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_history = {}
    
    def forward_with_gates(self, x):
        """
        Manual LSTM forward pass to capture gate values.
        
        The LSTM gates (input, forget, output) are the closest analog
        to biological gating variables (m_T, h_T, m_H). If the LSTM
        spontaneously learns gate dynamics that correlate with T-current
        gating, this is evidence of mechanistic convergence.
        """
        batch, time, _ = x.shape
        h = self.input_proj(x)
        
        # Manual LSTM unrolling for first layer
        hx = torch.zeros(batch, self.hidden_size, device=x.device)
        cx = torch.zeros(batch, self.hidden_size, device=x.device)
        
        gates_i = []  # input gate
        gates_f = []  # forget gate
        gates_o = []  # output gate
        gates_g = []  # cell candidate
        hidden_states = []
        cell_states = []
        
        # Access first LSTM layer weights
        w_ih = self.lstm.weight_ih_l0  # (4*hidden, input)
        w_hh = self.lstm.weight_hh_l0  # (4*hidden, hidden)
        b_ih = self.lstm.bias_ih_l0
        b_hh = self.lstm.bias_hh_l0
        
        for t in range(time):
            xt = h[:, t, :]
            gates = xt @ w_ih.T + b_ih + hx @ w_hh.T + b_hh
            
            i = torch.sigmoid(gates[:, :self.hidden_size])
            f = torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
            g = torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])
            o = torch.sigmoid(gates[:, 3*self.hidden_size:])
            
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            
            gates_i.append(i.detach())
            gates_f.append(f.detach())
            gates_o.append(o.detach())
            gates_g.append(g.detach())
            hidden_states.append(hx.detach())
            cell_states.append(cx.detach())
        
        # Stack
        self.gate_history = {
            'input_gate': torch.stack(gates_i, dim=1).cpu().numpy(),
            'forget_gate': torch.stack(gates_f, dim=1).cpu().numpy(),
            'output_gate': torch.stack(gates_o, dim=1).cpu().numpy(),
            'cell_candidate': torch.stack(gates_g, dim=1).cpu().numpy(),
            'hidden_states': torch.stack(hidden_states, dim=1).cpu().numpy(),
            'cell_states': torch.stack(cell_states, dim=1).cpu().numpy(),
        }
        
        # Continue through remaining layers and output
        lstm_input = torch.stack(hidden_states, dim=1)
        if self.n_layers > 1:
            lstm_out, _ = self.lstm(h)  # full forward for remaining layers
        else:
            lstm_out = lstm_input
        
        y = self.output_proj(lstm_out)
        return y, self.gate_history
```

### 3.3 LSTM Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| hidden_size | 128 | Start moderate; the circuit has ~40 neurons with ~6 state vars each → ~240 biological dimensions. 128 is deliberately smaller — tests if compression works |
| n_layers | 2 | First layer learns input transformation, second learns recurrent dynamics |
| dropout | 0.1 | Light regularization |
| learning_rate | 1e-3 | Adam optimizer |
| batch_size | 32 windows | |
| epochs | 200 | With early stopping on validation loss |
| window_length | 2000 ms | |
| Loss function | Binary cross-entropy | For spike prediction; alternatively MSE on smoothed rates |

### 3.4 Hidden Size Sweep

**Critical design choice.** Run a secondary sweep over hidden sizes:

| hidden_size | Biological dimensions | Ratio | Hypothesis |
|-------------|----------------------|-------|------------|
| 32 | ~240 | 0.13× | Severe compression; output may degrade |
| 64 | ~240 | 0.27× | Moderate compression |
| 128 | ~240 | 0.53× | Near-matching capacity |
| 256 | ~240 | 1.07× | Overcomplete — can learn arbitrary representation |
| 512 | ~240 | 2.13× | Highly overcomplete |

If latent-biological correlation is highest at hidden_size ≈ 240 (matching biological dimensionality), this suggests the model naturally discovers the biological representation when given matching capacity. If correlation is equally high at 512, the mapping is robust. If it's low everywhere — zombie.

---

## Phase 4: Model Architecture — Neural ODE

### 4.1 Architecture

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint  # pip install torchdiffeq

class ODEFunc(nn.Module):
    """
    Neural ODE vector field: dz/dt = f(z, t, u)
    
    z: latent state
    u: current input (retinal + GABA)
    """
    
    def __init__(self, latent_dim, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.input_dim = input_dim
        self.current_input = None  # set externally per integration step
    
    def forward(self, t, z):
        # Concatenate latent state with current input
        if self.current_input is not None:
            z_in = torch.cat([z, self.current_input], dim=-1)
        else:
            z_in = torch.cat([z, torch.zeros(z.shape[0], self.input_dim, 
                                              device=z.device)], dim=-1)
        return self.net(z_in)


class TCReplacementNeuralODE(nn.Module):
    """
    Neural ODE model replacing the TC-nRt circuit.
    
    Encodes retinal input into a latent space, evolves the latent
    state via a learned ODE, and decodes to TC spike predictions.
    
    The latent trajectory is the candidate for biological comparison.
    A continuous-time model may naturally discover dynamics resembling
    the T-current gating variables and nRt rebound oscillations.
    """
    
    def __init__(self, n_input=21, n_output=20, latent_dim=64, 
                 hidden_dim=128, solver='dopri5', rtol=1e-3, atol=1e-4):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
        # Encoder: input → initial latent state
        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # ODE function
        self.ode_func = ODEFunc(latent_dim, n_input, hidden_dim)
        
        # Decoder: latent state → TC spike probability
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output),
            nn.Sigmoid()
        )
    
    def forward(self, x, dt_ms=1.0):
        """
        Args:
            x: (batch, time, 21) input sequence
            dt_ms: timestep in ms
        
        Returns:
            y: (batch, time, 20) predicted TC spike probabilities
            z_traj: (batch, time, latent_dim) latent trajectories
        """
        batch, time, _ = x.shape
        
        # Initialize latent state from first input
        z0 = self.encoder(x[:, 0, :])  # (batch, latent_dim)
        
        # Integrate ODE step by step (input-driven)
        z_traj = [z0]
        z = z0
        
        t_span = torch.tensor([0.0, dt_ms], device=x.device)
        
        for t in range(1, time):
            self.ode_func.current_input = x[:, t, :]
            z_next = odeint(self.ode_func, z, t_span, 
                           method=self.solver,
                           rtol=self.rtol, atol=self.atol)[-1]
            z_traj.append(z_next)
            z = z_next
        
        z_traj = torch.stack(z_traj, dim=1)  # (batch, time, latent_dim)
        
        # Decode all timesteps
        y = self.decoder(z_traj)  # (batch, time, 20)
        
        return y, z_traj
    
    def get_latent_trajectory(self, x, dt_ms=1.0):
        """
        Extract latent trajectories for comparison against biological ground truth.
        
        The latent dimensions are the "hidden states" that will be compared
        to nRt voltages, T-current gating, etc.
        """
        with torch.no_grad():
            y, z_traj = self.forward(x, dt_ms)
        return {
            'latent_trajectory': z_traj.cpu().numpy(),
            'predictions': y.cpu().numpy()
        }
```

### 4.2 Neural ODE Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| latent_dim | 64 | Start smaller than biological (~240 dims). If 64 suffices for output AND recovers biology, strong compression result |
| hidden_dim | 128 | ODE function network width |
| solver | dopri5 | Adaptive Runge-Kutta; good default for moderate stiffness |
| rtol, atol | 1e-3, 1e-4 | Moderate tolerance; tighten if dynamics are stiff (oscillatory regime) |
| learning_rate | 5e-4 | Lower than LSTM — ODE gradients via adjoint method can be noisy |
| epochs | 300 | Neural ODEs converge slower; longer training |

### 4.3 Latent Dimension Sweep

Same logic as LSTM hidden_size sweep:

| latent_dim | Bio dims | Test |
|------------|----------|------|
| 16 | 240 | Extreme compression |
| 32 | 240 | Heavy compression |
| 64 | 240 | Moderate |
| 128 | 240 | Near-matching |
| 256 | 240 | Overcomplete |

---

## Phase 5: Training Protocol

### 5.1 Data Split

```python
# From A-R2 0%-replacement trials:
# 24 trials total (8 GABA × 3 seeds)
# Split: 16 train (2 seeds per GABA), 8 validation (1 seed per GABA)

# CRITICAL: Split by seed, not by time within trials
# Otherwise temporal autocorrelation leaks across train/val

train_seeds = [0, 1]  # first 2 seeds
val_seeds = [2]        # held-out seed

# Also hold out entire GABA conditions for extrapolation test
# Train on GABA = [0, 5, 10, 15, 25, 30, 35], validate on GABA = 20
# Tests whether the model can interpolate to unseen GABA levels
```

### 5.2 Training Loop (LSTM Example)

```python
def train_lstm(model, train_loader, val_loader, config):
    """Standard PyTorch training loop with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for X, Y in train_loader:
            X, Y = X.to(config['device']), Y.to(config['device'])
            optimizer.zero_grad()
            
            y_pred, _, _ = model(X)
            
            # Binary cross-entropy for spike prediction
            loss = nn.functional.binary_cross_entropy(y_pred, Y)
            
            # Optional: add MSE on smoothed rates as auxiliary loss
            # loss += 0.1 * nn.functional.mse_loss(
            #     gaussian_smooth(y_pred), gaussian_smooth(Y))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(config['device']), Y.to(config['device'])
                y_pred, _, _ = model(X)
                val_loss += nn.functional.binary_cross_entropy(y_pred, Y).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
    
    model.load_state_dict(torch.load('best_lstm.pt'))
    return model
```

### 5.3 Loss Function Considerations

| Loss | Pros | Cons |
|------|------|------|
| Binary CE | Natural for spike prediction | Sensitive to bin size; doesn't capture timing |
| MSE on smoothed rates | Captures rate dynamics smoothly | Loses spike-timing precision |
| Victor-Purpura metric | Gold standard for spike trains | Not differentiable; can't use as training loss |
| Hybrid: BCE + rate MSE | Best of both | Two hyperparameters to tune |

**Recommendation:** Train with **BCE + 0.1 × MSE on smoothed rates (σ=5 ms)**. Evaluate with **Victor-Purpura distance** post-hoc.

---

## Phase 6: Evaluation — Output Comparison

### 6.1 Spike-Level Metrics

```python
from scipy.spatial.distance import cdist

def victor_purpura_distance(spike_train_1, spike_train_2, q=0.5):
    """
    Victor-Purpura spike train distance.
    
    q: temporal precision parameter (1/ms)
       q=0: only spike count matters
       q→∞: exact timing required
       q=0.5: ~2 ms tolerance (appropriate for thalamic relay)
    """
    # Implementation via dynamic programming
    n1 = len(spike_train_1)
    n2 = len(spike_train_2)
    
    D = np.zeros((n1 + 1, n2 + 1))
    D[:, 0] = np.arange(n1 + 1)
    D[0, :] = np.arange(n2 + 1)
    
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost_move = q * abs(spike_train_1[i-1] - spike_train_2[j-1])
            D[i, j] = min(
                D[i-1, j] + 1,           # delete spike from train 1
                D[i, j-1] + 1,           # insert spike into train 1
                D[i-1, j-1] + cost_move  # move spike
            )
    
    return D[n1, n2]


def spike_train_correlation(pred_spikes, true_spikes, bin_ms=5.0, sigma_ms=10.0):
    """
    Pearson correlation between smoothed spike trains.
    Simpler metric than VP; good for overall rate tracking.
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Bin and smooth
    max_t = max(pred_spikes.max() if len(pred_spikes) else 0,
                true_spikes.max() if len(true_spikes) else 0) + 100
    n_bins = int(max_t / bin_ms)
    
    pred_binned = np.histogram(pred_spikes, bins=n_bins, range=(0, max_t))[0]
    true_binned = np.histogram(true_spikes, bins=n_bins, range=(0, max_t))[0]
    
    pred_smooth = gaussian_filter1d(pred_binned.astype(float), sigma_ms / bin_ms)
    true_smooth = gaussian_filter1d(true_binned.astype(float), sigma_ms / bin_ms)
    
    if pred_smooth.std() == 0 or true_smooth.std() == 0:
        return 0.0
    
    return np.corrcoef(pred_smooth, true_smooth)[0, 1]
```

### 6.2 Bifurcation Threshold Test

The critical population-level test from Rung 2, now applied to the transformation model:

```python
def test_bifurcation_preservation(model, retinal_data, gaba_range, 
                                   bio_threshold, bio_threshold_sd):
    """
    Test whether the transformation model reproduces the 
    bifurcation from tonic to oscillatory mode.
    
    For each GABA level, feed retinal input through the model
    and measure whether the output transitions from tonic (low PR)
    to oscillatory (high PR) at the same threshold as biology.
    """
    pause_rates = []
    
    for gaba in gaba_range:
        # Construct input with GABA parameter
        x = prepare_model_input(retinal_data, gaba)
        
        # Get model prediction
        y_pred = model.predict(x)  # (n_tc, T) spike probabilities
        
        # Threshold to get predicted spikes
        pred_spikes = threshold_to_spikes(y_pred, threshold=0.5)
        
        # Compute pause rate
        pr = compute_pause_rate(pred_spikes)
        pause_rates.append(pr)
    
    # Find threshold: GABA where PR crosses midpoint
    pred_threshold = find_bifurcation_threshold(gaba_range, pause_rates)
    
    # Compare to biological
    threshold_error = abs(pred_threshold - bio_threshold)
    within_1sd = threshold_error <= bio_threshold_sd
    within_2sd = threshold_error <= 2 * bio_threshold_sd
    
    return {
        'predicted_threshold': pred_threshold,
        'biological_threshold': bio_threshold,
        'error_nS': threshold_error,
        'within_1sd': within_1sd,
        'within_2sd': within_2sd,
        'pause_rate_curve': list(zip(gaba_range, pause_rates))
    }
```

### 6.3 Population Coherence Test

```python
def test_coherence_preservation(model, retinal_data, gaba_oscillatory,
                                 bio_coherence_baseline):
    """
    During oscillatory regime (high GABA), does the model's output
    show population-level coherence?
    
    This tests whether the transformation captures the TC-nRt 
    feedback that synchronizes the population, or whether it 
    produces individually correct but collectively incoherent output.
    """
    x = prepare_model_input(retinal_data, gaba_oscillatory)
    y_pred = model.predict(x)
    pred_spikes = threshold_to_spikes(y_pred)
    
    # Pairwise phase consistency
    coherence = compute_pairwise_phase_consistency(pred_spikes)
    
    # Compare to biological baseline
    ratio = coherence / bio_coherence_baseline
    
    return {
        'model_coherence': coherence,
        'bio_coherence': bio_coherence_baseline,
        'ratio': ratio,
        'preserved': ratio >= 0.7  # ≥70% of biological coherence
    }
```

---

## Phase 7: Evaluation — Latent Variable Comparison (NOVEL)

**This is the core ARIA COGITO contribution.** No prior study has performed this comparison.

### 7.1 Ground Truth Intermediate Variables

From the A-R2 biological (0% replacement) simulations, extract time-aligned vectors of all intermediate variables:

```python
def load_biological_ground_truth(trial_path):
    """
    Load biophysical ground truth intermediate variables.
    
    Returns dict of (n_neurons, T_sub) arrays at 0.1 ms resolution.
    """
    with h5py.File(trial_path, 'r') as f:
        gt = {
            # TC gating variables (the "hidden states" of biological TC neurons)
            'tc_mT': f['ground_truth/tc_mT'][:],    # T-current activation
            'tc_hT': f['ground_truth/tc_hT'][:],    # T-current inactivation
            'tc_mH': f['ground_truth/tc_mH'][:],    # H-current activation
            
            # nRt dynamics (entirely internal to the circuit)
            'nrt_V': f['ground_truth/nrt_V'][:],     # nRt membrane voltages
            'nrt_mT': f['ground_truth/nrt_mT'][:],   # nRt T-current activation
            'nrt_hT': f['ground_truth/nrt_hT'][:],   # nRt T-current inactivation
            
            # Synaptic conductances (the "communication channels")
            'gaba_a': f['ground_truth/gaba_a_cond'][:],
            'gaba_b': f['ground_truth/gaba_b_cond'][:],
            'ampa': f['ground_truth/ampa_cond'][:],
        }
    return gt
```

### 7.2 Model Latent Variable Extraction

```python
def extract_model_latents(model, model_type, input_data):
    """
    Extract internal representations from each architecture.
    
    Returns:
        latents: (latent_dim, T) matrix of model hidden states
    """
    if model_type == 'volterra':
        features = model.get_internal_features(
            input_data['retinal'], input_data['gaba'])
        # Reshape Laguerre coefficients: (n_input × n_basis, T)
        latents = features['laguerre_coeffs'].reshape(-1, features['laguerre_coeffs'].shape[-1])
        
    elif model_type == 'lstm':
        x = torch.tensor(input_data['x_tensor']).unsqueeze(0)
        y, gate_history = model.forward_with_gates(x)
        # Use hidden states: (hidden_size, T)
        latents = gate_history['hidden_states'][0].T  # (hidden_size, T)
        # Also extract gates separately for detailed analysis
        gates = {k: v[0].T for k, v in gate_history.items()}
        return latents, gates
        
    elif model_type == 'neural_ode':
        x = torch.tensor(input_data['x_tensor']).unsqueeze(0)
        result = model.get_latent_trajectory(x)
        latents = result['latent_trajectory'][0].T  # (latent_dim, T)
    
    return latents
```

### 7.3 Canonical Correlation Analysis (CCA)

The primary statistical method for comparing two sets of time series with different dimensionalities:

```python
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

def latent_bio_cca(model_latents, bio_ground_truth, n_components=10):
    """
    Canonical Correlation Analysis between model latent variables
    and biological ground truth intermediates.
    
    Args:
        model_latents: (latent_dim, T) model hidden states
        bio_ground_truth: dict of (n_neurons, T_sub) biological intermediates
    
    Returns:
        cca_results: dict with canonical correlations and component loadings
    """
    # Stack all biological variables into single matrix
    bio_vars = []
    bio_labels = []
    for name, data in bio_ground_truth.items():
        for i in range(data.shape[0]):
            bio_vars.append(data[i])
            bio_labels.append(f"{name}_{i}")
    
    bio_matrix = np.array(bio_vars)  # (n_bio_dims, T_sub)
    
    # Align temporal resolution
    # Model latents at dt_bin (1 ms), bio at 0.1 ms → subsample bio
    subsample = int(1.0 / 0.1)  # every 10th bio sample ≈ 1 ms
    bio_aligned = bio_matrix[:, ::subsample]
    
    # Trim to matching length
    T_common = min(model_latents.shape[1], bio_aligned.shape[1])
    X = model_latents[:, :T_common].T  # (T, latent_dim)
    Y = bio_aligned[:, :T_common].T     # (T, n_bio_dims)
    
    # Standardize
    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y)
    
    # CCA
    n_comp = min(n_components, X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_comp)
    X_c, Y_c = cca.fit_transform(X, Y)
    
    # Canonical correlations
    canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] 
                       for i in range(n_comp)]
    
    return {
        'canonical_correlations': canonical_corrs,
        'mean_cc': np.mean(canonical_corrs),
        'max_cc': np.max(canonical_corrs),
        'x_loadings': cca.x_loadings_,   # which model dims contribute
        'y_loadings': cca.y_loadings_,   # which bio dims contribute
        'bio_labels': bio_labels,
        'n_components': n_comp
    }
```

### 7.4 Representational Similarity Analysis (RSA)

Complementary to CCA — tests whether the *geometry* of representations matches:

```python
def representational_similarity(model_latents, bio_ground_truth, 
                                 n_timepoints=500):
    """
    Compare representational geometry between model and biology.
    
    Compute RDMs (representational dissimilarity matrices) for random
    timepoint pairs, then correlate the RDMs.
    
    High RSA correlation = model and biology represent different 
    conditions (GABA levels, oscillatory states) in similar geometric 
    relationships, even if individual dimensions don't align.
    """
    # Sample random timepoints
    T = min(model_latents.shape[1], 
            list(bio_ground_truth.values())[0].shape[1])
    idx = np.random.choice(T, size=min(n_timepoints, T), replace=False)
    idx.sort()
    
    # Model RDM
    X = model_latents[:, idx].T  # (n_timepoints, latent_dim)
    model_rdm = cdist(X, X, metric='correlation')
    
    # Biology RDM
    bio_vars = np.concatenate([v[:, idx] for v in bio_ground_truth.values()], axis=0)
    Y = bio_vars.T  # (n_timepoints, n_bio_dims)
    bio_rdm = cdist(Y, Y, metric='correlation')
    
    # Correlate RDMs (Spearman — rank correlation of off-diagonal elements)
    from scipy.stats import spearmanr
    mask = np.triu_indices(len(idx), k=1)
    rsa_corr, rsa_p = spearmanr(model_rdm[mask], bio_rdm[mask])
    
    return {
        'rsa_correlation': rsa_corr,
        'rsa_pvalue': rsa_p,
        'model_rdm': model_rdm,
        'bio_rdm': bio_rdm
    }
```

### 7.5 Individual Variable Recovery

The most interpretable test — can we find specific model dimensions that track specific biological variables?

```python
def individual_variable_recovery(model_latents, bio_ground_truth):
    """
    For each biological intermediate variable, find the model latent
    dimension with highest correlation. This is the simplest test of
    whether the model spontaneously discovers biologically meaningful
    representations.
    
    Reports:
        - Best correlation for each biological variable
        - Which model dimension it maps to
        - Whether correlations are significant after Bonferroni correction
    """
    from scipy.stats import pearsonr
    
    results = {}
    n_bio = sum(v.shape[0] for v in bio_ground_truth.values())
    bonferroni = n_bio * model_latents.shape[0]  # number of comparisons
    
    for var_name, var_data in bio_ground_truth.items():
        for i in range(var_data.shape[0]):
            bio_trace = var_data[i]
            
            # Align temporal resolution
            T = min(model_latents.shape[1], len(bio_trace))
            bio_aligned = np.interp(
                np.linspace(0, 1, T),
                np.linspace(0, 1, len(bio_trace)),
                bio_trace
            )
            
            best_corr = 0
            best_dim = -1
            best_p = 1.0
            
            for d in range(model_latents.shape[0]):
                model_trace = model_latents[d, :T]
                r, p = pearsonr(model_trace, bio_aligned[:T])
                if abs(r) > abs(best_corr):
                    best_corr = r
                    best_dim = d
                    best_p = p
            
            key = f"{var_name}_{i}"
            results[key] = {
                'best_correlation': best_corr,
                'best_model_dim': best_dim,
                'p_value': best_p,
                'significant_bonferroni': best_p < 0.05 / bonferroni,
                'abs_corr': abs(best_corr)
            }
    
    # Summary statistics
    all_corrs = [abs(r['best_correlation']) for r in results.values()]
    n_significant = sum(1 for r in results.values() if r['significant_bonferroni'])
    
    return {
        'per_variable': results,
        'mean_abs_correlation': np.mean(all_corrs),
        'median_abs_correlation': np.median(all_corrs),
        'max_abs_correlation': np.max(all_corrs),
        'n_significant': n_significant,
        'n_total': len(results),
        'fraction_significant': n_significant / len(results)
    }
```

### 7.6 Comparison Across Architectures

```python
def compare_architectures(results_dict):
    """
    Compare Volterra, LSTM, and Neural ODE on all metrics.
    
    results_dict: {
        'volterra': {'output_metrics': ..., 'cca': ..., 'rsa': ..., 'recovery': ...},
        'lstm': {...},
        'neural_ode': {...}
    }
    """
    comparison = {}
    
    for arch, results in results_dict.items():
        comparison[arch] = {
            # Output quality
            'spike_correlation': results['output_metrics']['mean_spike_corr'],
            'bifurcation_error': results['output_metrics']['threshold_error'],
            'coherence_ratio': results['output_metrics']['coherence_ratio'],
            
            # Latent alignment (THE NOVEL METRICS)
            'cca_mean': results['cca']['mean_cc'],
            'cca_max': results['cca']['max_cc'],
            'rsa_correlation': results['rsa']['rsa_correlation'],
            'var_recovery_mean': results['recovery']['mean_abs_correlation'],
            'var_recovery_significant': results['recovery']['fraction_significant'],
        }
    
    return comparison
```

### 7.7 Interpretation Guide

| Scenario | Output match? | Latent correlation | Interpretation |
|----------|:---:|:---:|---|
| A | ✅ High | ❌ Low (all archs) | **Computational zombie.** Format compatibility without mechanistic equivalence. Same output, alien internals. Strongest philosophical result. |
| B | ✅ High | ⚠️ Medium (ODE only) | **Inductive bias matters.** Continuous dynamics architecture naturally recovers biology; discrete sequence models don't. Architecture shapes representation. |
| C | ✅ High | ✅ High (all archs) | **Identifiability.** The I/O mapping uniquely constrains internals. No room for zombie solutions. Surprising but mathematically deep. |
| D | ❌ Low | N/A | **Transformation not learnable.** The TC-nRt feedback dynamics cannot be captured by feedforward mapping. Recurrence is essential. |
| E | ✅ High | ✅ High (ODE+LSTM) but ❌ Low (Volterra) | **Representational capacity matters.** Volterra's fixed basis is too rigid; flexible architectures discover biology. Expected result given Volterra's linearity. |
| F | ✅ Moderate | ⚠️ Varies by variable | **Partial equivalence.** Some biological intermediates are recovered (e.g., nRt timing) while others aren't (e.g., gating details). Maps which aspects of the computation are load-bearing. |

---

## Phase 8: Project Structure

```
Le-Masson-2002/
├── models/                              # ← Existing (Rung 1)
├── synapses/                            # ← Existing (Rung 1)
├── circuit/                             # ← Existing (Rung 1)
├── population/                          # ← Existing (Rung 2)
│   └── population_circuit.py            # MODIFIED: add intermediate recording
├── transformation/                      # ← NEW (Rung 3)
│   ├── data/
│   │   ├── preprocess.py               # Binning, smoothing, windowing
│   │   ├── dataset.py                  # PyTorch Dataset class
│   │   └── export_training_data.py     # Extract from A-R2 HDF5 → training format
│   ├── models/
│   │   ├── volterra_laguerre.py        # GLVM implementation
│   │   ├── lstm_replacement.py         # LSTM with gate access
│   │   └── neural_ode_replacement.py   # Neural ODE with latent access
│   ├── training/
│   │   ├── train_volterra.py           # Ridge regression fitting
│   │   ├── train_lstm.py              # PyTorch training loop
│   │   └── train_neural_ode.py        # Neural ODE training (adjoint method)
│   ├── evaluation/
│   │   ├── output_metrics.py          # VP distance, correlation, coherence
│   │   ├── bifurcation_test.py        # Threshold preservation
│   │   ├── latent_comparison.py       # CCA, RSA, individual recovery
│   │   └── architecture_comparison.py # Cross-model comparison table
│   ├── experiments/
│   │   ├── exp9_volterra_fit.py       # Train + evaluate Volterra
│   │   ├── exp10_lstm_fit.py          # Train + evaluate LSTM
│   │   ├── exp11_neural_ode_fit.py    # Train + evaluate Neural ODE
│   │   ├── exp12_hidden_sweep.py      # Hidden size / latent dim sweep
│   │   └── exp13_full_comparison.py   # All architectures head-to-head
│   └── figures/
│       ├── plot_bifurcation_comparison.py
│       ├── plot_latent_alignment.py
│       └── plot_architecture_comparison.py
├── analysis/
│   └── population_analysis.py          # ← Existing, extend
└── results/
    ├── population_results.json         # ← Existing (Rung 2)
    └── transformation_results.json     # ← NEW (Rung 3)
```

---

## Phase 9: Claude Code Task Sequence

### Task 0: Extend A-R2 Recording (PREREQUISITE)
```
Modify population/population_circuit.py to record intermediate variables
(tc_mT, tc_hT, tc_mH, nrt_V, nrt_mT, nrt_hT, gaba_a_cond, gaba_b_cond, ampa_cond)
at 0.1 ms resolution. Add save_trial_data() function using HDF5 format.

Re-run A-R2 at 0% replacement across all GABA levels × 3 seeds
with intermediate recording enabled. Save to data/rung2_trials/.
```

### Task 1: Preprocessing Pipeline
```
Create transformation/data/preprocess.py with:
- bin_spike_trains(spike_times, dt_bin, duration)
- smooth_spike_trains(binary, sigma_ms, dt_bin)
- create_training_windows(input, output, window_ms, stride_ms, dt_bin)

Create transformation/data/export_training_data.py:
- Read HDF5 trials from data/rung2_trials/
- Extract retinal inputs and TC outputs at 0% replacement
- Preprocess into training-ready format
- Save as PyTorch tensors / numpy arrays
- Split train/val by seed

Create transformation/data/dataset.py:
- PyTorch Dataset class for windowed spike train data
- DataLoader with shuffling and batching
```

### Task 2: Volterra-Laguerre Model
```
Create transformation/models/volterra_laguerre.py:
- laguerre_basis() function
- project_to_laguerre() function
- VoltLagModel class with fit() via ridge regression
- get_internal_features() for latent extraction

Create transformation/training/train_volterra.py:
- Load preprocessed data
- Fit GLVM per output neuron
- Save model parameters
- Generate predictions on validation set

Create transformation/experiments/exp9_volterra_fit.py:
- Full pipeline: load → preprocess → fit → predict → evaluate
- Report spike correlation, bifurcation test, coherence test
```

### Task 3: LSTM Model
```
Create transformation/models/lstm_replacement.py:
- TCReplacementLSTM class
- LSTMWithGateAccess class with forward_with_gates()

Create transformation/training/train_lstm.py:
- Training loop with early stopping
- Learning rate scheduling
- Gradient clipping
- Checkpoint saving

Create transformation/experiments/exp10_lstm_fit.py:
- Full pipeline with hidden_size=128
- Output metrics + latent extraction
```

### Task 4: Neural ODE Model
```
Create transformation/models/neural_ode_replacement.py:
- ODEFunc class
- TCReplacementNeuralODE class with get_latent_trajectory()
- Requires: pip install torchdiffeq

Create transformation/training/train_neural_ode.py:
- Training with adjoint method for memory efficiency
- Longer training schedule (300 epochs)
- Lower learning rate

Create transformation/experiments/exp11_neural_ode_fit.py:
- Full pipeline with latent_dim=64
```

### Task 5: Latent Comparison (NOVEL)
```
Create transformation/evaluation/latent_comparison.py:
- latent_bio_cca() — Canonical Correlation Analysis
- representational_similarity() — RSA
- individual_variable_recovery() — per-variable best correlation

CRITICAL: This is the code that produces the novel scientific result.
Ensure all statistical tests are correct (Bonferroni correction,
permutation tests for CCA significance, etc.)
```

### Task 6: Hidden Dimension Sweep
```
Create transformation/experiments/exp12_hidden_sweep.py:
- For LSTM: sweep hidden_size = [32, 64, 128, 256, 512]
- For Neural ODE: sweep latent_dim = [16, 32, 64, 128, 256]
- At each size: train, evaluate output, evaluate latent alignment
- Plot: output accuracy vs hidden_size AND latent correlation vs hidden_size
- Key question: does latent alignment peak near biological dimensionality (~240)?
```

### Task 7: Full Comparison
```
Create transformation/experiments/exp13_full_comparison.py:
- Load best models from each architecture
- Run all metrics head-to-head
- Generate comparison table (for Paper 1 results section)
- Generate figures:
  - Bifurcation curves: biological vs Volterra vs LSTM vs Neural ODE
  - Latent-bio correlation heatmap (which bio variables are recovered by which model dims)
  - CCA spectrum comparison across architectures
  - RSA matrices side by side
```

---

## Compute Estimates

| Task | Hardware | Time | Bottleneck |
|------|----------|------|------------|
| Task 0: Re-run A-R2 with recording | CPU (12 cores) | ~3 hours | Same as original A-R2; recording adds ~10% overhead |
| Task 1: Preprocessing | CPU | Minutes | I/O bound |
| Task 2: Volterra fit | CPU | ~10 min | Ridge regression is fast (closed-form) |
| Task 3: LSTM training | GPU (if available) or CPU | 1-4 hours | 200 epochs × training data |
| Task 4: Neural ODE training | GPU strongly preferred | 4-12 hours | ODE integration at each forward pass; adjoint backward pass |
| Task 5: Latent comparison | CPU | ~30 min | CCA and RSA on moderate-sized matrices |
| Task 6: Hidden sweep | GPU | 12-24 hours | 5 sizes × 3 architectures × training time |
| Task 7: Comparison + figures | CPU | Minutes | Analysis only |

**Total: ~2-3 days with GPU, ~4-5 days CPU only.**

---

## Dependencies

```bash
# Core (likely already installed from Rung 1-2)
pip install numpy scipy matplotlib h5py

# Machine learning
pip install torch  # or torch-cpu if no GPU
pip install torchdiffeq  # for Neural ODE
pip install scikit-learn  # for CCA, StandardScaler

# Optional but recommended
pip install tensorboard  # training visualization
pip install seaborn      # publication figures
```

---

## Key Design Decisions and Rationale

### Why three architectures?

Each architecture has a different **inductive bias** that shapes what internal representations it can discover:

- **Volterra-Laguerre:** Linear filter bank with polynomial nonlinearity. Fixed temporal basis. Internal "features" are Laguerre coefficients — basically a spectrogram-like representation. Unlikely to recover oscillatory dynamics spontaneously because Laguerre functions decay exponentially, not oscillate. This is the **null model** — if even Volterra recovers biological states, the signal is very strong.

- **LSTM:** Discrete-time recurrent model with gating. The input/forget/output gates are structurally analogous to ion channel gating variables (activation/inactivation). If LSTM gates correlate with T-current gating, this would be a remarkable convergence — evolution and gradient descent discovering the same computational motif.

- **Neural ODE:** Continuous-time dynamics. The learned vector field dz/dt = f(z) is directly analogous to the Hodgkin-Huxley formalism dV/dt = f(V, m, h, n). If any architecture should recover biological dynamics, it's this one — it speaks the same mathematical language.

### Why include GABA as input?

The GABA conductance parameter controls the operating regime of the circuit (tonic vs oscillatory). Including it as a model input lets us test a stronger claim: not just "the model reproduces output at one GABA level" but "the model captures how the transformation changes across operating regimes."

This mirrors how a real prosthesis would need to adapt to changing neuromodulatory state.

### Why diagonal k₂ only for Volterra?

Full second-order Volterra with 20 inputs × 7 basis functions = 140 features. Full k₂ has 140 × 141 / 2 ≈ 9,870 parameters per output neuron × 20 output neurons = ~200K parameters total. With ~1,440 seconds of data at 1 ms bins = ~1.4M timepoints, the ratio is workable but risks overfitting.

Diagonal k₂ (self-interactions only) reduces to 20 × 28 = 560 parameters per output. This captures within-channel nonlinearities (temporal summation, refractory effects) but not between-channel interactions. If diagonal k₂ suffices, the cross-channel coupling in the TC-nRt circuit is captured by the linear kernels alone. If it fails, add selected cross-channel terms guided by the known connectivity matrix.

---

## Success Criteria

### For Paper 1 (Thalamic Substrate Independence)

**Minimum:** At least one architecture reproduces bifurcation threshold within 2 SD of biological value AND latent comparison reveals interpretable results (even if negative — zombie confirmation is a result).

**Ideal:** Neural ODE reproduces threshold within 1 SD AND shows statistically significant latent-biological correlations for at least 3 biological variable categories (nRt voltage, T-current gating, GABA conductance dynamics).

### For the broader ARIA COGITO program

The A-R3 results determine the framing of Paper 2 (hippocampal MIMO ground truth test):
- If Scenario A (zombie): Paper 2 tests whether hippocampal MIMO is also a zombie
- If Scenario B (architecture matters): Paper 2 tests which architecture to use for hippocampal prostheses
- If Scenario C (identifiability): Paper 2 validates the identifiability result in a more complex circuit

---

## Essential References

1. **Le Masson et al. 2002** — Nature 417:854–858 — Foundation: single-neuron replacement
2. **Marmarelis 2004** — *Nonlinear Dynamic Modeling of Physiological Systems* — Volterra-Laguerre theory
3. **Berger et al. 2011** — J Neural Eng 8:046017 — MIMO hippocampal prosthesis
4. **Song et al. 2007** — J Neural Eng 4:S180–S186 — MIMO kernel estimation for CA1
5. **Beniaguev et al. 2021** — Neuron 109:2727–2739 — TCN on NEURON L5PC (closest prior work)
6. **Oláh et al. 2022** — eLife 11:e79535 — CNN-LSTM predicting dendritic variables
7. **Chen et al. 2018** — NeurIPS — Neural Ordinary Differential Equations
8. **Pandarinath et al. 2018** — Nature Methods 15:805–815 — LFADS
9. **Lei et al. 2021** — Phil Trans A 378:20190348 — Neural ODE ion channel dynamics
10. **Burghi et al. 2025** — PNAS — Recurrent mechanistic models (gray-box comparison)
