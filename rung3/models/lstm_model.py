"""
LSTM Model for thalamic circuit transformation.

Architecture:
  - 2-layer LSTM with hidden_size=128
  - Custom GateAccessLSTMCell exposing i/f/o/g gate activations
  - Input: 21D (20 retinal rates + 1 GABA scalar)
  - Output: 20D (TC smoothed rates, sigmoid activation)

The gate-access cell allows extraction of internal gate activations
for latent comparison with biological intermediates (the novel part).
"""

import torch
import torch.nn as nn
import numpy as np

from rung3.config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    INPUT_DIM, OUTPUT_DIM,
)


class GateAccessLSTMCell(nn.Module):
    """LSTM cell that exposes gate activations.

    Standard LSTM equations:
      i = σ(W_ii x + b_ii + W_hi h + b_hi)  — input gate
      f = σ(W_if x + b_if + W_hf h + b_hf)  — forget gate
      g = tanh(W_ig x + b_ig + W_hg h + b_hg) — cell candidate
      o = σ(W_io x + b_io + W_ho h + b_ho)  — output gate
      c = f * c_prev + i * g
      h = o * tanh(c)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weight matrices for efficiency
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, hc):
        """
        Parameters
        ----------
        x : (batch, input_size)
        hc : tuple of (h, c), each (batch, hidden_size)

        Returns
        -------
        h_new, c_new, gates : where gates is dict with i, f, o, g
        """
        h_prev, c_prev = hc

        # Combined linear transforms
        gates_x = self.W_x(x)
        gates_h = self.W_h(h_prev)
        gates = gates_x + gates_h

        # Split into 4 gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)

        gate_dict = {'i': i, 'f': f, 'o': o, 'g': g}
        return h_new, c_new, gate_dict


class ThalamicLSTM(nn.Module):
    """2-layer LSTM for thalamic circuit replacement.

    Shared interface:
      forward(x, return_latent=False)
        x: (batch, seq_len, 21)
        returns: (batch, seq_len, 20)  or  (output, latent_dict)
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, output_dim=OUTPUT_DIM,
                 dropout=LSTM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Custom LSTM cells for gate access
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_dim if layer == 0 else hidden_size
            self.cells.append(GateAccessLSTMCell(in_size, hidden_size))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, return_latent=False):
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)
        return_latent : bool
            If True, also return hidden states and gate activations.

        Returns
        -------
        output : (batch, seq_len, output_dim)
        latent_dict : dict (if return_latent)
            'hidden': (batch, seq_len, hidden_size * num_layers)
            'gates': dict of (batch, seq_len, hidden_size) per gate per layer
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden states
        h_states = [torch.zeros(batch, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch, self.hidden_size, device=device)
                     for _ in range(self.num_layers)]

        # Collect outputs
        all_outputs = []
        all_hidden = [] if return_latent else None
        all_gates = {f'layer{l}_{g}': [] for l in range(self.num_layers)
                     for g in ['i', 'f', 'o', 'g']} if return_latent else None

        for t in range(seq_len):
            inp = x[:, t, :]  # (batch, input_dim)

            for layer in range(self.num_layers):
                h_new, c_new, gates = self.cells[layer](
                    inp, (h_states[layer], c_states[layer]))
                h_states[layer] = h_new
                c_states[layer] = c_new
                inp = h_new

                if self.dropout and layer < self.num_layers - 1:
                    inp = self.dropout(inp)

                if return_latent:
                    for g_name, g_val in gates.items():
                        all_gates[f'layer{layer}_{g_name}'].append(
                            g_val.detach())

            # Output from last layer
            out = torch.sigmoid(self.output_proj(h_states[-1]))
            all_outputs.append(out)

            if return_latent:
                # Concatenate hidden states from all layers
                h_cat = torch.cat(h_states, dim=-1).detach()
                all_hidden.append(h_cat)

        output = torch.stack(all_outputs, dim=1)  # (batch, seq_len, output_dim)

        if return_latent:
            hidden = torch.stack(all_hidden, dim=1)  # (batch, seq_len, H*L)
            gate_tensors = {
                k: torch.stack(v, dim=1) for k, v in all_gates.items()
            }
            return output, {'hidden': hidden, 'gates': gate_tensors}

        return output

    def forward_fast(self, x):
        """Fast forward pass using PyTorch's built-in LSTM (no gate access).

        Use for training speed; use forward(return_latent=True) for evaluation.
        """
        batch, seq_len, _ = x.shape

        # Build standard LSTM for speed
        if not hasattr(self, '_fast_lstm'):
            self._fast_lstm = nn.LSTM(
                self.input_dim, self.hidden_size, self.num_layers,
                batch_first=True,
                dropout=self.dropout.p if self.dropout else 0.0,
            ).to(x.device)
            # Copy weights from custom cells
            self._sync_fast_weights()

        lstm_out, _ = self._fast_lstm(x)
        output = torch.sigmoid(self.output_proj(lstm_out))
        return output

    def _sync_fast_weights(self):
        """Copy weights from GateAccessLSTMCells to built-in LSTM."""
        if not hasattr(self, '_fast_lstm'):
            return
        with torch.no_grad():
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                # PyTorch LSTM weight layout: [i, f, g, o] but our cell uses [i, f, g, o]
                self._fast_lstm.all_weights[layer][0].copy_(cell.W_x.weight)
                self._fast_lstm.all_weights[layer][1].copy_(cell.W_h.weight)
                self._fast_lstm.all_weights[layer][2].copy_(cell.W_x.bias)
                # W_h has no bias in our cell, so zero it
                self._fast_lstm.all_weights[layer][3].zero_()
