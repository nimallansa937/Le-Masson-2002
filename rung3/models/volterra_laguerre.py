"""
Volterra-Laguerre Model (GLVM) for thalamic circuit transformation.

Architecture:
  1. Laguerre basis expansion of input channels (20 retinal + 1 GABA)
  2. Linear (1st order) + diagonal quadratic (2nd order) kernels
  3. Output feedback via Laguerre expansion of past predictions
  4. Ridge regression fitting (closed-form, no iterative training)

The Laguerre basis provides an orthonormal expansion of the Volterra kernels
with exponential memory decay, making the model both interpretable and
tractable. The 'alpha' parameter controls memory length.

Latent representation: the Laguerre coefficients c(t) at each timestep
form the model's hidden state — analogous to LSTM hidden states.
"""

import numpy as np
from scipy.linalg import solve

from rung3.config import (
    VOLTERRA_N_BASES, VOLTERRA_ALPHA, VOLTERRA_MEMORY_MS,
    VOLTERRA_ORDER, VOLTERRA_RIDGE_ALPHA,
    VOLTERRA_OUTPUT_FEEDBACK, VOLTERRA_FB_N_BASES,
    INPUT_DIM, OUTPUT_DIM, BIN_DT_MS,
)


def laguerre_basis(n_bases, alpha, memory_bins):
    """Compute discrete Laguerre basis functions.

    Parameters
    ----------
    n_bases : int
        Number of basis functions.
    alpha : float
        Laguerre parameter (0 < alpha < 1). Controls memory decay.
    memory_bins : int
        Number of time bins for the basis.

    Returns
    -------
    B : ndarray (n_bases, memory_bins)
        Laguerre basis functions.
    """
    B = np.zeros((n_bases, memory_bins))
    sqrt_a = np.sqrt(alpha)

    for k in range(memory_bins):
        # Laguerre polynomials via recursion
        if k == 0:
            B[0, k] = np.sqrt(1 - alpha)
        else:
            B[0, k] = np.sqrt(1 - alpha) * (-alpha) ** k

    # Higher order bases via recursion
    for j in range(1, n_bases):
        for k in range(memory_bins):
            if k == 0:
                B[j, k] = sqrt_a * B[j-1, k]
            else:
                B[j, k] = sqrt_a * B[j-1, k] + B[j, k-1] * sqrt_a \
                           - B[j-1, k-1]

    # Normalize
    for j in range(n_bases):
        norm = np.sqrt(np.sum(B[j]**2))
        if norm > 1e-10:
            B[j] /= norm

    return B


def laguerre_filter(x, basis):
    """Apply Laguerre basis expansion to a signal.

    Parameters
    ----------
    x : ndarray (n_timesteps,)
        Input signal.
    basis : ndarray (n_bases, memory_bins)
        Laguerre basis functions.

    Returns
    -------
    coeffs : ndarray (n_timesteps, n_bases)
        Laguerre coefficients at each timestep.
    """
    n_bases, memory_bins = basis.shape
    n_t = len(x)
    coeffs = np.zeros((n_t, n_bases), dtype=np.float32)

    for j in range(n_bases):
        # Causal convolution
        coeffs[:, j] = np.convolve(x, basis[j], mode='full')[:n_t]

    return coeffs


class VolterraLaguerre:
    """Generalized Laguerre-Volterra Model.

    Shared interface with LSTM and Neural ODE:
      forward(x, return_latent=False)
        x: (batch, seq_len, 21)
        returns: (batch, seq_len, 20)  or  (output, latent_dict)
    """

    def __init__(self, n_bases=VOLTERRA_N_BASES, alpha=VOLTERRA_ALPHA,
                 memory_ms=VOLTERRA_MEMORY_MS, order=VOLTERRA_ORDER,
                 ridge_alpha=VOLTERRA_RIDGE_ALPHA,
                 output_feedback=VOLTERRA_OUTPUT_FEEDBACK,
                 fb_n_bases=VOLTERRA_FB_N_BASES):
        self.n_bases = n_bases
        self.alpha = alpha
        self.memory_bins = int(memory_ms / BIN_DT_MS)
        self.order = order
        self.ridge_alpha = ridge_alpha
        self.output_feedback = output_feedback
        self.fb_n_bases = fb_n_bases

        # Compute basis functions
        self.basis = laguerre_basis(n_bases, alpha, self.memory_bins)
        if output_feedback:
            self.fb_basis = laguerre_basis(fb_n_bases, alpha, self.memory_bins)

        # Feature dimension (computed for default INPUT_DIM/OUTPUT_DIM,
        # but actual dimension is determined at fit() time from data):
        self.n_features_1st = INPUT_DIM * n_bases
        self.n_features_2nd = INPUT_DIM * n_bases if order >= 2 else 0
        self.n_features_fb = OUTPUT_DIM * fb_n_bases if output_feedback else 0
        self.n_features = (self.n_features_1st + self.n_features_2nd +
                           self.n_features_fb + 1)

        # Weights (fitted via ridge regression)
        self.W = None  # (n_features, output_dim)

    def _expand_features(self, x_seq, y_prev=None):
        """Expand input sequence into Volterra-Laguerre feature matrix.

        Parameters
        ----------
        x_seq : ndarray (seq_len, input_dim)
        y_prev : ndarray (seq_len, output_dim) or None
            Previous output for feedback.

        Returns
        -------
        F : ndarray (seq_len, n_features)
            Feature matrix.
        laguerre_coeffs : ndarray (seq_len, input_dim * n_bases)
            Raw Laguerre coefficients (for latent comparison).
        """
        seq_len, n_in = x_seq.shape

        # 1st order: Laguerre expansion of each input channel
        coeffs_1st = np.zeros((seq_len, n_in * self.n_bases), dtype=np.float32)
        for ch in range(n_in):
            c = laguerre_filter(x_seq[:, ch], self.basis)
            coeffs_1st[:, ch*self.n_bases:(ch+1)*self.n_bases] = c

        features = [coeffs_1st]

        # 2nd order: diagonal quadratic terms
        if self.order >= 2:
            coeffs_2nd = coeffs_1st ** 2
            features.append(coeffs_2nd)

        # Output feedback
        if self.output_feedback and y_prev is not None:
            n_out = y_prev.shape[1]
            fb_coeffs = np.zeros((seq_len, n_out * self.fb_n_bases),
                                  dtype=np.float32)
            for ch in range(n_out):
                c = laguerre_filter(y_prev[:, ch], self.fb_basis)
                fb_coeffs[:, ch*self.fb_n_bases:(ch+1)*self.fb_n_bases] = c
            features.append(fb_coeffs)

        # Bias
        features.append(np.ones((seq_len, 1), dtype=np.float32))

        F = np.concatenate(features, axis=1)
        return F, coeffs_1st

    def fit(self, X_train, Y_train):
        """Fit model via ridge regression.

        Parameters
        ----------
        X_train : list of ndarray (seq_len, 21) or ndarray (n_windows, seq_len, 21)
        Y_train : list of ndarray (seq_len, 20) or ndarray (n_windows, seq_len, 20)
        """
        # Concatenate all windows
        if isinstance(X_train, np.ndarray) and X_train.ndim == 3:
            X_list = [X_train[i] for i in range(X_train.shape[0])]
            Y_list = [Y_train[i] for i in range(Y_train.shape[0])]
        else:
            X_list = X_train
            Y_list = Y_train

        F_all = []
        Y_all = []

        for x_seq, y_seq in zip(X_list, Y_list):
            # Use y_seq as feedback for expansion (teacher forcing)
            F, _ = self._expand_features(x_seq, y_seq)
            F_all.append(F)
            Y_all.append(y_seq)

        F = np.concatenate(F_all, axis=0)
        Y = np.concatenate(Y_all, axis=0)

        # Ridge regression: W = (F^T F + αI)^{-1} F^T Y
        FtF = F.T @ F
        FtF += self.ridge_alpha * np.eye(FtF.shape[0])
        FtY = F.T @ Y

        self.W = solve(FtF, FtY, assume_a='pos')

        # Training residual
        Y_pred = F @ self.W
        mse = np.mean((Y - Y_pred) ** 2)
        corr = np.mean([np.corrcoef(Y[:, i], Y_pred[:, i])[0, 1]
                        for i in range(Y.shape[1])
                        if np.std(Y[:, i]) > 1e-8])

        return {'mse': float(mse), 'correlation': float(corr)}

    def forward(self, x, return_latent=False):
        """Predict output from input.

        Parameters
        ----------
        x : ndarray (batch, seq_len, 21) or (seq_len, 21)
        return_latent : bool

        Returns
        -------
        output : ndarray (batch, seq_len, 20) or (seq_len, 20)
        latent_dict : dict (if return_latent)
            'hidden': Laguerre coefficients (batch, seq_len, latent_dim)
        """
        if self.W is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        single = x.ndim == 2
        if single:
            x = x[np.newaxis]

        batch, seq_len, n_in = x.shape
        n_out = self.W.shape[1]  # Output dim from fitted weights
        latent_dim = n_in * self.n_bases
        outputs = np.zeros((batch, seq_len, n_out), dtype=np.float32)
        latents = np.zeros((batch, seq_len, latent_dim), dtype=np.float32)

        for b in range(batch):
            # Recursive prediction with output feedback
            y_prev = np.zeros((seq_len, n_out), dtype=np.float32)

            for iteration in range(3):  # Iterate for output feedback convergence
                F, coeffs = self._expand_features(x[b], y_prev)
                y_pred = F @ self.W
                y_pred = np.clip(y_pred, 0, None)  # Non-negative rates
                y_prev = y_pred

            outputs[b] = y_pred
            latents[b] = coeffs

        if single:
            outputs = outputs[0]
            latents = latents[0]

        if return_latent:
            return outputs, {'hidden': latents}
        return outputs

    def save(self, filepath):
        """Save model weights."""
        np.savez(filepath,
                 W=self.W,
                 n_bases=self.n_bases,
                 alpha=self.alpha,
                 memory_bins=self.memory_bins,
                 order=self.order,
                 ridge_alpha=self.ridge_alpha)

    def load(self, filepath):
        """Load model weights."""
        data = np.load(filepath)
        self.W = data['W']
        return self
