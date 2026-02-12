"""
Biological heterogeneity model for TC and nRt populations.

"Biological" TC neurons: heterogeneous instances from LogNormal distributions
matching real biological variability (CV 10-15%).

"Computational replacement" TC neurons: single standardized model validated
in Rung 1 (bifurcation threshold = 25.9 nS). All identical — the prosthetic
scenario.
"""

import numpy as np


# Validated Le Masson computational TC parameters (Rung 1 result)
VALIDATED_TC_PARAMS = {
    'C_m': 1.0,
    'g_Na': 90.0,
    'g_K': 10.0,
    'g_T': 2.0,
    'g_h': 0.05,
    'g_L': 0.02,
    'g_KL': 0.03,
    'E_Na': 50.0,
    'E_K': -100.0,
    'E_Ca': 120.0,
    'E_h': -40.0,
    'E_L': -70.0,
    'E_KL': -100.0,
    'area_cm2': 2.9e-4,
    'temperature': 35.0,
    'V_rest': -64.0,
}

# Base nRt parameters
BASE_NRT_PARAMS = {
    'C_m': 1.0,
    'g_Na': 100.0,
    'g_K': 10.0,
    'g_Ts': 3.0,
    'g_L': 0.05,
    'E_Na': 50.0,
    'E_K': -100.0,
    'E_Ca': 120.0,
    'E_L': -77.0,
    'area_cm2': 1.43e-4,
    'temperature': 35.0,
    'V_rest': -72.0,
}

# Heterogeneity specification
TC_CONDUCTANCE_CVS = {
    'g_T': 0.15,
    'g_h': 0.15,
    'g_Na': 0.10,
    'g_K': 0.10,
    'g_L': 0.10,
}

TC_CM_CV = 0.05

NRT_CONDUCTANCE_CVS = {
    'g_Ts': 0.15,
    'g_Na': 0.10,
    'g_K': 0.10,
    'g_L': 0.10,
}

NRT_CM_CV = 0.05


def _sample_lognormal(mean, cv, rng):
    """Sample from LogNormal with specified mean and CV."""
    sigma_ln = np.sqrt(np.log(1 + cv ** 2))
    mu_ln = np.log(mean) - 0.5 * sigma_ln ** 2
    return float(rng.lognormal(mu_ln, sigma_ln))


def sample_tc_parameters(n, rng, base_params=None):
    """Sample heterogeneous TC neuron parameters from biological distributions.

    Conductances: LogNormal with specified CV.
    Capacitance: LogNormal with CV=5%.
    Resting potential: Normal(−64, σ=2 mV).

    Parameters
    ----------
    n : int
        Number of neurons to sample.
    rng : np.random.Generator
        Random number generator.
    base_params : dict, optional
        Base parameters. Defaults to VALIDATED_TC_PARAMS.

    Returns
    -------
    params_list : list of dict
        One parameter dict per neuron.
    """
    base = (base_params or VALIDATED_TC_PARAMS).copy()
    params_list = []

    for _ in range(n):
        p = base.copy()

        # Conductance densities (lognormal)
        for key, cv in TC_CONDUCTANCE_CVS.items():
            p[key] = _sample_lognormal(base[key], cv, rng)

        # g_KL: same CV as g_L
        p['g_KL'] = _sample_lognormal(base['g_KL'], 0.10, rng)

        # Capacitance (lognormal, tighter)
        p['C_m'] = _sample_lognormal(base['C_m'], TC_CM_CV, rng)

        # Resting potential (normal)
        p['V_rest'] = float(rng.normal(-64.0, 2.0))

        params_list.append(p)

    return params_list


def sample_nrt_parameters(n, rng, base_params=None):
    """Sample heterogeneous nRt neuron parameters from biological distributions.

    Parameters
    ----------
    n : int
        Number of neurons.
    rng : np.random.Generator
    base_params : dict, optional
        Defaults to BASE_NRT_PARAMS.

    Returns
    -------
    params_list : list of dict
    """
    base = (base_params or BASE_NRT_PARAMS).copy()
    params_list = []

    for _ in range(n):
        p = base.copy()

        for key, cv in NRT_CONDUCTANCE_CVS.items():
            p[key] = _sample_lognormal(base[key], cv, rng)

        p['C_m'] = _sample_lognormal(base['C_m'], NRT_CM_CV, rng)
        p['V_rest'] = float(rng.normal(-72.0, 2.0))

        params_list.append(p)

    return params_list


def create_replacement_tc_params():
    """Return the validated Le Masson computational TC model parameters.

    Same parameters for every replacement neuron — no heterogeneity.
    This is the prosthetic scenario: one model, stamped out N times.
    """
    return VALIDATED_TC_PARAMS.copy()


def create_heterogeneous_replacement_tc_params(rng, cv=0.10):
    """Replacement model with controlled heterogeneity (for Control B).

    Parameters drawn from distribution around the validated Le Masson model.
    """
    p = VALIDATED_TC_PARAMS.copy()
    for key in ['g_T', 'g_h', 'g_Na', 'g_K', 'g_L']:
        p[key] = _sample_lognormal(p[key], cv, rng)
    return p
