"""
Phase 0: HDF5 save/load utilities for trial data with intermediate recordings.

File format per trial:
  /meta/gaba_gmax          scalar
  /meta/seed               scalar
  /meta/duration_s         scalar
  /meta/n_tc               scalar
  /meta/n_nrt              scalar
  /meta/dt_record          scalar (recording timestep)
  /time                    (n_timepoints,)
  /V_tc                    (n_tc, n_timepoints)
  /V_nrt                   (n_nrt, n_timepoints)
  /tc_spike_times/neuron_i (variable length)
  /nrt_spike_times/neuron_j (variable length)
  /retinal_spike_times/channel_i (variable length)
  /intermediates/tc_m_T    (n_tc, n_timepoints)
  /intermediates/tc_h_T    (n_tc, n_timepoints)
  /intermediates/tc_m_h    (n_tc, n_timepoints)
  /intermediates/nrt_m_Ts  (n_nrt, n_timepoints)
  /intermediates/nrt_h_Ts  (n_nrt, n_timepoints)
  /intermediates/gabaa_per_tc  (n_tc, n_timepoints) summed GABA_A conductance
  /intermediates/gabab_per_tc  (n_tc, n_timepoints) summed GABA_B conductance
  /intermediates/ampa_per_nrt  (n_nrt, n_timepoints) summed AMPA conductance
"""

import os
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def trial_filename(gaba_gmax, seed, data_dir="rung3_data"):
    """Generate standardized filename for a trial."""
    return os.path.join(data_dir, f"trial_gaba{gaba_gmax:05.1f}_seed{seed}.h5")


def save_trial_hdf5(filepath, sim_result, gaba_gmax, seed):
    """Save a complete trial (simulation output) to HDF5.

    Parameters
    ----------
    filepath : str
        Output HDF5 path.
    sim_result : dict
        Output from PopulationCircuit.simulate(record_intermediates=True).
    gaba_gmax : float
    seed : int
    """
    if not HAS_H5PY:
        raise ImportError("h5py required for HDF5 I/O: pip install h5py")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        # Metadata
        meta = f.create_group('meta')
        meta.attrs['gaba_gmax'] = float(gaba_gmax)
        meta.attrs['seed'] = int(seed)
        meta.attrs['duration_s'] = float(sim_result['duration_s'])
        meta.attrs['n_tc'] = int(sim_result['n_tc'])
        meta.attrs['n_nrt'] = int(sim_result['n_nrt'])
        dt_rec = sim_result['t'][1] - sim_result['t'][0] if len(sim_result['t']) > 1 else 0.001
        meta.attrs['dt_record'] = float(dt_rec)

        # Time and voltages
        f.create_dataset('time', data=sim_result['t'], compression='gzip')
        f.create_dataset('V_tc', data=sim_result['V_tc'], compression='gzip')
        f.create_dataset('V_nrt', data=sim_result['V_nrt'], compression='gzip')

        # Spike times (variable length)
        tc_grp = f.create_group('tc_spike_times')
        for i, spk in enumerate(sim_result['tc_spike_times']):
            tc_grp.create_dataset(f'neuron_{i}', data=np.asarray(spk))

        nrt_grp = f.create_group('nrt_spike_times')
        for i, spk in enumerate(sim_result['nrt_spike_times']):
            nrt_grp.create_dataset(f'neuron_{i}', data=np.asarray(spk))

        ret_grp = f.create_group('retinal_spike_times')
        for i, spk in enumerate(sim_result['retinal_spike_times']):
            ret_grp.create_dataset(f'channel_{i}', data=np.asarray(spk))

        # Intermediate biological variables
        if 'intermediates' in sim_result:
            inter = sim_result['intermediates']
            ig = f.create_group('intermediates')
            for key, arr in inter.items():
                ig.create_dataset(key, data=arr, compression='gzip')

    return filepath


def load_trial_hdf5(filepath):
    """Load a trial from HDF5.

    Returns
    -------
    data : dict
        Contains all fields from save_trial_hdf5 in the same format.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required for HDF5 I/O: pip install h5py")

    data = {}

    with h5py.File(filepath, 'r') as f:
        # Metadata
        meta = f['meta']
        data['gaba_gmax'] = float(meta.attrs['gaba_gmax'])
        data['seed'] = int(meta.attrs['seed'])
        data['duration_s'] = float(meta.attrs['duration_s'])
        data['n_tc'] = int(meta.attrs['n_tc'])
        data['n_nrt'] = int(meta.attrs['n_nrt'])
        data['dt_record'] = float(meta.attrs['dt_record'])

        # Time and voltages
        data['t'] = f['time'][:]
        data['V_tc'] = f['V_tc'][:]
        data['V_nrt'] = f['V_nrt'][:]

        # Spike times
        n_tc = data['n_tc']
        n_nrt = data['n_nrt']

        data['tc_spike_times'] = []
        tc_grp = f['tc_spike_times']
        for i in range(n_tc):
            key = f'neuron_{i}'
            if key in tc_grp:
                data['tc_spike_times'].append(tc_grp[key][:])
            else:
                data['tc_spike_times'].append(np.array([]))

        data['nrt_spike_times'] = []
        nrt_grp = f['nrt_spike_times']
        for i in range(n_nrt):
            key = f'neuron_{i}'
            if key in nrt_grp:
                data['nrt_spike_times'].append(nrt_grp[key][:])
            else:
                data['nrt_spike_times'].append(np.array([]))

        data['retinal_spike_times'] = []
        ret_grp = f['retinal_spike_times']
        for i in range(n_tc):
            key = f'channel_{i}'
            if key in ret_grp:
                data['retinal_spike_times'].append(ret_grp[key][:])
            else:
                data['retinal_spike_times'].append(np.array([]))

        # Intermediate variables
        data['intermediates'] = {}
        if 'intermediates' in f:
            ig = f['intermediates']
            for key in ig:
                data['intermediates'][key] = ig[key][:]

    return data


def list_trials(data_dir="rung3_data"):
    """List all trial HDF5 files in a directory.

    Returns
    -------
    trials : list of dict
        Each dict has 'filepath', 'gaba_gmax', 'seed'.
    """
    trials = []
    if not os.path.isdir(data_dir):
        return trials

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.h5') and fname.startswith('trial_'):
            fpath = os.path.join(data_dir, fname)
            # Parse from filename
            parts = fname.replace('.h5', '').split('_')
            gaba = None
            seed = None
            for p in parts:
                if p.startswith('gaba'):
                    try:
                        gaba = float(p[4:])
                    except ValueError:
                        pass
                elif p.startswith('seed'):
                    try:
                        seed = int(p[4:])
                    except ValueError:
                        pass
            trials.append({
                'filepath': fpath,
                'gaba_gmax': gaba,
                'seed': seed,
            })

    return trials
