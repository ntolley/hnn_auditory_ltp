import os.path as op
import numpy as np

import hnn_core
from hnn_core import calcium_model, simulate_dipole, read_params, pick_connection
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt

### Function to run batch simulation that takes new network for each simulation
def add_connectivity_drives(net, seed=0):
    seed_rng = np.random.default_rng(seed)
    seed_array = seed_rng.integers(0, 1e5, size=3)

    # Proximal 1 drive
    weights_ampa_p1 = {'L2_basket': 0.997291, 'L2_pyramidal':0.990722,'L5_basket':0.614932, 'L5_pyramidal': 0.004153}
    weights_nmda_p1 = {'L2_basket': 0.984337, 'L2_pyramidal':1.714247,'L5_basket':0.061868, 'L5_pyramidal': 0.010042}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,'L5_basket': 1., 'L5_pyramidal': 1.}
    net.add_evoked_drive('evprox1', mu=54.897936, sigma=5.401034, numspikes=1, weights_ampa=weights_ampa_p1, weights_nmda=weights_nmda_p1, location='proximal',synaptic_delays=synaptic_delays_prox, event_seed=seed_array[0])
    # Distal drive
    weights_ampa_d1 = {'L2_basket': 0.624131, 'L2_pyramidal': 0.606619, 'L5_pyramidal':0.258}
    weights_nmda_d1 = {'L2_basket': 0.95291, 'L2_pyramidal': 0.242383, 'L5_pyramidal': 0.156725}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}
    net.add_evoked_drive('evdist1', mu=82.9915, sigma=13.208408, numspikes=1, weights_ampa=weights_ampa_d1,weights_nmda=weights_nmda_d1, location='distal',synaptic_delays=synaptic_delays_d1, event_seed=seed_array[1]) 
    # Second proximal evoked drive.
    weights_ampa_p2 = {'L2_basket': 0.758537, 'L2_pyramidal': 0.854454,'L5_basket': 0.979846, 'L5_pyramidal': 0.012483}
    weights_nmda_p2 = {'L2_basket': 0.851444, 'L2_pyramidal':0.067491 ,'L5_basket': 0.901834, 'L5_pyramidal': 0.003818}
    net.add_evoked_drive('evprox2', mu=161.306837, sigma=19.843986, numspikes=1, weights_ampa=weights_ampa_p2,weights_nmda= weights_nmda_p2, location='proximal',synaptic_delays=synaptic_delays_prox, event_seed=seed_array[2])
    

def scale_connectivity(net, scale_factor=1.0):
    """Multiple AMPA and NMDA conductances by scale factor"""
    conn_indices = pick_connection(net, src_gids=['L2_pyramidal', 'L5_pyramidal'], receptor='ampa')
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= scale_factor
    conn_indices = pick_connection(net, src_gids=['L2_pyramidal', 'L5_pyramidal'], receptor='nmda')
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= scale_factor

def run_connectivity_simulation(net_base, connectivity_scale, tstop, seed):
    net = net_base.copy()
    add_connectivity_drives(net, seed)
    scale_connectivity(net, connectivity_scale)

    dpl = simulate_dipole(net, tstop=tstop, n_trials=1)
    return net, dpl

# === Simulation setup ===
n_trials = 50
tstop = 450
data_path = '../data/connectivity_simulations'

hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = '../data/L_Contra.param'
params = read_params(params_fname)

# === Build network variants with different calcium scaling ===
connectivity_list = [1.0, 2.0, 5.0, 10.0]
connectivity_sweep = np.repeat(connectivity_list, n_trials)

# Base network
net_base = hnn_core.calcium_model(params, add_drives_from_params=False)

# === Run all jobs in parallel ===
res = Parallel(n_jobs=n_trials)(
    delayed(run_connectivity_simulation)(net_base, scale, tstop, seed_idx) for seed_idx, scale in enumerate(connectivity_sweep))

dpl_dict = {
    'trial': np.tile(np.arange(n_trials), (len(connectivity_list))),
    'connectivity_scale': connectivity_sweep,
    'dpl': [res[sim_idx][1][0].copy().smooth(30).scale(1500).data['agg'] for sim_idx in range(len(res))],
    'spike_times': [res[sim_idx][0].cell_response.spike_times for sim_idx in range(len(res))],
    'spike_gids': [res[sim_idx][0].cell_response.spike_gids for sim_idx in range(len(res))],
    'spike_types': [res[sim_idx][0].cell_response.spike_types for sim_idx in range(len(res))],
    'times': [res[sim_idx][1][0].times for sim_idx in range(len(res))]

}

df = pd.DataFrame(dpl_dict)
df.to_pickle(f'{data_path}/connectivity_simulations.pkl')