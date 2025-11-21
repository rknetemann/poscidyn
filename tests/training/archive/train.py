import h5py

FILENAME = 'batch_2025-08-14_15:12:41_0.hdf5'

with h5py.File(FILENAME, 'r') as hdf5:
    
    driving_frequencies = hdf5['driving_frequencies']
    driving_amplitudes = hdf5['driving_amplitudes']
    
    simulations_group = hdf5['simulations']
    
    for sim_id in simulations_group:
        sim_data = simulations_group[sim_id]
        
        Q = sim_data.attrs['Q']
        gamma = sim_data.attrs['gamma']
        sweep_direction = sim_data.attrs['sweep_direction']
        max_steady_state_displacement = sim_data[:] # (n_driving_frequencies* n_driving_amplitudes)
        
        max_steady_state_displacement = max_steady_state_displacement.reshape(
            driving_frequencies.shape[0], driving_amplitudes.shape[0]
        ) # (n_driving_frequencies, n_driving_amplitudes)
        
        
        
        
        