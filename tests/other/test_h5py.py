import h5py
import numpy as np

# Create dummy data
N, M = 100, 100  # dimensions of displacement amplitude
num_sims = 10  # number of simulations

HDF5_FILE = '/home/raymo/Projects/parameter-identification-nanomechanical-resonators/tests/other/simulations.hdf5'

with h5py.File(HDF5_FILE, 'w') as f:
    sim_group = f.create_group("simulations")
    
    for i in range(num_sims):
        sim_id = f"sim_{i:05d}"
        g = sim_group.create_group(sim_id)

        # Example data
        amp_map = np.random.rand(N, M).astype(np.float32)
        min_freq = np.float32(1.0)
        max_freq = np.float32(5.0)
        min_amp = np.float32(0.1)
        max_amp = np.float32(1.0)

        # Store the array
        g.create_dataset("amplitude_map", data=amp_map, compression="gzip")

        # Store scalars as attributes or datasets
        g.attrs["min_driving_frequency"] = min_freq
        g.attrs["max_driving_frequency"] = max_freq
        g.attrs["min_driving_amplitude"] = min_amp
        g.attrs["max_driving_amplitude"] = max_amp