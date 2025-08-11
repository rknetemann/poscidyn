import h5py
import numpy as np

HDF5_FILE = '/home/raymo/Projects/parameter-identification-nanomechanical-resonators/tests/other/simulations.hdf5'

with h5py.File(HDF5_FILE, 'r') as f:
    for sim_id in f["simulations"]:
        g = f["simulations"][sim_id]
        amp_map = g["amplitude_map"][:]
        min_freq = g.attrs["min_driving_frequency"]
        max_freq = g.attrs["max_driving_frequency"]
        min_amp = g.attrs["min_driving_amplitude"]
        max_amp = g.attrs["max_driving_amplitude"]

        # Process the data as needed
        print(f"Simulation ID: {sim_id}")
        print(f"  Amplitude Map Shape: {amp_map.shape}")
        print(f"  Frequency Range: {min_freq} - {max_freq}")
        print(f"  Amplitude Range: {min_amp} - {max_amp}")
