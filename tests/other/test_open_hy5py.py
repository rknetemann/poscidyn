import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
HDF5_FILE = filedialog.askopenfilename(
    title="Select HDF5 file",
    filetypes=[("HDF5 files", "*.hdf5"), ("All files", "*.*")]
)

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
