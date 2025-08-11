import h5py
import numpy as np

import tkinter as tk
from tkinter import filedialog

# root = tk.Tk()
# root.withdraw()
# HDF5_FILE = filedialog.askopenfilename(
#     title="Select HDF5 file",
#     filetypes=[("HDF5 files", "*.hdf5"), ("All files", "*.*")]
# )

HDF5_FILE = '/home/raymo/Downloads/simulation_batch_2025-08-11T13-57-00.h5'

with h5py.File(HDF5_FILE, 'r', swmr=True) as f:
    grp_name = 'simulation_00003999'
    ds_name = 'maximum_steady_state_displacement'

    if grp_name not in f:
        print(f"Group '{grp_name}' not found in file.")
    else:
        grp = f[grp_name]
        if ds_name not in grp:
            print(f"Dataset '{ds_name}' not found in group '{grp_name}'.")
        else:
            ds = grp[ds_name]
            data = ds[()]  # load entire dataset into a NumPy array
            n_preview = min(10, data.size)
            preview = data[:n_preview]
            print(f"Preview of '{ds_name}' (first {n_preview} values):\n{preview}")
