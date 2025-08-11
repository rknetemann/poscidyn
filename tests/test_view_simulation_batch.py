import h5py
import numpy as np

HDF5_FILE = 'tests/simulation_batch_2025-08-11T11:54:18.h5'

with h5py.File(HDF5_FILE, 'r', swmr=True) as f:
    def print_groups(name, obj):
        if isinstance(obj, h5py.Group):
            print(name)

    f.visititems(print_groups)
