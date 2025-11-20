from pathlib import Path
import h5py

FILENAME = Path("/home/raymo/Downloads/batch_0_2025-11-17_17-03.hdf5")

with h5py.File(FILENAME, "r") as f:
        if "simulations" not in f or not isinstance(f["simulations"], h5py.Group):
            raise ValueError("Expected a 'simulations' group in the HDF5 file.")
        sims_grp: h5py.Group = f["simulations"]

        sim_names = sorted(sims_grp.keys())
        if not sim_names:
            raise ValueError("No simulations found in the provided HDF5 file.")
        X_list, Y_list, sim_id_list = [], [], []

        for sim_idx, nm in enumerate(sim_names):
            obj = sims_grp[nm]

            if isinstance(obj, h5py.Group):
                Q = obj.attrs.get("Q", None)
                omega_0 = obj.attrs.get("omega_0", None)
                gamma = obj.attrs.get("gamma", None)

                forward_sweep =  obj["sweeped_periodic_solutions"][...]




            else:
                raise TypeError(f"Unsupported object type {type(obj)} for '{nm}'.") 