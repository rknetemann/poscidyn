import h5py
from pathlib import Path
import numpy as np

HDF5_FILE_EXTENSIONS = ['.hdf5', '.h5']

class DataLoader:
    def __init__(self, hdf5_files: list[str] | Path , split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1)):
        self.hdf5_files: list[str] | Path = hdf5_files
        self.split_ratios: tuple[float, ...] = split_ratios

        self.n_modes: int = None
        self.n_files: int = None
        self.n_sims: int = None
        self.n_split_sims: tuple[int, ...] = None
        self.splits_sims_idxs: list[list[tuple[int, int]]] = None
        self.parameter_shapes: dict[str, tuple[int, ...]] = None
        self._opened_files: list[tuple[h5py.File, list[str]]] | None = None

        if isinstance(self.hdf5_files, Path):
            if self.hdf5_files.is_dir():
                self.hdf5_files = list(self.hdf5_files.glob('*.hdf5'))
                self.n_files = len(self.hdf5_files)
            else:
                raise ValueError(f"The provided path '{self.hdf5_files}' is not a directory.")
            
        elif isinstance(self.hdf5_files, list):
            for file in self.hdf5_files:
                file = Path(file)
                if not file.is_file() or file.suffix not in HDF5_FILE_EXTENSIONS:
                    raise ValueError(f"File '{file}' is not a valid HDF5 file.")
            self.n_files = len(self.hdf5_files)

        if sum(self.split_ratios) != 1:
            raise ValueError("The split ratio's must sum to 1.")

        self._get_files_info()
        self._split_data()

    def load_data(self, split_idx: int, split_sims_idxs_to_load: list[int] | str = 'all'):
        if not (0 <= split_idx < len(self.split_ratios)):
            raise ValueError(f"split_idx must be between 0 and {len(self.split_ratios) - 1}, inclusive.")
        
        if not (split_sims_idxs_to_load == 'all' or all(isinstance(i, int) and 0 <= i < self.n_split_sims[split_idx] for i in split_sims_idxs_to_load)):
            raise ValueError(f"split_sims_idxs_to_load must be 'all' or a list of valid indices for split {split_idx}.")

        opened_files = self._get_or_open_files()
        split_sims_idxs = self.splits_sims_idxs[split_idx]

        pairs: list[tuple[np.ndarray, np.ndarray]] = []

        requested_idxs = (
            split_sims_idxs_to_load
            if split_sims_idxs_to_load != 'all'
            else range(self.n_split_sims[split_idx])
        )

        for split_sim_idx in requested_idxs:
            file_idx, file_sim_idx = split_sims_idxs[split_sim_idx]
            file, sim_names = opened_files[file_idx]

            sim_name = sim_names[file_sim_idx]
            sim_grp: h5py.Group = file["simulations"][sim_name]
            forward_sweep = sim_grp['forward_sweep']
            backward_sweep = sim_grp['backward_sweep']

            # TO DO: Use backwards sweeps as well
            if isinstance(forward_sweep, h5py.Dataset) and isinstance(backward_sweep, h5py.Dataset):
                flattened_Q = sim_grp.attrs['Q'].flatten()
                flattened_omega_0 = sim_grp.attrs['omega_0'].flatten()
                flattened_gamma = forward_sweep.attrs['gamma_ndim'].flatten()
                flattened_f_amps = sim_grp.attrs['f_amps'].flatten()
                flattened_modal_forces = sim_grp.attrs['modal_forces'].flatten()
                flattened_f_omegas = sim_grp.attrs['f_omegas'].flatten() 

                flattened_forward_sweep = forward_sweep[:].flatten()

                x = np.concatenate([
                    flattened_Q,
                    flattened_omega_0,
                    flattened_gamma,
                    flattened_f_amps,
                    flattened_modal_forces,
                    np.array([np.min(flattened_f_omegas)]),
                    np.array([np.max(flattened_f_omegas)]),
                ])

                y = flattened_forward_sweep

                pairs.append((x, y))
            else:
                raise TypeError(f"Expected a dataset for forward and backward sweeps in '{sim_name}', but found {type(forward_sweep)} and {type(backward_sweep)}.")
        if not pairs:
            raise RuntimeError("No simulations were loaded for the requested split/indices.")
        return pairs

    def _get_or_open_files(self) -> list[tuple[h5py.File, list[str]]]:
        if self._opened_files is None:
            self._opened_files = []
            for hdf5_file_path, _ in self.hdf5_files:
                file_obj = h5py.File(hdf5_file_path, 'r')
                sims_grp: h5py.Group = file_obj["simulations"]
                sim_names = sorted(sims_grp.keys())
                self._opened_files.append((file_obj, sim_names))
        return self._opened_files

    def close(self):
        if self._opened_files is not None:
            for file_obj, _ in self._opened_files:
                file_obj.close()
            self._opened_files = None

    def __del__(self):
        self.close()

    def _split_data(self):
        # Calculate number of simulations per split based on ratios
        n_splits = len(self.split_ratios)
        n_split_sims = [0, ] * n_splits
        cumulative_ratio = 0

        for i, ratio in enumerate(self.split_ratios):
            cumulative_ratio += ratio
            if i == len(self.split_ratios) - 1:
                n_split_sims[i] = self.n_sims - sum(n_split_sims)
            else:
                n_split_sims[i] = round(self.n_sims * cumulative_ratio) - sum(n_split_sims)              

        if sum(n_split_sims) != self.n_sims:
            raise ValueError("Split calculations do not add up to the total number of simulations.")
        
        self.n_split_sims = tuple(n_split_sims)
        
        # Allocate simulations to splits
        n_files = len(self.hdf5_files)
        unallocated_files_sims: list[list[int]] = [None, ] * n_files
        for i, (file, n_file_sims) in enumerate(self.hdf5_files):
            sim_idxs = list(range(n_file_sims))
            unallocated_files_sims[i] = sim_idxs

        splits_sims_idxs: list[list[tuple[int, int]]] = [None, ] * n_splits
        for split_idx in range(n_splits):
            n_sims_to_allocate = n_split_sims[split_idx]
            allocated_sim_idxs = []

            while n_sims_to_allocate > 0:
                for file_i in range(n_files):
                    if n_sims_to_allocate == 0:
                        break

                    if unallocated_files_sims[file_i]:
                        sim_idx = unallocated_files_sims[file_i].pop(0)
                        allocated_sim_idxs.append((file_i, sim_idx))
                        n_sims_to_allocate -= 1

            splits_sims_idxs[split_idx] = allocated_sim_idxs

        self.splits_sims_idxs = splits_sims_idxs
    
    def _get_files_info(self):
        n_sims = 0
        n_modes = None
        checked_files = []
        for file in self.hdf5_files:
            with h5py.File(file, 'r') as f:
                if "simulations" not in f or not isinstance(f["simulations"], h5py.Group):
                    raise ValueError(f"Expected a 'simulations' group in the HDF5 file '{file}'.")
                
                sims_grp: h5py.Group = f["simulations"]

                sim_names = sorted(sims_grp.keys())
                if not sim_names:
                    raise ValueError(f"No simulations found in HDF5 file '{file}'.")
                
                test_sim: h5py.Group = sims_grp[sim_names[0]]
                shapes = {
                        'forward_sweep': test_sim["forward_sweep"].shape,
                        'Q': test_sim.attrs['Q'].shape,
                        'omega_0': test_sim.attrs['omega_0'].shape,
                        'gamma': test_sim["forward_sweep"].attrs['gamma_ndim'].shape,
                        'f_amps': test_sim.attrs['f_amps'].shape,
                        'modal_forces': test_sim.attrs['modal_forces'].shape,
                        'f_omegas': test_sim.attrs['f_omegas'].shape,
                }
                self.parameter_shapes = shapes
                
                n_sims += len(sim_names)
                checked_files.append((file, len(sim_names)))

                file_n_modes = f.attrs.get('n_modes', 2)
                if not n_modes:
                    n_modes = file_n_modes
                elif n_modes != file_n_modes:
                    raise ValueError(f"Inconsistent number of modes across files. Expected {n_modes}, but found {file_n_modes} in file '{file}'.")

        self.n_sims = n_sims
        self.hdf5_files = checked_files

if __name__ == "__main__":
    loader = DataLoader(Path('/home/raymo/Projects/oscidyn/data/simulations/'))

    test = loader.load_data(split_idx=0, split_sims_idxs_to_load=list(range(0, 5)))
    print(f"Loaded {len(test)} simulation pairs.")

    
