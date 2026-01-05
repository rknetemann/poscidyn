import h5py
from pathlib import Path
import numpy as np

HDF5_FILE_EXTENSIONS = [".hdf5", ".h5"]


class DataLoader:
    def __init__(self, hdf5_files: list[str] | Path, 
                 split_ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
                 params: dict[str, list[tuple[int, ...]] | tuple] | None = None,
                 *,
                 log_gamma: bool = True,
                 gamma_log_eps: float = 1e-12):
        
        self.hdf5_files: list[str] | Path = hdf5_files
        self.split_ratios: tuple[float, ...] = split_ratios
        self.params: dict[str, list[tuple[int, ...]] | None] | None = self._normalize_params_dict(params)
        self.log_gamma = bool(log_gamma)
        self.gamma_log_eps = float(gamma_log_eps)

        self.n_modes: int | None = None
        self.n_files: int | None = None
        self.n_sims: int | None = None
        self.n_split_sims: tuple[int, ...] | None = None
        self.splits_sims_idxs: list[list[tuple[int, int]]] | None = None
        self.parameter_shapes: dict[str, tuple[int, ...]] | None = None
        self._opened_files: list[tuple[h5py.File, list[str]]] | None = None

        self.x_dim: int | None = None
        self.y_dim: int | None = None
        self._alpha_indices: list[tuple[int, ...]] | None = None
        self._gamma_indices: list[tuple[int, ...]] | None = None
        self._alpha_shape: tuple[int, ...] | None = None
        self._gamma_shape: tuple[int, ...] | None = None
        self.y_slices: dict[str, slice] | None = None

        if isinstance(self.hdf5_files, Path):
            if self.hdf5_files.is_dir():
                self.hdf5_files = list(self.hdf5_files.glob("*.hdf5"))
                self.n_files = len(self.hdf5_files)
            else:
                raise ValueError(f"The provided path '{self.hdf5_files}' is not a directory.")

        elif isinstance(self.hdf5_files, list):
            checked = []
            for file in self.hdf5_files:
                p = Path(file)
                if not p.is_file() or p.suffix not in HDF5_FILE_EXTENSIONS:
                    raise ValueError(f"File '{p}' is not a valid HDF5 file.")
                checked.append(p)
            self.hdf5_files = checked
            self.n_files = len(self.hdf5_files)

        if not np.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("The split ratios must sum to 1.")

        self._get_files_info()
        self._split_data()

    def load_data(
        self,
        split_idx: int,
        split_sims_idxs_to_load: list[int] | str = "all",
        *,
        dtype: np.dtype = np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          x_batch: (batch, x_dim)  # sweep (forward_sweep)
          y_batch: (batch, y_dim)  # parameters vector
        """
        if not (0 <= split_idx < len(self.split_ratios)):
            raise ValueError(f"split_idx must be between 0 and {len(self.split_ratios) - 1}, inclusive.")

        if not (
            split_sims_idxs_to_load == "all"
            or (
                isinstance(split_sims_idxs_to_load, list)
                and all(isinstance(i, int) and 0 <= i < self.n_split_sims[split_idx] for i in split_sims_idxs_to_load)
            )
        ):
            raise ValueError(f"split_sims_idxs_to_load must be 'all' or a list of valid indices for split {split_idx}.")

        if self.x_dim is None or self.y_dim is None:
            raise RuntimeError("Internal error: x_dim/y_dim not initialized.")

        opened_files = self._get_or_open_files()
        split_sims_idxs = self.splits_sims_idxs[split_idx]

        requested_idxs = (
            list(range(self.n_split_sims[split_idx]))
            if split_sims_idxs_to_load == "all"
            else split_sims_idxs_to_load
        )
        if len(requested_idxs) == 0:
            raise ValueError("Requested zero indices; cannot build a batch.")

        x_batch = np.empty((len(requested_idxs), self.x_dim), dtype=dtype)
        y_batch = np.empty((len(requested_idxs), self.y_dim), dtype=dtype)
        sweep_difference_batch = np.empty((len(requested_idxs),), dtype=dtype)

        for row, split_sim_idx in enumerate(requested_idxs):
            file_idx, file_sim_idx = split_sims_idxs[split_sim_idx]
            file, sim_names = opened_files[file_idx]

            sim_name = sim_names[file_sim_idx]
            forward_sweep = file["forward_sweeps"][sim_name]
            backward_sweep = file["backward_sweeps"][sim_name]

            sweep_difference = np.sum(np.abs(forward_sweep[:] - backward_sweep[:]))

            # TO DO: Use backwards sweeps as well
            if not (isinstance(forward_sweep, h5py.Dataset) and isinstance(backward_sweep, h5py.Dataset)):
                raise TypeError(
                    f"Expected datasets for forward/backward sweep in '{sim_name}', "
                    f"but found {type(forward_sweep)} and {type(backward_sweep)}."
                )

            # x = sweep
            effective_f_omegas = np.asarray(forward_sweep.attrs["scaled_f_omegas"]).reshape(-1)
            x = np.concatenate(
                [
                    np.asarray(forward_sweep[:]).reshape(-1).astype(dtype, copy=False),
                    np.array([np.min(effective_f_omegas)], dtype=np.float64),
                    np.array([np.max(effective_f_omegas)], dtype=np.float64)
                ],
                axis=0,
            ).astype(dtype, copy=False)

            # y = parameters
            effective_Q = np.asarray(forward_sweep.attrs["Q"]).reshape(-1)
            effective_omega_0 = np.asarray(forward_sweep.attrs["scaled_omega_0"]).reshape(-1)
            effective_alpha = self._select_parameters(
                np.asarray(forward_sweep.attrs["scaled_alpha"]), self._alpha_indices, dtype
            )
            effective_gamma = self._select_parameters(
                np.asarray(forward_sweep.attrs["scaled_gamma"]), self._gamma_indices, dtype
            )
            if self.log_gamma:
                effective_gamma = np.log(effective_gamma + self.gamma_log_eps).astype(dtype, copy=False)
            effective_modal_forces = np.asarray(forward_sweep.attrs["modal_forces"]).reshape(-1)

            y = np.concatenate(
                [
                    effective_Q,
                    effective_omega_0,
                    effective_alpha,
                    effective_gamma,
                ],
                axis=0,
            ).astype(dtype, copy=False)

            if x.size != self.x_dim:
                raise RuntimeError(f"x dim mismatch for '{sim_name}': expected {self.x_dim}, got {x.size}")
            if y.size != self.y_dim:
                raise RuntimeError(f"y dim mismatch for '{sim_name}': expected {self.y_dim}, got {y.size}")

            x_batch[row, :] = x
            y_batch[row, :] = y

            sweep_difference_batch[row] = sweep_difference

        return x_batch, y_batch, sweep_difference_batch

    def close(self):
        opened = getattr(self, "_opened_files", None)
        if not opened:
            self._opened_files = None
            return
        for file_obj, _ in opened:
            try:
                if file_obj is not None:
                    file_obj.close()
            except Exception:
                # never throw during cleanup
                pass
        self._opened_files = None

    def _get_or_open_files(self) -> list[tuple[h5py.File, list[str]]]:
        if self._opened_files is None:
            self._opened_files = []
            # after _get_files_info(), self.hdf5_files is list[(Path, n_sims)]
            for hdf5_file_path, _ in self.hdf5_files:
                file_obj = h5py.File(hdf5_file_path, "r")
                forward_grp: h5py.Group = file_obj["forward_sweeps"]
                sim_names = sorted(forward_grp.keys())
                self._opened_files.append((file_obj, sim_names))
        return self._opened_files

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    def _split_data(self):
        n_splits = len(self.split_ratios)
        n_split_sims = [0] * n_splits
        cumulative_ratio = 0.0

        for i, ratio in enumerate(self.split_ratios):
            cumulative_ratio += ratio
            if i == n_splits - 1:
                n_split_sims[i] = self.n_sims - sum(n_split_sims)
            else:
                n_split_sims[i] = round(self.n_sims * cumulative_ratio) - sum(n_split_sims)

        if sum(n_split_sims) != self.n_sims:
            raise ValueError("Split calculations do not add up to the total number of simulations.")

        self.n_split_sims = tuple(n_split_sims)

        n_files = len(self.hdf5_files)
        unallocated_files_sims: list[list[int]] = [None] * n_files
        for i, (_, n_file_sims) in enumerate(self.hdf5_files):
            unallocated_files_sims[i] = list(range(n_file_sims))

        splits_sims_idxs: list[list[tuple[int, int]]] = [None] * n_splits
        for split_idx in range(n_splits):
            n_sims_to_allocate = n_split_sims[split_idx]
            allocated: list[tuple[int, int]] = []

            while n_sims_to_allocate > 0:
                for file_i in range(n_files):
                    if n_sims_to_allocate == 0:
                        break
                    if unallocated_files_sims[file_i]:
                        sim_idx = unallocated_files_sims[file_i].pop(0)
                        allocated.append((file_i, sim_idx))
                        n_sims_to_allocate -= 1

            splits_sims_idxs[split_idx] = allocated

        self.splits_sims_idxs = splits_sims_idxs

    def _get_files_info(self):
        n_sims = 0
        n_modes: int | None = None
        checked_files: list[tuple[Path, int]] = []

        for file in self.hdf5_files:
            file = Path(file)
            with h5py.File(file, "r") as f:
                if "forward_sweeps" not in f or not isinstance(f["forward_sweeps"], h5py.Group):
                    raise ValueError(f"Expected a 'forward_sweeps' group in the HDF5 file '{file}'.")
                if "backward_sweeps" not in f or not isinstance(f["backward_sweeps"], h5py.Group):
                    raise ValueError(f"Expected a 'backward_sweeps' group in the HDF5 file '{file}'.")
                if "unsweeped_modes" not in f or not isinstance(f["unsweeped_modes"], h5py.Group):
                    raise ValueError(f"Expected an 'unsweeped_modes' group in '{file}'.")
                if "unsweeped_total" not in f or not isinstance(f["unsweeped_total"], h5py.Group):
                    raise ValueError(f"Expected an 'unsweeped_total' group in '{file}'.")

                forward_grp: h5py.Group = f["forward_sweeps"]
                backward_grp: h5py.Group = f["backward_sweeps"]
                unsweeped_modes_grp: h5py.Group = f["unsweeped_modes"]
                unsweeped_total_grp: h5py.Group = f["unsweeped_total"]

                sim_names = sorted(forward_grp.keys())
                if not sim_names:
                    raise ValueError(f"No simulations found in HDF5 file '{file}'.")

                for grp_name, grp in [
                    ("backward_sweeps", backward_grp),
                    ("unsweeped_modes", unsweeped_modes_grp),
                    ("unsweeped_total", unsweeped_total_grp),
                ]:
                    if sorted(grp.keys()) != sim_names:
                        raise ValueError(
                            f"Simulation names differ between 'forward_sweeps' and '{grp_name}' in '{file}'."
                        )

                test_forward = forward_grp[sim_names[0]]

                shapes = {
                    "forward_sweep": test_forward.shape,
                    "Q": np.asarray(test_forward.attrs["Q"]).shape,
                    "omega_0": np.asarray(test_forward.attrs["omega_0"]).shape,
                    "f_omegas": np.asarray(test_forward.attrs["f_omegas"]).shape,
                    "f_amp": np.asarray(test_forward.attrs["f_amp"]).shape,
                    "modal_forces": np.asarray(test_forward.attrs["modal_forces"]).shape,
                }
                self.parameter_shapes = shapes if self.parameter_shapes is None else self.parameter_shapes

                current_alpha_shape = tuple(np.asarray(test_forward.attrs["scaled_alpha"]).shape)
                current_gamma_shape = tuple(np.asarray(test_forward.attrs["scaled_gamma"]).shape)

                if self._alpha_shape is None:
                    self._alpha_shape = current_alpha_shape
                elif self._alpha_shape != current_alpha_shape:
                    raise ValueError(
                        f"Inconsistent scaled_alpha shapes across files. Expected {self._alpha_shape}, "
                        f"but found {current_alpha_shape} in file '{file}'."
                    )

                if self._gamma_shape is None:
                    self._gamma_shape = current_gamma_shape
                elif self._gamma_shape != current_gamma_shape:
                    raise ValueError(
                        f"Inconsistent scaled_gamma shapes across files. Expected {self._gamma_shape}, "
                        f"but found {current_gamma_shape} in file '{file}'."
                    )

                if self.x_dim is None:
                    self._prepare_param_indices(current_alpha_shape, current_gamma_shape)

                    sweep_len = int(np.prod(test_forward.shape))

                    # x = sweep + fmin + fmax
                    self.x_dim = sweep_len + 2

                    # y = [Q, omega0, alpha(sel), gamma(sel)]
                    q_len = int(np.prod(np.asarray(test_forward.attrs["Q"]).shape))
                    omega_len = int(np.prod(np.asarray(test_forward.attrs["scaled_omega_0"]).shape))
                    alpha_len = self._param_length(self._alpha_indices, current_alpha_shape)
                    gamma_len = self._param_length(self._gamma_indices, current_gamma_shape)

                    self.y_dim = q_len + omega_len + alpha_len + gamma_len
                    self.y_slices = {
                        "Q": slice(0, q_len),
                        "omega_0": slice(q_len, q_len + omega_len),
                        "alpha": slice(q_len + omega_len, q_len + omega_len + alpha_len),
                        "gamma": slice(q_len + omega_len + alpha_len, q_len + omega_len + alpha_len + gamma_len),
                    }
                else:
                    expected_y_dim = (
                        int(np.prod(np.asarray(test_forward.attrs["Q"]).shape))
                        + int(np.prod(np.asarray(test_forward.attrs["scaled_omega_0"]).shape))
                        + self._param_length(self._alpha_indices, current_alpha_shape)
                        + self._param_length(self._gamma_indices, current_gamma_shape)
                    )
                    if expected_y_dim != self.y_dim:
                        raise ValueError(
                            f"Inconsistent parameter vector dimension across files. "
                            f"Expected {self.y_dim}, but got {expected_y_dim} for file '{file}'."
                        )

                n_sims += len(sim_names)
                checked_files.append((file, len(sim_names)))

                file_n_modes = int(f.attrs.get("n_modes", 2))
                if n_modes is None:
                    n_modes = file_n_modes
                elif n_modes != file_n_modes:
                    raise ValueError(
                        f"Inconsistent number of modes across files. Expected {n_modes}, "
                        f"but found {file_n_modes} in file '{file}'."
                    )

        self.n_sims = n_sims
        self.n_modes = n_modes
        self.hdf5_files = checked_files

    def _prepare_param_indices(
        self, alpha_shape: tuple[int, ...], gamma_shape: tuple[int, ...]
    ) -> None:
        self._alpha_indices = self._validate_param_indices("alpha", alpha_shape)
        self._gamma_indices = self._validate_param_indices("gamma", gamma_shape)

    def _validate_param_indices(
        self, key: str, shape: tuple[int, ...]
    ) -> list[tuple[int, ...]] | None:
        if self.params is None or key not in self.params:
            return None

        indices = self.params[key]
        if indices is None:
            return None

        validated: list[tuple[int, ...]] = []
        for idx in indices:
            if not isinstance(idx, tuple):
                raise TypeError(f"Entries for '{key}' must be tuples of indices, got {type(idx)}.")
            if len(idx) != len(shape):
                raise ValueError(
                    f"Index {idx} for '{key}' has {len(idx)} dimensions but expected {len(shape)}."
                )
            if any(i < 0 or i >= shape[dim] for dim, i in enumerate(idx)):
                raise ValueError(
                    f"Index {idx} for '{key}' is out of bounds for shape {shape}."
                )
            validated.append(idx)

        return validated

    def _param_length(self, indices: list[tuple[int, ...]] | None, shape: tuple[int, ...]) -> int:
        return int(np.prod(shape)) if indices is None else len(indices)

    def _select_parameters(
        self,
        tensor: np.ndarray,
        indices: list[tuple[int, ...]] | None,
        dtype: np.dtype,
    ) -> np.ndarray:
        if indices is None:
            return tensor.reshape(-1).astype(dtype, copy=False)
        if len(indices) == 0:
            return np.empty((0,), dtype=dtype)
        return np.asarray([tensor[idx] for idx in indices], dtype=dtype)

    def _normalize_params_dict(
        self, params: dict[str, list[tuple[int, ...]] | tuple] | None
    ) -> dict[str, list[tuple[int, ...]] | None] | None:
        if params is None:
            return None
        if not isinstance(params, dict):
            raise TypeError(f"params must be a dict or None, received {type(params)}.")

        allowed_keys = {"alpha", "gamma", "Q", "omega_0"}
        extra_keys = set(params.keys()) - allowed_keys
        if extra_keys:
            raise ValueError(f"Unsupported parameter keys in params: {sorted(extra_keys)}. Allowed keys: {sorted(allowed_keys)}.")

        normalized: dict[str, list[tuple[int, ...]] | None] = {}
        for key in allowed_keys:
            if key not in params:
                continue
            value = params[key]
            if value is None:
                normalized[key] = None
                continue

            if isinstance(value, tuple) and value and all(isinstance(v, int) for v in value):
                entries = [tuple(int(v) for v in value)]
            elif isinstance(value, tuple):
                entries = [tuple(int(v) for v in item) for item in value]
            elif isinstance(value, list):
                entries = [tuple(int(v) for v in item) for item in value]
            else:
                raise TypeError(
                    f"Expected list or tuple of index tuples for '{key}', got {type(value)}."
                )

            normalized[key] = entries

        return normalized
