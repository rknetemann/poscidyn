from __future__ import annotations
from typing import Iterator, Optional, Sequence, Tuple
import math
import h5py
import numpy as np

class H5Dataset:
    """
    Minimal dataset wrapper around an HDF5 file with datasets 'x' and 'y'.
    - Opens the file in read-only mode.
    - Supports integer indexing and slicing for single samples or batches of indices.
    """

    def __init__(
        self,
        h5_path: str,
        x_key: str = "x",
        y_key: str = "y",
        libver: str = "latest",
        swmr: bool = True,
    ):
        self.path = h5_path
        self.file = h5py.File(h5_path, mode="r", libver=libver, swmr=swmr)
        if x_key not in self.file or y_key not in self.file:
            raise KeyError(f"Expected datasets '{x_key}' and '{y_key}' in {h5_path}")
        self.x = self.file[x_key]
        self.y = self.file[y_key]

        if len(self.x.shape) < 1 or self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y must share the same length along axis 0.")

        self.length = self.x.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        """
        Supports:
        - int idx -> returns (x[idx], y[idx])
        - 1D np.ndarray or list of ints -> returns (x[idxs], y[idxs]) as arrays
        """
        return self.x[idx], self.y[idx]

    def close(self):
        try:
            # Close datasets first (not strictly required but tidy).
            if hasattr(self, "x"): self.x.id.close()
            if hasattr(self, "y"): self.y.id.close()
        except Exception:
            pass
        try:
            if hasattr(self, "file"): self.file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


class DataLoader:
    """
    Simple DataLoader for H5Dataset.
    - Shuffles indices once per epoch.
    - Supports drop_last and deterministic seeding.
    - Vectorized batch reads via fancy indexing.
    """

    def __init__(
        self,
        dataset: H5Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = 42,
        indices: Optional[Sequence[int]] = None,  # subset (e.g., train/val/test split)
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        if indices is None:
            self.indices = np.arange(len(dataset), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
        self._rng = np.random.default_rng(seed) if seed is not None else None

        if self.drop_last:
            self.num_batches = len(self.indices) // self.batch_size
        else:
            self.num_batches = math.ceil(len(self.indices) / self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = self.indices.copy()
        if self.shuffle and self._rng is not None:
            self._rng.shuffle(idx)

        # Iterate in contiguous chunks; h5py fancy indexing (array of ints) is supported.
        for start in range(0, len(idx), self.batch_size):
            stop = start + self.batch_size
            if stop > len(idx):
                if self.drop_last:
                    break
                stop = len(idx)
            batch_idx = idx[start:stop]
            x_batch, y_batch = self.dataset[batch_idx]  # vectorized read
            # Ensure NumPy arrays (h5py returns array-likes already)
            x_batch = np.asarray(x_batch)
            y_batch = np.asarray(y_batch)
            yield x_batch, y_batch
