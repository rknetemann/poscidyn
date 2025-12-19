# dataset_normalizer.py

from __future__ import annotations

from dataclasses import dataclass
import equinox as eqx
import jax.numpy as jnp


class DatasetNormalizer(eqx.Module):
    x_mean: jnp.ndarray  # shape (1, x_dim)
    x_std: jnp.ndarray   # shape (1, x_dim)
    y_mean: jnp.ndarray  # shape (1, y_dim)
    y_std: jnp.ndarray   # shape (1, y_dim)

    @classmethod
    def from_data(cls, X: jnp.ndarray, Y: jnp.ndarray, eps: float = 1e-8) -> "DatasetNormalizer":
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)

        x_mean = jnp.mean(X, axis=0, keepdims=True)
        x_std = jnp.std(X, axis=0, keepdims=True) + eps

        y_mean = jnp.mean(Y, axis=0, keepdims=True)
        y_std = jnp.std(Y, axis=0, keepdims=True) + eps

        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def norm_X(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.asarray(X)
        return (X - self.x_mean) / self.x_std

    def norm_Y(self, Y: jnp.ndarray) -> jnp.ndarray:
        Y = jnp.asarray(Y)
        return (Y - self.y_mean) / self.y_std

    def denorm_Y(self, Y: jnp.ndarray) -> jnp.ndarray:
        Y = jnp.asarray(Y)
        return Y * self.y_std + self.y_mean


@dataclass
class _RunningStats:
    n: int
    mean: jnp.ndarray  # shape (dim,)
    m2: jnp.ndarray    # shape (dim,)


def _init_running(dim: int, dtype=jnp.float32) -> _RunningStats:
    return _RunningStats(n=0, mean=jnp.zeros((dim,), dtype=dtype), m2=jnp.zeros((dim,), dtype=dtype))


def _update_running(stats: _RunningStats, batch: jnp.ndarray) -> _RunningStats:
    """
    Welford update for vector-valued data.
    batch: (B, dim)
    """
    batch = jnp.asarray(batch)
    b_n = batch.shape[0]
    if b_n == 0:
        return stats

    b_mean = jnp.mean(batch, axis=0)
    b_m2 = jnp.sum((batch - b_mean) ** 2, axis=0)

    if stats.n == 0:
        return _RunningStats(n=int(b_n), mean=b_mean, m2=b_m2)

    n_a = stats.n
    n_b = int(b_n)
    n = n_a + n_b

    delta = b_mean - stats.mean
    mean = stats.mean + delta * (n_b / n)

    # Parallel/merge form of Welford
    m2 = stats.m2 + b_m2 + (delta**2) * (n_a * n_b / n)

    return _RunningStats(n=n, mean=mean, m2=m2)


def _finalize_running(stats: _RunningStats, eps: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    if stats.n < 2:
        # Degenerate; avoid div-by-zero
        mean = stats.mean
        std = jnp.ones_like(mean)
        return mean, std
    var = stats.m2 / (stats.n - 1)
    std = jnp.sqrt(var) + eps
    return stats.mean, std


def build_normalizer_from_dataloader(
    dataloader,
    *,
    split_idx: int = 0,
    batch_size: int = 512,
    dtype=jnp.float32,
    eps: float = 1e-8,
) -> DatasetNormalizer:
    """
    Streaming computation of mean/std over a split using DataLoader.load_data.
    Does NOT materialize the full dataset in memory.
    """
    n = dataloader.n_split_sims[split_idx]
    if n is None or n <= 0:
        raise RuntimeError(f"Split {split_idx} has no simulations.")

    x_dim = dataloader.x_dim
    y_dim = dataloader.y_dim
    if x_dim is None or y_dim is None:
        raise RuntimeError("DataLoader x_dim/y_dim not initialized.")

    x_stats = _init_running(int(x_dim), dtype=dtype)
    y_stats = _init_running(int(y_dim), dtype=dtype)

    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        x_np, y_np = dataloader.load_data(split_idx, idxs)
        x = jnp.asarray(x_np, dtype=dtype)
        y = jnp.asarray(y_np, dtype=dtype)

        x_stats = _update_running(x_stats, x)
        y_stats = _update_running(y_stats, y)

    x_mean, x_std = _finalize_running(x_stats, eps)
    y_mean, y_std = _finalize_running(y_stats, eps)

    # Store as (1, dim) for broadcasting over (B, dim)
    return DatasetNormalizer(
        x_mean=x_mean[None, :],
        x_std=x_std[None, :],
        y_mean=y_mean[None, :],
        y_std=y_std[None, :],
    )


def save_normalizer(path, normalizer: DatasetNormalizer) -> None:
    eqx.tree_serialise_leaves(path, normalizer)


def try_load_normalizer(path, template: DatasetNormalizer) -> DatasetNormalizer | None:
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return None
    return eqx.tree_deserialise_leaves(path, template)
