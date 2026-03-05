
from equinox import filter_jit
import jax.numpy as jnp

from .abstract_response_measure import AbstractResponseMeasure

class Min(AbstractResponseMeasure):
    def __init__(self):
        super().__init__()

    def __call__(self, xs: jnp.ndarray, ts: jnp.ndarray, drive_omega: jnp.ndarray):
        return self.min(xs, ts)

    @filter_jit
    def min(self, xs: jnp.ndarray, ts: jnp.ndarray):
        xs = jnp.asarray(xs)
        _ = jnp.asarray(ts)

        if xs.ndim == 1:
            xs = xs[:, None]
        elif xs.ndim != 2:
            raise ValueError("xs must have shape (n_ts, n_modes) or (n_ts,)")

        amplitudes = jnp.min(xs, axis=0)
        phases = jnp.full_like(amplitudes, jnp.nan)
        
        return amplitudes, phases
