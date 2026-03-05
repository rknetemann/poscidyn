
from equinox import filter_jit
import jax.numpy as jnp

from .abstract_response_measure import AbstractResponseMeasure

class Min(AbstractResponseMeasure):
    def __init__(self, mode_shape: jnp.ndarray | None = None):
        super().__init__(mode_shape=mode_shape)

    def __call__(
        self,
        xs: jnp.ndarray,
        ts: jnp.ndarray,
        drive_omega: jnp.ndarray,
    ):
        return self.min(xs, ts)

    @filter_jit
    def min(
        self,
        xs: jnp.ndarray,
        ts: jnp.ndarray,
    ):
        xs = jnp.asarray(xs)
        _ = jnp.asarray(ts)

        if xs.ndim == 1:
            xs = xs[:, None]
        elif xs.ndim != 2:
            raise ValueError("xs must have shape (n_ts, n_modes) or (n_ts,)")

        n_modes = xs.shape[1]
        mode_shape = self._resolve_mode_shape(n_modes=n_modes)
        mode_shape = jnp.asarray(mode_shape, dtype=xs.dtype)

        modal_amplitudes = jnp.min(xs, axis=0)
        modal_phases = jnp.full_like(modal_amplitudes, jnp.nan)
        modal_response_frequency = jnp.full_like(modal_amplitudes, jnp.nan)

        x_total = jnp.sum(xs * mode_shape[None, :], axis=1)
        total_amplitude = jnp.min(x_total, axis=0)
        total_phase = jnp.full_like(total_amplitude, jnp.nan)
        total_response_frequency = jnp.full_like(total_amplitude, jnp.nan)

        return {
            "modal": {
                "amplitude": modal_amplitudes,
                "phase": modal_phases,
                "response_frequency": modal_response_frequency,
            },
            "total": {
                "amplitude": total_amplitude,
                "phase": total_phase,
                "response_frequency": total_response_frequency,
            },
        }
