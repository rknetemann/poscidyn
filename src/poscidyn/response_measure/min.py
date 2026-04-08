
from equinox import filter_jit
import jax.numpy as jnp

from .abstract_response_measure import AbstractResponseMeasure

class Min(AbstractResponseMeasure):
    def __init__(
        self,
        modal_contributions: jnp.ndarray | None = None,
        mode_shape: jnp.ndarray | None = None,
    ):
        if modal_contributions is not None and mode_shape is not None:
            raise ValueError("Pass either modal_contributions or mode_shape, not both.")
        if modal_contributions is None:
            modal_contributions = mode_shape
        super().__init__(modal_contributions=modal_contributions)

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
        modal_contributions = self._resolve_modal_contributions(n_modes=n_modes)
        modal_contributions = jnp.asarray(modal_contributions, dtype=xs.dtype)

        modal_amplitudes = jnp.min(xs, axis=0)
        modal_phases = jnp.full_like(modal_amplitudes, jnp.nan)
        modal_response_frequency = jnp.full_like(modal_amplitudes, jnp.nan)

        x_total = jnp.sum(xs * modal_contributions[None, :], axis=1)
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
