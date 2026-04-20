
from equinox import filter_jit
import jax.numpy as jnp
from collections.abc import Sequence

from .abstract_response_measure import AbstractResponseMeasure

WINDOWS = [None, "hann", "hamming"]

class Demodulation(AbstractResponseMeasure):

    def __init__(
        self,
        multiples: Sequence[float] = (1.0,),
        window: str | None = None,
        modal_contributions: jnp.ndarray | None = None,
    ):
        super().__init__(modal_contributions=modal_contributions)

        if len(multiples) == 0:
            raise ValueError("multiples cannot be empty")

        self.multiples = jnp.asarray(multiples)

        if window is None:
            self.window = jnp.ones
        elif window == "hann":
            self.window = jnp.hanning
        elif window == "hamming":
            self.window = jnp.hamming
        else:
            raise ValueError(f"Unsupported window: {window}. Supported windows are: {WINDOWS}")

    def __call__(
        self,
        xs: jnp.ndarray,
        ts: jnp.ndarray,
        drive_omega: jnp.ndarray,
    ):
        return self.dft(xs, ts, drive_omega)

    @filter_jit
    def dft(
        self,
        xs: jnp.ndarray,
        ts: jnp.ndarray,
        drive_omega: jnp.ndarray,
    ):
        xs = jnp.asarray(xs)
        ts = jnp.asarray(ts)

        n_ts = xs.shape[0]
        n_modes = xs.shape[1]
        modal_contributions = self._resolve_modal_contributions(n_modes=n_modes)
        modal_contributions = jnp.asarray(modal_contributions, dtype=xs.dtype)

        drive_omega = jnp.full((n_modes,), drive_omega)

        w = self.window(n_ts)
        demod_omegas = self.multiples[:, None] * drive_omega[None, :]

        # exp_term shape: (n_multiples, n_ts, n_modes)
        exp_term = jnp.exp(-1j * demod_omegas[:, None, :] * ts[None, :, None])

        # C shape: (n_multiples, n_modes)
        C = jnp.sum(exp_term * (xs[None, :, :] * w[None, :, None]), axis=1)
        # Total displacement phasor at the measurement point, per multiple.
        C_total = jnp.sum(C * modal_contributions[None, :], axis=1)

        cg = w.sum()
        modal_amplitudes = 2.0 * jnp.abs(C) / cg
        modal_phase_rad = jnp.angle(C)
        total_amplitudes = 2.0 * jnp.abs(C_total) / cg
        total_phase_rad = jnp.angle(C_total)

        total_demod_omegas = self.multiples * jnp.max(drive_omega)

        return {
            "modal": {
                "amplitude": modal_amplitudes,
                "phase": modal_phase_rad,
                "response_frequency": demod_omegas,
            },
            "total": {
                "amplitude": total_amplitudes,
                "phase": total_phase_rad,
                "response_frequency": total_demod_omegas,
            },
        }
