from equinox import filter_jit
import jax.numpy as jnp
from collections.abc import Sequence

from .abstract_response_measure import AbstractResponseMeasure

WINDOWS = [None, "hann", "hamming", "blackman", "bartlett"]


class Demodulation(AbstractResponseMeasure):
    def __init__(
        self,
        multiples: Sequence[float] = (1.0,),
        window: str | None = "hann",
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
        elif window == "blackman":
            self.window = jnp.blackman
        elif window == "bartlett":
            self.window = jnp.bartlett
        else:
            raise ValueError(
                f"Unsupported window: {window}. Supported windows are: {WINDOWS}"
            )

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

        n_ts, n_modes = xs.shape

        modal_contributions = self._resolve_modal_contributions(n_modes=n_modes)
        modal_contributions = jnp.asarray(modal_contributions)

        drive_omega = jnp.asarray(drive_omega)
        if drive_omega.ndim == 0:
            drive_omega_modes = jnp.full((n_modes,), drive_omega)
        else:
            drive_omega_modes = drive_omega

        w = jnp.asarray(self.window(n_ts), dtype=xs.dtype)
        cg = jnp.sum(w)

        demod_omegas = self.multiples[:, None] * drive_omega_modes[None, :]
        exp_term = jnp.exp(-1j * demod_omegas[:, None, :] * ts[None, :, None])
        C = jnp.sum(
            exp_term * (xs[None, :, :] * w[None, :, None]),
            axis=1,
        )

        x_total = jnp.sum(xs * modal_contributions[None, :], axis=1)

        total_demod_omegas = self.multiples * jnp.max(drive_omega_modes)
        exp_total = jnp.exp(-1j * total_demod_omegas[:, None] * ts[None, :])
        C_total = jnp.sum(
            exp_total * (x_total[None, :] * w[None, :]),
            axis=1,
        )

        modal_amplitudes = 2.0 * jnp.abs(C) / cg
        modal_phase_rad = jnp.angle(C)

        total_amplitudes = 2.0 * jnp.abs(C_total) / cg
        total_phase_rad = jnp.angle(C_total)

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