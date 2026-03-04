
from equinox import filter_jit
import jax.numpy as jnp
from collections.abc import Sequence

from .abstract_response_measure import AbstractResponseMeasure

WINDOWS = [None, "hann", "hamming"]

class Demodulation(AbstractResponseMeasure):

    def __init__(self, multiples: Sequence[float] = (1.0,), window: str | None = None):
        super().__init__()

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

    def __call__(self, xs: jnp.ndarray, ts: jnp.ndarray, drive_omega: jnp.ndarray):
        return self.dft(xs, ts, drive_omega)

    @filter_jit
    def dft(self, xs: jnp.ndarray, ts: jnp.ndarray, drive_omega: jnp.ndarray):
        xs = jnp.asarray(xs)
        ts = jnp.asarray(ts)

        if xs.ndim == 1:
            xs = xs[:, None]
        elif xs.ndim != 2:
            raise ValueError("xs must have shape (n_ts, n_modes) or (n_ts,)")

        if ts.ndim != 1:
            raise ValueError("ts must have shape (n_ts,)")

        n_ts = xs.shape[0]
        if ts.shape[0] != n_ts:
            raise ValueError("xs and ts must have matching time dimension")

        n_modes = xs.shape[1]
        drive_omega = jnp.asarray(drive_omega)
        if drive_omega.ndim == 0:
            drive_omega = jnp.full((n_modes,), drive_omega)
        elif drive_omega.ndim == 1 and drive_omega.shape[0] == 1 and n_modes > 1:
            drive_omega = jnp.full((n_modes,), drive_omega[0])
        elif drive_omega.ndim != 1:
            raise ValueError("drive_omega must be scalar or shape (n_modes,)")

        if drive_omega.shape[0] != n_modes:
            raise ValueError("drive_omega must be scalar or match n_modes")

        w = self.window(n_ts)
        demod_omegas = self.multiples[:, None] * drive_omega[None, :]

        # exp_term shape: (n_multiples, n_ts, n_modes)
        exp_term = jnp.exp(-1j * demod_omegas[:, None, :] * ts[None, :, None])

        # C shape: (n_multiples, n_modes)
        C = jnp.sum(exp_term * (xs[None, :, :] * w[None, :, None]), axis=1)

        cg = w.sum()
        amplitudes = 2.0 * jnp.abs(C) / cg
        return amplitudes, demod_omegas
