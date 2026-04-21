from abc import ABC, abstractmethod
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp

from .. import constants as const

class AbstractExcitation(ABC):
    def __init__(self, drive_frequencies, drive_amplitudes, modal_forces):
        if drive_frequencies.ndim != 1:
            raise ValueError("drive_frequencies must be a 1D array")
        if drive_amplitudes.ndim != 1:
            raise ValueError("drive_amplitudes must be a 1D array")
        if modal_forces.ndim != 1:
            raise ValueError("modal_forces must be a 1D array")

        self.drive_frequencies = drive_frequencies
        self.drive_amplitudes = drive_amplitudes
        self.modal_forces = modal_forces
                
        self.f_omegas = jnp.asarray(drive_frequencies)
        self.f_amps = jnp.outer(jnp.asarray(drive_amplitudes), jnp.asarray(modal_forces))

    @abstractmethod
    def f_d(self, t: Float, state: Array, args: PyTree) -> float:
        return 0.0

    @abstractmethod
    def f_p(self, t: Float, state: Array, args: PyTree) -> float:
        return 0.0