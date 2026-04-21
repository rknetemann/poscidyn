import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_excitation import AbstractExcitation

class ParametricExcitation(AbstractExcitation):
    def __init__(self, drive_frequencies, drive_amplitudes, modal_forces):
        super().__init__(drive_frequencies, drive_amplitudes, modal_forces)

    def f_e(self, t: Float, y: Array, args: PyTree) -> float:
        """Parametric external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
        """
        q, dq_dt = jnp.split(y, 2)
        return 0.5 * jnp.cos(args['f_omega'] * t) * q + args['f_amp'] * jnp.cos(args['f_omega'] * t)
