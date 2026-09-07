import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_excitation import AbstractExcitation

class ParametricHarmonicExcitation(AbstractExcitation):
    def __init__(self, f_p: Array, omegas: Array, lambdas: Array):
        super().__init__(omegas, lambdas)

        self.f_p = f_p

    def f_e(self, t: Float, y: Array, args: PyTree) -> float:
        """Parametric external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
        """
        q, dq_dt = jnp.split(y, 2)
        f_amp = args.get("f_amp")
        if f_amp is None:
            f_amp = self.f_p * args["lambda"]
        return f_amp * jnp.cos(args["omega"] * t) * q
