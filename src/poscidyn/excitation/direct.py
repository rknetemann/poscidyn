import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_excitation import AbstractExcitation

class DirectExcitation(AbstractExcitation):
    def __init__(self, f_d: Array, omegas: Array, lambdas: Array = jnp.array([1.0])):
        super().__init__(omegas, lambdas)

        self.f_d = f_d

    def f_e(self, t: Float, y: Array, args: PyTree) -> float:
        """Direct external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
        """
        f_amp = args.get("f_amp")
        if f_amp is None:
            f_amp = self.f_d * args["lambda"]
        return f_amp * jnp.cos(args["omega"] * t)
        
    
    
