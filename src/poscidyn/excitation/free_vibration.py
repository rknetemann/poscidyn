import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_excitation import AbstractExcitation

class FreeVibration(AbstractExcitation):
    """Free vibration excitation class. No external forces are applied to the system, i.e., f_e = 0.
    
    """
    def __init__(self):
        super().__init__()

    def f_e(self, t: Float, y: Array, args: PyTree) -> float:
        """Direct external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
        """

        f_amp = 0.0 * args["lambda"]
        return f_amp
        
    
    
