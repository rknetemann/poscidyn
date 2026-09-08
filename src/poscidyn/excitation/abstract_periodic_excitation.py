from abc import ABC, abstractmethod
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp

from .abstract_excitation import AbstractExcitation

class AbstractPeriodicExcitation(AbstractExcitation):
    """ Abstract interface for excitation functions.

    Subclasses implement specific excitation functions, such as free vibration, harmonic excitation, etc.
    
    """

    @abstractmethod
    def f_e(self, t: Float, y: Array, args: PyTree, **kwargs) -> Array:
        """External forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): State vector
            args (PyTree): Additional arguments
            **kwargs: Additional keyword arguments
        """
        omega = args.get("omega")
        pass