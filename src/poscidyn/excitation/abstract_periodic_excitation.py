from abc import ABC, abstractmethod
from typing import Optional

from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp

from .abstract_excitation import AbstractExcitation

class AbstractPeriodicExcitation(AbstractExcitation):
    """Interface for excitations with one harmonic drive frequency.

    Subclasses implement specific excitation functions, such as free vibration, harmonic excitation, etc.
    
    """

    def __init__(
        self,
        omega: Optional[Float] = None,
        lambdas: Array = jnp.array([1.0]),
    ):
        super().__init__(lambdas=lambdas)
        self.omega = omega

    @abstractmethod
    def f_e(self, t: Float, y: Array, args: PyTree, **kwargs) -> Array:
        """External forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): State vector
            args (PyTree): Additional arguments
            **kwargs: Additional keyword arguments
        """
        pass
