from abc import ABC, abstractmethod
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp

class AbstractExcitation(ABC):
    def __init__(self, omegas: Array, lambdas: Array = jnp.array([1.0])):
        
        self.omegas = omegas
        self.lambdas = lambdas

    @abstractmethod
    def f_e(self, t: Float, y: Array, args: PyTree) -> float:
        """External forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): State vector
            args (PyTree): Additional arguments
        """
        pass