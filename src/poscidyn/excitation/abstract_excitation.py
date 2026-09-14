from abc import ABC, abstractmethod
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp

class AbstractExcitation(ABC):
    """ Abstract interface for excitation functions.

    Subclasses implement specific excitation functions, such as free vibration, harmonic excitation, etc.

    """
    def __init__(self, lambdas: Array = jnp.array([1.0])):
        """Initialize an excitation.

        Args:
            lambdas (Array): Scaling factors for the excitation. Defaults to
                ``jnp.array([1.0])``.
        """
        
        self.lambdas = lambdas

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