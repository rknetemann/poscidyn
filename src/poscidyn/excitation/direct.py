import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_periodic_excitation import AbstractPeriodicExcitation

class DirectHarmonicExcitation(AbstractPeriodicExcitation):
    """ Direct harmonic excitation class.

    The external forces are applied to the system as a harmonic function, i.e., f_e = f_d * cos(omega * t).
    
    """
    def __init__(self, f_d: Array, lambdas: Array = jnp.array([1.0])):
        """Initialize DirectHarmonicExcitation.

        Args:
            f_d (Array): Amplitude of the harmonic excitation.
            lambdas (Array): Scaling factors for the excitation. Defaults to
                ``jnp.array([1.0])``.
        """
        super().__init__(lambdas)

        self.f_d = f_d

    def f_e(self, t: Float, y: Array, args: PyTree, **kwargs) -> Array:
        """Direct external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
            omega (Float): Periodic excitation frequency
        """
        omega = args.get("omega")
        return self.f_d * self.lambdas * jnp.cos(omega * t)
        
    
    
