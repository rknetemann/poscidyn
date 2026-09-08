import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_periodic_excitation import AbstractPeriodicExcitation

class ParametricHarmonicExcitation(AbstractPeriodicExcitation):
    """ Parametric harmonic excitation class.

    The external forces are applied to the system as a parametric harmonic function, i.e., f_e = f_p * cos(omega * t) * q.

    """
    def __init__(self, f_p: Array, lambdas: Array):
        """Initialize ParametricHarmonicExcitation.
        
        Args:
            f_p (Array): Amplitude of the parametric harmonic excitation.
            lambdas (Array): Scaling factors for the excitation. Defaults to
                ``jnp.array([1.0])``.
        """
        super().__init__(lambdas)

        self.f_p = f_p

    def f_e(self, t: Float, y: Array, args: PyTree, **kwargs) -> Array:
        """Parametric external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
            omega (Float): Periodic excitation frequency
        """
        omega = args.get("omega")   
        q, dq_dt = jnp.split(y, 2)
        return self.f_p * self.lambdas * jnp.cos(omega * t) * q
