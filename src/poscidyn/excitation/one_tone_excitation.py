import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_excitation import AbstractExcitation

class OneToneExcitation(AbstractExcitation):
    def __init__(self, drive_frequencies, drive_amplitudes, modal_forces):
        super().__init__(drive_frequencies, drive_amplitudes, modal_forces)

    def f_e(self, t: Float, y: Array, args: PyTree) -> float:
        return self.f_d(t, y, args) + self.f_p(t, y, args)
        
    def f_d(self, t: Float, y: Array, args: PyTree):
        """Direct external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
        """
        return args['f_amp'] * jnp.cos(args['f_omega'] * t) 
    
    def f_p(self, t: Float, y: Array, args: PyTree):
        """Parametric external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
        """
        return 0.0
    
    