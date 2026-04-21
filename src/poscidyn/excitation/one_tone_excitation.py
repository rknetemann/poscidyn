import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array

from .abstract_excitation import AstractExcitation

class OneToneExcitation(AstractExcitation):
    def __init__(self, drive_frequencies, drive_amplitudes, modal_forces):
        super().__init__(drive_frequencies, drive_amplitudes, modal_forces)
        
    def f_d(self, t: Float, state: Array, args: PyTree):
        return args['f_amp'] * jnp.cos(args['f_omega'] * t) 
    
    