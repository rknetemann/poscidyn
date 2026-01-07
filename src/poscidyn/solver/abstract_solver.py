import jax
import diffrax
from abc import ABC, abstractmethod

from ..oscillator.abstract_oscillator import AbstractOscillator
from ..excitation.abstract_excitation import AbstractExcitation
from .. import constants as const

class AbstractSolver(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def time_response(self,
            excitation: AbstractExcitation, 
            init_disp: jax.Array,  
            init_vel: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def frequency_sweep(self,
            excitation: AbstractExcitation,
            sweep_direction: const.SweepDirection,
        ) -> tuple[jax.Array, jax.Array]:
        pass
