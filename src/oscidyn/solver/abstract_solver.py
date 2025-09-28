import jax
import diffrax
from abc import ABC, abstractmethod

from ..model.abstract_model import AbstractModel
from .. import constants as const

class AbstractSolver(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def time_response(self,
            drive_freq: jax.Array,  
            drive_amp: jax.Array, 
            init_disp: jax.Array,  
            init_vel: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def frequency_sweep(self,
            drive_freq: jax.Array, 
            drive_amp: jax.Array, 
            sweep_direction: const.SweepDirection,
        ) -> tuple[jax.Array, jax.Array]:
        pass
