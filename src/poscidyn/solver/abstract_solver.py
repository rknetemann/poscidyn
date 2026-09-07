import jax
from abc import ABC, abstractmethod

from ..oscillator.abstract_oscillator import AbstractOscillator
from ..excitation.abstract_excitation import AbstractExcitation
from ..excitation.free_vibration import FreeVibration
from ..response_measure.abstract_response_measure import AbstractResponseMeasure

class AbstractSolver(ABC):
    """Abstract interface for oscillator response solvers.

    Subclasses implement time-domain response calculations and
    frequency sweeps.

    For initialization we won't (yet) require the excitation and response_measure to be provided, 
    as they may not be needed for all solvers. However, the oscillator is required for all solvers.
    For example a time-domain solver may not need an excitation or response_measure, but a frequency sweep solver will need both.

    Later we will default to certain classes of excitation and response_measure if they are not provided. 

    """
    def __init__(self, oscillator: AbstractOscillator, 
                 excitation: AbstractExcitation = FreeVibration(), 
                 response_measure: AbstractResponseMeasure = None):
        
        self.oscillator = oscillator
        self.excitation = excitation
        self.response_measure = response_measure
    
    @abstractmethod
    def time_response(self, x0: jax.Array, v0: jax.Array) -> tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def frequency_sweep(self) -> tuple[jax.Array, jax.Array]:
        pass
