from jaxtyping import Array, PyTree
from jax.core import Tracer
from abc import ABC, abstractmethod

from ..oscillator.abstract_oscillator import AbstractOscillator
from ..excitation.abstract_excitation import AbstractExcitation
from ..excitation.abstract_periodic_excitation import AbstractPeriodicExcitation
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
    def time_response(self, x0: Array, v0: Array, **kwargs) -> tuple[Array, Array]:
        pass

    @abstractmethod
    def frequency_sweep(self, omegas: Array, **kwargs) -> tuple[Array, Array]:
        """Perform a frequency sweep of the oscillator response.

        If the excitation is not periodic, this method should raise an error. For this it can be checked using _validate_frequency_sweep().
        
        Args:
            omegas (jax.Array): Array of frequencies to sweep over.
            **kwargs: Additional keyword arguments for the solver.
        """
        pass

    def _validate_frequency_sweep(self) -> None:
        if not isinstance(self.excitation, AbstractPeriodicExcitation):
            raise TypeError(
                "Frequency sweep requires a periodic excitation."
            )

    @staticmethod
    def _is_tracer(value) -> bool:
        """Check whether a value is being traced by JAX (e.g. inside vmap/jit)."""
        return isinstance(value, Tracer)
