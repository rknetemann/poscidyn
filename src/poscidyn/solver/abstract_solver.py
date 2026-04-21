import jax
from abc import ABC, abstractmethod

class AbstractSolver(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def time_response(self, init_disp: jax.Array, init_vel: jax.Array) -> tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def frequency_sweep(self) -> tuple[jax.Array, jax.Array]:
        pass
