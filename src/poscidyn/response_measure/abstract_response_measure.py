import jax.numpy as jnp
from abc import ABC, abstractmethod

class AbstractResponseMeasure(ABC):
    @abstractmethod
    def __call__(self, xs: jnp.ndarray, ts: jnp.ndarray, drive_omega: jnp.ndarray):
        pass
