import jax.numpy as jnp
from abc import ABC, abstractmethod

class AbstractResponseMeasure(ABC):
    def __init__(self):
        # Default to the fundamental component so callers can treat
        # all response measures uniformly (Demodulation overrides this).
        self.multiples = jnp.asarray((1.0,))

    @abstractmethod
    def __call__(self, xs: jnp.ndarray, ts: jnp.ndarray, drive_omega: jnp.ndarray):
        pass
