import jax.numpy as jnp
from abc import ABC, abstractmethod

class AbstractResponseMeasure(ABC):
    def __init__(self, mode_shape: jnp.ndarray | None = None):
        # Default to the fundamental component so callers can treat
        # all response measures uniformly (Demodulation overrides this).
        self.multiples = jnp.asarray((1.0,))
        self.mode_shape = None if mode_shape is None else jnp.asarray(mode_shape)

    def _resolve_mode_shape(
        self,
        n_modes: int,
    ) -> jnp.ndarray:
        shape = self.mode_shape
        if shape is None:
            # Backward-compatible default: equal weighting of all modal coordinates.
            return jnp.ones((n_modes,))
        shape = jnp.asarray(shape)
        if shape.ndim != 1:
            raise ValueError(f"mode_shape must be 1D, got shape {shape.shape}")
        if shape.shape[0] != n_modes:
            raise ValueError(
                f"mode_shape length ({shape.shape[0]}) does not match n_modes ({n_modes})."
            )
        return shape

    @abstractmethod
    def __call__(
        self,
        xs: jnp.ndarray,
        ts: jnp.ndarray,
        drive_omega: jnp.ndarray,
    ):
        pass
