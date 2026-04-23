import jax.numpy as jnp
from abc import ABC, abstractmethod

class AbstractResponseMeasure(ABC):
    def __init__(self, modal_contributions: jnp.ndarray | None = None):
        self.multiples = jnp.asarray((1.0,))
        self.modal_contributions = modal_contributions

    def _resolve_modal_contributions(
        self,
        n_modes: int,
    ) -> jnp.ndarray:
        shape = self.modal_contributions
        if shape is None:
            return jnp.ones((n_modes,))
        shape = jnp.asarray(shape)
        if shape.ndim != 1:
            raise ValueError(f"modal_contributions must be 1D, got shape {shape.shape}")
        if shape.shape[0] != n_modes:
            raise ValueError(
                f"modal_contributions length ({shape.shape[0]}) does not match n_modes ({n_modes})."
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
