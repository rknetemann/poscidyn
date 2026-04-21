from __future__ import annotations
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from ..abstract_oscillator import AbstractOscillator

class VanDerPolOscillator(AbstractOscillator):
    mu: jax.Array

    def __post_init__(self):
        jnp.asarray(self.mu)

    def f_i(self, t: Float, y: Array, args: PyTree, omega_ref: float = 1.0, x_ref: float = 1.0) -> Array:
        q, dq_dtau   = jnp.split(y, 2)

        damping_term = self.mu * (1 - q**2) * dq_dtau
        linear_stiffness_term = q

        d2q_dtau2 = (
            + damping_term
            - linear_stiffness_term
        ) 
        return d2q_dtau2

    def f_y(self, t: Float, y: Array, args: PyTree) -> Array:
        pass
    
    @property
    def n_modes(self) -> int:
        return self.mu.shape[0]

    def t_steady_state(self, driving_frequency: jax.Array, ss_tol: float) -> float:

        #driving_frequency = 1.0
        t_steady_state = 100.0

        return t_steady_state
