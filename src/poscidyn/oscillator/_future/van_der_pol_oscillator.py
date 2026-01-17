from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from jax import tree_util

from .. import constants as const
from .abstract_oscillator import AbstractOscillator, oscillator

@oscillator
class VanDerPolOscillator(AbstractOscillator):
    mu: jax.Array

    def __post_init__(self):
        jnp.asarray(self.mu)

    def f(self, tau, state, args, omega_ref=1.0, x_ref=1.0):
        q, dq_dtau   = jnp.split(state, 2)
        f_amp, f_omega = [jnp.asarray(v).squeeze(()) for v in args]

        #omega_ref = f_omega[0]

        damping_term = self.mu * (1 - q**2) * dq_dtau
        linear_stiffness_term = q
        forcing_term = f_amp / jnp.cos(f_omega * tau)

        d2q_dtau2 = (
            + damping_term
            - linear_stiffness_term
            + forcing_term
        ) 
        return jnp.concatenate([dq_dtau, d2q_dtau2])

    def f_y(self, tau, state, args):
        pass
    
    @property
    def n_modes(self) -> int:
        return self.mu.shape[0]
    
    @property
    def n_states(self) -> int:
        return self.n_modes * 2

    def t_steady_state(self, driving_frequency: jax.Array, ss_tol: float) -> float:

        #driving_frequency = 1.0
        t_steady_state = 100.0

        return t_steady_state
    
    def to_dtype(self, dtype: jnp.dtype) -> VanDerPolOscillator:
        return VanDerPolOscillator(
            mu=self.mu.astype(dtype)
        )
    
