# examples/models.py
from __future__ import annotations
import jax
import jax.numpy as jnp

from ..models import AbstractModel, oscimodel
from .. import constants as const
from dataclasses import field
    
@oscimodel
class BaseDuffingOscillator(AbstractModel):

    g1: jax.Array
    g2: jax.Array
    g3: jax.Array

    def rhs(self, t, state, args):
        q, dq_dt   = jnp.split(state, 2)
        g4, g5, g6 = [jnp.asarray(v).reshape(()) for v in args]

        damping_term = self.g1 * dq_dt
        linear_stiffness_term = self.g2 * q
        cubic_stiffness_term = self.g3 * q**3
        forcing_term = jnp.zeros((self.n_modes,)).at[:1].set(g4 * jnp.cos(g5 * t) + g4 * jnp.cos(g6 * t))

        d2q_dt2 = (
            - damping_term
            - linear_stiffness_term
            - cubic_stiffness_term
            + forcing_term
        ) 

        return jnp.concatenate([dq_dt, d2q_dt2])
    
    @property
    def n_modes(self) -> int:
        return self.g1.shape[0]

    def t_steady_state(self, driving_frequency: float, ss_tol: float) -> float:
        '''driving_frequency
        Calculates the settling time for a given Q-factor and driving frequency.
        Equation from Eq.5.10b Vibrations 2nd edition by Balakumar Balachandran | Edward B. Magrab
        '''
        Q = jnp.max(jnp.sqrt(self.g2) / self.g1)
        driving_frequency = jnp.asarray(driving_frequency).reshape(())
        tau_d = -2 * Q * jnp.log(ss_tol * jnp.sqrt(1 - 1 / (4 * Q**2)) / jnp.max(driving_frequency)) * 1.4

        three_periods = 3 * (2 * jnp.pi / jnp.max(driving_frequency))
        t_steady_state = (tau_d + three_periods)  * const.SAFETY_FACTOR_T_STEADY_STATE
        
        return t_steady_state
    
    @classmethod
    def from_physical_params(cls, Q: jax.Array, omega_0: jax.Array, gamma: jax.Array) -> BaseDuffingOscillator:
        g1 = omega_0/Q
        g2 = omega_0**2
        g3 = gamma
        
        return cls(g1=g1, g2=g2, g3=g3)