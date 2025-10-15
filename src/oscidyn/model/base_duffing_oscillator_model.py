from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from .. import constants as const
from .abstract_model import AbstractModel, oscimodel

@oscimodel
class BaseDuffingOscillator(AbstractModel):
    # Physical parameters
    Q: jax.Array
    omega_0: jax.Array
    gamma: jax.Array

    # Reference parameters for non-dimensionalization
    omega_ref: Optional[jax.Array] = None
    x_ref: Optional[jax.Array] = None
    
    def __post_init__(self):
        if self.omega_ref is None:
            self.omega_ref = jnp.max(self.omega_0)
            self.omega_ref = 1.0
        if self.x_ref is None:
            self.x_ref = jnp.max(self.Q) / (self.omega_ref**2)
            self.x_ref = 1.0

        jnp.asarray(self.Q)
        jnp.asarray(self.omega_0)
        jnp.asarray(self.gamma)
        jnp.asarray(self.omega_ref)
        jnp.asarray(self.x_ref)

    def f(self, tau, state, args):
        q, dq_dtau   = jnp.split(state, 2)
        f, omega = [jnp.asarray(v).reshape(()) for v in args]

        damping_term = (self.omega_0/self.omega_ref) * 1/self.Q * dq_dtau
        linear_stiffness_term = (1/self.omega_ref**2) * self.omega_0**2 * q
        cubic_stiffness_term = (self.x_ref**2 / self.omega_ref**2) * self.gamma * q**3
        forcing_term = jnp.zeros((self.n_modes,)).at[:1].set(f / (self.omega_ref**2 * self.x_ref) * jnp.cos(omega/self.omega_ref * tau))

        d2q_dtau2 = (
            - damping_term
            - linear_stiffness_term
            - cubic_stiffness_term
            + forcing_term
        ) 
        return jnp.concatenate([dq_dtau, d2q_dtau2])

    def f_y(self, tau, state, args):
        q, dq_dtau = jnp.split(state, 2)

        zero_block = jnp.zeros((self.n_modes, self.n_modes))
        identity_block = jnp.eye(self.n_modes)
        A_bottom_left = -jnp.diag(self.g2 + 3 * self.g3 * q**2)
        A_bottom_right = -jnp.diag(self.g1)

        A = jnp.block([[zero_block, identity_block],
                       [A_bottom_left, A_bottom_right]])
        return A
    
    @property
    def n_modes(self) -> int:
        return self.Q.shape[0]
    
    @property
    def n_states(self) -> int:
        return self.n_modes * 2

    @property
    def _t_steady_state(self, driving_frequency: float, ss_tol: float) -> float:
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
    
    def to_dtype(self, dtype: jnp.dtype) -> BaseDuffingOscillator:
        return BaseDuffingOscillator(
            Q=self.Q.astype(dtype),
            omega_0=self.omega_0.astype(dtype),
            gamma=self.gamma.astype(dtype)
        )
    
    def __repr__(self):
        return (f"BaseDuffingOscillator(n_modes={self.n_modes}, "
                f"Q={self.Q}, omega_0={self.omega_0}, gamma={self.gamma})")