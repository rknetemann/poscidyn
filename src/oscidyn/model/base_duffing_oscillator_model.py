from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from jax import tree_util

from .. import constants as const
from .abstract_model import AbstractModel, oscimodel

@oscimodel
class BaseDuffingOscillator(AbstractModel):
    # Physical parameters
    Q: jax.Array
    omega_0: jax.Array
    alpha: jax.Array
    gamma: jax.Array

    # Reference parameters for non-dimensionalization
    omega_ref: Optional[jax.Array] = 1.0
    x_ref: Optional[jax.Array] = 1.0
    
    def __post_init__(self):
        if self.omega_ref is None:
            self.omega_ref = jnp.max(self.omega_0)
            self.omega_ref = 1.0
        if self.x_ref is None:
            self.x_ref = jnp.max(self.Q) / (self.omega_ref**2)
            self.x_ref = 1.0

        jnp.asarray(self.Q)
        jnp.asarray(self.omega_0)
        jnp.asarray(self.alpha)
        jnp.asarray(self.gamma)
        jnp.asarray(self.omega_ref)
        jnp.asarray(self.x_ref)

    def f(self, tau, state, args):
        q, dq_dtau   = jnp.split(state, 2)
        f_amp, f_omega = [jnp.asarray(v).squeeze(()) for v in args]

        damping_term = (self.omega_0/self.omega_ref) * 1/self.Q * dq_dtau
        linear_stiffness_term = (1/self.omega_ref**2) * self.omega_0**2 * q
        quadratic_stiffness_term = (self.x_ref / self.omega_ref**2) * jnp.einsum("ijk,j,k->i", self.alpha, q, q)
        cubic_stiffness_term = (self.x_ref**2 / self.omega_ref**2) * jnp.einsum("ijkl,j,k,l->i", self.gamma, q, q, q) # Shape: (n_modes,)
        forcing_term = f_amp / (self.omega_ref**2 * self.x_ref) * jnp.cos(f_omega/self.omega_ref * tau)

        d2q_dtau2 = (
            - damping_term
            - linear_stiffness_term
            - quadratic_stiffness_term
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

    def t_steady_state(self, driving_frequency: jax.Array, ss_tol: float) -> float:
        '''driving_frequency
        Calculates the settling time for a given Q-factor and driving frequency.
        Equation from Eq.5.10b Vibrations 2nd edition by Balakumar Balachandran | Edward B. Magrab
        '''
        t_steady_state = jnp.max(-2 * jnp.max(self.Q) * jnp.log(ss_tol * jnp.sqrt(1 - 1 / (4 * jnp.max(self.Q)**2)) / (driving_frequency))).reshape(())

        return t_steady_state
    
    def to_dtype(self, dtype: jnp.dtype) -> BaseDuffingOscillator:
        return BaseDuffingOscillator(
            Q=self.Q.astype(dtype),
            omega_0=self.omega_0.astype(dtype),
            alpha=self.alpha.astype(dtype),
            gamma=self.gamma.astype(dtype)
        )
    
    def __repr__(self):
        Q_terms = ", ".join([f"Q[{i}]={float(v):.6f}" for i, v in enumerate(self.Q)])
        omega_0_terms = ", ".join([f"omega_0[{i}]={float(v):.6f}" for i, v in enumerate(self.omega_0)])

        alpha_indices, alpha_value = jnp.where(self.alpha != 0.0), self.alpha[jnp.where(self.alpha != 0.0)]
        alpha_terms = [f"alpha[{i[0]},{i[1]},{i[2]}]={float(v):.6f}" for i, v in zip(zip(*alpha_indices), alpha_value)]
        if len(alpha_terms) > 20:
            alpha_terms = alpha_terms[:20] + ["... (truncated)"]

        gamma_indices, gamma_value = jnp.where(self.gamma != 0.0), self.gamma[jnp.where(self.gamma != 0.0)]
        gamma_terms = [f"gamma[{i[0]},{i[1]},{i[2]},{i[3]}]={float(v):.6f}" for i, v in zip(zip(*gamma_indices), gamma_value)]
        if len(gamma_terms) > 20:
            gamma_terms = gamma_terms[:20] + ["... (truncated)"]

        return (f"BaseDuffingOscillator(n_modes={self.n_modes}, "
                f"{Q_terms}, {omega_0_terms}, "
                f"{', '.join(alpha_terms)}, {', '.join(gamma_terms)})")

# Register BaseDuffingOscillator as a pytree
def _tree_flatten(obj):
    leaves = (obj.Q, obj.omega_0, obj.alpha, obj.gamma, obj.omega_ref, obj.x_ref)
    aux_data = None
    return leaves, aux_data

def _tree_unflatten(aux_data, leaves):
    Q, omega_0, alpha, gamma, omega_ref, x_ref = leaves
    return BaseDuffingOscillator(Q=Q, omega_0=omega_0, alpha=alpha, gamma=gamma, omega_ref=omega_ref, x_ref=x_ref)

tree_util.register_pytree_node(BaseDuffingOscillator, _tree_flatten, _tree_unflatten)