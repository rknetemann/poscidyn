from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from jax import tree_util

from .. import constants as const
from .abstract_lienard_oscillator import AbstractLienardOscillator
from .abstract_oscillator import AbstractOscillator, oscillator

@oscillator
class NonlinearOscillator(AbstractOscillator):
    Q: jax.Array
    omega_0: jax.Array
    alpha: jax.Array
    gamma: jax.Array

    def __post_init__(self):
        jnp.asarray(self.Q)
        jnp.asarray(self.omega_0)
        jnp.asarray(self.alpha)
        jnp.asarray(self.gamma)

    def damping_term(self, t, state, args):
        return (self.omega_0) * 1/self.Q
    
    def stiffness_term(self, t, state, args):
        q, dq_dt = jnp.split(state, 2)
        linear_stiffness_term = self.omega_0**2 * q
        quadratic_stiffness_term = jnp.einsum("ijk,j,k->i", self.alpha, q, q)
        cubic_stiffness_term = jnp.einsum("ijkl,j,k,l->i", self.gamma, q, q, q)

        return (linear_stiffness_term
                + quadratic_stiffness_term
                + cubic_stiffness_term)
    
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
        #driving_frequency = 1.0
        t_steady_state = jnp.max(-2 * jnp.max(self.Q) * jnp.log(ss_tol * jnp.sqrt(1 - 1 / (4 * jnp.max(self.Q)**2)) / (driving_frequency))).reshape(())

        return t_steady_state
    
    def to_dtype(self, dtype: jnp.dtype) -> NonlinearOscillator:
        return NonlinearOscillator(
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

        return (f"NonlinearOscillator(n_modes={self.n_modes}, "
                f"{Q_terms}, {omega_0_terms}, "
                f"{', '.join(alpha_terms)}, {', '.join(gamma_terms)})")
