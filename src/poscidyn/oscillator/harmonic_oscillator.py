from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from jax import tree_util

from .. import constants as const
from .abstract_oscillator import AbstractOscillator, oscillator

@oscillator
class HarmonicOscillator(AbstractOscillator):
    zeta: Optional[jax.Array] = None
    Q: Optional[jax.Array] = None

    omega_0: jax.Array

    def __post_init__(self):
        if self.Q is None:
            if self.zeta is None:
                raise ValueError("Either Q or zeta must be provided.")
            self.Q = 1 / (2 * self.zeta)
        elif self.zeta is None:
            self.zeta = 1 / (2 * self.Q)
        else:
            raise ValueError("Only one of Q or zeta should be provided.")

        self.omega_0 = jnp.asarray(self.omega_0)

    def damping_term(self, t, state, args):
        q, dq_dt = jnp.split(state, 2)
        return self.omega_0 / self.Q * dq_dt
    
    def stiffness_term(self, t, state, args):
        q, dq_dt = jnp.split(state, 2)
        linear_stiffness_term = self.omega_0**2 * q

        return (linear_stiffness_term)
    
    def t_steady_state(self, driving_frequency: jax.Array, ss_tol: float) -> float:
        '''driving_frequency
        Calculates the settling time for a given Q-factor and driving frequency.
        Equation from Eq.5.10b Vibrations 2nd edition by Balakumar Balachandran | Edward B. Magrab
        '''
        #driving_frequency = 1.0
        t_steady_state = jnp.max(-2 * jnp.max(self.Q) * jnp.log(ss_tol * jnp.sqrt(1 - 1 / (4 * jnp.max(self.Q)**2)) / (driving_frequency))).reshape(())

        return t_steady_state
    
    @property
    def n_dof(self) -> int:
        return self.Q.shape[0]
    
    def to_dtype(self, dtype: jnp.dtype) -> HarmonicOscillator:
        return HarmonicOscillator(
            Q=self.Q.astype(dtype),
            omega_0=self.omega_0.astype(dtype),
        )
    
    def __repr__(self):
        Q_terms = ", ".join([f"Q[{i}]={float(v):.6f}" for i, v in enumerate(self.Q)])
        omega_0_terms = ", ".join([f"omega_0[{i}]={float(v):.6f}" for i, v in enumerate(self.omega_0)])

        return (f"HarmonicOscillator(n_dof={self.n_dof}, "
                f"{Q_terms}, {omega_0_terms})")