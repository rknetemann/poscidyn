# examples/models.py
from __future__ import annotations
import jax
import jax.numpy as jnp

from ..models import AbstractModel, oscimodel

@oscimodel
class DuffingOscillator(AbstractModel):
                  
    Q:              jax.Array # Shape: (n_modes,)
    gamma:          jax.Array # Shape: (n_modes,)                    
    
    def rhs(self, t, state, args):
        q, v   = jnp.split(state, 2)
        omega, F = [jnp.asarray(v).reshape(()) for v in args]

        damping_term = (1 / self.Q) * v # Shape: (n_modes,)
                
        linear_stiffness_term = q # Shape: (n_modes,)

        cubic_stiffness_term = self.gamma * q**3
        
        forcing_term = jnp.zeros((self.n_modes,)).at[:1].set(
            F * jnp.cos(omega * t)
        )

        a = (
            - damping_term
            - linear_stiffness_term
            - cubic_stiffness_term
            + forcing_term
        ) 

        return jnp.concatenate([v, a])
    
    @property
    def n_modes(self) -> int:
        return self.Q.shape[0]
    
    @classmethod
    def from_example(cls, n_modes: int) -> DuffingOscillator:
        if n_modes == 1:
            omega_ref = 1.0
            x_ref = 1.0
            Q = jnp.array([10000.0])
            gamma = jnp.array([0.010])
        
        return cls(
            omega_ref=omega_ref,
            x_ref=x_ref,
            Q=Q,
            gamma=gamma,
        )