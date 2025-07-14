# examples/models.py
from __future__ import annotations
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from ..models import Model, oscimodel

@oscimodel
class NonlinearOscillator(Model): 
    omega_ref:      float                                   # reference frequency
    x_ref:          float                                   # reference displacement
                    
    N:              int                                     # number of modes   
    Q:              jax.Array                               # quality-factor (N,)
    eta_hat:        jax.Array                               # non-dimensionalized damping (N,)
    alpha_hat:      jax.Array                               # non-dimensionalized quadratic stiffness (N,N,N)
    gamma_hat:      jax.Array                               # non-dimensionalized cubic stiffness (N,N,N)
    delta_hat:      jax.Array                               # non-dimensionalized fith order stiffness (N,N,N,N)
    F_amp_hat:      jax.Array                               # non-dimensionalized forcing amplitude (N,)
    F_omega_hat:    jax.Array                               # non-dimensionalized forcing frequency (1,)
                    
    omega_0_hat: jax.Array                                  # non-dimensionalised natural frequencies (N,) 
    
    def rhs(self, tau, state, args):
        q, v   = jnp.split(state, 2)
        F_omega_hat_arg, F_amp_hat_arg = args
        
        damping_term = (self.omega_0_hat / self.Q) * v
        
        nonlinear_damping_term = self.eta_hat * q**2 * v
        
        linear_stiffness_term = self.omega_0_hat**2 * q

        quadratic_stiffness_term = jnp.einsum("ijk,j,k->i", self.alpha_hat, q, q)
        
        cubic_stiffness_term = jnp.einsum("ijkl,j,k,l->i", self.gamma_hat, q, q, q)
        
        fith_order_stiffness_term = jnp.einsum("ijklm,j,k,l,m->i", self.delta_hat, q, q, q, q)
        
        forcing_term = F_amp_hat_arg * jnp.cos(F_omega_hat_arg * tau)
        
        a = (
            - damping_term
            - nonlinear_damping_term
            - linear_stiffness_term
            - quadratic_stiffness_term
            - cubic_stiffness_term
            - fith_order_stiffness_term
            + forcing_term
        )
        return jnp.concatenate([v, a])
    
    @classmethod
    def from_example(cls, N: int) -> NonlinearOscillator:
        if N == 1:
            omega_ref = 1.0
            x_ref = 1.0
            omega_0_hat = jnp.array([1.0])
            Q = jnp.array([1000.0])
            eta_hat = jnp.array([0.0])
            alpha_hat = jnp.zeros((N, N, N)).at[0,0,0].set(0.0)
            gamma_hat = jnp.zeros((N, N, N, N)).at[0, 0, 0, 0].set(+0.1)
            delta_hat = jnp.zeros((N, N, N, N, N)).at[0, 0, 0, 0, 0].set(0.0)
            F_amp_hat = jnp.array([1.0]) 
            F_omega_hat = jnp.array([1.0])
        else:
            raise ValueError("Example not found for N={N}.")
        
        return cls(
            omega_ref=omega_ref,
            x_ref=x_ref,
            N=N,
            Q=Q,
            eta_hat=eta_hat,
            alpha_hat=alpha_hat,
            gamma_hat=gamma_hat,
            delta_hat=delta_hat,
            F_amp_hat=F_amp_hat,
            F_omega_hat=F_omega_hat,
            omega_0_hat=omega_0_hat
        )
        
