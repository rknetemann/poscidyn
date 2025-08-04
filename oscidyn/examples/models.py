# examples/models.py
from __future__ import annotations
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from ..models import AbstractModel, oscimodel

@oscimodel
class NonlinearOscillator(AbstractModel): 
    omega_ref:      float
    x_ref:          float 
                    
    Q:              jax.Array                               # Shape: (n_modes,)
    eta_hat:        jax.Array                               # Shape: (n_modes,)
    alpha_hat:      jax.Array                               # Shape: (n_modes,n_modes,n_modes)
    gamma_hat:      jax.Array                               # Shape: (n_modes,n_modes,n_modes)
    delta_hat:      jax.Array                               # Shape: (n_modes,n_modes,n_modes,n_modes)
                    
    omega_0_hat: jax.Array                                  # Shape: (n_modes,) 
    
    def rhs(self, tau, state, args):
        q, v   = jnp.split(state, 2)
        F_omega_hat_arg, F_amp_hat_arg = args
        F_omega_hat_arg = jnp.asarray(F_omega_hat_arg).reshape(())
        F_amp_hat_arg   = jnp.asarray(F_amp_hat_arg).reshape(())
        
        damping_term = (self.omega_0_hat / self.Q) * v # Shape: (n_modes,)
        
        nonlinear_damping_term = self.eta_hat * q**2 * v # Shape: (n_modes,)
        
        linear_stiffness_term = self.omega_0_hat**2 * q # Shape: (n_modes,)

        quadratic_stiffness_term = jnp.einsum("ijk,j,k->i", self.alpha_hat, q, q) # Shape: (n_modes,)
        
        cubic_stiffness_term = jnp.einsum("ijkl,j,k,l->i", self.gamma_hat, q, q, q) # Shape: (n_modes,)
        
        fith_order_stiffness_term = jnp.einsum("ijklm,j,k,l,m->i", self.delta_hat, q, q, q, q) # Shape: (n_modes,)
        
        forcing_term = jnp.zeros((self.n_modes,))

        forcing_term = forcing_term.at[:1].set(
            F_amp_hat_arg * jnp.cos(F_omega_hat_arg * tau)
        )

        a = (
            - damping_term
            - nonlinear_damping_term
            - linear_stiffness_term
            - quadratic_stiffness_term
            - cubic_stiffness_term
            - fith_order_stiffness_term
            + forcing_term
        ) # Shape: (n_modes,)
        return jnp.concatenate([v, a]) # Shape: (2 * n_modes,)
    
    @classmethod
    def from_example(cls, n_modes: int) -> NonlinearOscillator:
        if n_modes == 1:
            omega_ref = 1.0
            x_ref = 1.0
            omega_0_hat = jnp.array([1.0])
            Q = jnp.array([10.0])
            eta_hat = jnp.array([0.005])
            alpha_hat = jnp.zeros((n_modes, n_modes, n_modes)).at[0,0,0].set(0.00)
            gamma_hat = jnp.zeros((n_modes, n_modes, n_modes, n_modes)).at[0, 0, 0, 0].set(0.03)
            delta_hat = jnp.zeros((n_modes, n_modes, n_modes, n_modes, n_modes)).at[0, 0, 0, 0, 0].set(-0.00)
        elif n_modes == 2:
            omega_ref = 1.0
            x_ref = 1.0
            omega_0_hat = jnp.array([1.0, 1.5])
            Q = jnp.array([80.0, 120.0])
            eta_hat = jnp.array([0.0, 0.0])

            # Quadratic stiffness (mode self- and cross-coupling)
            alpha_hat = jnp.zeros((n_modes, n_modes, n_modes))
            alpha_hat = (
                alpha_hat
                .at[0, 0, 0].set(-0.2)
                .at[1, 1, 1].set(-0.3)
                .at[0, 1, 1].set(-0.1)
                .at[1, 0, 0].set(-0.1)
            )

            # Cubic stiffness (self- and cross-coupling)
            gamma_hat = jnp.zeros((n_modes, n_modes, n_modes, n_modes))
            gamma_hat = (
                gamma_hat
                .at[0, 0, 0, 0].set(0.5)
                .at[1, 1, 1, 1].set(0.4)
                .at[0, 0, 0, 1].set(0.05)
                .at[0, 1, 1, 1].set(0.05)
            )

            # Fifth-order stiffness (only primary self-terms here)
            delta_hat = jnp.zeros((n_modes, n_modes, n_modes, n_modes, n_modes))
            delta_hat = (
                delta_hat
                .at[0, 0, 0, 0, 0].set(-0.05)
                .at[1, 1, 1, 1, 1].set(-0.04)
            )
        else:
            raise ValueError("Example not found for n_modes={n_modes}.")
        
        return cls(
            n_modes=n_modes,
            Q=Q,
            eta_hat=eta_hat,
            alpha_hat=alpha_hat,
            gamma_hat=gamma_hat,
            delta_hat=delta_hat,
            omega_0_hat=omega_0_hat,
            omega_ref=omega_ref,
            x_ref=x_ref,
        )
        
class DuffingOscillator(AbstractModel):
    omega_ref: float
    x_ref: float
    
    Q: jax.Array  # Shape: (n_modes,)
    gamma_hat: jax.Array  # Shape: (n_modes,)
    
    def rhs(self, tau, state, args):
        q, v = jnp.split(state, 2)
        F_omega_hat_arg, F_amp_hat_arg = args
        F_omega_hat_arg = jnp.asarray(F_omega_hat_arg).reshape(())
        F_amp_hat_arg = jnp.asarray(F_amp_hat_arg).reshape(())
        
        damping_term = (self.omega_ref / self.Q) * v  # Shape: (n_modes,)
       
        duffing_term = self.gamma_hat * q**3  # Shape: (n_modes,)

        forcing_term = jnp.zeros((self.n_modes,))
        forcing_term = forcing_term.at[:1].set(
            F_amp_hat_arg * jnp.cos(F_omega_hat_arg * tau)
        )

        a = -damping_term - duffing_term + forcing_term  # Shape: (n_modes,)
        return jnp.concatenate([v, a])  # Shape: (2 * n_modes,)
    
    @classmethod
    def from_example(cls, n_modes: int) -> DuffingOscillator:
        if n_modes == 1:
            omega_ref = 1.0
            x_ref = 1.0
            Q = jnp.array([10.0])
            eta_hat = jnp.array([0.005])
        else:
            raise ValueError("Example not found for n_modes={n_modes}.")
        
        return cls(
            n_modes=n_modes,
            Q=Q,
            eta_hat=eta_hat,
            omega_ref=omega_ref,
            x_ref=x_ref,
        )

