# ───────────────────────── models.py ──────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from .utils.random_generation import random_uniform
from .constants import Damping
    
@dataclass(eq=False)
class NonDimensionalisedModel: 
    # -----------------------non-dimensionalised parameters---------------------------- 
    omega_ref:      float                                   # reference frequency
    x_ref:          float                                   # reference displacement
                    
    N:              int                                     # number of modes   
    Q:              jax.Array                               # quality-factor (N,)
    alpha_hat:          jax.Array                           # non-dimensionalized quadratic stiffness (N,N,N)
    gamma_hat:          jax.Array                           # non-dimensionalized cubic stiffness (N,N,N)
    F_amp_hat:      jax.Array                               # non-dimensionalized forcing amplitude (N,)
    F_omega_hat:    jax.Array                               # non-dimensionalized forcing frequency (1,)
                    
    omega_0_hat: jax.Array                                  # non-dimensionalised natural frequencies (N,) 
    
    rhs_jit: callable = field(init=False, repr=False)
    
    def __post_init__(self):
        self._build_rhs()
    
    def _build_rhs(self):
        def _rhs(tau, state, args):
            q, v   = jnp.split(state, 2)
            F_omega_hat_arg, F_amp_hat_arg = args
            
            damping_term = (self.omega_0_hat / self.Q) * v
            
            linear_stiffness_term = self.omega_0_hat**2 * q

            quadratic_stiffness_term = jnp.einsum("ijk,j,k->i", self.alpha_hat, q, q)
            
            cubic_stiffness_term = jnp.einsum("ijkl,j,k,l->i", self.gamma_hat, q, q, q)
            
            forcing_term = F_amp_hat_arg * jnp.cos(F_omega_hat_arg * tau)
            
            a = (
                - damping_term
                - linear_stiffness_term
                - quadratic_stiffness_term
                - cubic_stiffness_term
                + forcing_term
            )
            return jnp.concatenate([v, a])

        self.rhs_jit = jax.jit(_rhs)
        
    def get_steady_state_t_end(self) -> float:
        raise NotImplementedError("get_steady_state_t_end is not implemented yet.")
        
    def dimensionalise(self) -> PhysicalModel:
        # Use unit masses as default
        m = jnp.ones(self.N)
        
        # Calculate physical parameters from non-dimensional ones
        omega_0 = self.omega_0_hat * self.omega_ref
        c = m * omega_0 / self.Q
        k = m * omega_0**2
        
        # Expand m for broadcasting with the tensor parameters
        m_expanded_alpha = jnp.reshape(m, (self.N, 1, 1))  # Shape: (N, 1, 1)
        m_expanded_gamma = jnp.reshape(m, (self.N, 1, 1, 1))  # Shape: (N, 1, 1, 1)
        
        # Calculate tensor parameters
        alpha = self.alpha_hat * (m_expanded_alpha * self.omega_ref**2) / self.x_ref
        gamma = self.gamma_hat * (m_expanded_gamma * self.omega_ref**2) / self.x_ref**2
        
        # Calculate forcing parameters
        F_amp = self.F_amp_hat * (m * self.omega_ref**2 * self.x_ref)
        F_omega = self.F_omega_hat * self.omega_ref
        
        # Create and return the physical model
        return PhysicalModel(
            N=self.N,
            m=m,
            c=c,
            k=k,
            alpha=alpha,
            gamma=gamma,
            F_amp=F_amp,
            F_omega=F_omega
        )
    
@dataclass(eq=False)
class PhysicalModel:    
    N:         int                                          # number of modes N
    m:         jax.Array                                    # mass (N,)
    c:         jax.Array                                    # damping (N,)
    k:         jax.Array                                    # linear stiffness (N,)
    alpha:     jax.Array                                    # quadratic stiffness (N,N,N)
    gamma:     jax.Array                                    # cubic stiffness (N,N,N,N)
    F_amp:     jax.Array                                    # forcing amplitude (N,)
    F_omega:   jax.Array                                    # forcing frequency (1,)

    omega_0:   jax.Array = field(init=False, repr=False)    # natural frequencies (N,)

    rhs_jit: callable = field(init=False, repr=False)       # right-hand side function
    
    @classmethod
    def from_random(cls, N: int, seed: int = 0) -> "PhysicalModel":
        raise NotImplementedError("Random generation is not implemented yet.")

    @classmethod
    def from_example(cls, N) -> "PhysicalModel":
        if N == 1:
            m     = jnp.array([1.0])
            c     = jnp.array([2.0 * 0.01 * 5.0])
            k     = jnp.array([10.0])
            alpha = jnp.zeros((N, N, N))
            gamma = jnp.zeros((N, N, N, N)).at[0].set(jnp.array([[0.1]]))
            F_amp = jnp.array([15.0]) 
            F_omega = jnp.array([1.0])
        elif N == 2:
            m     = jnp.array([1.0, 2.0])
            c     = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
            k     = jnp.array([10.0, 12.0])
            alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.0],
                                                            [0.0, 0.0]]))
            gamma = jnp.zeros((N, N, N, N)).at[0].set(jnp.array([[0.1, 0.0],
                                                            [0.0, 0.0]]))
            F_amp = jnp.array([15.0, 4.0]) 
            F_omega = jnp.array([1.0])
        elif N == 3:
            m     = jnp.array([0.204, 0.195, 0.183])
            c     = jnp.array([1.0, 1.0, 1.0])
            k     = jnp.array([5.78, 16.5, 34.6])
            alpha = jnp.zeros((N, N, N))
            gamma = jnp.zeros((N, N, N, N))
            F_amp = jnp.array([15.0, 10.0, 5.0]) 
            F_omega = jnp.array([1.0])
        elif N == 4:
            # Mass values for a 4-DOF system (kg)
            m = jnp.array([1.2, 1.5, 1.8, 2.0])
            
            # Damping coefficients with realistic modal damping ratios (~1-3%)
            c = jnp.array([0.8, 1.2, 1.5, 1.8])
            
            # Stiffness matrix representing a chain-like structure with coupling (N/m)
            k = jnp.array([250.0, 200.0, 180.0, 150.0])
            # Quadratic nonlinearity - light coupling between modes
            alpha = jnp.zeros((N, N, N))
            alpha = alpha.at[0, 0, 0].set(2.5)
            alpha = alpha.at[1, 1, 1].set(5.8)
            
            # Cubic nonlinearity - typical in mechanical systems with geometric nonlinearity
            gamma = jnp.zeros((N, N, N, N))
            gamma = gamma.at[0, 0, 0, 0].set(5.0)
            gamma = gamma.at[1, 1, 1, 1].set(4.0)
            gamma = gamma.at[2, 2, 2, 2].set(3.0)
            
            # External forcing - decreasing amplitude for higher modes
            F_amp = jnp.array([180.0, 15.0, 100.0, 5.0])
            F_omega = jnp.array([8.0])  # Forcing frequency
        else:
            raise ValueError("Example not found for N={N}.")
        
        return cls(N, m, c, k, alpha, gamma, F_amp, F_omega)
    
    @classmethod
    def from_file(cls, path:str) -> "PhysicalModel":
        raise NotImplementedError("Loading from file is not implemented yet.")
    
    # --------------------------------------------------- helpers
    def __post_init__(self):
        if isinstance(self.c, Damping):
            if self.c == Damping.NONE:
                self.c = jnp.zeros(self.N)
            elif self.c == Damping.LIGHTLY_DAMPED:
                self.c = 2.0 * 0.1 * jnp.sqrt(jnp.matmul(self.m, self.k_1))
            elif self.c == Damping.MODERATELY_DAMPED:
                self.c = 2.0 * 0.2 * jnp.sqrt(jnp.matmul(self.m, self.k_1))
            else:
                raise ValueError(f"Unknown damping type: {self.c}")
            
        self._calc_eigenfrequencies()
        #self._build_rhs()
        
    def _calc_eigenfrequencies(self):
        M_inv_K_diag = self.k / self.m
        M_inv_K = jnp.diag(M_inv_K_diag)  # Convert to a diagonal matrix
        eigvals, _ = jnp.linalg.eig(M_inv_K)
        idx = jnp.argsort(eigvals)
        self.omega_0 = jnp.sqrt(jnp.abs(eigvals[idx]))

    def _build_rhs(self):
        raise NotImplementedError("RHS is not implemented yet.")
        
    def non_dimensionalise(self) -> NonDimensionalisedModel:
        omega_0_1 = self.omega_0[0]
        omega_ref = omega_0_1
        
        Q_1 = self.m[0] * omega_0_1 / self.c[0]
        x_ref = self.F_amp[0] * Q_1 / (self.m[0] * omega_ref**2)
        
        Q = self.m * self.omega_0 / self.c
        alpha_hat = self.alpha * x_ref / (self.m * omega_ref**2)
        gamma_hat = self.gamma * x_ref**2 / (self.m * omega_ref**2)
        F_amp_hat = self.F_amp / (self.m * omega_ref**2 * x_ref)
        F_omega_hat = self.F_omega / omega_ref    
        
        omega_0_hat = self.omega_0 / omega_ref
        
        non_dimensionalised_model = NonDimensionalisedModel(
            omega_ref=omega_ref,  
            x_ref=x_ref,
            N=self.N,
            Q=Q,
            alpha_hat=alpha_hat,
            gamma_hat=gamma_hat,
            F_amp_hat=F_amp_hat,
            F_omega_hat=F_omega_hat,
            omega_0_hat=omega_0_hat
        )
        return non_dimensionalised_model
