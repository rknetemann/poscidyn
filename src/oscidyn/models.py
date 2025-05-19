# ───────────────────────── models.py ──────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from .utils.random import random_uniform
from .constants import Damping
    
@dataclass(eq=False)
class NonDimensionalisedModel: 
    # --------------------------------------------------- non-dimensionalised parameters
    omega_ref:      float
    x_ref:          float
    
    N:              int
    Q:              jax.Array            # (N,)
    kappa:          jax.Array            # (N,)
    alpha:          jax.Array            # (N,)
    gamma:          jax.Array            # (N,N,N)
    F_amp_hat:      jax.Array            # (N,)
    F_omega_hat:    jax.Array            # (1,)
    
    omega_0_hat: jax.Array = field(init=False, repr=False)
    
    rhs_jit: callable = field(init=False, repr=False)
    
    def __post_init__(self):
        self._calc_eigenfrequencies()
        self._build_rhs()
        
    def _calc_eigenfrequencies(self):
        self.omega_0_hat = self.kappa
    
    def _build_rhs(self):
        def _rhs(tau, state, args):
            q, v   = jnp.split(state, 2)
            F_omega_hat_arg, F_amp_hat_arg = args
            
            damping_term = (self.omega_0_hat / self.Q) * v
            
            linear_stiffness_term = (self.kappa**2 * q)

            quadratic_stiffness_term = jnp.einsum("ijk,j,k->i",    self.alpha, q, q)
            
            cubic_stiffness_term = jnp.einsum("ijkl,j,k,l->i", self.gamma, q, q, q)
            
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
        raise NotImplementedError("Dimensionalisation is not implemented yet.")
    
@dataclass(eq=False)
class PhysicalModel:    
    # --------------------------------------------------- physical parameters
    N:         int
    m:         jax.Array            # (N,)
    c:         jax.Array            # (N,)
    k:         jax.Array            # (N,)
    a:         jax.Array            # (N,N,N)
    g:         jax.Array            # (N,N,N,N)
    F_amp:     jax.Array            # (N,)
    F_omega:   jax.Array            # (1,)
    
    omega_0: jax.Array = field(init=False, repr=False)
        
    rhs_jit: callable = field(init=False, repr=False)
    
    # --------------------------------------------------- constructors
    @classmethod
    def from_random(cls, N: int, seed: int = 0) -> "PhysicalModel":
        key = jax.random.PRNGKey(seed)
        m,       key = random_uniform(key, (N,),           0.1, 10.0)
        c,       key = random_uniform(key, (N,),           80.0, 100.0)
        k,       key = random_uniform(key, (N, N),         0.4, 1.0)
        alpha,   key = random_uniform(key, (N, N, N),     -1.0, 1.0)
        gamma,   key = random_uniform(key, (N, N, N, N),  -1.0, 1.0)
        F_amp,   key = random_uniform(key, (N,),          0.5, 1.0)
        F_omega, key = random_uniform(key, (1,),          2.0, 10.0)
        
        return cls(N, m, c, k, alpha, gamma, F_amp, F_omega)

    @classmethod
    def from_example(cls, N) -> "PhysicalModel":
        if N == 1:
            m     = jnp.array([1.0])
            c     = jnp.array([2.0 * 0.01 * 5.0])
            k     = jnp.array([[10.0]])
            alpha = jnp.zeros((N, N, N))
            gamma = jnp.zeros((N, N, N, N)).at[0].set(jnp.array([[0.1]]))
            F_amp = jnp.array([15.0]) 
            F_omega = jnp.array([1.0])
        elif N == 2:
            m     = jnp.array([1.0, 2.0])
            c     = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
            k     = jnp.array([[10.0, 0],
                               [0, 12.0]])
            alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.0],
                                                            [0.0, 0.0]]))
            gamma = jnp.zeros((N, N, N, N)).at[0].set(jnp.array([[0.1, 0.0],
                                                            [0.0, 0.0]]))
            F_amp = jnp.array([15.0, 4.0]) 
            F_omega = jnp.array([1.0])
        elif N == 3:
            m     = jnp.array([0.204, 0.195, 0.183])
            c     = jnp.array([1.0, 1.0, 1.0])
            k     = jnp.array([[5.78, 0.0, 0],
                               [0.0,16.5, 0],
                               [0,   0,   34.6]])
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
            k = jnp.array([
            [250.0, 200.0,   100.0,   0.0],
            [200.0, 200.0, -60.0,   0.0],
            [  100.0, -60.0, 180.0, -40.0],
            [  0.0,   0.0, -40.0, 150.0]
            ])
            
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
            F_amp = jnp.array([20.0, 15.0, 10.0, 5.0])
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
                self.c = 2.0 * 0.1 * jnp.sqrt(jnp.matmul(self.m, self.k))
            elif self.c == Damping.MODERATELY_DAMPED:
                self.c = 2.0 * 0.2 * self.m
            else:
                raise ValueError(f"Unknown damping type: {self.c}")
            
        self._calc_eigenfrequencies()
        self._build_rhs()
        
    def _calc_eigenfrequencies(self):
        M_inv_K = self.k / self.m[:, None]
        eigvals, _ = jnp.linalg.eig(M_inv_K)
        idx       = jnp.argsort(eigvals)
        self.omega_0 = jnp.sqrt(jnp.abs(eigvals[idx]))

    def _build_rhs(self):
        def _rhs(tau, state, args):
            q, v   = jnp.split(state, 2)
            F_omega_hat_arg, F_amp_hat_arg = args
            
            damping_term = -(self.non_dimensionalised_model.omega_0_hat / self.non_dimensionalised_model.Q) * v
            
            linear_stiffness_term = -(self.non_dimensionalised_model.kappa**2 * q)

            quadratic_stiffness_term = -jnp.einsum("ijk,j,k->i",    self.non_dimensionalised_model.alpha, q, q)
            
            cubic_stiffness_term = -jnp.einsum("ijkl,j,k,l->i", self.non_dimensionalised_model.gamma, q, q, q)
            
            forcing_term = F_amp_hat_arg * jnp.cos(F_omega_hat_arg * tau)
            
            a = (
                damping_term
                + linear_stiffness_term
                + quadratic_stiffness_term
                + cubic_stiffness_term
                + forcing_term
            )
            return jnp.concatenate([v, a])

        self.rhs_jit = jax.jit(_rhs)
        
    def get_steady_state_t_end(self) -> float:
        raise NotImplementedError("get_steady_state_t_end is not implemented yet.")
        
    def non_dimensionalise(self) -> NonDimensionalisedModel:
        omega_0_1 = self.omega_0[0]
        Q_1 = self.m[0] * omega_0_1 / self.c[0]
        
        omega_ref = omega_0_1
        x_ref = self.F_amp[0] * Q_1 / omega_0_1**2
        
        Q = self.m * self.omega_0 / self.c
        kappa = self.omega_0 / omega_ref
        alpha = self.a * x_ref / (self.m * omega_ref**2)
        gamma = self.g * x_ref**2 / (self.m * omega_ref**2)
        F_amp_hat = self.F_amp / (self.m * omega_ref**2 * x_ref)
        F_omega_hat = self.F_omega / omega_ref    
        
        non_dimensionalised_model = NonDimensionalisedModel(
            omega_ref=omega_ref,  
            x_ref=x_ref,
            N=self.N,
            Q=Q,
            kappa=kappa,
            alpha=alpha,
            gamma=gamma,
            F_amp_hat=F_amp_hat,
            F_omega_hat=F_omega_hat
        )
        return non_dimensionalised_model
    