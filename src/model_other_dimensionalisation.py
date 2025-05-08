# ───────────────────────── modal_eom_improved.py ──────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from utils.random import random_uniform

jax.config.update("jax_enable_x64", False)
CALCULATE_DIMMLESS = True
    
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
    
    omega_0_hat:        jax.Array
    
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

    
    
@dataclass(eq=False)
class Model:    
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
    
    name:      str = field(default="physical_model")
        
    non_dimensionalised_model: NonDimensionalisedModel = field(init=False, repr=False)
    
    rhs_jit: callable = field(init=False, repr=False)
    
    # --------------------------------------------------- constructors
    @classmethod
    def from_random(cls, N: int, seed: int = 0) -> "Model":
        key = jax.random.PRNGKey(seed)
        m,       key = random_uniform(key, (N,),           0.1, 10.0)
        c,       key = random_uniform(key, (N,),           20.0, 30.0)
        k,       key = random_uniform(key, (N, N),         0.4, 1.0)
        alpha,   key = random_uniform(key, (N, N, N),     -1.0, 1.0)
        gamma,   key = random_uniform(key, (N, N, N, N),  -1.0, 1.0)
        F_amp,   key = random_uniform(key, (N,),          0.5, 1.0)
        F_omega, key = random_uniform(key, (1,),          2.0, 10.0)
        
        return cls(N, m, c, k, alpha, gamma, F_amp, F_omega, name="from_random")

    @classmethod
    def from_example(cls, N) -> "Model":
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
            m     = jnp.array([1.0, 2.0, 3.0, 4.0])
            c     = jnp.array([2.0 * 0.05 * 5.0, 2.0 * 0.05 * 8.0, 2.0 * 0.06 * 10.0, 2.0 * 0.06 * 12.0])
            k     = jnp.array([[10.0, 1.0, 0.5, 0],
                            [1.0,12.0, 1.5, 1],
                            [0.5, 1.5,13.0, 1],
                            [0,   1,   1,   14]])
            alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.5, 1.5, 2],
                                                            [0.5, 0.0, 1.5, 2],
                                                            [1.5, 1.5, 0.0, 2],
                                                            [2,   2,   2,   0]]))
            gamma = jnp.zeros((N, N, N, N))
            F_amp = jnp.array([15.0, 10.0, 5.0, 2.0])  # Fixed to have 4 elements
            F_omega = jnp.array([1.0])
        else:
            raise ValueError("N must be 1, 2, 3, 4 or 5")
        
        return cls(N, m, c, k, alpha, gamma, F_amp, F_omega, name="from_example")
    
    @classmethod
    def from_file(cls, path:str) -> "Model":
        raise NotImplementedError("Loading from file is not implemented yet.")
    
    # --------------------------------------------------- helpers
    def __post_init__(self):
        self._calc_eigenfrequencies()
        self._non_dimensionalise()
        self._build_rhs()
        
    def _calc_eigenfrequencies(self):
        M_inv_K = self.k / self.m[:, None]          # (N,N)
        eigvals, _ = jnp.linalg.eig(M_inv_K)
        idx       = jnp.argsort(eigvals)
        self.omega_0 = jnp.sqrt(jnp.abs(eigvals[idx]))
    
    def _non_dimensionalise(self):
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
        
        omega_0_hat = self.omega_0 / omega_ref
        
        self.non_dimensionalised_model = NonDimensionalisedModel(
            omega_ref=omega_ref,  
            x_ref=x_ref,
            N=self.N,
            Q=Q,
            kappa=kappa,
            alpha=alpha,
            gamma=gamma,
            F_amp_hat=F_amp_hat,
            F_omega_hat=F_omega_hat,
            omega_0_hat=omega_0_hat,
        )

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

    def _get_steady_state_t_end(self) -> float:
        t_end_dl = 8 / jnp.max(self.non_dimensionalised_model.zeta * self.non_dimensionalised_model.eigenfrequencies_rad)
        return t_end_dl * self.non_dimensionalised_model.T0
            
    def _get_steady_state(self, q, v, discard_frac):
        n_steps = q.shape[1]
        
        q_steady = q[:, int(discard_frac * n_steps):]
        q_steady = jnp.max(jnp.abs(q_steady), axis=1)
        v_steady = v[:, int(discard_frac * n_steps):]
        v_steady = jnp.max(jnp.abs(v_steady), axis=1)
        return q_steady, v_steady
        
    def _steady_state_event(self, t, state, args, **kwargs) -> jax.Array:
        del kwargs
        t_theoretical_steady_state = self._get_steady_state_t_end()
        
        return t > t_theoretical_steady_state
        
    # --------------------------------------------------- internal solver
    def _solve_rhs(
        self,
        F_omega_hat: jax.Array,
        F_amp_hat: jax.Array,
        y0_hat: jax.Array,
        tau_end: float,
        n_steps: int,
        steady_state: bool = False,
    ) -> jax.Array:
        
        if steady_state:
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=jnp.inf,
                dt0=None,
                max_steps=None,
                y0=y0_hat,
                event=diffrax.Event(cond_fn=self._steady_state_event),
                adjoint=diffrax.ImplicitAdjoint(),
                progress_meter=diffrax.TqdmProgressMeter(),
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-7),
                args=(F_omega_hat, F_amp_hat),
            )
            t = sol.ts
            q = sol.ys[:, :self.N]
            v = sol.ys[:, self.N:]
            
        else:
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=tau_end,
                dt0=None,
                max_steps=400096,
                y0=y0_hat,
                progress_meter=diffrax.TqdmProgressMeter(),
                saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, tau_end, n_steps)),
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
                args=(F_omega_hat, F_amp_hat),
            )
            t = sol.ts
            q = sol.ys[:, : self.N]
            v = sol.ys[:, self.N :]
        
        return t, q, v
    
    # --------------------------------------------------- public wrappers
    
    def time_response(
        self,
        y0: jax.Array,
        n_steps: int,
        t_end: float = None,
        F_amp: jax.Array = None,
        F_omega_hz: jax.Array = None,
    ) -> jax.Array:
        
        if t_end is None:
            tau_end = float(self._get_steady_state_t_end()) * 2
        else:
            tau_end = self.non_dimensionalised_model.omega_ref * t_end
        if F_amp is None:
            F_amp = self.F_amp
        if F_omega_hz is None:
            F_omega_hz = self.F_omega / (2 * np.pi)
        
        F_omega_hat = jnp.atleast_1d(F_omega_hz) *  2 * np.pi / self.non_dimensionalised_model.omega_ref
        F_amp_hat = jnp.atleast_1d(F_amp) / (self.m * self.non_dimensionalised_model.omega_ref**2 * self.non_dimensionalised_model.x_ref)
                
        def solve_rhs(F_omega_hat, F_amp_hat):
            return self._solve_rhs(F_omega_hat, F_amp_hat, y0, t_end_dl, n_steps, steady_state=False)
        
        t_dl, q_dl, v_dl = jax.vmap(solve_rhs, in_axes=(0, None))(F_omega_hat, F_amp_hat)
        q_dl,v_dl = jnp.abs(q_dl), jnp.abs(v_dl)
        
        return t_dl, q_dl, v_dl
    
    def frequency_response(
        self,
        F_omega: jax.Array = None,
        F_omega_hat: jax.Array = None,
        F_amp : jax.Array = None,
        F_amp_hat: jax.Array = None,
        y0: jax.Array = None,
        y0_hat: jax.Array = None,
        t_end: float = None,
        tau_end: float = None,
        n_steps: int = None,
        discard_frac: float = None,
    ) -> tuple:
        
        if F_omega and F_omega_hat:
            raise ValueError("Either F_omega or F_omega_hat must be provided, not both.")
        
        if F_omega_hat is None:
            if F_omega is None:
                raise ValueError("Either F_omega or F_omega_hat must be provided.")
            F_omega_hat = F_omega / self.non_dimensionalised_model.omega_ref
            
        if F_amp_hat is None:
            if F_amp is None:
                F_amp = self.F_amp
            F_amp_hat = F_amp / (self.m * self.non_dimensionalised_model.omega_ref**2 * self.non_dimensionalised_model.x_ref)

        if y0_hat and y0:
            raise ValueError("Either y0 or y0_hat must be provided, not both.")
            
        if y0_hat is None:
            if y0 is None:
                y0 = jnp.zeros(2 * self.N)
            
            q0 = y0[:self.N] / self.non_dimensionalised_model.x_ref
            v0 = y0[self.N:] / (self.non_dimensionalised_model.x_ref * self.non_dimensionalised_model.omega_ref)
            y0_hat = jnp.concatenate([q0, v0])
            
        if tau_end and t_end:
            raise ValueError("Either t_end or tau_end must be provided, not both.")
            
        if tau_end is None:
            if t_end is None:
                t_end = float(self._get_steady_state_t_end())
            tau_end = self.non_dimensionalised_model.omega_ref * t_end    
            
        if n_steps is None:
            n_steps = 4000        
            
        if discard_frac is None:
            discard_frac = 0.8
            
        def solve_rhs(F_omega_hat, F_amp_hat):
            return self._solve_rhs(F_omega_hat, F_amp_hat, y0_hat, tau_end, n_steps, steady_state=False)

        tau, q, v = jax.vmap(solve_rhs, in_axes=(0, None))(F_omega_hat, F_amp_hat)
        q_st, v_st = self._get_steady_state(q, v, discard_frac)

        q_st_total = jnp.sum(q_st, axis=1)
        
        return F_omega_hat, q_st, q_st_total, v_st
    
    def force_sweep(
        self,
        y0: jax.Array,
        n_steps: int,
        discard_frac: float,
        F_amp: jax.Array,
        F_amp_levels: jax.Array,
        F_omega_hz: jax.Array,
        t_end: float = None,
    ) -> tuple:
        
        if t_end is None:
            t_end = float(self._get_steady_state_t_end())
        else:
            t_end = t_end / self.non_dimensionalised_model.T0
        
        F_omega_rad_dl = jnp.atleast_1d(F_omega_hz) * 2 * np.pi * self.non_dimensionalised_model.T0
        F_amp_dl_base = jnp.atleast_1d(F_amp) * self.non_dimensionalised_model.T0**2 / (self.m * self.non_dimensionalised_model.Q0)
        
        q_st_forces_dl = []
        
        def solve_rhs(F_omega, F_amp):
            return self._solve_rhs(F_omega, F_amp, y0, t_end, n_steps)
        
        for level in F_amp_levels:
            F_amp_dl = F_amp_dl_base * level
            
            t_dl, q_dl, v_dl = jax.vmap(solve_rhs, in_axes=(0, None))(F_omega_rad_dl, F_amp_dl)
            
            q_st_dl, v_st_dl = self._get_steady_state(q_dl, v_dl, discard_frac)
            q_st_forces_dl.append(q_st_dl)

        amplitude_responses_forces = jnp.stack(q_st_forces_dl, axis=0)
            
        return F_omega_rad_dl, amplitude_responses_forces
    
    def __repr__(self):
        return f"Model(N={self.N}, name={self.name})"
    
    def __str__(self):
        return (
            f"Model(N={self.N}, name={self.name})\n"
            f"Masses (m):\n{self.m}\n\n"
            f"Damping Coefficients (c):\n{self.c}\n\n"
            f"Stiffness Matrix (k):\n{self.k}\n\n"
            f"Nonlinear Coefficients (alpha):\n{self.alpha}\n\n"
            f"Nonlinear Coefficients (gamma):\n{self.gamma}\n\n"
            f"Force Amplitudes (F_amp):\n{self.F_amp}\n\n"
            f"Force Frequencies (F_omega):\n{self.F_omega}\n"
        )