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
    N:          int
    zeta:       jax.Array            # (N,)
    K:          jax.Array            # (N,)
    A:          jax.Array            # (N,)
    G:          jax.Array            # (N,N,N)
    F_amp:      jax.Array            # (N,)
    F_omega:    jax.Array            # (1,)

    T0:         float
    Q0:         float
    
    eigenfrequencies_rad: jax.Array
    
    name:      str = "modal_system"
    
@dataclass(eq=False)
class Model:       
    N:         int
    m:         jax.Array            # (N,)
    c:         jax.Array            # (N,)
    k:         jax.Array            # (N,)
    alpha:     jax.Array            # (N,N,N)
    gamma:     jax.Array            # (N,N,N,N)
    f_amp:     jax.Array            # (N,)
    f_omega_rad:   jax.Array            # (1,)
    
    name:      str = "modal_system"
    
    eigenfrequencies_rad: jax.Array = field(init=False, repr=False)
    
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
        f_amp,   key = random_uniform(key, (N,),          0.5, 1.0)
        f_omega, key = random_uniform(key, (1,),          2.0, 10.0)
        
        return cls(N, m, c, k, alpha, gamma, f_amp, f_omega, name="from_random")

    @classmethod
    def from_example(cls, N) -> "Model":
        if N == 1:
            m     = jnp.array([1.0])
            c     = jnp.array([2.0 * 0.01 * 5.0])
            k     = jnp.array([[10.0]])
            alpha = jnp.zeros((N, N, N))
            gamma = jnp.zeros((N, N, N, N)).at[0].set(jnp.array([[0.1]]))
            f_amp = jnp.array([15.0]) 
            f_omega = jnp.array([1.0])
        elif N == 2:
            m     = jnp.array([1.0, 2.0])
            c     = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
            k     = jnp.array([[10.0, 0],
                               [0, 12.0]])
            alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.0],
                                                            [0.0, 0.0]]))
            gamma = jnp.zeros((N, N, N, N)).at[0].set(jnp.array([[0.1, 0.0],
                                                            [0.0, 0.0]]))
            f_amp = jnp.array([15.0, 4.0]) 
            f_omega = jnp.array([1.0])
        elif N == 3:
            m     = jnp.array([0.204, 0.195, 0.183])
            c     = jnp.array([1.0, 1.0, 1.0])
            k     = jnp.array([[5.78, 0.0, 0],
                               [0.0,16.5, 0],
                               [0,   0,   34.6]])
            alpha = jnp.zeros((N, N, N))
            gamma = jnp.zeros((N, N, N, N))
            f_amp = jnp.array([15.0, 10.0, 5.0]) 
            f_omega = jnp.array([1.0])
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
            f_amp = jnp.array([15.0, 10.0, 5.0, 2.0])  # Fixed to have 4 elements
            f_omega = jnp.array([1.0])
        else:
            raise ValueError("N must be 1, 2, 3, 4 or 5")
        
        return cls(N, m, c, k, alpha, gamma, f_amp, f_omega, name="from_example")
    
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
        self.eigenfrequencies_rad = jnp.sqrt(jnp.abs(eigvals[idx]))
    
    def _non_dimensionalise(self):
        T0 = 1.0 / jnp.max(self.eigenfrequencies_rad)
        Q0 = jnp.max(self.f_amp) / jnp.min(jnp.diag(self.k))
        
        eigenfrequencies_rad = self.eigenfrequencies_rad * T0
        
        zeta = self.c * T0 / (2.0 * self.m)
        K = self.k * T0**2 / self.m[:, None]
        A = self.alpha * Q0 * T0**2 / self.m[:, None, None]
        G = self.gamma * Q0**2 * T0**2 / self.m[:, None, None, None]
        F_amp = self.f_amp * T0**2 / (self.m * Q0)
        F_omega_rad = self.f_omega_rad * T0
        
        self.non_dimensionalised_model = NonDimensionalisedModel(
            N=self.N,
            zeta=zeta,
            K=K,
            A=A,
            G=G,
            F_amp=F_amp,
            F_omega=F_omega_rad,
            T0=float(T0),
            Q0=float(Q0),
            eigenfrequencies_rad=eigenfrequencies_rad,
        )

    def _build_rhs(self):
        def _rhs(t, state, args):
            f_omega_rad_dl, f_amp_dl = args
            q, v   = jnp.split(state, 2)

            a = (-2.0 * self.non_dimensionalised_model.zeta * v
                - jnp.matmul(self.non_dimensionalised_model.K, q)
                - jnp.einsum("ijk,j,k->i",    self.non_dimensionalised_model.A, q, q)
                - jnp.einsum("ijkl,j,k,l->i", self.non_dimensionalised_model.G, q, q, q)
                + f_amp_dl * jnp.cos(f_omega_rad_dl * t)
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
        f_omega_rad: jax.Array,
        f_amp: jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        steady_state: bool = False,
    ) -> jax.Array:
        
        f_omega_rad = jnp.atleast_1d(f_omega_rad)
        
        if steady_state:
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=jnp.inf,
                dt0=None,
                max_steps=None,
                y0=y0,
                event=diffrax.Event(cond_fn=self._steady_state_event),
                adjoint=diffrax.ImplicitAdjoint(),
                progress_meter=diffrax.TqdmProgressMeter(),
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-7),
                args=(f_omega_rad, f_amp),
            )
            t = sol.ts
            q = sol.ys[:, :self.N]
            v = sol.ys[:, self.N:]
            
        else:
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=t_end,
                dt0=None,
                max_steps=400096,
                y0=y0,
                progress_meter=diffrax.TqdmProgressMeter(),
                saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
                args=(f_omega_rad, f_amp),
            )
            t = sol.ts
            q = sol.ys[:, : self.N]
            v = sol.ys[:, self.N :]
        
        return t, q, v
    
    # --------------------------------------------------- public wrappers
    
    def eigenfrequencies(self, dimensionless=True) -> jax.Array:
        if dimensionless: 
            eigenfrequencies_rad = self.non_dimensionalised_model.eigenfrequencies_rad
        else:
            eigenfrequencies_rad = self.eigenfrequencies_rad 

        return eigenfrequencies_rad
    
    def quality_factors(self) -> jax.Array:
        eigenfreq = self.eigenfrequencies(dimensionless=False)
        Q = self.m * eigenfreq / self.c
        return Q
    
    def time_response(
        self,
        y0: jax.Array,
        n_steps: int,
        t_end: float = None,
        f_amp: jax.Array = None,
        f_omega_hz: jax.Array = None,
    ) -> jax.Array:
        
        if t_end is None:
            t_end_dl = float(self._get_steady_state_t_end()) * 2
        else:
            t_end_dl = t_end / self.non_dimensionalised_model.T0
        if f_amp is None:
            f_amp = self.f_amp
        if f_omega_hz is None:
            f_omega_hz = self.f_omega_rad / (2 * np.pi)
            
        
        f_omega_dl = jnp.atleast_1d(f_omega_hz) *  2 * np.pi * self.non_dimensionalised_model.T0
        f_amp_dl = jnp.atleast_1d(f_amp) * self.non_dimensionalised_model.T0**2 / (self.m * self.non_dimensionalised_model.Q0)
                
        def solve_rhs(f_omega_dl, f_amp_dl):
            return self._solve_rhs(f_omega_dl, f_amp_dl, y0, t_end_dl, n_steps, steady_state=False)
        
        t_dl, q_dl, v_dl = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega_dl, f_amp_dl)
        q_dl,v_dl = jnp.abs(q_dl), jnp.abs(v_dl)
        
        return t_dl, q_dl, v_dl
    
    def frequency_response(
        self,
        f_omega_hz: jax.Array,
        y0: jax.Array = None,
        n_steps: int = None,
        discard_frac: float = 0.8,
        t_end: float = None,
        f_amp : jax.Array = None,
    ) -> tuple:
        
        if y0 is None:
            y0 = jnp.zeros(2 * self.N)
        if n_steps is None:
            n_steps = len(f_omega_hz) * 5
        if t_end is None:
            t_end = self._get_steady_state_t_end()
        if f_amp is None:
            f_amp = self.f_amp
            
        f_omega_rad_dl = jnp.atleast_1d(f_omega_hz) *  2 * np.pi * self.non_dimensionalised_model.T0
        f_amp_dl = jnp.atleast_1d(f_amp) * self.non_dimensionalised_model.T0**2 / (self.m * self.non_dimensionalised_model.Q0)
        
        def solve_rhs(f_omega_rad_dl, f_amp_dl):
            return self._solve_rhs(f_omega_rad_dl, f_amp_dl, y0, t_end, n_steps, steady_state=False)
        
        t_dl, q_dl, v_dl = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega_rad_dl, f_amp_dl)
        q_st_dl, v_st_dl = self._get_steady_state(q_dl, v_dl, discard_frac)
        
        # t_dl, q_st_dl, v_st_dl = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega_rad_dl, f_amp_dl)
        # q_st_dl, v_st_dl = jnp.abs(q_st_dl), jnp.abs(v_st_dl)

        q_st_total_dl = jnp.sum(q_st_dl, axis=1)
        
        return f_omega_rad_dl, q_st_dl, q_st_total_dl, v_st_dl
    
    def force_sweep(
        self,
        y0: jax.Array,
        n_steps: int,
        discard_frac: float,
        f_amp: jax.Array,
        f_amp_levels: jax.Array,
        f_omega_hz: jax.Array,
        t_end: float = None,
    ) -> tuple:
        
        if t_end is None:
            t_end = float(self._get_steady_state_t_end())
        else:
            t_end = t_end / self.non_dimensionalised_model.T0
        
        f_omega_rad_dl = jnp.atleast_1d(f_omega_hz) * 2 * np.pi * self.non_dimensionalised_model.T0
        f_amp_dl_base = jnp.atleast_1d(f_amp) * self.non_dimensionalised_model.T0**2 / (self.m * self.non_dimensionalised_model.Q0)
        
        q_st_forces_dl = []
        
        def solve_rhs(f_omega, f_amp):
            return self._solve_rhs(f_omega, f_amp, y0, t_end, n_steps)
        
        for level in f_amp_levels:
            f_amp_dl = f_amp_dl_base * level
            
            t_dl, q_dl, v_dl = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega_rad_dl, f_amp_dl)
            
            q_st_dl, v_st_dl = self._get_steady_state(q_dl, v_dl, discard_frac)
            q_st_forces_dl.append(q_st_dl)

        amplitude_responses_forces = jnp.stack(q_st_forces_dl, axis=0)
            
        return f_omega_rad_dl, amplitude_responses_forces
    
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
            f"Force Amplitudes (f_amp):\n{self.f_amp}\n\n"
            f"Force Frequencies (f_omega_rad):\n{self.f_omega_rad}\n"
        )