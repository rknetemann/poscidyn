from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import diffrax

from utils.random import random_uniform

@dataclass(eq=False)
class ModalEOM:
    N:         int
    c:         jax.Array            # (N,)
    k:         jax.Array            # (N,N)
    alpha:     jax.Array            # (N,N,N)
    gamma:     jax.Array            # (N,N,N,N)
    f_amp:     jax.Array            # (N,)
    f_omega:   jax.Array            # (1,)
    
    name:      str = "modal_system"

    rhs_jit: callable = field(init=False, repr=False)

    # --------------------------------------------------- constructors
    @classmethod
    def from_random(cls, N: int, seed: int = 0) -> "ModalEOM":
        key = jax.random.PRNGKey(seed)
        c,       key = random_uniform(key, (N,),           0.0, 1.0)
        k,       key = random_uniform(key, (N, N),         0.0, 1.0)
        alpha,   key = random_uniform(key, (N, N, N),     -1.0, 1.0)
        gamma,   key = random_uniform(key, (N, N, N, N),  -1.0, 1.0)
        f_amp,   key = random_uniform(key, (N,),          -1.0, 1.0)
        f_omega, key = random_uniform(key, (1,),          0.0, 10.0)
        
        return cls(N, c, k, alpha, gamma, f_amp, f_omega, name="from_random")

    @classmethod
    def from_example(cls, N) -> "ModalEOM":
        if N == 1:
            c     = jnp.array([2.0 * 0.01 * 5.0])
            k     = jnp.array([[10.0]])
            alpha = jnp.zeros((N, N, N))
            gamma = jnp.zeros((N, N, N, N)).at[0].set(jnp.array([[0.1]]))
            f_amp = jnp.array([15.0]) 
            f_omega = jnp.array([1.0])
        elif N == 2:
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
            c     = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0, 2.0 * 0.03 * 10.0])
            k     = jnp.array([[10.0, 1.0, 0.5],
                            [1.0,12.0, 1.5],
                            [0.5, 1.5,13.0]])
            alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.5, 1.5],
                                                            [0.5, 0.0, 1.5],
                                                            [1.5, 1.5, 0.0]]))
            gamma = jnp.zeros((N, N, N, N))
            f_amp = jnp.array([15.0, 10.0, 5.0]) 
            f_omega = jnp.array([1.0])
        
        return cls(N, c, k, alpha, gamma, f_amp, f_omega, name="from_example")
    
    @classmethod
    def from_duffing(cls) -> "ModalEOM":
        N = 2
        c       = jnp.array([0.1, 0.1])
        k       = jnp.array([[1.0, 0.0],
                           [0.0, 1.0]])
        alpha   = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.5],
                                                          [0.5, 0.0]]))
        gamma   = jnp.zeros((N, N, N, N))
        f_amp   = jnp.array([1.0, 1.0])
        f_omega = jnp.array([1.0])
        return cls(N, c, k, alpha, gamma, f_amp, f_omega, name="from_duffing")
    
    # --------------------------------------------------- helpers
    def __post_init__(self):
        self._build_rhs()

    def _build_rhs(self):
        @jax.jit
        def rhs(t, state, args):
            f_omega, f_amp = args
            
            f_omega = jnp.atleast_1d(f_omega)
            f_amp   = jnp.atleast_1d(f_amp)
            
            q, v = jnp.split(state, 2)

            a = (- jnp.multiply(self.c, v) # (N,) * (N,) = (N,)
                 - jnp.matmul(self.k, q)   # (N,N) @ (N,) = (N,)
                 - jnp.einsum("ijk,j,k->i",    self.alpha, q, q)
                 - jnp.einsum("ijkl,j,k,l->i", self.gamma, q, q, q)
                 + jnp.multiply(f_amp, jnp.cos(f_omega * t))
                )

            return jnp.concatenate([v, a]) # The solver will integrate to get q and v

        self.rhs_jit = rhs
        
    def _get_steady_state(self, q, v, discard_frac):
        n_steps = q.shape[1]
        
        q_steady = q[:, int(discard_frac * n_steps):]
        q_steady = jnp.max(jnp.abs(q_steady), axis=1)
        v_steady = v[:, int(discard_frac * n_steps):]
        v_steady = jnp.max(jnp.abs(v_steady), axis=1)
        return q_steady, v_steady
        
    # --------------------------------------------------- internal solvers
    @partial(jax.jit, static_argnames=("self", "t_end", "n_steps"))
    def _solve_rhs(
        self,
        f_omega: jax.Array,
        f_amp: jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
    ) -> jax.Array:
        
        f_omega = jnp.atleast_1d(f_omega)
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.rhs_jit),
            #solver=diffrax.ImplicitEuler(root_finder=diffrax.VeryChord(rtol=1e-8, atol=1e-8)),
            #solver=diffrax.Kvaerno5(),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=t_end,
            dt0=None,
            max_steps=4096,
            y0=y0,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            args=(f_omega, f_amp),
        )
        
        ts = sol.ts
        qs = sol.ys[:, : self.N]
        vs = sol.ys[:, self.N :]
        return ts, qs, vs
    
    # --------------------------------------------------- public wrappers
    def eigenfrequencies(self) -> jax.Array:
        eigvals, _ = jnp.linalg.eig(jnp.asarray(self.k))
        idx = jnp.argsort(eigvals)
        return jnp.abs(jnp.sqrt(eigvals[idx]))
    
    def quality_factors(self) -> jax.Array:
        eigvals, _ = jnp.linalg.eig(jnp.asarray(self.k))
        idx = jnp.argsort(eigvals)
        eigvals = eigvals[idx]
        return 2 * jnp.pi * jnp.abs(eigvals) / (2 * jnp.pi * self.c)
    
    @partial(jax.jit, static_argnames=("self", "t_end", "n_steps"))
    def time_response(
        self,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        f_amp: jax.Array = None,
        f_omega: jax.Array = None,
    ) -> jax.Array:
        
        if f_amp is None:
            f_amp = self.f_amp
        if f_omega is None:
            f_omega = self.f_omega
        
        f_omega = jnp.atleast_1d(f_omega)
        f_amp = jnp.atleast_1d(f_amp)
        
        def solve_rhs(f_omega, f_amp):
            return self._solve_rhs(f_omega, f_amp, y0, t_end, n_steps)
        
        ts, qs, vs = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega, f_amp)
        qs = jnp.abs(qs)
        vs = jnp.abs(vs)
        
        return ts, qs, vs
    
    @partial(jax.jit, static_argnames=("self", "t_end", "n_steps", "discard_frac"))
    def frequency_response(
        self,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        discard_frac: float,
        f_omega: jax.Array,
        f_amp : jax.Array = None,
    ) -> tuple:
        
        if f_amp is None:
            f_amp = self.f_amp
            
        f_omega = jnp.atleast_1d(f_omega)
        f_amp = jnp.atleast_1d(f_amp)
        
        def solve_rhs(f_omega, f_amp):
            return self._solve_rhs(f_omega, f_amp, y0, t_end, n_steps)
        
        t, q, v = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega, f_amp)
        q_steady, v_steady = self._get_steady_state(q, v, discard_frac)
        
        return f_omega, q_steady, v_steady
    
    @partial(jax.jit, static_argnames=("self", "t_end", "n_steps", "discard_frac"))
    def force_sweep(
        self,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        discard_frac: float,
        f_amp : jax.Array,
        f_omega: jax.Array,
    ) -> tuple:
        
        f_omega = jnp.atleast_1d(f_omega)
        f_amp = jnp.atleast_1d(f_amp)
        
        def solve_rhs(f_omega, f_amp):
            return self._solve_rhs(f_omega, f_amp, y0, t_end, n_steps)
        
        qs_steady_forces = []
        
        for amp in f_amp:
            t, q, v = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega, amp)
            q_steady, v_steady = self._get_steady_state(q, v, discard_frac)
            qs_steady_forces.append(q_steady)

        amplitude_responses_forces = jnp.stack(qs_steady_forces, axis=0)
            
        return f_omega, amplitude_responses_forces

