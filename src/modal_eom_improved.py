from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import diffrax

from utils.random import random_uniform

@dataclass(eq=False)
class ModalEOM:
    N:     int
    c:     jax.Array            # (N,)
    k:     jax.Array            # (N,N)
    alpha: jax.Array            # (N,N,N)
    gamma: jax.Array            # (N,N,N,N)
    f:     jax.Array            # (N,)
    name:  str = "modal_system"

    rhs_jit: callable = field(init=False, repr=False)

    # --------------------------------------------------- constructors
    @classmethod
    def from_random(cls, N: int, seed: int = 0) -> "ModalEOM":
        key = jax.random.PRNGKey(seed)
        c,       key = random_uniform(key, (N,),           0.0, 1.0)
        k,       key = random_uniform(key, (N, N),         0.0, 1.0)
        alpha,   key = random_uniform(key, (N, N, N),     -1.0, 1.0)
        gamma,   key = random_uniform(key, (N, N, N, N),  -1.0, 1.0)
        f,       key = random_uniform(key, (N,),          -1.0, 1.0)
        return cls(N, c, k, alpha, gamma, f, name="from_random")

    @classmethod
    def from_example(cls) -> "ModalEOM":
        N = 2
        c     = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
        k     = jnp.array([[10.0, 1.0],
                           [ 1.0,12.0]])
        alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.5],
                                                          [0.5, 0.0]]))
        gamma = jnp.zeros((N, N, N, N))
        f     = jnp.array([15.0, 0.5])     # drive only mode 1
        return cls(N, c, k, alpha, gamma, f, name="from_example")
    
    @classmethod
    def from_duffing(cls) -> "ModalEOM":
        N = 2
        c     = jnp.array([0.1, 0.1])
        k     = jnp.array([[1.0, 0.0],
                           [0.0, 1.0]])
        alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.5],
                                                          [0.5, 0.0]]))
        gamma = jnp.zeros((N, N, N, N))
        f     = jnp.array([1.0, 1.0])
        return cls(N, c, k, alpha, gamma, f, name="from_duffing")
    
    # --------------------------------------------------- helpers
    
    def __post_init__(self):
        self._build_rhs()

    def __hash__(self):
        return id(self)

    def _build_rhs(self):
        @jax.jit
        def rhs(t, y, args):
            ω_d, f_amp = args      # drive frequency and force amplitude
            q, v = jnp.split(y, 2)

            a = (- self.c * v
                 - self.k @ q
                 - jnp.einsum("ijk,j,k->i",    self.alpha, q, q)
                 - jnp.einsum("ijkl,j,k,l->i", self.gamma, q, q, q)
                 + self.f.at[0].set(f_amp) * jnp.cos(ω_d * t))

            return jnp.concatenate([v, a])

        self.rhs_jit = rhs
        
    # --------------------------------------------------- internal solvers
    @partial(jax.jit, static_argnames=("self", "y0", "t_end", "n_steps"))
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
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=t_end,
            dt0=1e-1,
            max_steps=500_000,
            y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            args=(f_omega, f_amp),
        )
        q1 = sol.ys[:, 0]
        
        return q1
    
    # --------------------------------------------------- public wrappers
    def eigenfrequencies(self) -> jax.Array:
        eigvals, _ = jnp.linalg.eig(jnp.asarray(self.k))
        idx = jnp.argsort(eigvals)
        return jnp.abs(jnp.sqrt(eigvals[idx]))
    
    @partial(jax.jit, static_argnames=("self", "t_end", "n_steps"))
    def time_response(
        self,
        f_omega: jax.Array,
        f_amp: jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
    ) -> jax.Array:
        
        def solve_rhs(f_omega, f_amp):
            return self._solve_rhs(f_omega, f_amp, y0, t_end, n_steps)
        
        amplitude_responses = jax.vmap(solve_rhs)(f_omega, f_amp)
        
        return amplitude_responses
    
    @partial(jax.jit, static_argnames=("self", "t_end", "n_steps", "discard_frac"))
    def frequency_response(
        self,
        f_omega: jax.Array,
        f_amp : jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        discard_frac: float,
    ) -> tuple:
        
        f_omega = jnp.atleast_1d(f_omega)
        f_amp = jnp.atleast_1d(f_amp)
        
        def solve_rhs(f_omega, f_amp):
            return self._solve_rhs(f_omega, f_amp, y0, t_end, n_steps)
        
        amplitude_responses = jax.vmap(solve_rhs)(f_omega, f_amp)
        amplitude_responses_steady_state = amplitude_responses[:, int(discard_frac * n_steps):]
        max_amplitudes = jnp.max(jnp.abs(amplitude_responses_steady_state), axis=1)
        
        return f_omega, max_amplitudes

