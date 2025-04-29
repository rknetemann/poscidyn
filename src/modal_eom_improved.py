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
    def from_example(cls) -> "ModalEOM":
        N = 2
        c     = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
        k     = jnp.array([[10.0, 1.0],
                           [ 1.0,12.0]])
        alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.5],
                                                          [0.5, 0.0]]))
        gamma = jnp.zeros((N, N, N, N))
        f_amp = jnp.array([1.0, 0.5]) 
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

    def __hash__(self):
        return id(self)

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

            return jnp.concatenate([v, a])

        self.rhs_jit = rhs
        
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
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=t_end,
            dt0=None,
            max_steps=500_000,
            y0=y0,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            args=(f_omega, f_amp),
        )

        qs = sol.ys[:, : self.N]
        return qs
    
    # --------------------------------------------------- public wrappers
    def eigenfrequencies(self) -> jax.Array:
        eigvals, _ = jnp.linalg.eig(jnp.asarray(self.k))
        idx = jnp.argsort(eigvals)
        return jnp.abs(jnp.sqrt(eigvals[idx]))
    
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
        
        amplitude_responses = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega, f_amp)
        amplitude_responses = jnp.abs(amplitude_responses)
        
        return amplitude_responses
    
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
        
        amplitude_responses = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega, f_amp)
        amplitude_responses_steady_state = amplitude_responses[:, int(discard_frac * n_steps):]
        max_amplitudes = jnp.max(jnp.abs(amplitude_responses_steady_state), axis=1)
        
        return f_omega, max_amplitudes
    
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
        
        amplitude_responses_forces = []
        
        for amp in f_amp:
            # compute response for each frequency at the current force amplitude vector
            amplitude_responses = jax.vmap(solve_rhs, in_axes=(0, None))(f_omega, amp)
            # discard transient
            idx = int(discard_frac * n_steps)
            steady = amplitude_responses[:, idx:]
            # max steady‐state amplitude for each ω
            max_amplitudes = jnp.max(jnp.abs(steady), axis=1)
            amplitude_responses_forces.append(max_amplitudes)
        # shape: (n_forces, n_omegas)
        amplitude_responses_forces = jnp.stack(amplitude_responses_forces, axis=0)
            
        return f_omega, amplitude_responses_forces

