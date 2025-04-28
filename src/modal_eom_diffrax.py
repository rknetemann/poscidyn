from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import diffrax                              # pip install diffrax

Array = jax.Array


# --------------------------------------------------------------------------------------
# utilities ----------------------------------------------------------------------------
def _random_uniform(key, shape, lo, hi):
    key, sub = jax.random.split(key)
    return jax.random.uniform(sub, shape, minval=lo, maxval=hi), key


# --------------------------------------------------------------------------------------
# main class ---------------------------------------------------------------------------
@dataclass(eq=False) 
class ModalEOM:
    N:             int
    c:             Array          # (N,)
    k:             Array          # (N, N)
    alpha:         Array          # (N, N, N)
    gamma:         Array          # (N, N, N, N)
    f:             Array          # (N,)
    name:          str = "modal_system"

    # private field to store the jit-compiled RHS
    rhs_jit: callable = field(init=False, repr=False)

    # ------------------------------------------------------------------ constructors --
    @classmethod
    def random(cls, N: int, seed: int = 0) -> "ModalEOM":
        key = jax.random.PRNGKey(seed)
        c,       key = _random_uniform(key, (N,),            0.0, 1.0)
        k,       key = _random_uniform(key, (N, N),          0.0, 1.0)
        alpha,   key = _random_uniform(key, (N, N, N),      -1.0, 1.0)
        gamma,   key = _random_uniform(key, (N, N, N, N),   -1.0, 1.0)
        f,       key = _random_uniform(key, (N,),           -1.0, 1.0)
        return cls(N, c, k, alpha, gamma, f, name="random")

    @classmethod
    def example(cls) -> "ModalEOM":
        N = 2
        c     = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
        k     = jnp.array([[10.0, 1.0],
                           [ 1.0,12.0]])
        alpha = jnp.zeros((N, N, N)).at[0].set(jnp.array([[0.0, 0.5],
                                                          [0.5, 0.0]]))
        gamma = jnp.zeros((N, N, N, N))
        f     = jnp.array([1.0, 0.5])
        return cls(N, c, k, alpha, gamma, f, name="example")

    # ------------------------------------------------------------- post-init & helpers --
    def __post_init__(self):
        self._build_rhs()
        
    def __hash__(self):
        return id(self)

    def _build_rhs(self):
        @jax.jit
        def rhs(t, y, args):
            ω_d, = args
            q, v = jnp.split(y, 2)

            a = (- self.c * v
                 - self.k @ q
                 - jnp.einsum("ijk,j,k->i",   self.alpha, q, q)
                 - jnp.einsum("ijkl,j,k,l->i", self.gamma, q, q, q)
                 + self.f * jnp.cos(ω_d * t))

            return jnp.concatenate([v, a])

        self.rhs_jit = rhs

    # ---------------------------------------------------------------- eigenfrequencies --
    def eigenfrequencies(self) -> np.ndarray:
        """
        Return the undamped natural frequencies ω_n (sorted ascending).
        """
        eigvals, _ = np.linalg.eig(np.asarray(self.k))
        idx        = np.argsort(eigvals)
        return np.sqrt(eigvals[idx])

    # ---------------------------------------------------------------- ODE integration --
    @partial(jax.jit, static_argnames=("self",
                                       "t_end", "n_steps", "discard_frac"))
    def _steady_state_amp_vec(self,
                              omega_d: Array,   # (batch,)
                              y0: Array,
                              t_end: float,
                              n_steps: int,
                              discard_frac: float) -> Array:
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps))

        def _solve_single(ω):
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(self.rhs_jit),
                solver,
                t0   = 0.0,
                t1   = t_end,
                dt0  = 1e-3,
                max_steps=50000,
                y0   = y0,
                saveat = saveat,
                stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
                args  = (ω,))
            q1 = sol.ys[:, 0]
            return jnp.max(jnp.abs(q1[int(discard_frac * q1.size):]))

        return jax.vmap(_solve_single)(omega_d)

    # -------------------------------------------------- public convenience wrappers ----
    def steady_state_amp(self,
                         omega_d: float | Array,
                         y0: np.ndarray,
                         t_end: float,
                         n_steps: int = 2000,
                         discard_frac: float = 0.5):
        ω   = jnp.atleast_1d(omega_d)
        amp = self._steady_state_amp_vec(ω, jnp.asarray(y0),
                                         t_end, n_steps, discard_frac)
        return float(amp[0]) if np.isscalar(omega_d) else np.asarray(amp)

    def frequency_response(self,
                           omega_d_min: float,
                           omega_d_max: float,
                           n_omega_d: int,
                           y0: np.ndarray,
                           t_end: float,
                           n_steps: int = 2000,
                           discard_frac: float = 0.5):
        ω_grid = jnp.linspace(omega_d_min, omega_d_max, n_omega_d)
        amps   = self._steady_state_amp_vec(ω_grid, jnp.asarray(y0),
                                            t_end, n_steps, discard_frac)
        return np.asarray(ω_grid), np.asarray(amps)