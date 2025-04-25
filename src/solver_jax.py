import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

def rhs_jax(state, t, c, k, alpha, gamma, f, omega_d):
    """
    JAX-compatible RHS for NormalForm dynamics.
    state: [q, v] concatenated, shape (2*N,)
    t: scalar time
    c, k, alpha, gamma, f: JAX arrays of appropriate shapes
    omega_d: drive frequency
    """
    N = c.shape[0]
    q = state[:N]
    v = state[N:]
    a_lin = - jnp.matmul(k, q)
    a_quad = - jnp.einsum('ijk,j,k->i', alpha, q, q)
    a_cub  = - jnp.einsum('ijkl,j,k,l->i', gamma, q, q, q)
    a_forc = f * jnp.cos(omega_d * t)
    a = a_lin + a_quad + a_cub + a_forc
    return jnp.concatenate([v, a])

def steady_state_amp_jax(c, k, alpha, gamma, f, y0, omega_d, ts, discard_idx):
    """
    Compute steady-state amplitude for a single omega_d using RK4 over given time grid.
    ts: array of times, shape (n_steps,)
    discard_idx: index to start steady-state sampling
    """
    # time step from grid
    dt = ts[1] - ts[0]

    def step(y, t):
        k1 = rhs_jax(y, t, c, k, alpha, gamma, f, omega_d)
        k2 = rhs_jax(y + dt/2 * k1, t + dt/2, c, k, alpha, gamma, f, omega_d)
        k3 = rhs_jax(y + dt/2 * k2, t + dt/2, c, k, alpha, gamma, f, omega_d)
        k4 = rhs_jax(y + dt * k3, t + dt, c, k, alpha, gamma, f, omega_d)
        return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # integrate over time grid, collect states
    def scan_fn(y, t):
        y_new = step(y, t)
        return y_new, y_new
    ys = lax.scan(scan_fn, y0, ts)[1]
    # include initial state and align lengths
    ys = jnp.vstack([y0, ys[:-1]])
    q1 = ys[:, 0]
    # mask using dynamic discard_idx to avoid dynamic slicing
    n = q1.shape[0]
    idxs = jnp.arange(n)
    masked = jnp.where(idxs >= discard_idx, jnp.abs(q1), -jnp.inf)
    return jnp.max(masked)

# Vectorize and JIT the steady-state amplitude over omega grid (map over omega_d)
steady_state_amps_jax = jit(vmap(steady_state_amp_jax,
                                 in_axes=(None, None, None, None, None, None, 0, None, None)))
