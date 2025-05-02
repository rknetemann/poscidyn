import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use the CPU backend

import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# right-hand side  y = [q, v]  ->  dy/dt = [v, a(q,v,t)]
# ----------------------------------------------------------------------
def rhs(t, y, args):
    """
    Parameters
    ----------
    t    : scalar time
    y    : (2N,)  = [q_1 … q_N  v_1 … v_N]
    args : (m, c, K, A, G, f_drive, omega_d)
           m        (N,)            diagonal masses (positive!)
           c        (N,)
           K        (N,N)           linear stiffness
           A        (N,N,N)         quadratic stiffness
           G        (N,N,N,N)       cubic stiffness
           f_drive  (N,)            force amplitude for each dof
           omega_d  scalar          drive frequency
    """
    m, c, K, A, G, f_drive, omega_d = args
    N = m.shape[0]

    q, v = y[:N], y[N:]                      # split state vector

    # external drive ----------------------------------------------------
    F_drive = f_drive * jnp.cos(omega_d * t)          # (N,)

    # internal forces ---------------------------------------------------
    F_damp  = c * v                                   # (N,)
    F_lin   = jnp.einsum('ij,j->i',       K, q)        # K q
    F_quad  = jnp.einsum('ijk,j,k->i',   A, q, q)      # α q²
    F_cub   = jnp.einsum('ijkl,j,k,l->i', G, q, q, q)  # γ q³

    # acceleration ------------------------------------------------------
    a = (F_drive - F_damp - F_lin - F_quad - F_cub) / m

    return jnp.concatenate([v, a])                    # (2N,)


# ----------------------------------------------------------------------
# integrate & return time series of displacements only
# ----------------------------------------------------------------------
@eqx.filter_jit          # keeps solver adaptive inside the jit
def solve(t_span=(0.0, 100.0), nt=2001):
    N   = 3
    key = jax.random.PRNGKey(0)

    # problem parameters ------------------------------------------------
    m       = 1.0 + jax.random.uniform(key, (N,))     # random masses > 1
    c       = 0.02 * jnp.ones(N)
    K       = jnp.diag(jnp.array([4., 5., 6.]))
    A       = jnp.zeros((N, N, N))
    G       = jnp.zeros((N, N, N, N))
    f_drive = jnp.array([1.0, 0.0, 0.0])              # drive 1st dof only
    omega_d = 3.0

    args = (m, c, K, A, G, f_drive, omega_d)

    # initial state y0 = [q(0), v(0)]
    y0 = jnp.zeros(2 * N)

    # times at which to save ------------------------------------------------
    ts = jnp.linspace(*t_span, nt)

    # save only the displacement block (the first N entries of y)
    saveat = SaveAt(ts=ts, fn=lambda t, y, args: y[:N])

    sol = diffeqsolve(
        ODETerm(rhs),
        Tsit5(),
        t0=t_span[0],
        t1=t_span[1],
        dt0=0.3,                  # reasonable initial step
        y0=y0,
        args=args,
        saveat=saveat             # we get an (nt, N) array back
    )
    return sol.ts, sol.ys         # shapes: (nt,)  (nt, N)


# ----------------------------------------------------------------------
# run and plot
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ts, q = map(jax.device_get, solve())   # move data to host for Matplotlib

    for i in range(q.shape[1]):
        plt.plot(ts, q[:, i], label=f"Mode {i+1}")

    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.title("Displacement of modes over time")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
