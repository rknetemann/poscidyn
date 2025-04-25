# freq_response.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"          # run JAX on CPU

import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve


# --------------------------------------------------------------------
# right-hand side   y = [q, v]  ->  dy/dt = [v, a(q,v,t)]
# --------------------------------------------------------------------
def rhs(t, y, args):
    m, c, K, A, G, f_drive, omega = args
    N = m.shape[0]

    q, v = y[:N], y[N:]

    F_drive = f_drive * jnp.cos(omega * t)
    F_damp  = c * v
    F_lin   = jnp.einsum('ij,j->i',       K, q)
    F_quad  = jnp.einsum('ijk,j,k->i',   A, q, q)
    F_cub   = jnp.einsum('ijkl,j,k,l->i', G, q, q, q)

    a = (F_drive - F_damp - F_lin - F_quad - F_cub) / m
    return jnp.concatenate([v, a])


# --------------------------------------------------------------------
# integrate for a single drive frequency, return displacement history
# --------------------------------------------------------------------
@eqx.filter_jit                # keeps Diffrax adaptive
def solve_one(omega, *, t_span=(0., 200.), nt=4001):
    N   = 3
    key = jax.random.PRNGKey(0)

    # model parameters (keep the same for all ω) ----------------------
    m       = 1.0 + jax.random.uniform(key, (N,))
    c       = 0.02 * jnp.ones(N)
    K       = jnp.diag(jnp.array([4., 5., 6.]))
    A       = jnp.zeros((N, N, N))
    G       = jnp.zeros((N, N, N, N))
    f_drive = jnp.array([1.0, 0.0, 0.0])

    args = (m, c, K, A, G, f_drive, omega)

    y0 = jnp.zeros(2 * N)
    ts = jnp.linspace(*t_span, nt)
    saveat = SaveAt(ts=ts, fn=lambda t, y, args: y[:N])   # keep q only

    sol = diffeqsolve(
        ODETerm(rhs), Tsit5(),
        t0=t_span[0], t1=t_span[1],
        dt0=0.2,
        y0=y0, args=args,
        saveat=saveat
    )
    return sol.ts, sol.ys          # shapes (nt,), (nt, N)


# --------------------------------------------------------------------
# steady-state amplitude from the last `frac` fraction of the record
# --------------------------------------------------------------------
def _amplitude(q_hist, frac=0.10):
    start = int(q_hist.shape[0] * (1.0 - frac))
    q_ss  = q_hist[start:, :]                  # (nt_ss, N)
    amp   = 0.5 * (jnp.max(q_ss, axis=0) -
                   jnp.min(q_ss, axis=0))
    return amp                                 # (N,)


# --------------------------------------------------------------------
# main frequency sweep & print table
# --------------------------------------------------------------------
if __name__ == "__main__":
    omega_vec = jnp.linspace(0.5, 3.0, 20)       # rad s⁻¹
    amp_mat   = []

    for w in omega_vec:
        _, q = map(jax.device_get, solve_one(w))
        amp_mat.append(_amplitude(q))

    amp_mat = jnp.vstack(amp_mat)                # shape (nfreq, N)

    # -------- pretty printing ----------------------------------------
    header = "ω(rad/s)     A1      A2      A3"
    print(header)
    print("-" * len(header))
    for w, A in zip(omega_vec, amp_mat):
        print(f"{float(w):7.3f}   {float(A[0]):6.3f}  "
              f"{float(A[1]):6.3f}  {float(A[2]):6.3f}")
