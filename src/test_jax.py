import jax.numpy as jnp
from diffrax import ODETerm, Tsit5, diffeqsolve, SaveAt
import matplotlib.pyplot as plt
import jax
import equinox as eqx

def rhs(t, state, args):
    """
    Parameters
    ----------
    t : float
    state : (2N,)  concatenated [q, v]
    args : tuple   (m, c, K, A, G, f_drive, omega_d)
    """
    m, c, K, A, G, f_drive, omega_d = args      # shapes: (N,) (N,) (N,N) (N,N,N) (N,N,N,N) (N,)
    N = m.shape[0]

    q = state[:N]           # displacements
    v = state[N:]           # velocities

    # --- external forcing ----------------------------------
    F_drive = f_drive * jnp.cos(omega_d * t)     # (N,)

    # --- internal forces -----------------------------------
    F_damp   = c * v                                           # (N,)
    F_lin    = jnp.einsum('ij,j->i',     K, q)                 # K q
    F_quad   = jnp.einsum('ijk,j,k->i', A, q, q)               # α q²
    F_cubic  = jnp.einsum('ijkl,j,k,l->i', G, q, q, q)         # γ q³

    # --- acceleration  m a = external − internal ------------
    a = (F_drive - F_damp - F_lin - F_quad - F_cubic) / m      # (N,)

    # derivative of the whole state vector ------------------
    return jnp.concatenate([v, a])          # length 2N

@eqx.filter_jit     
def solve():
    N = 3
    key = jax.random.PRNGKey(0)
    m      = 1.0 + jax.random.uniform(key,   (N,))               # positive masses
    c      = 0.02 * jnp.ones(N)
    K      = jnp.diag(jnp.array([4., 5., 6.]))                   # any N×N matrix
    A      = jnp.zeros((N, N, N))                                # fill as needed
    G      = jnp.zeros((N, N, N, N))
    f_drive = jnp.array([1.0, 0.0, 0.0])
    omega_d = 1.2

    args = (m, c, K, A, G, f_drive, omega_d)

    y0 = jnp.concatenate([jnp.zeros(N), jnp.zeros(N)])           # q(0)=0, v(0)=0

    term   = ODETerm(rhs)
    solver = Tsit5()
    ts     = jnp.linspace(0.0, 50.0, 2001)

    out = diffeqsolve(term, solver, t0=0.0, t1=50.0,
                    dt0=0.1, y0=y0, args=args,
                    saveat=SaveAt(ts=ts))
    
    return out

sol = solve()
plt.plot(sol.ts, sol.ys[0, :], label='Mode 1')
plt.plot(sol.ts, sol.ys[1, :], label='Mode 2')
plt.plot(sol.ts, sol.ys[2, :], label='Mode 3')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement of Modes Over Time')
plt.legend()
plt.show()