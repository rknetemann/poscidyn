import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use the CPU backend

import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import ODETerm, Tsit5, SaveAt, diffeqsolve
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
@eqx.filter_jit
def solve(omega_d, f_drive_val=6.0, t_span=(0.0, 100.0), nt=2001):
    N   = 1                  
    key = jax.random.PRNGKey(0)
    # Duffing parameters: x″ + δx′ + αx + βx³ = γ cos(ω t)
    m       = jnp.array([1.0])                         
    delta   = 0.2                                      
    alpha   = 1.0                                      
    beta    = 5.0                                      
    gamma   = 8.0                                      
    c       = jnp.array([delta])
    K       = jnp.array([[alpha]])
    A       = jnp.zeros((N, N, N))
    G       = jnp.array([[[[beta]]]])                  
    f_drive = jnp.array([f_drive_val])

    eigenfrequencies = jnp.sqrt(K / m)                   
    print(f"Natural frequency: {jnp.array(eigenfrequencies)}")

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
        saveat=saveat,             # we get an (nt, N) array back
        max_steps=100000,        # max steps to avoid infinite loop
    )
    return sol.ts, sol.ys         # shapes: (nt,)  (nt, N)


# ----------------------------------------------------------------------
# run and plot
# ----------------------------------------------------------------------
if __name__ == "__main__":
    #ts, q = map(jax.device_get, solve(1.0))   # move data to host for Matplotlib

    # for i in range(q.shape[1]):
    #     plt.plot(ts, q[:, i], label=f"Mode {i+1}")

    # plt.xlabel("Time")
    # plt.ylabel("Displacement")
    # plt.title("Displacement of modes over time")
    # plt.legend(loc="upper right")
    # plt.tight_layout()
    # plt.show()

    # Frequency‐response sweep ------------------------------------------
    # omegas = np.linspace(13, 15.0, 2000)              # finer frequency grid
    # amps = []
    # for w in omegas:
    #     ts, q = map(jax.device_get, solve(w, t_span=(0.0, 500.0), nt=4001))
    #     discard = int(0.80 * len(ts))      
    #     amps.append(q[discard:, 0].max())

    # plt.figure()
    # plt.plot(omegas, amps, '-', linewidth=2)         # smooth line
    # plt.grid(True)
    # plt.xlabel("Drive frequency ω")
    # plt.ylabel("Steady‐state amplitude")
    # plt.title("Frequency response (steady‐state)")
    # plt.tight_layout()
    # plt.show()

    # 3D Frequency‐response sweep over forces
    omegas = np.linspace(13, 15.0, 2000)
    forces = [3, 4, 5, 6, 7, 8]
    amps_2d = []
    for f in forces:
        amps = []
        for w in omegas:
            ts, q = map(jax.device_get, solve(w, f, t_span=(0.0, 500.0), nt=4001))
            discard = int(0.80 * len(ts))
            amps.append(q[discard:, 0].max())
        amps_2d.append(amps)

    # meshgrid for ω vs f
    X, Y = np.meshgrid(omegas, forces)
    Z = np.array(amps_2d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # draw each force-response as a black line
    for idx, f in enumerate(forces):
        ax.plot(omegas, [f] * len(omegas), amps_2d[idx], color='k')

    ax.set_xlabel("Drive frequency ω")
    ax.set_ylabel("Force amplitude f")
    ax.set_zlabel("Steady‐state amplitude")
    plt.tight_layout()
    plt.show()
